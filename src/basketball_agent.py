"""
NBA Basketball Game Prediction Agent
======================================
Supervised ML model that predicts NBA game outcomes using:
  - Web scraping from stats.nba.com & ESPN
  - Last 10 game averages & win percentage
  - Days of rest since last game
  - Home / away performance splits
  - ELO rating system
  - Injury reports
  - Logistic Regression with evaluation via Accuracy, Log Loss, AUC-ROC,
    Backtesting, and Calibration
 
Install dependencies:
    pip install requests beautifulsoup4 pandas numpy scikit-learn python-dotenv
"""
 
import os
import json
import time
import datetime
import warnings
import requests
import numpy as np
import pandas as pd
 
from dotenv import load_dotenv
#reads env file and loads content into program environment
from bs4 import BeautifulSoup
#an html parser used to scrap the espn injury page by finding
specific elements from the webpages html
 
from sklearn.linear_model import LogisticRegression
# the actual model that the agent is using
from sklearn.preprocessing import StandardScaler
# makes features use the same scale so that the model does not
get confused from large differences in numbers
from sklearn.model_selection import train_test_split
# splits data into training and testing sets so you can evaluate how 
well the model performs on games it hasnt seen
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
 
warnings.filterwarnings("ignore")
load_dotenv()
 

# CONFIG

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")
ANTHROPIC_URL     = "https://api.anthropic.com/v1/messages"
MODEL             = "claude-sonnet-4-20250514"
 
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}
 
NBA_STATS_HEADERS = {
    **HEADERS,
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token":  "true",
    "Origin":             "https://www.nba.com",
    "Referer":            "https://www.nba.com/",
}
# ELO SYSTEM 
class EloSystem:
    """
    Tracks and updates ELO ratings for all NBA teams.
    Starting ELO = 1500. K-factor = 20.
    Home court advantage = +100 ELO points when computing expected score.
    """
 
    DEFAULT_ELO   = 1500
    K_FACTOR      = 20
    HOME_ADVANTAGE = 100
 
 # These are the tuning knobs of the agent, 1500 is starting point for each team
 # 20 is how much ratings move after each game, and 100 is the bonus given to home teams


    def __init__(self):
        self.ratings: dict[str, float] = {}

        #creates an empty directory to store ratings, teams are not added
        #until their first game. Before that they are assumed 1500 elo 
 
    def get(self, team: str) -> float:
        return self.ratings.get(team, self.DEFAULT_ELO)

        #this returns a teams current rating 
 
    def expected(self, team_a: float, team_b: float) -> float:
        return 1 / (1 + 10 ** ((team_b - team_a) / 400))

        #the core math, takes two ratings and returns a probability between 1 and 0.
 
    def update(self, winner: str, loser: str, home_team: str) -> None:
        r_w = self.get(winner)
        r_l = self.get(loser)

        #called after every game, upsets move ratings more than expected results. 
        #If a heavy underdog wins, both teams ratings shift dramatically, if the 
        #favored team wins, the shift is tiny. 


        # Apply home advantage to expected scores
        if home_team == winner:
            e_w = self.expected(r_w + self.HOME_ADVANTAGE, r_l)
        else:
            e_w = self.expected(r_w, r_l + self.HOME_ADVANTAGE)
        e_l = 1 - e_w
 
        self.ratings[winner] = r_w + self.K_FACTOR * (1 - e_w)
        self.ratings[loser]  = r_l + self.K_FACTOR * (0 - e_l)
 
    def win_probability(self, home_team: str, away_team: str) -> float:
        """Returns probability that the HOME team wins."""
        r_home = self.get(home_team) + self.HOME_ADVANTAGE
        r_away = self.get(away_team)
        return self.expected(r_home, r_away)

        #public facing predicition method, adds the home court bonus to the home 
        #teams ratings, then runs expected() to get a clean win probability percentage.

# WEBSCRAPER
class NBAScraper:
    """
    Scrapes game logs, injury reports, and team stats from
    stats.nba.com and ESPN. Falls back to synthetic data
    when endpoints are rate-limited or unavailable.
    """

    # NBA Stats API endpoint for team game logs
    GAME_LOG_URL = (
        "https://stats.nba.com/stats/teamgamelogs"
        "?Season={season}&SeasonType=Regular+Season&TeamID={team_id}"
    )

    # ESPN injury report
    ESPN_INJURIES_URL = "https://www.espn.com/nba/injuries"

    # NBA Stats team list
    TEAMS_URL = "https://stats.nba.com/stats/commonteamyears?LeagueID=00"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(NBA_STATS_HEADERS)
        self._team_id_map: dict[str, int] = {}

 #Team ID lookup 
    def fetch_team_ids(self) -> dict[str, int]:
        """Return {team_abbreviation: team_id} from NBA Stats API."""

        #if the map has already been fetched, return the cached version, instead of making another network request.
        if self._team_id_map:
            return self._team_id_map

        try:
            #makes a get request to the NBA stats API to fetch teams endpoint
            r = self.session.get(self.TEAMS_URL, timeout=10)
            #raise an exception if the HTTP response fails (4xx/5xx)
            r.raise_for_status()
            #parse the JSON response body  

            data = r.json()
            headers = data["resultSets"][0]["headers"]
            rows    = data["resultSets"][0]["rowSet"]
            abbr_idx = headers.index("ABBREVIATION")
            #finds which column index holds the team abbreviation 
            id_idx   = headers.index("TEAM_ID")
            #finds which column index holds the numberic team ID 
            self._team_id_map = {row[abbr_idx]: row[id_idx] for row in rows}
        except Exception:
            # Fallback: common team IDs hard-coded
            self._team_id_map = {
                "LAL": 1610612747, "BOS": 1610612738, "GSW": 1610612744,
                "MIA": 1610612748, "DEN": 1610612743, "MIL": 1610612749,
                "PHX": 1610612756, "DAL": 1610612742, "BKN": 1610612751,
                "NYK": 1610612752, "PHI": 1610612755, "CLE": 1610612739,
                "OKC": 1610612760, "MIN": 1610612750, "SAC": 1610612758,
            }
        return self._team_id_map


 # Game log scraper 
    def fetch_game_log(self, team_abbr: str, season: str = "2023-24") -> list[dict]:
        """
        Returns a list of game dicts with keys:
          date, home, pts_for, pts_against, win, rest_days
        """
        #get the full abbreviation to ID map, then look up the specific team. 
        team_ids = self.fetch_team_ids()
        team_id  = team_ids.get(team_abbr.upper()) # the .upper() handles lowercase input 

        #only proceed if the team abbreviation was recognized. 
        if team_id:
            try:
                #build the URL by injecting the season string and numeric team ID
                url = self.GAME_LOG_URL.format(season=season, team_id=team_id)
                r   = self.session.get(url, timeout=12)]
                #raise an exception if the HTTP response fails (4xx/5xx)
                r.raise_for_status()
                data    = r.json()
                headers = data["resultSets"][0]["headers"]
                rows    = data["resultSets"][0]["rowSet"]

                games = []
                prev_date = None
                for row in reversed(rows):   # oldest → newest
                    g = dict(zip(headers, row))
                    game_date = datetime.datetime.strptime(g["GAME_DATE"], "%Y-%m-%dT%H:%M:%S")
                    rest      = (game_date - prev_date).days if prev_date else 3
                    prev_date = game_date
                    games.append({
                        "date":        game_date,
                        "home":        "vs." in g.get("MATCHUP", ""),
                        "pts_for":     g["PTS"],
                        "pts_against": g["PTS"] - g["PLUS_MINUS"],
                        "win":         g["WL"] == "W",
                        "rest_days":   min(rest, 7),
                        "fg_pct":      g.get("FG_PCT", 0.45),
                        "fg3_pct":     g.get("FG3_PCT", 0.35),
                        "reb":         g.get("REB", 43),
                        "ast":         g.get("AST", 24),
                        "tov":         g.get("TOV", 13),
                    })
                    #field goal percent, 3 point percent, rebounds, assists, and turnovers. 
                return games

            except Exception:
                pass  # fall through to synthetic data


 # Synthetic fallback (realistic distributions) 
        return self._synthetic_game_log(team_abbr)

    def _synthetic_game_log(self, team_abbr: str) -> list[dict]:
        """Generate realistic synthetic game history when API is unavailable."""
        # Seed the RNG from the team abbreviation so the fallback is repeatable
        # for a given team instead of changing on every run.
        rng      = np.random.default_rng(abs(hash(team_abbr)) % (2**31))
        # Give each team a slightly different scoring baseline to keep the
        # synthetic logs from looking identical across franchises.
        base_ppg = rng.integers(108, 122)
        games    = []
        # Start from opening night and move forward through a full season.
        date     = datetime.datetime(2023, 10, 24)

        for i in range(82):
            # Simulate a normal NBA rest pattern: 1-4 days between games,
            # with a default 3-day rest before the first game.
            rest      = rng.integers(1, 5) if i > 0 else 3
            home      = bool(rng.integers(0, 2))
            # Home teams get a small scoring bump; game-to-game variation comes
            # from a normal distribution around the team's baseline offense.
            pts_for   = int(rng.normal(base_ppg + (2 if home else -1), 8))
            # Opponent scoring is also sampled around a similar baseline, but
            # shifted slightly to create stronger and weaker matchups.
            pts_ag    = int(rng.normal(base_ppg - rng.integers(-3, 6), 8))
            date     += datetime.timedelta(days=int(rest))
            games.append({
                "date":        date,
                "home":        home,
                # Clamp extreme low scores so the synthetic data stays within a
                # believable modern NBA range.
                "pts_for":     max(85, pts_for),
                "pts_against": max(85, pts_ag),
                "win":         pts_for > pts_ag,
                "rest_days":   min(int(rest), 7),
                # Add a few box-score style stats using league-ish averages.
                "fg_pct":      round(float(rng.normal(0.46, 0.03)), 3),
                "fg3_pct":     round(float(rng.normal(0.36, 0.04)), 3),
                "reb":         int(rng.normal(43, 4)),
                "ast":         int(rng.normal(24, 3)),
                "tov":         int(rng.normal(13, 2)),
            })
        return games


         #Injury report scraper 
    def fetch_injuries(self) -> dict[str, list[str]]:
        """
        Scrapes ESPN injury page. Returns {team_name: [player_status, ...]}
        Falls back to empty dict on failure.
        """
        try:
            #actual request for the espn injury page 
            r    = self.session.get(self.ESPN_INJURIES_URL, timeout=10)
            r.raise_for_status()
            #parses the HTML so we can search it with CSS selectors 
            soup = BeautifulSoup(r.text, "html.parser")

            injuries: dict[str, list[str]] = {}
            for section in soup.select(".TableBase"):
                #find the team name shown above the table
                team_el = section.select_one(".injuries__teamName")
                if not team_el:
                    #skip if it does not contain a team header
                    continue
                team = team_el.get_text(strip=True)
                players = []
                #walk through each player row in the injury table 
                for row in section.select("tbody tr"):
                    cols = row.select("td")
                    if len(cols) >= 4:
                        name   = cols[0].get_text(strip=True)
                        status = cols[3].get_text(strip=True)
                        players.append(f"{name} ({status})")
                if players:
                    injuries[team] = players
            return injuries
        except Exception:
            return {}
