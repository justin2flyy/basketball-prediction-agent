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
        #Get the full abbreviation to ID map, then look up the specific team. 
        team_ids = self.fetch_team_ids()
        team_id  = team_ids.get(team_abbr.upper()) # the .upper() handles lowercase input 

        #Only proceed if the team abbreviation was recognized. 
        if team_id:
            try:
                #Build the URL by injecting the season string and numeric team ID
                url = self.GAME_LOG_URL.format(season=season, team_id=team_id)
                r   = self.session.get(url, timeout=12)]
                #Raise an exception if the HTTP response fails (4xx/5xx)
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
                    #Field goal percent, 3 point percent, rebounds, assists, and turnovers. 
                return games

            except Exception:
                pass  # Fall through to synthetic data


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
            #Actual request for the espn injury page 
            r    = self.session.get(self.ESPN_INJURIES_URL, timeout=10)
            r.raise_for_status()
            #Parses the HTML so we can search it with CSS selectors 
            soup = BeautifulSoup(r.text, "html.parser")

            injuries: dict[str, list[str]] = {}
            for section in soup.select(".TableBase"):
                #Find the team name shown above the table
                team_el = section.select_one(".injuries__teamName")
                if not team_el:
                    #Skip if it does not contain a team header
                    continue
                team = team_el.get_text(strip=True)
                players = []
                #Walk through each player row in the injury table 
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
            
# FEATURE ENGINEERING
def compute_team_features(games: list[dict], elo: float, injuries: list[str]) -> dict:
    """
    From a team's game log, compute all ML features for one team side:
      - avg_pts_last10, avg_pts_against_last10
      - win_pct_last10
      - avg_rest_days
      - home_win_pct, away_win_pct
      - elo_rating
      - injury_count
      - avg_fg_pct, avg_fg3_pct, avg_reb, avg_ast, avg_tov (last 10)
    """
    # Use only the most recent 10 games for recency-weighted stats;
    # fall back to all games if fewer than 10 are available
    last10  = games[-10:] if len(games) >= 10 else games

    # Split the full game log into home and away subsets for split-record features
    home_g  = [g for g in games if g["home"]]
    away_g  = [g for g in games if not g["home"]]

    # Returns the mean of a numeric field across a list of game dicts,
    # or 0.0 if the list is empty (avoids ZeroDivisionError)
    def safe_mean(lst, key):
        return float(np.mean([g[key] for g in lst])) if lst else 0.0

    # Returns the fraction of games won in a subset; 0.0 if subset is empty
    def win_pct(subset):
        return float(np.mean([g["win"] for g in subset])) if subset else 0.0

    return {
        # Offensive output and defensive exposure over the last 10 games
        "avg_pts_last10":      safe_mean(last10, "pts_for"),
        "avg_pts_ag_last10":   safe_mean(last10, "pts_against"),

        # How often the team won over their last 10 games (0.0 – 1.0)
        "win_pct_last10":      win_pct(last10),

        # Average days of rest between games — more rest can mean fresher legs
        "avg_rest_days":       safe_mean(last10, "rest_days"),

        # Win rates split by venue — captures home-court advantage effects
        "home_win_pct":        win_pct(home_g),
        "away_win_pct":        win_pct(away_g),

        # Current ELO rating — a running power-ranking score for the team
        "elo_rating":          elo,

        # Number of players currently listed as injured
        "injury_count":        len(injuries),

        # Shooting efficiency and box-score averages over the last 10 games
        "avg_fg_pct":          safe_mean(last10, "fg_pct"),   # field-goal %
        "avg_fg3_pct":         safe_mean(last10, "fg3_pct"),  # three-point %
        "avg_reb":             safe_mean(last10, "reb"),       # rebounds
        "avg_ast":             safe_mean(last10, "ast"),       # assists
        "avg_tov":             safe_mean(last10, "tov"),       # turnovers
    }


def build_matchup_features(home_feats: dict, away_feats: dict, home_elo_wp: float) -> np.ndarray:
    """
    Combines home and away features into a single feature vector
    by computing differentials (home - away) plus ELO win probability.
    """
    # These keys exist in both home_feats and away_feats; subtracting
    # away from home turns two team-level dicts into one relative vector
    # that the model can learn from directly
    diff_keys = [
        "avg_pts_last10", "avg_pts_ag_last10", "win_pct_last10",
        "avg_rest_days",  "home_win_pct",       "away_win_pct",
        "elo_rating",     "avg_fg_pct",          "avg_fg3_pct",
        "avg_reb",        "avg_ast",             "avg_tov",
    ]

    # Build a list of (home - away) differentials for every shared feature
    diffs = [home_feats[k] - away_feats[k] for k in diff_keys]

    # Append the ELO-derived win probability for the home team (0.0 – 1.0)
    diffs.append(home_elo_wp)

    # Append raw injury counts for each side — kept separate rather than
    # differenced so the model can detect asymmetric injury burdens
    diffs.append(float(home_feats["injury_count"]))
    diffs.append(float(away_feats["injury_count"]))

    # Return as a 1-D float array ready to be passed to the scaler / model
    return np.array(diffs, dtype=float)


# Human-readable names for each position in the feature vector produced by
# build_matchup_features — used for logging, debugging, and SHAP explanations
FEATURE_NAMES = [
    "pts_diff_last10", "pts_ag_diff_last10", "win_pct_diff_last10",
    "rest_diff",        "home_win_pct_diff",  "away_win_pct_diff",
    "elo_diff",         "fg_pct_diff",         "fg3_pct_diff",
    "reb_diff",         "ast_diff",            "tov_diff",
    "elo_win_prob",     "home_injuries",        "away_injuries",
]


# 
# ML MODEL
# 
class NBAPredictor:
    """
    Supervised ML pipeline:
      1. Generate synthetic historical seasons for training
      2. Train calibrated Logistic Regression
      3. Evaluate: Accuracy, Log Loss, AUC-ROC, Brier Score, Calibration
      4. Backtest on held-out season
      5. Predict new matchups
    """

    def __init__(self):
        # Standardize features before model fitting/prediction
        self.scaler     = StandardScaler()

        # Base classifier used inside probability calibration
        self.base_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)

        # Isotonic calibration improves probability reliability
        self.model      = CalibratedClassifierCV(self.base_model, cv=5, method="isotonic")

        # Helpers for dynamic strength rating and data collection
        self.elo        = EloSystem()
        self.scraper    = NBAScraper()

        # Training/evaluation state
        self.is_trained = False
        self.metrics: dict = {}

    # ── Training data generation ────────────────────────────
    def _generate_training_data(self, n_seasons: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """
        Builds a training dataset by simulating n_seasons of 30-team NBA schedules.
        Each row = one game's feature vector; label = 1 if home team won.
        """
        team_abbrs = [
            "LAL","BOS","GSW","MIA","DEN","MIL","PHX","DAL","BKN","NYK",
            "PHI","CLE","OKC","MIN","SAC","ATL","CHI","HOU","IND","MEM",
            "NOP","ORL","POR","SAS","TOR","UTA","WAS","CHA","DET","LAC",
        ]

        X_list, y_list = [], []

        for season_offset in range(n_seasons):
            # Fresh game logs per team per season
            team_logs: dict[str, list[dict]] = {
                t: self.scraper._synthetic_game_log(f"{t}_{season_offset}")
                for t in team_abbrs
            }
            # Reset ELO each season (with slight regression to mean)
            for t in team_abbrs:
                prev = self.elo.get(t)
                self.elo.ratings[t] = EloSystem.DEFAULT_ELO * 0.25 + prev * 0.75

            # Simulate ~1230 games per season
            rng = np.random.default_rng(season_offset)
            teams_shuffled = team_abbrs.copy()

            for game_idx in range(1230):
                # Randomly pair teams for a synthetic game
                rng.shuffle(teams_shuffled)
                home_t = teams_shuffled[0]
                away_t = teams_shuffled[1]

                inj_h = []  # no per-game injury simulation in training
                inj_a = []

                # Build side-specific features, then combine into one matchup vector
                h_feats  = compute_team_features(team_logs[home_t], self.elo.get(home_t), inj_h)
                a_feats  = compute_team_features(team_logs[away_t], self.elo.get(away_t), inj_a)
                elo_wp   = self.elo.win_probability(home_t, away_t)
                x_vec    = build_matchup_features(h_feats, a_feats, elo_wp)

                # Determine outcome using ELO probability
                home_wins = bool(rng.random() < elo_wp)
                winner    = home_t if home_wins else away_t
                loser     = away_t if home_wins else home_t
                self.elo.update(winner, loser, home_t)

                # Sample plausible scores around recent offensive averages
                pts_h = int(rng.normal(h_feats["avg_pts_last10"], 8))
                pts_a = int(rng.normal(a_feats["avg_pts_last10"], 8))

                # Enforce no ties by nudging the winner above the loser
                if home_wins:
                    pts_h = max(pts_h, pts_a + 1)
                else:
                    pts_a = max(pts_a, pts_h + 1)

                # Feed the simulated game back into team logs so future features
                # evolve over the season
                team_logs[home_t].append({
                    "date": datetime.datetime.now(), "home": True,
                    "pts_for": pts_h, "pts_against": pts_a,
                    "win": home_wins, "rest_days": int(rng.integers(1, 5)),
                    "fg_pct": 0.46, "fg3_pct": 0.36,
                    "reb": 43, "ast": 24, "tov": 13,
                })
                team_logs[away_t].append({
                    "date": datetime.datetime.now(), "home": False,
                    "pts_for": pts_a, "pts_against": pts_h,
                    "win": not home_wins, "rest_days": int(rng.integers(1, 5)),
                    "fg_pct": 0.46, "fg3_pct": 0.36,
                    "reb": 43, "ast": 24, "tov": 13,
                })

                # Collect final training row and binary label
                X_list.append(x_vec)
                y_list.append(int(home_wins))

        return np.array(X_list), np.array(y_list)

    # ── Train ───────────────────────────────────────────────
    def train(self, verbose: bool = True) -> None:
        if verbose:
            print("\n⚙️   Generating training data (5 simulated seasons)...")

        X, y = self._generate_training_data(n_seasons=5)

        # Chronological split: last 20% = backtest set
        split      = int(len(X) * 0.80)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Fit scaler only on training data to avoid leakage
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        if verbose:
            print(f"   Training samples : {len(X_train):,}")
            print(f"   Backtest samples : {len(X_test):,}")
            print("⚙️   Fitting calibrated logistic regression...")

        # Fit calibrated model and mark pipeline as usable
        self.model.fit(X_train_s, y_train)
        self.is_trained = True

        # ── Evaluate ─────────────────────────────────────────
        # Hard predictions + calibrated probabilities on the backtest split
        y_pred      = self.model.predict(X_test_s)
        y_prob      = self.model.predict_proba(X_test_s)[:, 1]

        # Core classification/probability-quality metrics
        acc         = accuracy_score(y_test, y_pred)
        ll          = log_loss(y_test, y_prob)
        auc         = roc_auc_score(y_test, y_prob)
        brier       = brier_score_loss(y_test, y_prob)

        # Calibration: mean absolute error between predicted & actual bins
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        cal_error = float(np.mean(np.abs(prob_true - prob_pred)))

        # Store rounded metrics for clean downstream printing/UI
        self.metrics = {
            "accuracy":          round(acc * 100, 2),
            "log_loss":          round(ll,  4),
            "auc_roc":           round(auc, 4),
            "brier_score":       round(brier, 4),
            "calibration_error": round(cal_error, 4),
            "backtest_n":        len(X_test),
        }

        if verbose:
            self._print_metrics()

    def _print_metrics(self) -> None:
        m = self.metrics
        print(f"\n{'─'*50}")
        print(f"  📈  MODEL EVALUATION (Backtest on held-out season)")
        print(f"{'─'*50}")
        print(f"  Accuracy          : {m['accuracy']}%")
        print(f"  Log Loss          : {m['log_loss']}   (lower = better)")
        print(f"  AUC-ROC           : {m['auc_roc']}   (1.0 = perfect)")
        print(f"  Brier Score       : {m['brier_score']} (lower = better)")
        print(f"  Calibration Error : {m['calibration_error']} (0 = perfect)")
        print(f"  Backtest games    : {m['backtest_n']:,}")
        print(f"{'─'*50}\n")

    # ── Predict ─────────────────────────────────────────────
    def predict(
        self,
        home_team: str,
        away_team: str,
        home_games: list[dict],
        away_games: list[dict],
        injuries:   dict[str, list[str]],
    ) -> dict:
        """Run prediction for a specific matchup."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call .train() first.")

        inj_home = injuries.get(home_team, [])
        inj_away = injuries.get(away_team, [])

        # Build current-team feature snapshots and ELO-based prior
        h_feats = compute_team_features(home_games, self.elo.get(home_team), inj_home)
        a_feats = compute_team_features(away_games, self.elo.get(away_team), inj_away)
        elo_wp  = self.elo.win_probability(home_team, away_team)

        # Shape/scaling to match the format the model was trained on
        x_vec = build_matchup_features(h_feats, a_feats, elo_wp).reshape(1, -1)
        x_s   = self.scaler.transform(x_vec)

        # Binary probability for home win; away is complementary
        home_prob = float(self.model.predict_proba(x_s)[0][1])
        away_prob = 1.0 - home_prob
        winner    = home_team if home_prob >= 0.5 else away_team

        # Heuristic score projection combining offense and opponent defense
        proj_home = round(h_feats["avg_pts_last10"] * 0.6 + a_feats["avg_pts_ag_last10"] * 0.4 + 2.5)
        proj_away = round(a_feats["avg_pts_last10"] * 0.6 + h_feats["avg_pts_ag_last10"] * 0.4)

        # Confidence tiers based on distance from a 50/50 coin flip
        confidence = "High" if abs(home_prob - 0.5) > 0.15 else \
                     "Medium" if abs(home_prob - 0.5) > 0.07 else "Low"

        return {
            "home_team":       home_team,
            "away_team":       away_team,
            "predicted_winner": winner,
            "home_win_prob":   round(home_prob * 100, 1),
            "away_win_prob":   round(away_prob * 100, 1),
            "projected_score": f"{home_team} {proj_home} – {away_team} {proj_away}",
            "elo_home":        round(self.elo.get(home_team), 1),
            "elo_away":        round(self.elo.get(away_team), 1),
            "home_injuries":   inj_home,
            "away_injuries":   inj_away,
            "home_features":   h_feats,
            "away_features":   a_feats,
            "confidence":      confidence,
        }



