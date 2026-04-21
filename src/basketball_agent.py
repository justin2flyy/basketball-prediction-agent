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
