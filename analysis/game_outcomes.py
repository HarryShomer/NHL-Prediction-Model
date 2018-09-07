"""
Used the Html pbp files (there are much better ways....) to get the outcomes for each game.
Data deposited in a CSV file
"""


import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup, SoupStrainer
import helpers


def get_soup(game_html):
    """
    Uses Beautiful soup to parses the html document.
    Some parsers work for some pages but don't work for others....I'm not sure why so I just try them all here in order

    :param game_html: html doc

    :return: "soupified" html 
    """
    for parser in ["lxml", "html.parser", "html5lib"]:
        strainer = SoupStrainer('table', attrs={'id': ['Visitor', 'Home']})
        soup = BeautifulSoup(game_html, parser, parse_only=strainer)

        if soup:
            break

    return soup


def get_game_info(soup):
    """
    Using the "souped-up" file, get the final scores for both teams and the team names
    """
    teams_info = {}

    # Scores
    score_soup = soup.find_all('td', {'align': 'center', 'style': "font-size: 40px;font-weight:bold"})
    teams_info['Visitor_Score'] = int(score_soup[0].get_text())
    teams_info['Home_Score'] = int(score_soup[1].get_text())

    # Team Name
    team_soup = soup.find_all('td', {'align': 'center', 'style': "font-size: 10px;font-weight:bold"})
    regex = re.compile(r'>(.*)<br/?>')
    teams_info['Visitor_Team'] = helpers.TEAMS[regex.findall(str(team_soup[0]))[0]]
    teams_info['Home_Team'] = helpers.TEAMS[regex.findall(str(team_soup[1]))[0]]

    return teams_info


# NOTE: About 1.34s per game
games_list = []
# For each game in each season get the scores and store in a Csv
for season in range(2007, 2018):
    season_df = pd.read_csv("../data/nhl_pbp{}{}.csv".format(season, season+1))
    game_ids = list(season_df['Game_Id'].unique())
    game_ids.sort()

    for game_id in game_ids:
        file_name = ''.join([str(season), '0', str(game_id), '.txt'])
        print(file_name)

        file = open("../hockey_scraper_data/docs/{}/html_pbp/{}".format(season, file_name))
        soup = get_soup(file)

        game_info = get_game_info(soup)
        game_info['Game_Id'] = ''.join([str(season), '0', str(game_id)])
        games_list.append(game_info)


outcomes_df = pd.DataFrame(games_list)
outcomes_df['Winner'] = np.where(outcomes_df['Home_Score'] > outcomes_df['Visitor_Score'],
                                 outcomes_df['Home_Team'], outcomes_df['Visitor_Team'])

outcomes_df.to_csv("game_outcomes.csv", sep=',')


