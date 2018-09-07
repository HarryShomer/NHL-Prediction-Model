"""
Code for creating the Elo Ratings
"""

import pandas as pd
import helpers


def team_preprocessing():
    """
    Get the Data and process it how I want it
    
    :return DataFrame
    """
    cols = ["Team", "Season", "Game.ID", "Date", "Opponent", "Venue", "TOI", "GF", 'GA']

    # Get Game Data for Teams
    df = pd.read_csv("./data/teams_all_sits.csv")
    for team_col in ['Team', "Opponent", "Venue"]:
        df = helpers.fix_team(df, team_col)
    df = df[cols]

    # Get "correct" game id
    df['game_id'] = df.apply(lambda x: int(str(x['Season']) + "0" + str(x['Game.ID'])), axis=1)
    df = df.drop(['Game.ID'], axis=1)

    # Sort by game_id...lowest is first
    df = df.sort_values(by=['game_id'])

    # Merge in game outcomes:
    df = helpers.merge_outcomes(df)

    # Only keeps games from the home team perspective!!!!!!!!
    df = df[df['Team'] == df['Venue']]

    return df


def regress_elo(team_elo):
    """
    Regresses the Elo Ratings of each team 50% to the mean after each season (mean = 1500)
    
    :param team_elo: Dictionary of elo ratings for each team
    
    :return regressed elo ratings
    
    """
    elo_avg = 1500
    regress_amount = .5

    for team in team_elo.keys():
        team_elo[team] = (1 - regress_amount) * team_elo[team] + regress_amount * elo_avg

    return team_elo


def get_home_prob(game, team_elo):
    """
    Get the probability that the home team will win for a given game
    
    *** Home Advantage ***
    dr = -400log10(1/prob-1) 
    Home Advantage = 33.5 points. 
    Derived from dr = -400log10(1/prob-1) where prob = .548
    
    *** Get Home Probability ***
    Prob = 1 / (1 + 10^(dr/400)) ; where dr = away_elo - (home_elo + 33.5)
    
    :param game: Dict with home and away team
    :param team_elo: Dict of Elo Ratings
    
    :return probability of home team winning
    """
    home_advantage = 33.5
    dr = team_elo[game['Opponent']] - (team_elo[game['Team']] + home_advantage)

    return 1 / (1 + 10 ** (dr/400))


def update_elo(game, team_elo):
    """
    Update the elo ratings for both teams after a game
    
    The k-rating formula is taken from Cole Anderson - http://crowdscoutsports.com/team_elo.php
    
    :param game: Dict with details of game
    :param team_elo: Dict of Elo Ratings
    
    :return Updated Elo Ratings 
    """
    # k is the constant for how much the ratings should change from this game
    win_margin = abs(game['GF'] - game['GA']) if abs(game['GF'] - game['GA']) != 0 else 1
    if_shootout = 1 if game['TOI'] == 65 else 0
    k_rating = 4 + (4 * win_margin) - (if_shootout * 2)

    # New Rating = Old Rating + k * (actual - expected)
    elo_change = k_rating * (game['if_home_win'] - game['home_prob'])
    team_elo[game['Team']] += elo_change
    team_elo[game['Opponent']] -= elo_change

    return team_elo


def get_elo():
    """
    Iterate through every game and:
    1. Get the Probability of the home team winning every game
    2. Using that probability and the outcome of the game update the elo ratings for both teams
    3. At the start of every game regress the ratings 50% towards the mean (1500)
    
    :return DataFrame of home prob given elo ratings for every game
    """
    df = team_preprocessing()

    # Every Tem Starts off at 1500
    team_elo = {'ANA': 1500, 'ARI': 1500, 'ATL': 1500, 'BOS': 1500, 'BUF': 1500, 'CAR': 1500, 'CBJ': 1500, 'CGY': 1500,
                'CHI': 1500, 'COL': 1500, 'DAL': 1500, 'DET': 1500, 'EDM': 1500, 'FLA': 1500, 'L.A': 1500, 'MIN': 1500,
                'MTL': 1500, 'N.J': 1500, 'NSH': 1500, 'NYI': 1500, 'NYR': 1500, 'OTT': 1500, 'PHI': 1500, 'PHX': 1500,
                'PIT': 1500, 'S.J': 1500, 'STL': 1500, 'T.B': 1500, 'TOR': 1500, 'VAN': 1500, 'VGK': 1500, 'WPG': 1500,
                'WSH': 1500
    }

    games_list = []
    for row in df.to_dict("records"):
        # Regress the ratings at the start of each season
        if str(row['game_id'])[-5:] == "20001":
            team_elo = regress_elo(team_elo)

        # Get the Probability of the Home Team winning and update then ratings based on the game outcome and probability
        row['home_prob'] = get_home_prob(row, team_elo)
        row['home_rating'] = team_elo[row['Team']]
        row['away_rating'] = team_elo[row['Opponent']]
        team_elo = update_elo(row, team_elo)

        games_list.append(row)

    # Create df and drop 2007
    df = pd.DataFrame(games_list)
    df = df[df.Season > 2007]

    return df


def main():
    # 12703 Rows
    df = get_elo()


if __name__ == "__main__":
    main()