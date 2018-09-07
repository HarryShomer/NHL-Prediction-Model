import math

import pandas as pd

import helpers
from models import get_goalie_stats as ggs


def get_all_sits_data():
    """
    Get and prepare the All Situations data
    
    :return: DataFrame of data
    """
    cols = ["Team", "Season", "Game.ID", "Date", "Opponent", "Venue", "TOI_all", "PENT_all", "PEND_all"]

    df = pd.read_csv("./data/teams_all_sits.csv")
    df = df.sort_values(by=['Season', 'Game.ID', 'Team'])
    for team_col in ['Team', "Opponent", "Venue"]:
        df = helpers.fix_team(df, team_col)

    df = df.rename(index=str, columns={"TOI": "TOI_all", "PENT": "PENT_all", "PEND": "PEND_all"})

    return df[cols]


def get_even_data():
    """
    Get and prepare the even strength data
    
    :return: DataFrame of data
    """
    cols = ["Team", "Season", "Game.ID", "Date", "TOI",
            'GF', 'GA', 'FF', 'FA', 'xGF', 'xGA', 'CF', 'CA', 'wshF', 'wshA']

    df = pd.read_csv("../projection_data/teams/teams_even.csv")
    df = helpers.fix_team(df, "Team")
    df = df.sort_values(by=['Season', 'Game.ID', 'Team'])

    df['wshF'] = ((df['CF'] - df['GF']) * .2 + df['GF'])
    df['wshA'] = ((df['CA'] - df['GA']) * .2 + df['GA'])

    df = df[cols]
    df = df.rename(index=str, columns={col: col + "_even" for col in cols[4:]})

    return df


def get_pp_data():
    """
    Get and prepare the power play data
    
    :return: DataFrame of data
    """
    cols = ["Team", "Season", "Game.ID", "Date", "TOI", 'GF', 'FF', 'xGF', 'CF', 'wshF']

    df = pd.read_csv("../projection_data/teams/teams_pp.csv")
    df = helpers.fix_team(df, "Team")
    df = df.sort_values(by=['Season', 'Game.ID', 'Team'])

    df['wshF'] = ((df['CF'] - df['GF']) * .2 + df['GF'])

    df = df[cols]
    df = df.rename(index=str, columns={col: col + "_pp" for col in cols[4:]})

    return df


def get_pk_data():
    """
    Get and prepare the penalty kill data
    
    :return: DataFrame of data
    """
    cols = ["Team", "Season", "Game.ID", "Date", "TOI", 'GA', 'FA', 'xGA', 'CA', 'wshA', 'wshF']

    df = pd.read_csv("../projection_data/teams/teams_pk.csv")
    df = helpers.fix_team(df, "Team")
    df = df.sort_values(by=['Season', 'Game.ID', 'Team'])

    df['wshF'] = ((df['CF'] - df['GF']) * .2 + df['GF'])
    df['wshA'] = ((df['CA'] - df['GA']) * .2 + df['GA'])

    df = df[cols]
    df = df.rename(index=str, columns={col: col + "_pk" for col in cols[4:]})

    return df


def team_preprocessing():
    """
    Get All the Data foe each strength and combine them
    
    :return: DataFrame of data
    """
    df_all = get_all_sits_data()
    df_even = get_even_data()
    df_pp = get_pp_data()
    df_pk = get_pk_data()

    # Merge them all into one DataFrame
    df2 = pd.merge(df_all, df_even, how="left",
                   left_on=["Team", "Season", "Game.ID", "Date"],
                   right_on=["Team", "Season", "Game.ID", "Date"],
                   suffixes=['', "_even"])
    df3 = pd.merge(df2, df_pp, how="left",
                   left_on=["Team", "Season", "Game.ID", "Date"],
                   right_on=["Team", "Season", "Game.ID", "Date"],
                   suffixes=['', "_pp"])
    df_merged = pd.merge(df3, df_pk, how="left",
                         left_on=["Team", "Season", "Game.ID", "Date"],
                         right_on=["Team", "Season", "Game.ID", "Date"],
                         suffixes=['', "_pk"])

    df_merged = df_merged.sort_values(by=['Season', 'Game.ID', 'Team'])

    df_merged['game_id'] = df_merged.apply(lambda x: str(x['Season']) + "0" + str(x['Game.ID']), axis=1)

    return df_merged


def add_goalie_data(df):
    """
    Add the weighted avg for each teams's goalies
    
    It's -> adj_fsv(team) = adj_fsv(starter) * .946 + adj_fsv(backup) * .053
    
    :param df: DataFrame of data
    
    :return DataFrame with weighted adj_fsv. Also drop unwanted cols
    """
    df['home_adj_fsv'] = df['Home_Starter_adj_fsv'] * .946 + df['Home_Backup_adj_fsv'] * .053
    df['away_adj_fsv'] = df['Away_Starter_adj_fsv'] * .946 + df['Away_Backup_adj_fsv'] * .053

    return df.drop(['Away_Backup', 'Away_Backup_adj_fsv', 'Away_Starter', 'Away_Starter_adj_fsv', 'Home_Backup',
                    'Home_Backup_adj_fsv', 'Home_Starter', 'Home_Starter_adj_fsv'], axis=1)


def get_last_game(row, df):
    """
    Get the last game for a team **THAT** season
    NOTE: If it's the first game of the season I just put 5
    
    :param row: Given game
    :param df: DataFrame of all games
    
    :return [home_days_rest, away_days_rest, home_b2b, away_b2b]
    """
    home_col = "Venue"
    away_col = "Opponent" if row['Team'] == row['Venue'] else "Team"
    home_b2b, away_b2b = 0, 0
    days = []

    # Home
    prev_games = df[(df["Team"] == row[home_col]) & (df['Date'] < row['Date']) & (df['Season'] == row['Season'])]
    days.append(5 if prev_games.empty else (row['Date'] - prev_games.iloc[prev_games.shape[0] - 1]['Date']).days)
    if not prev_games.empty and (row['Date'] - prev_games.iloc[prev_games.shape[0] - 1]['Date']).days == 1 and \
       row['Home_Starter'] in [prev_games.iloc[prev_games.shape[0] - 1]['Home_Starter'],
                                               prev_games.iloc[prev_games.shape[0] - 1]['Away_Starter']]:
                home_b2b = 1

    # Away
    prev_games = df[(df["Team"] == row[away_col]) & (df['Date'] < row['Date']) & (df['Season'] == row['Season'])]
    days.append(5 if prev_games.empty else (row['Date'] - prev_games.iloc[prev_games.shape[0] - 1]['Date']).days)
    if not prev_games.empty and (row['Date'] - prev_games.iloc[prev_games.shape[0] - 1]['Date']).days == 1 and \
       row['Away_Starter'] in [prev_games.iloc[prev_games.shape[0] - 1]['Home_Starter'],
                               prev_games.iloc[prev_games.shape[0] - 1]['Away_Starter']]:
            away_b2b = 1

    return days[0], days[1], home_b2b, away_b2b


def get_days_since_last(df):
    """
    Get days since last game for each team 
    
    :param df: dataFrame of stats until this point
    
    :return DataFrame with added info
    """
    # Convert to date
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df['Date'] = df['Date'].dt.date

    df['days_rest_home'], df['days_rest_away'], df['home_b2b'], df['away_b2b'] = zip(*[get_last_game(row, df)
                                                                                       for row in df.to_dict("records")])

    return df


def calc_stats(df):
    """
    Calculate stats (model features) for given sample
    
    :param df: DataFrame of given team sample
    
    :return DataFrame with added stats
    """
    # All
    df['PENT60'] = df['PENT_all'] * 60 / df['TOI_all']
    df['PEND60'] = df['PEND_all'] * 60 / df['TOI_all']

    # Even
    df['FF60_even'] = df['FF_even'] * 60 / df['TOI_even']
    df['FA60_even'] = df['FA_even'] * 60 / df['TOI_even']
    df['xGF60/FF60_even'] = df['xGF_even'] / df['FF_even']
    df['xGA60/FA60_even'] = df['xGA_even'] / df['FA_even']
    df['GF60/xGF60_even'] = df['GF_even'] / df['xGF_even']

    # PP
    df['FF60_pp'] = df['FF_pp'] * 60 / df['TOI_pp']
    df['xGF60/FF60_pp'] = df['xGF_pp'] / df['FF_pp']
    df['GF60/xGF60_pp'] = df['GF_pp'] / df['xGF_pp']

    # PK
    df['FA60_pk'] = df['FA_pk'] * 60 / df['TOI_pk']
    df['xGA60/FA60_pk'] = df['GA_pk'] / df['FA_pk']

    return df


def get_prev_stats_row(row, df, sum_cols, stats_cols):
    """
    Get the Stats for this row (so home and away team).
    
    It works as follows: 
    1. Each game played that season. Their importance is weighed by e^-.05x where x is the number of games between that
    game and this one. For example, for the previous game x=0 so the weight is 0.
    2. If a team has played less than 25 games that season we also incorporate the previous season stats. The importance
    of the previous season's stats in this scenario is also governed by an exponential function. Here it's given by
    e^-.175x. So, for example, if a team has played 0 games that seaosn then the previous season's stats are worth 1.  
    
    :param row: Given Game -> we need to get stats for both teams
    :param df: DataFrame of all games 
    :param sum_cols: Columns for summing stats
    :param stats_cols: Columns for stats we are calculating
    
    :return Dict with stats for both sides
    """
    print(row['Date'])

    row_dict = {"Team": row['Team'], "Season": row['Season'], "Date": row['Date'], "Opponent": row['Opponent'],
                "Venue": row['Venue'], "game_id": row['game_id']}

    for team_col in ['Team', 'Opponent']:
        if_less_than_25 = True

        # Get that year's numbers prior to that game
        prev_stats_df = df[(df["Team"] == row[team_col]) & (df['Season'] == row['Season']) & (df['Date'] < row['Date'])]
        if not prev_stats_df.empty:
            if_less_than_25 = False if prev_stats_df.shape[0] > 24 else True

            # We go to -1 to get 0 (which necessitate us starting one under the number of games)
            # TODO: Look at the constant a!!!
            prev_stats_df['game_weight'] = [math.e ** (-.05 * x) for x in range(prev_stats_df.shape[0]-1, -1, -1)]

            # Get Weighted Average for each number
            weight_sum = prev_stats_df["game_weight"].sum()
            for col in sum_cols:
                prev_stats_df[col] *= (prev_stats_df["game_weight"] / weight_sum)

            # Get Stats for that year
            df_same_sum = prev_stats_df[sum_cols].sum()
            df_same = calc_stats(df_same_sum)

        # Check if need last years numbers..if so add in
        if if_less_than_25:
            prev_season_df = df[(df["Team"] == row[team_col]) & (df['Season'] == row['Season'] - 1)]
            if not prev_season_df.empty:
                df_last_sum = prev_season_df[sum_cols].sum()
            else:
                # Just take the average when we got nothing for last year
                df_last_sum = df[sum_cols].sum()
            # Get Stats for previous year
            df_last = calc_stats(df_last_sum)

        # Assign the stats
        # If Less than 25 add in by given weight
        for stat in stats_cols:
            gp = prev_stats_df.shape[0]
            prev_yr_weight = math.e ** (-.175 * gp)
            if gp > 24:
                row_dict["_".join([stat, team_col])] = df_same[stat]
            elif gp > 0:
                row_dict["_".join([stat, team_col])] = (df_same[stat] * (1 - prev_yr_weight)) + (df_last[stat] * prev_yr_weight)
            else:
                row_dict["_".join([stat, team_col])] = df_last[stat]

    return row_dict


def get_previous_stats(df):
    """
    Get the previous stats for each game
    Note: For a better explanation see the function - 'get_prev_stats_row'
    
    :param df: DataFrame of all games
    
    :return DataFrame of the previous stats for each game for both teams
    """
    sum_cols = ['TOI_all', 'PENT_all', 'PEND_all', 'TOI_even', 'GF_even', 'GA_even', 'FF_even', 'FA_even', 'xGF_even',
                'xGA_even', 'CF_even', 'CA_even', 'TOI_pp', 'GF_pp', 'FF_pp', 'xGF_pp', 'CF_pp', 'TOI_pk', 'GA_pk',
                'FA_pk', 'xGA_pk', 'CA_pk', ]

    stats_cols = ['PENT60', 'PEND60',
                  'FF60_even', 'FA60_even',
                  'xGF60/FF60_even', 'xGA60/FA60_even', 'GF60/xGF60_even',
                  'FF60_pp',
                  'xGF60/FF60_pp', 'GF60/xGF60_pp',
                  'FA60_pk',
                  'xGA60/FA60_pk']

    return pd.DataFrame([get_prev_stats_row(row, df, sum_cols, stats_cols) for row in df.to_dict("records")])


def get_model_data():
    """
    Get The Data required for building the model
    """
    all_cols = ['game_id', 'Season',
                'FA60_even_Opponent', 'FA60_even_Team', 'FA60_pk_Opponent', 'FA60_pk_Team',
                'FF60_even_Opponent', 'FF60_even_Team', 'FF60_pp_Opponent', 'FF60_pp_Team',
                'GF60/xGF60_even_Opponent',
                'GF60/xGF60_even_Team', 'GF60/xGF60_pp_Opponent', 'GF60/xGF60_pp_Team', 'PEND60_Opponent',
                'PEND60_Team', 'PENT60_Opponent', 'PENT60_Team', 'xGA60/FA60_even_Opponent', 'xGA60/FA60_even_Team',
                'xGA60/FA60_pk_Opponent', 'xGA60/FA60_pk_Team', 'xGF60/FF60_even_Opponent', 'xGF60/FF60_even_Team',
                'xGF60/FF60_pp_Opponent', 'xGF60/FF60_pp_Team', 'days_rest_home', 'days_rest_away', 'home_b2b',
                'away_b2b', 'home_adj_fsv', 'away_adj_fsv', 'if_playoff', 'if_home_win']

    df = team_preprocessing()
    df = df.fillna(0)

    df = get_previous_stats(df)
    df = ggs.get_goalies(df)
    df = get_days_since_last(df)
    df = add_goalie_data(df)

    # Only keeps games from the home team perspective!!!!!!!!
    df = df[df['Team'] == df['Venue']]

    # Add if a playoff game
    df['if_playoff'] = df.apply(lambda x: 1 if int(str(x['game_id'])[-5:]) > 30000 else 0, axis=1)

    # Merge in outcomes
    df = helpers.merge_outcomes(df)

    # Only Data from 2008 onwards!!!!
    df = df[df['Season'] > 2007]

    # Fill in any missing value with the column average
    df = df.fillna(df.mean())

    return df[all_cols]


def main():
    #df = get_model_data()
    pass


if __name__ == "__main__":
    main()
