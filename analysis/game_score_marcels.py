"""
Using the Game Score model created by Dom Luszczyszyn to create a marcels based projection system
https://hockey-graphs.com/2016/07/13/measuring-single-game-productivity-an-introduction-to-game-score/
https://www.baseball-reference.com/about/marcels.shtml
"""
import pandas as pd
import numpy as np
import json
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression
pd.options.mode.chained_assignment = None  # default='warn'


def get_previous_yr(df, df2, years):
    """
    Get stats for previous year for on player level
    
    :param df: DataFrame with data used to predict
    :param df2: DataFrame with data to predict
    :param years: How many years of data to include in predictor
    
    :return Merged DataFrame with Years 1-n used to predict season n+1
    """
    # Get n+_ year
    df["season_n-{}_tmp".format(years)] = df["season"] - years
    df_merged = pd.merge(df, df2, how="left", left_on=["player", "player_id", "season_n-{}_tmp".format(years)],
                         right_on=["player", "player_id", "season"],
                         suffixes=['', "_n-{}".format(years)])

    df_merged = df_merged.drop(["season_n-{}_tmp".format(years)], axis=1)

    return df_merged


def skater_preprocessing(pos):
    """
    Transfer from raw Data to how I want it (have to group team by season)
    
    :param pos: Position - D/F

    :return: DataFrame
    """
    even_cols = ['player', 'player_id', 'season', 'toi_on', 'corsi_f', 'corsi_a', 'goals_f', 'goals_a']
    all_cols = ['player', 'player_id', 'season', 'toi_on', 'goals', 'a1', 'a2', 'icors', 'iblocks', 'pend', 'pent',
                'ifac_win', 'ifac_loss', 'games']

    with open("skaters/{}_even.json".format(pos)) as file_even:
        df_even = pd.DataFrame(json.load(file_even)['data'], columns=even_cols)

        # Convert from string to float for some reason
        for col in ["toi_on", "corsi_f", "corsi_a"]:
            df_even[col] = df_even[col].astype(float)
        df_even = df_even.groupby(['player', 'player_id', 'season'], as_index=False).sum()
        df_even = df_even.sort_values(['player', 'player_id', 'season'])

    with open("skaters/{}_all_sits.json".format(pos)) as file_all_sits:
        df_all_sits = pd.DataFrame(json.load(file_all_sits)['data'], columns=all_cols)
        df_all_sits['toi_on'] = df_all_sits['toi_on'].astype(float)
        df_all_sits = df_all_sits.groupby(['player', 'player_id', 'season'], as_index=False).sum()
        df_all_sits = df_all_sits.sort_values(['player', 'player_id', 'season'])

    # Just transfer over corsi straight to All Situations
    df_all_sits['corsi_f'] = df_even['corsi_f']
    df_all_sits['corsi_a'] = df_even['corsi_a']
    df_all_sits['goals_f'] = df_even['goals_f']
    df_all_sits['goals_a'] = df_even['goals_a']
    df_all_sits['even_toi_on'] = df_even['toi_on']

    df_all_sits['gs'] = (.75 * df_all_sits['goals']) + (.7 * df_all_sits['a1']) + (.55 * df_all_sits['a2'])\
                        + (.049 * df_all_sits['icors']) + (.05 * df_all_sits['iblocks']) + (.15 * df_all_sits['pend'])\
                        - (.15 * df_all_sits['pent']) + (.01 * df_all_sits['ifac_win']) - (.01 * df_all_sits['ifac_win'])\
                        + (.05 * df_all_sits['corsi_f']) - (.05 * df_all_sits['corsi_a']) + (.15 * df_all_sits['goals_f'])\
                        - (.15 * df_all_sits['goals_a'])

    # Get Per 60
    df_all_sits['gs60'] = df_all_sits['gs'] * 60 / df_all_sits['toi_on']

    # Toi per game
    df_all_sits['toi/gp'] = df_all_sits['toi_on'] / df_all_sits['games']

    return df_all_sits


def calc_marcel_weights(df):
    """
    Get the marcel weights using 3 years to predict 1

    :param df: DataFrame of data for all seasons

    :return: None
    """
    cols = ['toi_on', 'gs60', 'toi/gp']

    # Only essential columns
    df = df[["player", "player_id", "season"] + cols]

    # Copy over ALL Data
    predict_df = df.copy()

    # To get fours years in a row (3 predicting 1).. we run 'get_previous_year' 3 times
    # Each time we get n-_ by using predict_col
    # NOTE: I'm writing over df here!!!!
    for seasons in range(1, 4):
        df = get_previous_yr(df, predict_df, seasons)
        df = df[~df['toi_on_n-{}'.format(seasons)].isnull()]

    # Filter for minimum toi
    # 400 for first 3 and 800 for last
    df = df[(df['toi_on'] >= 800) & (df['toi_on_n-1'] >= 400) & (df['toi_on_n-2'] >= 400) & (df['toi_on_n-3'] >= 400)]

    print("\nPlayers: {}".format(df.shape[0]))

    for col in ['gs60', 'toi/gp']:
        print("Getting the Weights for: ", col)
        # Prepare shit
        model_features = df[['{}_n-1'.format(col), '{}_n-2'.format(col), '{}_n-3'.format(col)]].values.tolist()
        model_target = df[col].values.tolist()
        model_features, model_target = np.array(model_features), np.array(model_target).ravel()

        lr = LinearRegression()
        lr.fit(model_features, model_target)

        # Print all the Coefficient neatly
        print("Coefficients:")
        for season, coef in zip(range(1, 4), lr.coef_):
            print("Season n-{}:".format(season), round(coef, 3))

        print("")


def get_reliability(df, pos):
    """
    Get Reliability using the weights 
    
    """
    weights = {
        "F": {
            'gs60': {"n-1": .62, "n-2": .22, "n-3": .16},
            'toi/gp': {"n-1": .9, "n-2": .1, "n-3": 0}
        },
        "D": {
            'gs60': {"n-1": .6, "n-2": .25, "n-3": .15},
            'toi/gp': {"n-1": .85, "n-2": .15, "n-3": 0}
        }
    }

    # Only essential columns
    df = df[["player", "player_id", "season", 'toi_on', 'gs60', 'toi/gp']]

    # Copy over ALL Data
    predict_df = df.copy()

    # To get fours years in a row (3 predicting 1).. we run 'get_previous_year' 3 times
    # Each time we get n-_ by using predict_col
    # NOTE: I'm writing over df here!!!!
    for seasons in range(1, 4):
        df = get_previous_yr(df, predict_df, seasons)
        # Instead of dropping we just fill it
        df = df.fillna(0)

    # Filter for minimum toi over the previous 3 years
    # I'll do 1200 over three years to qualify (because b4 we did at least 400 for each 3)
    # Also need 800 in year 4
    df = df[(df['toi_on'] >= 800) & (df['toi_on_n-1'] + df['toi_on_n-2'] + df['toi_on_n-3'] >= 1200)]

    print("\nPlayers: {}".format(df.shape[0]))

    # Apply weights and predict
    for col in ['gs60', 'toi/gp']:
        df['weighted_{}'.format(col)] = df['{}_n-1'.format(col)] * weights[pos][col]["n-1"]\
                                      + df['{}_n-2'.format(col)] * weights[pos][col]["n-2"]\
                                      + df['{}_n-3'.format(col)] * weights[pos][col]["n-3"]

        df['weighted_sample_{}'.format(col)] = df['toi_on_n-1'] * weights[pos][col]["n-1"]\
                                             + df['toi_on_n-2'] * weights[pos][col]["n-2"]\
                                             + df['toi_on_n-3'] * weights[pos][col]["n-3"]

        # Prepare shit
        model_features = df['weighted_{}'.format(col)].values.tolist()
        model_target = df[col].values.tolist()
        model_features, model_target = np.array(model_features), np.array(model_target).ravel()
        corr = pearsonr(model_features, model_target)[0]

        print("The Correlation for {}:".format(col), round(corr, 2))
        print("The Constant for {}:".format(col), round((1-corr)/corr * df['weighted_sample_{}'.format(col)].mean(), 0))


def age_adjustment(df):
    consts = {
        "F": {"gs60": 540, "toi/gp": 370},
        "D": {"gs60": 657, "toi/gp": 710},
    }


def main():
    df_forwards = skater_preprocessing("forwards")
    df_defensemen = skater_preprocessing("defensemen")

    print("Forwards:")
    calc_marcel_weights(df_forwards)
    get_reliability(df_forwards, "F")
    print("\nDefensemen:")
    calc_marcel_weights(df_defensemen)
    get_reliability(df_defensemen, "D")


if __name__ == "__main__":
    main()