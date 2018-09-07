"""
Gets the goalies for both teams and their "true" adj_Fsv% using marcels.  
"""
import pandas as pd
import json


def get_players(df):
    """
    Get all Goalies who dressed for a game
    
    :param df: DataFrame of all games
    
    :return Dict - player and stats for each season (as of right now there is obviously nothing)
    """
    total_players = []
    for goalie in ['Home_Starter', 'Away_Starter', 'Home_Backup', 'Away_Backup']:
        total_players.extend(df[goalie].unique())

    players_dict = {}
    for player in total_players:
        players_dict[player] = {"2007": {"xg": 0, "fen": 0, "goals": 0}, "2008": {"xg": 0, "fen": 0, "goals": 0},
                                "2009": {"xg": 0, "fen": 0, "goals": 0}, "2010": {"xg": 0, "fen": 0, "goals": 0},
                                "2011": {"xg": 0, "fen": 0, "goals": 0}, "2012": {"xg": 0, "fen": 0, "goals": 0},
                                "2013": {"xg": 0, "fen": 0, "goals": 0}, "2014": {"xg": 0, "fen": 0, "goals": 0},
                                "2015": {"xg": 0, "fen": 0, "goals": 0}, "2016": {"xg": 0, "fen": 0, "goals": 0},
                                "2017": {"xg": 0, "fen": 0, "goals": 0}
                                }
    return players_dict


def assign_shots(goalie_col, row, players):
    """
    Assign Shot for given play

    NOTE: This includes playoff data!!!!!!!!!!!!!!!
    """
    players[row[goalie_col]][row['game_id'][:4]]["xg"] += row["xGA_{}".format(goalie_col)]
    players[row[goalie_col]][row['game_id'][:4]]["goals"] += row["GA_{}".format(goalie_col)]
    players[row[goalie_col]][row['game_id'][:4]]["fen"] += row['FA_{}'.format(goalie_col)]


def get_goalie_marcels(df):
    """
    Get marcels for each goalie each game
    
    :param df: DataFrame with data from every game
    
    :return: DataFrame -> game_id, each player and their adj_fsv
    """
    players = get_players(df)

    # 0 = that year, 1 is year b4 ....
    marcel_weights = [.36, .29, .21, .14]
    reg_const = 2000
    reg_avg = 0  # Where to regress to

    games_list = []
    for row in df.to_dict("records"):
        game_dict = {"game_id": row['game_id']}
        season = row['game_id'][:4]

        for goalie in ['Home_Starter', 'Away_Starter', 'Home_Backup', 'Away_Backup']:
            weighted_goals_sum, weighted_fen_sum, weighted_xg_sum, weights_sum = 0, 0, 0, 0

            # Past 4 Seasons
            for i in range(0, 4):
                if int(season) - i > 2006:
                    weighted_goals_sum += players[row[goalie]][str(int(season) - i)]["goals"] * marcel_weights[i]
                    weighted_fen_sum += players[row[goalie]][str(int(season) - i)]["fen"] * marcel_weights[i]
                    weighted_xg_sum += players[row[goalie]][str(int(season) - i)]["xg"] * marcel_weights[i]

                    # -> To divide by at end...normalize everything
                    weights_sum += marcel_weights[i]

            # Normalize weighted sums
            weighted_xg_sum = weighted_xg_sum / weights_sum if weights_sum != 0 else 0
            weighted_goals_sum = weighted_goals_sum / weights_sum if weights_sum != 0 else 0
            weighted_fen_sum = weighted_fen_sum / weights_sum if weights_sum != 0 else 0

            # Get Regressed
            if weighted_fen_sum != 0:
                weighted_adj_fsv = ((1 - weighted_goals_sum / weighted_fen_sum) - (1 - weighted_xg_sum / weighted_fen_sum)) * 100
            else:
                weighted_adj_fsv = 0
            reg_adj_fsv = weighted_adj_fsv - ((weighted_adj_fsv - reg_avg) * (reg_const / (reg_const + weighted_fen_sum)))

            # Add to game dictionary for that goalie
            game_dict[goalie] = row[goalie]
            game_dict[goalie + "_adj_fsv"] = reg_adj_fsv

            # Now we can assign the numbers for that game
            assign_shots(goalie, row, players)

        # Add to bullshit
        games_list.append(game_dict)

    return pd.DataFrame(games_list)


def merge_goalie_data(df_games):
    """
    Merge the goalie data for each game into the DataFrame with the Starter and Backup info
    
    :param df_games: DataFrame we are merging the goalie data into
    
    :return DataFrame with some data with goalie data included
    """
    df_goalies = pd.read_csv("goalies_even_game.csv")
    df_goalies = df_goalies.sort_values(by=['Date'])
    df_goalies['game_id'] = df_goalies.apply(lambda x: str(x['Season']) + "0" + str(x['Game.ID']), axis=1)
    df_goalies = df_goalies[["game_id", "Player", 'GA', 'FA', 'xGA']]

    # I do these separately because I have no idea how to get the suffix on them
    df_games = pd.merge(df_games, df_goalies, how="left", left_on=["game_id", "Home_Starter"], right_on=["game_id", "Player"])
    df_games = df_games.drop(["Player"], axis=1)

    for goalie_col in ["Away_Starter", "Home_Backup", "Away_Backup"]:
        df_games = pd.merge(df_games, df_goalies, how="left", left_on=["game_id", goalie_col],
                            right_on=["game_id", "Player"], suffixes=['', '_{}'.format(goalie_col)])
        df_games = df_games.drop(["Player"], axis=1)

    # Fix Home_Starter columns
    df_games['GA_Home_Starter'] = df_games['GA']
    df_games['FA_Home_Starter'] = df_games['FA']
    df_games['xGA_Home_Starter'] = df_games['xGA']
    df_games = df_games.drop(['GA', 'FA', 'xGA'], axis=1)

    # Fill all empty ones with 0
    df_games = df_games.fillna(0)

    return df_games


def convert_goalies_to_df(rosters):
    """
    Convert the full rosters dict to a DataFrame of goalies for each game
    NOTE: Throws out games without two goalies for each side!!!!!!! (there are very few...)
    
    :param rosters: Rosters for every game
    
    :return List - holds dict with game id and starting and backup goalie for each team
    """
    game_goalies = []
    for game_id in rosters.keys():
        # Only if 2 each!!!!!!!!
        if len(rosters[game_id]['players']["Home"]["G"].keys()) == 2 and len(rosters[game_id]['players']["Away"]["G"].keys()) == 2:
            game_goalies.append({"game_id": game_id,
                                 "Home_Starter": rosters[game_id]['players']["Home"]["G"]["Starter"],
                                 "Away_Starter": rosters[game_id]['players']["Away"]["G"]["Starter"],
                                 "Home_Backup": rosters[game_id]['players']["Home"]["G"]["Backup"],
                                 "Away_Backup": rosters[game_id]['players']["Away"]["G"]["Backup"]})

    return pd.DataFrame(game_goalies, columns=['game_id', "Home_Starter", "Away_Starter", "Home_Backup", "Away_Backup"])


def get_goalies(df):
    """
    Get goalies for both teams (indicate if starter) and their marcel adj_fsv%
    
    :param df: Either the player/team model data
    
    :return Pandas -> With goalie data included for every game
    """
    with open("game_rosters.json") as file:
        rosters = json.loads(file.read())

    # Get goalies and merge them in
    df_games = convert_goalies_to_df(rosters)
    df_games = merge_goalie_data(df_games)

    marcels_df = get_goalie_marcels(df_games)

    # Final Merge...this merges with whatever df we were provided (either team or player)
    merged_df = pd.merge(df, marcels_df, how="left", on="game_id")

    return merged_df


def main():
    pass


if __name__ == "__main__":
    main()

