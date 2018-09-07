import json
from statistics import pstdev

import pandas as pd

import helpers
from models import get_goalie_stats as ggs
from models.skater_marcels_consts import *


def get_stats_dict():
    """
    Get the structure for the dictionary used in the players stats
    """
    return {'toi_on_all': 0, 'goals': 0, 'a1': 0, 'a2': 0, 'icors': 0, 'pend': 0, 'pent': 0, 'iblocks': 0,
            'ifac_win': 0, 'ifac_loss': 0, 'games': 0, 'toi_on_even': 0, 'corsi_f': 0, 'corsi_a': 0, 'goals_f': 0,
            'goals_a': 0, 'toi_off_even': 0, 'gp': 0}


def get_players(df):
    """
    Get all players who played in a game
    
    :param df: DataFrame - contains all players
    
    :return dict - players with stats dict
    """
    players_dict = dict()
    for player in df["player"].unique():
        players_dict[player] = {str(season): get_stats_dict() for season in range(2007, 2018)}

    return players_dict


def assign_shots(game_id, stats_df, players):
    """
    Assign Shot for a given game

    NOTE: This includes playoff data!!!!!!!!!!!!!!!
    """
    stat_cols = ['toi_on_all', 'goals', 'a1', 'a2', 'icors', 'iblocks', 'pend', 'pent', 'ifac_win', 'ifac_loss', 'games',
                 'corsi_f', 'corsi_a', 'goals_f', 'goals_a']

    for row in stats_df[stats_df.game_id == game_id].to_dict("records"):
        for stat in stat_cols:
            players[row['player']][str(row['season'])][stat] += row[stat]
        players[row['player']][str(row['season'])]['gp'] += 1


def calc_game_score(player):
    """
    Calculate game score per 60 for a player (it's the weighted sample)
    
    :param player: Some Asshole
    
    :return: weighted game score per 60 given by marcel weighting
    """
    # Calculate Game Score and Game Score per 60
    player['gs'] = (.75 * player['goals']) + (.7 * player['a1']) + (.55 * player['a2']) + (.049 * player['icors']) \
                   + (.05 * player['iblocks']) + (.15 * player['pend']) - (.15 * player['pent']) \
                   + (.01 * player['ifac_win']) - (.01 * player['ifac_loss']) + (.05 * player['corsi_f']) \
                   - (.05 * player['corsi_a']) + (.15 * player['goals_f']) - (.15 * player['goals_a'])

    return player['gs'] * 60 / player['toi_on_all']


def get_marcels_player(players, row, player_col, season):
    """
    Get the Marcels for a given player
    """
    pos = player_col[5]
    weighted_stats = {'toi_on_all': 0, 'goals': 0, 'a1': 0, 'a2': 0, 'icors': 0, 'iblocks': 0, 'pend': 0, 'pent': 0,
                      'ifac_win': 0, 'ifac_loss': 0, 'toi_on_even': 0, 'corsi_f': 0, 'corsi_a': 0, 'goals_f': 0,
                      'goals_a': 0, 'toi_all_gp': 0, 'gs_sum': 0, 'toi_sum': 0, 'gp': 0}

    # Get Stats (and weight them) for the Past 3 Seasons
    for i in range(0, 3):
        if int(season) - i > 2006:
            # Add all stats in by weight
            weighted_stats['toi_on_all'] += players[row[player_col]][str(int(season) - i)]["toi_on_all"] * marcel_weights[pos]['gs60'][i]
            weighted_stats['goals'] += players[row[player_col]][str(int(season) - i)]["goals"] * marcel_weights[pos]['gs60'][i]
            weighted_stats['a1'] += players[row[player_col]][str(int(season) - i)]["a1"] * marcel_weights[pos]['gs60'][i]
            weighted_stats['a2'] += players[row[player_col]][str(int(season) - i)]["a2"] * marcel_weights[pos]['gs60'][i]
            weighted_stats['icors'] += players[row[player_col]][str(int(season) - i)]["icors"] * marcel_weights[pos]['gs60'][i]
            weighted_stats['iblocks'] += players[row[player_col]][str(int(season) - i)]["iblocks"] * marcel_weights[pos]['gs60'][i]
            weighted_stats['pend'] += players[row[player_col]][str(int(season) - i)]["pend"] * marcel_weights[pos]['gs60'][i]
            weighted_stats['pent'] += players[row[player_col]][str(int(season) - i)]["pent"] * marcel_weights[pos]['gs60'][i]
            weighted_stats['ifac_win'] += players[row[player_col]][str(int(season) - i)]["ifac_win"] * marcel_weights[pos]['gs60'][i]
            weighted_stats['ifac_loss'] += players[row[player_col]][str(int(season) - i)]["ifac_loss"] * marcel_weights[pos]['gs60'][i]
            weighted_stats['corsi_f'] += players[row[player_col]][str(int(season) - i)]["corsi_f"] * marcel_weights[pos]['gs60'][i]
            weighted_stats['corsi_a'] += players[row[player_col]][str(int(season) - i)]["corsi_a"] * marcel_weights[pos]['gs60'][i]
            weighted_stats['goals_f'] += players[row[player_col]][str(int(season) - i)]["goals_f"] * marcel_weights[pos]['gs60'][i]
            weighted_stats['goals_a'] += players[row[player_col]][str(int(season) - i)]["goals_a"] * marcel_weights[pos]['gs60'][i]

            weighted_stats['toi_all_gp'] += players[row[player_col]][str(int(season) - i)]["toi_on_all"] * marcel_weights[pos]['toi/gp'][i]
            weighted_stats['gp'] += players[row[player_col]][str(int(season) - i)]["gp"] * marcel_weights[pos]['toi/gp'][i]

            # -> To divide by at end...normalize everything
            weighted_stats['gs_sum'] += marcel_weights[pos]['gs60'][i]
            weighted_stats['toi_sum'] += marcel_weights[pos]['toi/gp'][i]

    # Normalize for Game Score
    norm_cols = ['toi_on_all', 'goals', 'a1', 'a2', 'icors', 'iblocks', 'pend', 'pent', 'ifac_win', 'ifac_loss',
                 'corsi_f', 'corsi_a', 'goals_f', 'goals_a']
    for key in norm_cols:
        weighted_stats[key] = weighted_stats[key] / weighted_stats['gs_sum'] if weighted_stats['gs_sum'] != 0 else 0

    # Normalize for toi/gp
    weighted_stats['toi_all_gp'] = weighted_stats['toi_all_gp'] / weighted_stats['toi_sum'] if weighted_stats['toi_sum'] != 0 else 0
    weighted_stats['gp'] = weighted_stats['gp'] / weighted_stats['toi_sum'] if weighted_stats['toi_sum'] != 0 else 0
    toi_per_gp = weighted_stats['toi_all_gp'] / weighted_stats['gp'] if weighted_stats['gp'] != 0 else 0

    # Calculate regressed game score and toi for player
    weighted_wpaa = calc_game_score(weighted_stats) if weighted_stats['toi_on_all'] != 0 else 0
    reg_wpaa = weighted_wpaa - ((weighted_wpaa - reg_avgs[pos]["gs60"]) * (reg_consts[pos]["gs60"] /
                                                                           (reg_consts[pos]["gs60"] + weighted_stats['toi_on_all'])))
    reg_toi = toi_per_gp - ((toi_per_gp - reg_avgs[pos]["toi/gp"]) * (reg_consts[pos]["toi/gp"] /
                                                                      (reg_consts[pos]["toi/gp"] + weighted_stats['toi_all_gp'])))

    return {'gs': reg_wpaa, 'toi': reg_toi}


def get_marcels(df_rosters, stats_df, players):
    """
    1. Go through the roster for that game and calculate the marcels for each player given what we already have stored
    2. Then Filter all from stats df of that specific game_id and assign the stats
    
    :param df_rosters: DataFrame of Rosters for each game
    :param stats_df: DataFrame of all stats for players by game
    :param players: Dict of all players
    
    :return list of dict of players and their marcels for each game
    """
    player_cols = ['Away_D_1', 'Away_D_2', 'Away_D_3', 'Away_D_4', 'Away_D_5', 'Away_D_6', 'Away_D_7', 'Away_D_8',
                   'Away_F_1', 'Away_F_10', 'Away_F_11', 'Away_F_12', 'Away_F_13', 'Away_F_14', 'Away_F_2', 'Away_F_3',
                   'Away_F_4', 'Away_F_5', 'Away_F_6', 'Away_F_7', 'Away_F_8', 'Away_F_9', 'Home_D_1', 'Home_D_2',
                   'Home_D_3', 'Home_D_4', 'Home_D_5', 'Home_D_6', 'Home_D_7', 'Home_D_8', 'Home_F_1', 'Home_F_10',
                   'Home_F_11', 'Home_F_12', 'Home_F_13', 'Home_F_14', 'Home_F_2', 'Home_F_3', 'Home_F_4', 'Home_F_5',
                   'Home_F_6', 'Home_F_7', 'Home_F_8', 'Home_F_9']

    games_list = []
    for row in df_rosters.to_dict("records"):
        print(row['game_id'])
        season = row['game_id'][:4]
        game_dict = {"game_id": row['game_id'], 'season': int(season), 'Home': {'F': [], 'D': []}, 'Away': {'F': [], 'D': []}}

        for player_col in player_cols:
            # If it's empty just skip over this pass (float means it's nan)
            if type(row[player_col]) == str:
                player_marcels = get_marcels_player(players, row, player_col, season)
            else:
                continue

            # Add to game dictionary for that skater
            game_dict['Home' if player_col[:4] == "Home" else 'Away'][player_col[5]].append(player_marcels)

        # Now we can assign the numbers for that game
        assign_shots(row['game_id'], stats_df, players)

        # Add to bullshit
        games_list.append(game_dict)

    return games_list


def convert_marcels_to_df(marcels_games):
    """
    Convert the list of games of marcels to a DataFrame.
    
    The order is determined by projected toi/gp. So F_1 is the player with the highest toi/gp marcels and F_12 is the 
    lowest. 

    For some games, teams will differ from the usual 12 F and 6 D. In these cases any extra forwards are added to 
    the backend of the defensemen and any extra defensemen the same for the forward group.

    Ex: When - 13 F and 5 D, the 13th forward will be put in the 6th defensemen slot

    Possibilities (for home & away):
    1. 14 F & 4 D
    2. 13 F & 5 D
    3. 12 F & 6 D
    4. 11 F & 7 D
    5. 10 F & 8 D
    
    :param marcels_games: List of games with marcels (game score and toi/gp) for every player
    
    :return DataFrame of estimated rosters for each game
    """
    from operator import itemgetter

    games_list = []
    for game in marcels_games:
        game_dict = {'game_id': game['game_id'], 'season': game['season']}
        for venue in ['Home', 'Away']:
            # Sort so that the player with the most projected toi is gets popped first for each position
            sorted_forwards = sorted(game[venue]["F"], key=itemgetter('toi'), reverse=False)
            sorted_defensemen = sorted(game[venue]["D"], key=itemgetter('toi'), reverse=False)

            # Assign forwards in order
            f_index = 1
            while sorted_forwards:
                game_dict['_'.join([venue, "F", str(f_index)])] = sorted_forwards.pop()['gs']
                f_index += 1

                # Can't have more than 12 Forwards
                if f_index > 12:
                    break

            # Assign defensemen in order
            d_index = 1
            while sorted_defensemen:
                game_dict['_'.join([venue, "D", str(d_index)])] = sorted_defensemen.pop()['gs']
                d_index += 1

                # Can't have more than 6 Defensemen
                if d_index > 6:
                    break

            # We check if we have any players for each position still available
            # If we do we assign them the other position continuing with the previous index
            # Ex: If still 1 Defensemen left they get put in as the 13th Forward
            # Note: This only runs when they are 18 skaters specified. I don't bother if there are extra. So if we
            # already have 6 defensemen I just break from that loop
            while sorted_forwards:
                if d_index > 6:
                    break
                game_dict['_'.join([venue, "D", str(d_index)])] = sorted_forwards.pop()['gs']
                d_index += 1
            while sorted_defensemen:
                if f_index > 12:
                    break
                game_dict['_'.join([venue, "F", str(f_index)])] = sorted_defensemen.pop()['gs']
                f_index += 1

        games_list.append(game_dict)

    return pd.DataFrame(games_list)


def get_even_data():
    """
    Process the even strength DataFrame
    
    :return: Processed Even Strength DataFrame
    """
    cols = ['player', 'player_id', 'season', 'game_id',
            'toi_on', 'corsi_f', 'corsi_a', 'goals_f', 'goals_a']

    # Get All Data
    dfs = []
    for season in range(2007, 2018):
        for pos in ['forwards', 'defensemen']:
            print("even", pos, season)
            with open("../projection_data/skaters/{}_even_{}.json".format(pos, season)) as file:
                dfs.append(pd.DataFrame(json.load(file)['data'], columns=cols))

    # Combine All Forward and Defensemen Data
    df = pd.concat(dfs)

    # Convert from string to float for some reason
    for col in ["toi_on", "corsi_f", "corsi_a"]:
        df[col] = df[col].astype(float)

    # Change over some names for merging
    df = df.rename(index=str, columns={"toi_on": "toi_on_even"})

    # Get the correct game_id
    df['game_id'] = df.apply(lambda x: str(x['season']) + "0" + str(x['game_id']), axis=1)

    return df


def get_all_sits_data():
    """
    Process the All Situations DataFrame
    
    :return: Processed All Situations DataFrame
    """
    cols = ['player', 'player_id', 'season', 'game_id', 'date', 'position', 'team', 'opponent', 'venue',
            'toi_on', 'goals', 'a1', 'a2', 'icors', 'iblocks', 'pend', 'pent', 'ifac_win', 'ifac_loss', 'games']

    # Get All Data
    dfs = []
    for season in range(2007, 2018):
        for pos in ['forwards', 'defensemen']:
            print("all", pos, season)
            with open("../projection_data/skaters/{}_all_sits_{}.json".format(pos, season)) as file:
                dfs.append(pd.DataFrame(json.load(file)['data'], columns=cols))

    # Combine All Forward and Defensemen Data
    df = pd.concat(dfs)

    # Fix the Fucking Names!!!!!
    for team_col in ['team', "opponent", "venue"]:
        df = helpers.fix_team(df, team_col)

    # Idk
    df['toi_on'] = df['toi_on'].astype(float)

    # Change over some names for merging
    df = df.rename(index=str, columns={"toi_on": "toi_on_all"})

    # Get the correct game_id
    df['game_id'] = df.apply(lambda x: str(x['season']) + "0" + str(x['game_id']), axis=1)

    return df


def get_raw_data():
    """
    Get the raw data by position and strength

    :return: Dictionary for each DataFrame
    """
    df_all = get_all_sits_data()
    df_even = get_even_data()

    df = pd.merge(df_all, df_even, how="left", on=['player', 'player_id', 'season', 'game_id'])

    # Convert to date and sort
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['date'] = df['date'].dt.date
    df = df.sort_values(['season', 'game_id', 'player', 'player_id'])

    return df


def convert_roster_to_df(rosters):
    """
    Convert Roster to DataFrame (don't include Goalies...will be added in later)
    """
    game_skaters_list = []
    for game_id in rosters.keys():
        skaters = {"game_id": game_id,
                   "home_team": rosters[game_id]['teams']['Home'],
                   'away_team': rosters[game_id]['teams']['Away']}
        for venue in ['Home', 'Away']:
            for pos in ['F', 'D']:
                for player_index in range(len(rosters[game_id]['players'][venue][pos])):
                    # Deal with 2nd Sebastian Aho
                    if rosters[game_id]['players'][venue][pos][player_index]['player'] == 'Sebastian Aho' and \
                                    rosters[game_id]['players'][venue][pos][player_index][venue] == 'NYI':
                        skaters['_'.join([venue, pos, str(player_index + 1)])] = '5ebastian Aho'
                    else:
                        skaters['_'.join([venue, pos, str(player_index + 1)])] = \
                            rosters[game_id]['players'][venue][pos][player_index]['player']

        game_skaters_list.append(skaters)

    return pd.DataFrame(game_skaters_list)


def get_model_data():
    """
    Get the data required for the model

    NOTE: This is missing b2b and days of rest...the roster_df is used to construct the DataFrame and it doesn't include
    the Date. I could just merge it then calculate them but it's easier to just merge the player and team df's for the
    appropriate data. 
    
    :return model data DataFrame
    """
    with open("./data/game_rosters.json") as file:
        rosters = json.loads(file.read())

    df_rosters = convert_roster_to_df(rosters)

    # Get the initial data
    df_stats = get_raw_data()
    df_stats['player'] = df_stats.apply(lambda x: '5ebastian Aho' if x['player_id'] == 8480222 else x['player'], axis=1)
    df_stats['position'] = df_stats.apply(lambda x: 'F' if x['position'] != 'D' else 'D', axis=1)
    players = get_players(df_stats)

    # Get Skater Marcels
    game_marcels = get_marcels(df_rosters, df_stats, players)
    df = convert_marcels_to_df(game_marcels)
    df = df.fillna(0)

    # Add if a playoff game
    df['if_playoff'] = df.apply(lambda x: 1 if int(str(x['game_id'])[-5:]) > 30000 else 0, axis=1)

    # Add Goalie Data for starter and backup
    df = ggs.get_goalies(df)
    df = df.drop(["Home_Starter", "Away_Starter", "Home_Backup", "Away_Backup"], axis=1)

    # Fill in any missing value with the column average
    df = df.fillna(df.mean())

    # Get Game Outcomes
    df = helpers.merge_outcomes(df)

    # Drop 2007 season
    df = df[df.season > 2007]

    return df


def main():
    pass


if __name__ == "__main__":
    main()