"""
Get The starting rosters for a given list of games
"""

import pandas as pd
from bs4 import BeautifulSoup
import helpers
import hockey_scraper as hs


def fix_name(player):
    """
    Get rid of (A) or (C) when a player has it attached to their name
    
    Also fix to "correct" name -> The full list is in helpers.py

    :param player: list of player info -> [number, position, name]

    :return: fixed list
    """
    if player.find('(A)') != -1:
        player = player[:player.find('(A)')].strip()
    elif player.find('(C)') != -1:
        player = player[:player.find('(C)')].strip()

    return helpers.fix_name(player)


def fix_players_names(players):
    """
    Fix players names ... not done by hockey_scraper
    
    :return "Fixed" name
    """
    for team in ['Home', 'Away']:
        for index in range(len(players[team])):
            players[team][index][2] = helpers.fix_name(players[team][index][2])

    return players


def transform_data(players, goalies):
    """
    Combine the players and goalies list. Also move stuff into the way I want it
    """
    combined_players_list = {"Home":  {"F": [], "D": [], "G": dict()},
                             "Away":  {"F": [], "D": [], "G": dict()}
                             }

    for venue in ['Home', 'Away']:
        # First deal with players
        for player in players[venue]:
            if not player[3]:
                if player[1] in ["R", "C", "L", "RW", "LW", "F"]:
                    combined_players_list[venue]["F"].append({"player": player[2], 'number': player[0]})
                if player[1] in ["D", "LD", "RD", "DR", "DL"]:
                    combined_players_list[venue]["D"].append({"player": player[2], 'number': player[0]})

        # Now I add goalies
        for goalie_type in goalies[venue].keys():
            combined_players_list[venue]["G"][goalie_type] = goalies[venue][goalie_type]

    return combined_players_list


def get_soup(roster):
    """
    Get the "souped" up doc
    
    NOTE: Scrapping combines this and getting the players and head coaches...not the cleanest design decision
    
    :return: Soup and players
    """
    soup = BeautifulSoup(roster, "lxml")
    players = hs.playing_roster.get_players(soup)

    if len(players) == 0:
        soup = BeautifulSoup(roster, "html.parser")
        players = hs.playing_roster.get_players(soup)

        if len(players) == 0:
            soup = BeautifulSoup(roster, "html5lib")
            players = hs.playing_roster.get_players(soup)

    return soup, players


def get_teams(soup):
    """
    Get the home and away teams from the file
    Away = <td align="center" width="50%" class="teamHeading + border ">NEW JERSEY DEVILS</td> y 
    Home = <td align="center" width="50%" class="teamHeading + border">PHILADELPHIA FLYERS</td>
    
    :param soup: doc
    
    :return Dict -> Home and Away teams
    """
    away_team = soup.find_all("td", {"class": "teamHeading + border "})[0].get_text()
    home_team = soup.find_all("td", {"class": "teamHeading + border"})[0].get_text()

    return {"Away": helpers.TEAMS[away_team], "Home": helpers.TEAMS[home_team]}


def get_goalies(soup, players):
    """
    Get the starting and backup goalie for each team for a given game
    
    :param soup: doc
    :param players: Players in that game
    
    :return Dict -> Starters/Backup for each game
    """
    goalies = {"Home": dict(), "Away": dict()}

    # Should just be 2
    player_tables = soup.find_all("td", {"valign": "top", "width": "50%", "class": "border"})

    # Get the bolded players for each team (with fixed names)
    bold_players = dict()
    for table, team in zip(player_tables, ["Away", "Home"]):
        bolds = table.find_all("td", {"align": "left", "width": "70%", "class": "bold"})
        bold_players[team] = [fix_name(player.get_text()) for player in bolds]

    # Go through each list of players and pluck out both goalies (and differentiate between them)
    for team in ['Home', 'Away']:
        for player in players[team]:
            # First check if a Goalie - then place as a Stater or a Backup
            if player[1] == "G":
                if player[2] in bold_players[team]:
                    goalies[team]["Starter"] = player[2]
                else:
                    goalies[team]["Backup"] = player[2]

    return goalies


# TODO: This only works on the season level...swap this out in production code!!!!!!!
def get_games(season, games):
    """
    Is given a collections of game_id's and returns the corresponding rosters for each.
    NOTE: Works on the season level
    
    :param season: Given Season
    :param games: list of games
    
    :return Dict -> games with rosters
    """
    game_players = dict()

    for game in games:
        game_id = ''.join([str(season), '0', str(game)])
        file_name = game_id + '.txt'
        print(file_name)

        file = open("../hockey_scraper_data/docs/{}/html_roster/{}".format(season, file_name))
        soup, players = get_soup(file)

        teams = get_teams(soup)
        players = fix_players_names(players)
        goalies = get_goalies(soup, players)

        combined_list = transform_data(players, goalies)

        # Make sure have two goalies for each team
        if len(combined_list['Home']['G'].keys()) != 2 or len(combined_list['Away']['G'].keys()) != 2:
            print("For game", game_id, "the number of goalies is wrong.")
        # Make sure have 20 players for each team
        if (len(combined_list['Home']['F']) + len(combined_list['Home']['D'])) != 18 or \
           (len(combined_list['Away']['F']) + len(combined_list['Away']['D'])) != 18:
            print("For game", game_id, "the number of players is wrong.")

        game_players[str(game_id)] = {'players': combined_list, 'teams': teams}

    return game_players


def main():
    games_players = dict()

    for season in range(2007, 2018):
        season_df = pd.read_csv("../data/nhl_pbp{}{}.csv".format(season, season + 1))
        game_ids = list(season_df['Game_Id'].unique())
        game_ids.sort()

        players = get_games(season, game_ids)
        games_players.update(players)

    import json
    with open("game_rosters.json", "w") as file:
        file.write(json.dumps(games_players))


if __name__ == "__main__":
    main()