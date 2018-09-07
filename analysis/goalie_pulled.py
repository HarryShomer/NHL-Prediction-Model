"""
Gets the percentage of a game is played by the 1st or 2nd goalie (apparently there had never been 3 goalies in one 
game...not sure I believe that). 

We get: Starters: .946
        Backups:  .053
 """
import pandas as pd


df = pd.read_csv("/Users/Student/Desktop/projection_data/goalies/goalies_all_sits_game.csv")

df['team_game_id'] = df.apply(lambda row: row['Team'] + str(row['Season']) + "0" + str(row['Game.ID']), axis=1)
df = df.sort_values(by=["team_game_id", "Team", "TOI"])
df = df.reset_index(drop=True)
games_list = list(df['team_game_id'].unique())


goalies = {"1": [], "2": [], "3": []}
for game in games_list:
    df2 = df[df['team_game_id'] == game].reset_index(drop=True)
    for index, row in df2.iterrows():
        goalies[str(index+1)].append(row['TOI'])

total_toi = sum(goalies["1"]) + sum(goalies["2"]) + sum(goalies["3"])
print("Goalie 1%:", sum(goalies["1"])/total_toi)
print("Goalie 2%:", sum(goalies["2"])/total_toi)
print("Goalie 3%:", sum(goalies["3"])/total_toi)

