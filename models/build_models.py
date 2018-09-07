"""
Build the models - team, player, meta
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from models import team_model
from models import player_model
from models import elo_ratings

pd.options.mode.chained_assignment = None  # default='warn'


def get_team_features(df):
    """
    Get the features for the team model

    :param df: Team df of all features and labels

    :return: features
    """
    continuous_vars = ['FA60_even_Opponent', 'FA60_even_Team',
                       'FA60_pk_Opponent', 'FA60_pk_Team',
                       'FF60_even_Opponent', 'FF60_even_Team',
                       'FF60_pp_Opponent', 'FF60_pp_Team',
                       'GF60/xGF60_even_Opponent', 'GF60/xGF60_even_Team',
                       'GF60/xGF60_pp_Opponent', 'GF60/xGF60_pp_Team',
                       'PEND60_Opponent', 'PEND60_Team',
                       'PENT60_Opponent', 'PENT60_Team',
                       'xGA60/FA60_even_Opponent', 'xGA60/FA60_even_Team',
                       'xGA60/FA60_pk_Opponent', 'xGA60/FA60_pk_Team',
                       'xGF60/FF60_even_Opponent', 'xGF60/FF60_even_Team',
                       'xGF60/FF60_pp_Opponent', 'xGF60/FF60_pp_Team',
                       'days_rest_home', 'days_rest_away',
                       'home_adj_fsv', 'away_adj_fsv']

    non_scaled = ['elo_prob']
    dummies = ['home_b2b', 'away_b2b']

    # Switch it over -> Don't want to overwrite anything
    df_scaled = df[continuous_vars + non_scaled + dummies]

    # Scale only continuous vars
    scaler = StandardScaler().fit(df_scaled[continuous_vars])
    df_scaled[continuous_vars] = scaler.transform(df_scaled[continuous_vars])

    # Save Scaler
    pickle.dump(scaler, open("team_scaler.pkl", 'wb'))

    return df_scaled[continuous_vars + non_scaled + dummies].values.tolist()


def get_player_features(df):
    """
    Get the features for the player model

    :param df: Team df of all features and labels
    
    :return: features 
    """
    continuous_vars = ['Away_D_1', 'Away_D_2', 'Away_D_3', 'Away_D_4', 'Away_D_5', 'Away_D_6',
                       'Away_F_1', 'Away_F_2', 'Away_F_3', 'Away_F_4', 'Away_F_5', 'Away_F_6', 'Away_F_7', 'Away_F_8',
                       'Away_F_9', 'Away_F_10', 'Away_F_11', 'Away_F_12',
                       'Home_D_1', 'Home_D_2', 'Home_D_3', 'Home_D_4', 'Home_D_5', 'Home_D_6',
                       'Home_F_1', 'Home_F_2', 'Home_F_3', 'Home_F_4', 'Home_F_5', 'Home_F_6', 'Home_F_7', 'Home_F_8',
                       'Home_F_9', 'Home_F_10', 'Home_F_11', 'Home_F_12',
                       'Away_Backup_adj_fsv', 'Away_Starter_adj_fsv', 'Home_Backup_adj_fsv', 'Home_Starter_adj_fsv',
                       ]
    dummies = ['home_b2b', 'away_b2b']

    # Switch it over -> Don't want to overwrite anything
    df_scaled = df[continuous_vars + dummies]

    # Scale only continuous vars
    scaler = StandardScaler().fit(df_scaled[continuous_vars])
    df_scaled[continuous_vars] = scaler.transform(df_scaled[continuous_vars])

    # Save Scaler
    pickle.dump(scaler, open("player_scaler.pkl", 'wb'))

    return df_scaled[continuous_vars + dummies].values.tolist()


def test_model(preds_probs, labels):
    """
    Test the log loss and accuracy of the given model

    :param preds_probs: List of probabilities for test set
    :param labels: Corresponding labels for each 

    :return: None
    """
    # Convert test labels to a list instead of lists of lists
    # I honestly have no idea...
    if type(labels[0]) == list:
        flat_test_labels = [label[0] for label in labels]
    else:
        flat_test_labels = labels

    print("Log Loss: ", round(log_loss(flat_test_labels, preds_probs), 4))
    print("Accuracy:", round(accuracy_score(flat_test_labels, [round(prob[1]) for prob in preds_probs]), 4))


def train_test_clf(clf_type, features, labels):
    """
    Train the data on specific classifier. Then test how it does. 

    :param clf_type: Type of Classifier we are using
    :param features: Vars for model
    :param labels: What we are predicting

    :return: Trained Classifier
    """
    # 75/25 Split
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=.25, random_state=36)

    # Fix Data
    features_train, labels_train = np.array(features_train), np.array(labels_train).ravel()

    # FIT MODEL
    clf = clf_type.fit(features_train, labels_train)

    # Get Probs for training set and get test results
    print("\nTraining Set Results:")
    train_probs = clf.predict_proba(features_train)
    test_model(train_probs, labels_train)

    # Get Probs for test set and get test results
    print("\nTesting Set Results:")
    test_probs = clf.predict_proba(features_test)
    test_model(test_probs, labels_test)

    return clf


def build_team_model(df):
    """
    Build the Team model

    :param df: Team model DataFrame
    
    :return None 
    """
    labels = ['if_home_win']

    # Flip Penalties Taken and for
    # Fucked up in some earlier code and have drawn and taken mixed up
    for venue in ["Opponent", "Team"]:
        tmp_col = df['PEND60_{}'.format(venue)]
        df['PEND60_{}'.format(venue)] = df['PENT60_{}'.format(venue)]
        df['PENT60_{}'.format(venue)] = tmp_col

    # Transform to test and training arrays
    features = get_team_features(df)
    labels = df[labels].values.tolist()

    # Set up CV
    param_grid = {'n_estimators': range(2, 22, 2)}
    clf_type = BaggingClassifier(LogisticRegression(penalty='l1', solver='liblinear', random_state=36), random_state=36)

    # Tune hyperparameters
    cv_clf = GridSearchCV(estimator=clf_type, param_grid=param_grid, cv=5, verbose=2)

    # Train and Test Classifier
    print("Fitting Team Model...")
    clf = train_test_clf(cv_clf, features, labels)

    print("\nThe estimator chosen was:")
    print(clf.best_estimator_)

    # Save model
    pickle.dump(clf, open("team_classifier.pkl", 'wb'))


def build_player_model(df):
    """
    Build the player model
    
    :param df: Player model DataFrame
    
    :return None
    """
    labels = ['if_home_win']

    # Transform to test and training arrays
    features = get_player_features(df)
    labels = df[labels].values.tolist()

    # Set up CV
    param_grid = {'n_estimators': range(2, 22, 2)}
    clf_type = BaggingClassifier(LogisticRegression(penalty='l1', solver='liblinear', random_state=36), random_state=36)

    # Tune hyperparameters
    cv_clf = GridSearchCV(estimator=clf_type, param_grid=param_grid, cv=5, verbose=2)

    # Train and Test Classifier
    print("\nFitting Player Model...")
    clf = train_test_clf(cv_clf, features, labels)

    print("\nThe estimator chosen was:")
    print(clf.best_estimator_)

    # Save model
    pickle.dump(clf, open("player_classifier.pkl", 'wb'))


def build_meta_clf(team_df, player_df, elo_df):
    """
    Build the meta classifier -> made up of the Team and Player models
    
    :param team_df: DataFrame of team features 
    :param player_df: DataFrame of player features
    :param elo_df: DataFrame of elo ratings and some other info
    
    :return None
    """
    ensemble_df = pd.DataFrame()

    # Get both classifiers and feature sets
    models = {
        'team': {"clf": joblib.load("team_classifier.pkl"), "features": get_team_features(team_df)},
        'player': {"clf": joblib.load("player_classifier.pkl"), "features": get_player_features(player_df)}
    }

    # Add probs for each model
    ensemble_df['game_id'] = elo_df['game_id']
    ensemble_df['Season'] = elo_df['Season']
    ensemble_df['if_home_win'] = elo_df['if_home_win']
    ensemble_df['team'] = pd.Series([x[1] for x in models['team']['clf'].predict_proba(np.array(models['team']['features']))])
    ensemble_df['player'] = pd.Series([x[1] for x in models['player']['clf'].predict_proba(np.array(models['player']['features']))])
    ensemble_df['elo'] = elo_df['home_prob']

    # Get Data for meta clf
    features = ensemble_df[['team', 'player']].values.tolist()
    labels = team_df['if_home_win'].values.tolist()

    # Set up CV
    param_grid = {'n_estimators': range(2, 22, 2)}
    clf_type = BaggingClassifier(LogisticRegression(penalty='l1', solver='liblinear', random_state=36), random_state=36)

    # Tune hyperparameters
    cv_clf = GridSearchCV(estimator=clf_type, param_grid=param_grid, cv=5)

    # Train and Test Classifier
    print("\nFitting Meta Classifier")
    clf = train_test_clf(cv_clf, features, labels)

    print("\nThe estimator chosen was:")
    print(clf.best_estimator_)

    # Save model
    pickle.dump(clf, open("meta_classifier.pkl", 'wb'))


def build_separate_models():
    """
    Build each separate model
    1. Team 
    2. Player
    3. Ensemble of 2
    
    Note: Some data gets 
    """
    # NOTE: The models should be ready to be built here.
    # Also if you have the means of getting the data yourself then you can uncomment the below and get the info from
    # those functions (only do this if you made any changes or don't trust the data I provided).
    #team_df = team_model.get_model_data()
    #player_df = player_model.get_model_data()
    #elo_df = elo_ratings.get_elo().reset_index(drop=True)
    team_df = pd.read_csv("./data/team_model_data.csv", index_col=0).reset_index(drop=True)
    player_df = pd.read_csv("./data/player_model_data.csv", index_col=0).reset_index(drop=True)
    elo_df = pd.read_csv("./data/elo_df.csv", index_col=0).reset_index(drop=True)

    # Add b2b from teams into the players model data
    player_df = player_df.merge(team_df[['game_id', 'home_b2b', 'away_b2b']], how='inner', on=['game_id'])

    # Add in elo probability to the team model
    team_df['elo_prob'] = elo_df['home_prob']

    # Train and Test the Team, Player, elo, and the meta
    build_team_model(team_df)
    build_player_model(player_df)
    build_meta_clf(team_df, player_df, elo_df)


def main():
    build_separate_models()


if __name__ == "__main__":
    main()