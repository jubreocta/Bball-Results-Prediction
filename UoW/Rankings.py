import numpy as np
np.seterr(divide="ignore", invalid="ignore")
import pandas as pd
pd.set_option("expand_frame_repr", False)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)

class Common():
    def __init__(self, df):
        self.df = df
        self.n = len(set(self.df["LeftTeam"]).union(set(self.df["RightTeam"])))

    def team_order(self):
        teams = sorted(set(self.df["LeftTeam"]).union(set(self.df["RightTeam"])))
        return {key: value for value,key in enumerate(teams)}
    
    def total_no_won(self, dataset):
        """Creates a diagonal matrix containing the total number of games won
        by each team over the time span of the supplied dataset"""
        team_ordering = self.team_order()
        array = np.array([[0] * self.n] * self.n)
        for i in range(len(dataset)):
            #add 1 for each game won
            if dataset["LeftScore"][i] > dataset["RightScore"][i]:
                home_index = team_ordering[dataset["LeftTeam"][i]]
                array[home_index, home_index] += 1
            else:
                away_index = team_ordering[dataset["RightTeam"][i]]
                array[away_index, away_index] += 1
        return array
    
    def total_no_played(self, dataset):
        """Creates a diagonal matrix containing the total number of games played
        by each team over the time span of the supplied dataset"""
        team_ordering = self.team_order()
        array = np.array([[0] * self.n] * self.n)
        for i in range(len(dataset)):
            #add 1 for each game played
            home_index = team_ordering[dataset["LeftTeam"][i]]
            away_index = team_ordering[dataset["RightTeam"][i]]
            array[home_index, home_index] += 1
            array[away_index, away_index] += 1
        return array
    
    def pairwise_matchups(self, dataset):
        """Creates an off-diagonal matrix containing the number of pairwise
        matchups between teams over the time span of the supplied dataset"""
        team_ordering = self.team_order()
        array = np.array([[0] * self.n] * self.n)
        for i in range(len(dataset)):
            #add 1 for each game between distinct oponents
            home_index = team_ordering[dataset["LeftTeam"][i]]
            away_index = team_ordering[dataset["RightTeam"][i]]
            array[home_index, away_index] += 1
            array[away_index, home_index] += 1
        return array
    
    def subtract_losses_from_wins(self, dataset):
        """Creates a vector containing the total number of losses subtracted from
        the total number of wins for each team over the time span of the supplied
        dataset"""
        team_ordering = self.team_order()
        array = np.array([0] * self.n)
        for i in range(len(dataset)):
            home_index = team_ordering[dataset["LeftTeam"][i]]
            away_index = team_ordering[dataset["RightTeam"][i]]
            #checks who won the game
            if dataset["LeftScore"][i] > dataset["RightScore"][i]:
                array[home_index] += 1
                array[away_index] -= 1
            else:
                array[home_index] -= 1
                array[away_index] += 1     
        return array

class Colley():
    def __init__(self, df):
        self.df = df
        self.common_functions = Common(df)
        self.team_ordering = self.common_functions.team_order()
        self.n = len(self.team_ordering)
        self.home_rating   = []
        self.away_rating   = []
        self.unique_dates = df.Date.unique()
   
    def rank(self):
        for date_index in range(len(self.unique_dates)):
            data_used_for_ranking = self.df[self.df.Date == self.unique_dates[date_index - 1]].reset_index(drop = True)
            data_to_be_ranked = self.df[self.df.Date == self.unique_dates[date_index]].reset_index(drop = True)
            if date_index == 0:
                subtract_losses_from_wins_array = np.array([0] * self.n)
                pairwise_matchups_array = np.array([[0] * self.n] * self.n)
                total_no_played_array = np.array([[0] * self.n] * self.n)
                np.fill_diagonal(total_no_played_array, 2)
            else:
                total_no_played_array   += self.common_functions.total_no_played(data_used_for_ranking)
                pairwise_matchups_array -= self.common_functions.pairwise_matchups(data_used_for_ranking)
                subtract_losses_from_wins_array += self.common_functions.subtract_losses_from_wins(data_used_for_ranking)
            #Colley rating calculations start here
            T = total_no_played_array
            P = pairwise_matchups_array
            C = T + P
            b = 1 + 0.5 * subtract_losses_from_wins_array
            #letters are used inline with book to make referencing easy
            try:
                r = np.linalg.solve(C, b)
            except np.linalg.LinAlgError:
                r = ([None] * self.n)
            #Placing ranks of teams in the dataset)
            for game_index in range(len(data_to_be_ranked)):
                home_index = self.team_ordering[data_to_be_ranked["LeftTeam"][game_index]]
                self.home_rating.append(r[home_index])
                away_index = self.team_ordering[data_to_be_ranked["RightTeam"][game_index]]
                self.away_rating.append(r[away_index])
        self.df = self.df.assign(Left_C = self.home_rating)
        self.df = self.df.assign(Right_C = self.away_rating)
        return self.df

class WinPercentage():
    def __init__(self, df):
        self.df = df
        self.common_functions = Common(df)
        self.team_ordering = self.common_functions.team_order()
        self.n = len(self.team_ordering)
        self.home_rating   = []
        self.away_rating   = []
        self.unique_dates = df.Date.unique()

    def rank(self):
        for date_index in range(len(self.unique_dates)):
            data_used_for_ranking = self.df[self.df.Date == self.unique_dates[date_index - 1]].reset_index(drop = True)
            data_to_be_ranked = self.df[self.df.Date == self.unique_dates[date_index]].reset_index(drop = True)
            if date_index == 0:
                total_no_won_array = np.array([[0] * self.n] * self.n)
                total_no_played_array = np.array([[0] * self.n] * self.n)
            else:
                total_no_won_array   += self.common_functions.total_no_won(data_used_for_ranking)
                total_no_played_array   += self.common_functions.total_no_played(data_used_for_ranking)            
            r = []
            for i in range(len(total_no_played_array.diagonal())):
                if total_no_played_array.diagonal()[i] == 0:
                    r.append(0)
                else:
                    r.append(total_no_won_array.diagonal()[i] / total_no_played_array.diagonal()[i])
            r = np.array(r)    
            for game_index in range(len(data_to_be_ranked)):
                home_index = self.common_functions.team_order()[data_to_be_ranked["LeftTeam"][game_index]]
                self.home_rating.append(r[home_index])
                away_index = self.common_functions.team_order()[data_to_be_ranked["RightTeam"][game_index]]
                self.away_rating.append(r[away_index])
        self.df = self.df.assign(Left_WP = self.home_rating)
        self.df = self.df.assign(Right_WP = self.away_rating)
        return self.df

class OneSeason:
    def __init__(self, df):
        self.df = df
        self.seasons = df.Season.unique()
        self.final_df = []

    def do_seasonal_ranking(self):
        for season in self.seasons:
            season_data = self.df[self.df.Season == season]
            self.final_df.append(
                WinPercentage(
                    Colley(season_data).rank()
                ).rank()
            )
        return pd.concat(self.final_df, ignore_index=True)

        # massey = massey_for_a_season(data)
        # #print(massey)
        # od = od_for_a_season(data)
        # #print(od)
        # markov1 = markov_for_a_season(data,1)
        # #print(markov1)
        # markov2 = markov_for_a_season(data,2)
        # #print(markov2)
        # markov3 = markov_for_a_season(data,3)
        # #print(markov3)
    #return markov3