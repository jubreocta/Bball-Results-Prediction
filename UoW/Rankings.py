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
    
    def pairwise_matchups(self, dataset):
        """Creates an off-diagonal matrix containing the number of pairwise
        matchups between teams over the time span of the supplied dataset"""
        team_ordering = self.team_order()
        array = np.array([[0] * self.n] * self.n)
        for i in range(len(dataset)):
            #add 1 for each game between distinct oponents
            left_index = team_ordering[dataset["LeftTeam"][i]]
            right_index = team_ordering[dataset["RightTeam"][i]]
            array[left_index, right_index] += 1
            array[right_index, left_index] += 1
        return array
            
    def points_against(self, dataset):
        '''Creates a vector containing the total number of points conceded by each
        team over the time span of the supplied dataset'''
        team_ordering = self.team_order()
        array = np.array([0] * self.n)
        for i in range(len(dataset)):
            #add value of point conceeded to array
            array[team_ordering[dataset['LeftTeam'][i]]] += dataset['RightScore'][i]
            array[team_ordering[dataset['RightTeam'][i]]] += dataset['LeftScore'][i]
        return array
        
    def points_for(self, dataset, ):
        '''Creates a vector containing the total number of points scored by each
        team over the time span of the supplied dataset.'''
        team_ordering = self.team_order()
        array = np.array([0] * self.n)
        for i in range(len(dataset)):
            #add value of point scored to array
            array[team_ordering[dataset['LeftTeam'][i]]] += dataset['LeftScore'][i]
            array[team_ordering[dataset['RightTeam'][i]]] += dataset['RightScore'][i]
        return array

    def subtract_losses_from_wins(self, dataset):
        """Creates a vector containing the total number of losses subtracted from
        the total number of wins for each team over the time span of the supplied
        dataset"""
        team_ordering = self.team_order()
        array = np.array([0] * self.n)
        for i in range(len(dataset)):
            left_index = team_ordering[dataset["LeftTeam"][i]]
            right_index = team_ordering[dataset["RightTeam"][i]]
            #checks who won the game
            if dataset["LeftScore"][i] > dataset["RightScore"][i]:
                array[left_index] += 1
                array[right_index] -= 1
            else:
                array[left_index] -= 1
                array[right_index] += 1     
        return array

    def team_order(self):
        teams = sorted(set(self.df["LeftTeam"]).union(set(self.df["RightTeam"])))
        return {key: value for value,key in enumerate(teams)}    
    
    def total_no_played(self, dataset):
        """Creates a diagonal matrix containing the total number of games played
        by each team over the time span of the supplied dataset"""
        team_ordering = self.team_order()
        array = np.array([[0] * self.n] * self.n)
        for i in range(len(dataset)):
            #add 1 for each game played
            left_index = team_ordering[dataset["LeftTeam"][i]]
            right_index = team_ordering[dataset["RightTeam"][i]]
            array[left_index, left_index] += 1
            array[right_index, right_index] += 1
        return array
    
    def total_no_won(self, dataset):
        """Creates a diagonal matrix containing the total number of games won
        by each team over the time span of the supplied dataset"""
        team_ordering = self.team_order()
        array = np.array([[0] * self.n] * self.n)
        for i in range(len(dataset)):
            #add 1 for each game won
            if dataset["LeftScore"][i] > dataset["RightScore"][i]:
                left_index = team_ordering[dataset["LeftTeam"][i]]
                array[left_index, left_index] += 1
            else:
                right_index = team_ordering[dataset["RightTeam"][i]]
                array[right_index, right_index] += 1
        return array
    
class Colley():
    def __init__(self, df):
        self.df = df
        self.common_functions = Common(df)
        self.team_ordering = self.common_functions.team_order()
        self.n = len(self.team_ordering)
        self.left_rating    = []
        self.right_rating   = []
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
                left_index = self.team_ordering[data_to_be_ranked["LeftTeam"][game_index]]
                self.left_rating.append(r[left_index])
                right_index = self.team_ordering[data_to_be_ranked["RightTeam"][game_index]]
                self.right_rating.append(r[right_index])
        self.df = self.df.assign(Left_C = self.left_rating)
        self.df = self.df.assign(Right_C = self.right_rating)
        return self.df


class Massey():
    def __init__(self, df):
        self.df = df
        self.common_functions = Common(df)
        self.team_ordering = self.common_functions.team_order()
        self.n = len(self.team_ordering)
        self.left_offensive_rating    = []
        self.left_defensive_rating    = []
        self.left_rating              = []
        self.right_offensive_rating   = []
        self.right_defensive_rating   = []
        self.right_rating             = []
        self.unique_dates = df.Date.unique()
   
    def rank(self):
        for date_index in range(len(self.unique_dates)):
            data_used_for_ranking = self.df[self.df.Date == self.unique_dates[date_index - 1]].reset_index(drop = True)
            data_to_be_ranked = self.df[self.df.Date == self.unique_dates[date_index]].reset_index(drop = True)
            if date_index == 0:
                points_for_array = np.array([0] * self.n)
                points_against_array = np.array([0] * self.n)
                pairwise_matchups_array = np.array([[0] * self.n] * self.n)
                total_no_played_array = np.array([[0] * self.n] * self.n)
            else:
                total_no_played_array   += self.common_functions.total_no_played(data_used_for_ranking)
                pairwise_matchups_array += self.common_functions.pairwise_matchups(data_used_for_ranking)
                points_for_array        += self.common_functions.points_for(data_used_for_ranking)
                points_against_array    += self.common_functions.points_against(data_used_for_ranking)
            #Massy rating calculations start here
            T = total_no_played_array
            P = pairwise_matchups_array
            f = points_for_array
            a = points_against_array
            p = f - a
            M = T - P
            #letters are used inline with book to make referencing easy
            M[self.n - 1, :] = 1
            p[self.n - 1] = 0
            try:
                r = np.linalg.solve(M, p)
                d = np.linalg.solve(T + P, np.dot(T, r) - f)
                o = r - d            
            except np.linalg.LinAlgError:
                d, o, r = ([[None] * self.n] * 3)
            #Placing ranks of teams in the dataset
            for game_index in range(len(data_to_be_ranked)):
                left_index = self.team_ordering[data_to_be_ranked['LeftTeam'][game_index]]
                self.left_rating.append(r[left_index])
                self.left_offensive_rating.append(o[left_index])
                self.left_defensive_rating.append(d[left_index])
                right_index = self.team_ordering[data_to_be_ranked['RightTeam'][game_index]]
                self.right_rating.append(r[right_index])
                self.right_offensive_rating.append(o[right_index])
                self.right_defensive_rating.append(d[right_index])
        self.df = self.df.assign(Left_MO  = self.left_offensive_rating)
        self.df = self.df.assign(Right_MO = self.right_offensive_rating)
        self.df = self.df.assign(Left_MD = self.left_defensive_rating)
        self.df = self.df.assign(Right_MD = self.right_defensive_rating)
        self.df = self.df.assign(Left_M = self.left_rating)
        self.df = self.df.assign(Right_M = self.right_rating)
        return self.df
    
class WinPercentage():
    def __init__(self, df):
        self.df = df
        self.common_functions = Common(df)
        self.team_ordering = self.common_functions.team_order()
        self.n = len(self.team_ordering)
        self.left_rating    = []
        self.right_rating   = []
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
                left_index = self.common_functions.team_order()[data_to_be_ranked["LeftTeam"][game_index]]
                self.left_rating.append(r[left_index])
                right_index = self.common_functions.team_order()[data_to_be_ranked["RightTeam"][game_index]]
                self.right_rating.append(r[right_index])
        self.df = self.df.assign(Left_WP = self.left_rating)
        self.df = self.df.assign(Right_WP = self.right_rating)
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
                    Massey(
                        Colley(season_data).rank()
                    ).rank()
                ).rank()
            )
        return pd.concat(self.final_df, ignore_index=True)


        # od = od_for_a_season(data)
        # #print(od)
        # markov1 = markov_for_a_season(data,1)
        # #print(markov1)
        # markov2 = markov_for_a_season(data,2)
        # #print(markov2)
        # markov3 = markov_for_a_season(data,3)
        # #print(markov3)
    #return markov3