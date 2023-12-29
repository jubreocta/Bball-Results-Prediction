import numpy as np
np.seterr(divide="ignore", invalid="ignore")
import pandas as pd
#pd.set_option("expand_frame_repr", False)
#pd.set_option("display.max_rows", 500)
#pd.set_option("display.max_columns", 500)


class Ranking():
    def __init__(self, df):
        self.df = df
        self.n = len(set(self.df["LeftTeam"]).union(set(self.df["RightTeam"])))
        self.left_rating            = []
        self.left_offensive_rating  = []
        self.left_defensive_rating  = []
        self.right_rating           = []
        self.right_offensive_rating = []
        self.right_defensive_rating = []
        self.unique_dates = df.Date.unique()
    
    def pairwise_matchups(self, dataset):
        """Creates an off-diagonal matrix containing the number of pairwise
        matchups between teams over the time span of the supplied dataset"""
        team_ordering = self.team_ordering()
        array = np.array([[0] * self.n] * self.n)
        for i in range(len(dataset)):
            #add 1 for each game between distinct oponents
            left_index = team_ordering[dataset["LeftTeam"][i]]
            right_index = team_ordering[dataset["RightTeam"][i]]
            array[left_index, right_index] += 1
            array[right_index, left_index] += 1
        return array
            
    def points_against(self, dataset):
        """Creates a vector containing the total number of points conceded by each
        team over the time span of the supplied dataset"""
        team_ordering = self.team_ordering()
        array = np.array([0] * self.n)
        for i in range(len(dataset)):
            #add value of point conceeded to array
            array[team_ordering[dataset["LeftTeam"][i]]] += dataset["RightScore"][i]
            array[team_ordering[dataset["RightTeam"][i]]] += dataset["LeftScore"][i]
        return array
        
    def points_for(self, dataset, ):
        """Creates a vector containing the total number of points scored by each
        team over the time span of the supplied dataset."""
        team_ordering = self.team_ordering()
        array = np.array([0] * self.n)
        for i in range(len(dataset)):
            #add value of point scored to array
            array[team_ordering[dataset["LeftTeam"][i]]] += dataset["LeftScore"][i]
            array[team_ordering[dataset["RightTeam"][i]]] += dataset["RightScore"][i]
        return array

    def points_given_up(self, dataset_of_interest, kind):
        """Creates a matrix containing the total number of points given up to each
        team over the time span of the supplied dataset. The kinds represent the
        different forms of voting described in the textbook.
        1 -- loser votes only one point for winner
        2 -- loser votes point differential
        3 -- both winner and looser vote points given up"""
        team_ordering = self.team_ordering()
        matrix = np.array([[0] * self.n] * self.n)
        for i in range(len(dataset_of_interest)):
            #add value of point conceeded to array
            left_index = team_ordering[dataset_of_interest["LeftTeam"][i]]
            right_index = team_ordering[dataset_of_interest["RightTeam"][i]]
            if kind == 3:
                matrix[left_index, right_index] += dataset_of_interest["RightScore"][i]
                matrix[right_index, left_index] += dataset_of_interest["LeftScore"][i]
            elif kind == 2:
                if dataset_of_interest["LeftScore"][i] < dataset_of_interest["RightScore"][i]:
                    matrix[left_index, right_index] += dataset_of_interest["RightScore"][i] - dataset_of_interest["LeftScore"][i]
                else:
                    matrix[right_index, left_index] += dataset_of_interest["LeftScore"][i] - dataset_of_interest["RightScore"][i]
            elif kind == 1:
                if dataset_of_interest["LeftScore"][i] < dataset_of_interest["RightScore"][i]:
                    matrix[left_index, right_index] += 1
                else:
                    matrix[right_index, left_index] += 1
        return matrix

    def subtract_losses_from_wins(self, dataset):
        """Creates a vector containing the total number of losses subtracted from
        the total number of wins for each team over the time span of the supplied
        dataset"""
        team_ordering = self.team_ordering()
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

    def team_ordering(self):
        teams = sorted(set(self.df["LeftTeam"]).union(set(self.df["RightTeam"])))
        return {key: value for value,key in enumerate(teams)}    
    
    def total_no_played(self, dataset):
        """Creates a diagonal matrix containing the total number of games played
        by each team over the time span of the supplied dataset"""
        team_ordering = self.team_ordering()
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
        team_ordering = self.team_ordering()
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

class Colley(Ranking):
    def rank(self):
        for date_index in range(len(self.unique_dates)):
            data_used_for_ranking = self.df[self.df.Date == self.unique_dates[date_index - 1]].reset_index(drop=True)
            data_to_be_ranked = self.df[self.df.Date == self.unique_dates[date_index]].reset_index(drop=True)
            if date_index == 0:
                subtract_losses_from_wins_array = np.array([0] * self.n)
                pairwise_matchups_array = np.array([[0] * self.n] * self.n)
                total_no_played_array = np.array([[0] * self.n] * self.n)
                np.fill_diagonal(total_no_played_array, 2)
            else:
                total_no_played_array   += self.total_no_played(data_used_for_ranking)
                pairwise_matchups_array -= self.pairwise_matchups(data_used_for_ranking)
                subtract_losses_from_wins_array += self.subtract_losses_from_wins(data_used_for_ranking)
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
                left_index = self.team_ordering()[data_to_be_ranked["LeftTeam"][game_index]]
                self.left_rating.append(r[left_index])
                right_index = self.team_ordering()[data_to_be_ranked["RightTeam"][game_index]]
                self.right_rating.append(r[right_index])
        self.df = self.df.assign(Left_C = self.left_rating)
        self.df = self.df.assign(Right_C = self.right_rating)
        return self.df

class Markov(Ranking):
    def rank(self, beta, kind):
        for date_index in range(len(self.unique_dates)):
            data_used_for_ranking = self.df[self.df.Date == self.unique_dates[date_index - 1]].reset_index(drop=True)
            data_to_be_ranked = self.df[self.df.Date == self.unique_dates[date_index]].reset_index(drop=True)
            if date_index == 0:
                voting_matrix = np.array([[0] * self.n] * self.n)  
            else:
                voting_matrix += self.points_given_up(data_used_for_ranking, kind)
            #create stocastic matrix
            #line below uses equal voting to other teams by team that has not lost
            #for loop uses vote to self there reorganises the matrix.
            S = np.nan_to_num(voting_matrix/voting_matrix.sum(axis=1, keepdims=True), nan=1 / self.n)
            for i in range(len(S)):
                if (S[i] == np.array([1 / self.n] * self.n)).all():
                    S[i] = np.array([0] * self.n)
                    S[i][i] = 1
            #calculate the stationary vector
            S = beta * S + (1 - beta) / self.n * np.array([[1] * self.n] * self.n)
            A = np.append(np.transpose(S) - np.identity(self.n), [[1] * self.n], axis=0)
            b = np.transpose(np.append(np.array([0] * self.n), 1))
            try:
                r = np.linalg.solve(np.transpose(A).dot(A), np.transpose(A).dot(b))
            except np.linalg.LinAlgError:
                r = ([None] * self.n)
            #Placing ranks of teams in the dataset)
            for game_index in range(len(data_to_be_ranked)):
                left_index = self.team_ordering()[data_to_be_ranked["LeftTeam"][game_index]]
                self.left_rating.append(r[left_index])
                right_index = self.team_ordering()[data_to_be_ranked["RightTeam"][game_index]]
                self.right_rating.append(r[right_index])
        self.df[f"Left_MV{kind}"]  = self.left_rating
        self.df[f"Right_MV{kind}"] =  self.right_rating
        return self.df
        
class Massey(Ranking):
    def rank(self):
        for date_index in range(len(self.unique_dates)):
            data_used_for_ranking = self.df[self.df.Date == self.unique_dates[date_index - 1]].reset_index(drop=True)
            data_to_be_ranked = self.df[self.df.Date == self.unique_dates[date_index]].reset_index(drop=True)
            if date_index == 0:
                points_for_array = np.array([0] * self.n)
                points_against_array = np.array([0] * self.n)
                pairwise_matchups_array = np.array([[0] * self.n] * self.n)
                total_no_played_array = np.array([[0] * self.n] * self.n)
            else:
                total_no_played_array   += self.total_no_played(data_used_for_ranking)
                pairwise_matchups_array += self.pairwise_matchups(data_used_for_ranking)
                points_for_array        += self.points_for(data_used_for_ranking)
                points_against_array    += self.points_against(data_used_for_ranking)
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
                left_index = self.team_ordering()[data_to_be_ranked["LeftTeam"][game_index]]
                self.left_rating.append(r[left_index])
                self.left_offensive_rating.append(o[left_index])
                self.left_defensive_rating.append(d[left_index])
                right_index = self.team_ordering()[data_to_be_ranked["RightTeam"][game_index]]
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

class OffensiveDefensive(Ranking):
    def rank(self):
        for date_index in range(len(self.unique_dates)):
            data_used_for_ranking = self.df[self.df.Date == self.unique_dates[date_index - 1]].reset_index(drop=True)
            data_to_be_ranked = self.df[self.df.Date == self.unique_dates[date_index]].reset_index(drop=True)
            if date_index == 0:    
                voting_matrix = np.array([[0] * self.n] * self.n)
            else:
                voting_matrix += self.points_given_up(data_used_for_ranking, 3)
            #Offensive Defensive rating calculations start here
            A = voting_matrix
            d = np.array([1.0] * self.n).reshape(self.n, 1)
            old_d = np.array([0.9] * self.n).reshape(self.n, 1)
            k = 1
            while k < 10 and np.allclose(old_d,d) is False: #k used to be less than 1000 but this produces rankings that are too high. extreme outlier
                old_d = d
                o = np.transpose(A).dot(np.reciprocal(old_d))
                d = A.dot(np.reciprocal(o))
                k += 1
            d, o, r = d, o, o/d
            #Placing ranks of teams in the dataset
            for game_index in range(len(data_to_be_ranked)):
                left_index = self.team_ordering()[data_to_be_ranked["LeftTeam"][game_index]]
                self.left_rating.append(r[left_index][0])
                self.left_offensive_rating.append(o[left_index][0])
                self.left_defensive_rating.append(d[left_index][0])
                right_index = self.team_ordering()[data_to_be_ranked["RightTeam"][game_index]]
                self.right_rating.append(r[right_index][0])
                self.right_offensive_rating.append(o[right_index][0])
                self.right_defensive_rating.append(d[right_index][0])
        self.df = self.df.assign(Left_ODO  = self.left_offensive_rating)
        self.df = self.df.assign(Right_ODO = self.right_offensive_rating)
        self.df = self.df.assign(Left_ODD  = self.left_defensive_rating)
        self.df = self.df.assign(Right_ODD = self.right_defensive_rating)
        self.df = self.df.assign(Left_OD   = self.left_rating)
        self.df = self.df.assign(Right_OD  = self.right_rating)
        return self.df

class WinPercentage(Ranking):
    def rank(self):
        for date_index in range(len(self.unique_dates)):
            data_used_for_ranking = self.df[self.df.Date == self.unique_dates[date_index - 1]].reset_index(drop=True)
            data_to_be_ranked = self.df[self.df.Date == self.unique_dates[date_index]].reset_index(drop=True)
            if date_index == 0:
                total_no_won_array = np.array([[0] * self.n] * self.n)
                total_no_played_array = np.array([[0] * self.n] * self.n)
            else:
                total_no_won_array   += self.total_no_won(data_used_for_ranking)
                total_no_played_array   += self.total_no_played(data_used_for_ranking)            
            r = []
            for i in range(len(total_no_played_array.diagonal())):
                if total_no_played_array.diagonal()[i] == 0:
                    r.append(0)
                else:
                    r.append(total_no_won_array.diagonal()[i] / total_no_played_array.diagonal()[i])
            r = np.array(r)    
            for game_index in range(len(data_to_be_ranked)):
                left_index = self.team_ordering()[data_to_be_ranked["LeftTeam"][game_index]]
                self.left_rating.append(r[left_index])
                right_index = self.team_ordering()[data_to_be_ranked["RightTeam"][game_index]]
                self.right_rating.append(r[right_index])
        self.df = self.df.assign(Left_WP = self.left_rating)
        self.df = self.df.assign(Right_WP = self.right_rating)
        return self.df

class Fatigue:
    def __init__(self, df):
        self.df = df
        self.calendar = pd.date_range(start=min(self.df.Date), end=max(self.df.Date))
        self.teams = sorted(set(self.df.LeftTeam).union(set(self.df.RightTeam)))
    
    def EWMA(self, lambda_):
        load_dict = dict()
        for team in self.teams:
            all_team_data = []
            for date_index in range(len(self.calendar)):
                row = []
                row.append(self.calendar[date_index])
                data_for_date = self.df[self.df.Date == self.calendar[date_index]].reset_index(drop=True)
                if len(data_for_date) == 0:
                    loading = 0
                else:
                    for index in range(len(data_for_date)):
                        if data_for_date.loc[index, "LeftTeam"] == team or data_for_date.loc[index, "RightTeam"] == team:
                            loading = 48 + int(data_for_date.loc[index, "Overtime"][0]) * 5
                            break
                        else:
                            loading = 0
                row.append(loading)
                all_team_data.append(row)
            load_dataset = pd.DataFrame.from_records(data = all_team_data, columns = ["Date", "Load"])
            ewma = []
            ewma_yesterday = 0
            for index in range(len(load_dataset)):
                ewma_today = load_dataset.loc[index, "Load"] * lambda_ + ((1 - lambda_) * ewma_yesterday)
                ewma.append(ewma_today)
                ewma_yesterday = ewma_today
            load_dataset["ewma"] = ewma
            load_dict[team] = load_dataset
        return load_dict

    def ReturnEWMA(self, days):
        lambda_ = 2 / (days + 1)
        left_load = []
        right_load = []
        for index in range(len(self.df)):
            left_team_load_dataset = self.EWMA(lambda_)[self.df.loc[index, "LeftTeam"]]
            right_team_load_dataset = self.EWMA(lambda_)[self.df.loc[index, "RightTeam"]]
            date = self.df.loc[index, "Date"]
            try:
                h_load = left_team_load_dataset.loc[left_team_load_dataset["Date"] == date - pd.DateOffset(1), "ewma"].iloc[0]        
                a_load = right_team_load_dataset.loc[right_team_load_dataset["Date"] == date - pd.DateOffset(1), "ewma"].iloc[0]        
                left_load.append(h_load)
                right_load.append(a_load)
            except:
                left_load.append(0)
                right_load.append(0)            
        self.df[f"Left_EWMA_{days}"] = left_load
        self.df[f"Right_EWMA_{days}"] = right_load
        return self.df

    def Back2Back(self):
        left_info = []
        right_info = []
        for index in range(len(self.df)):
            date = self.df.loc[index, "Date"]
            left_team = self.df.loc[index, "LeftTeam"]
            right_team = self.df.loc[index, "RightTeam"]
            yday = date - pd.DateOffset(1)
            yday_data = self.df.where(self.df.Date == yday)
            yday_teams = set(yday_data["LeftTeam"]).union(set(yday_data["RightTeam"]))
            if left_team in yday_teams:
                left_info.append(1)
            else:
                left_info.append(0)
            if right_team in yday_teams:
                right_info.append(1)
            else:
                right_info.append(0)
        self.df = self.df.assign(Left_B2B = left_info)
        self.df = self.df.assign(Right_B2B = right_info)
        return self.df
    
class SeasonRanks:
    def __init__(self, df):
        self.df = df
        self.seasons = df.Season.unique()

    def do_1_seasonal_ranking(self):
        final_df = []
        for season in self.seasons:
            season_data = self.df[self.df.Season == season]
            final_df.append(
                WinPercentage(
                    OffensiveDefensive(
                        Massey(
                            Markov(
                                Markov(
                                    Markov(
                                        Colley(season_data).rank()
                                    ).rank(0.6, 1)
                                ).rank(0.6, 2)
                            ).rank(0.6, 3)
                        ).rank()
                    ).rank()
                ).rank()
            )
        print(len(final_df))
        return pd.concat(final_df, ignore_index=True)
    
    def do_2_seasonal_ranking(self):
        df = self.df.copy()
        final_df = []
        df["Year"] = df.Season.str[:4].astype(int)
        data_length = []
        for year in df.Year.unique():
            previous_year = year - 1
            season_data = df[df.Year.isin([year])]
            two_season_data = df[df.Year.isin([previous_year, year])].drop("Year", axis=1)
            data_length.append(len(season_data))
            final_df.append(
                WinPercentage(
                    OffensiveDefensive(
                        Massey(
                            Markov(
                                Markov(
                                    Markov(
                                        Colley(two_season_data).rank()
                                    ).rank(0.6, 1)
                                ).rank(0.6, 2)
                            ).rank(0.6, 3)
                        ).rank()
                    ).rank()
                ).rank()
            )
        final_df = [final_df[i].tail(data_length[i]) for i in range(len(data_length))]
        return pd.concat(final_df, ignore_index=True).reset_index(drop=True)
    
    def do_fatigue(self):
            Fatigue(
                Fatigue(
                    Fatigue(self.df).Back2Back()
                ).ReturnEWMA(7)
            ).ReturnEWMA(28)
                    
    def do_seasonal_ranking(self):
        self.df["Date"] = pd.to_datetime(self.df["Date"], format = "%Y/%m/%d")
        #one_season = self.do_1_seasonal_ranking()
        #two_season = self.do_2_seasonal_ranking()
        fatigue    = self.do_fatigue()
        return fatigue
        return pd.merge(one_season, two_season, on=self.df.columns.to_list(), how="inner", suffixes=("_1", "_2"))