from sklearn.metrics import accuracy_score

class Model:
    def __init__(self, df):
        self.df = df
        self.left_score = df.LeftScore
        self.right_score = df.RightScore
        self.y = (df.LeftScore - df.RightScore > 0).astype(int)
        self.features = df.iloc[:, -50:].fillna(0)
        self.feature_names = {name.split("_", 1)[1] for name in self.features.columns}

    def accuracy(self, y, y_pred):
        return accuracy_score(y, y_pred)
    
    def feature_voting(self, left_feature, right_feature):
        return (self.features[left_feature] - self.features[right_feature] > 0).astype(int)

    def features_model(self):
        results = []
        for feature in self.feature_names:
            left_feature = f"Left_{feature}"
            right_feature = f"Right_{feature}"
            y_pred = self.feature_voting(left_feature, right_feature)
            results.append((feature, self.accuracy(self.y, y_pred)))
        results.sort(key=lambda x: x[1])
        #for result in results:
        #    print(f"{result[0]} Feature - {result[1]:.2%}")
        return results

    def MyCV(self):
        seasons = self.df.Season.unique()
        result = []
        #starts from 3 so that even the first season considered for prediction 
        #has 3 previous seasons to train with
        for season_index in range(3, len(seasons)):
            train_seasons = seasons[season_index-3: season_index]
            test_season = seasons[season_index]
            test_data = self.df[self.df.Season == test_season].index.values.astype(int)
            train_data = self.df[self.df.Season.isin(train_seasons)].index.values.astype(int)
            result.append((train_data, test_data))
        return result