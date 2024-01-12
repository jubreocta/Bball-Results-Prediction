from sklearn.metrics import accuracy_score

class Model:
    def __init__(self, df):
        self.left_score = df.LeftScore
        self.right_score = df.RightScore
        self.y = (df.LeftScore - df.RightScore > 0).astype(int)
        self.df = df.iloc[:, -50:]
        self.feature_names = {name.split("_", 1)[1] for name in self.df.columns}

    def accuracy(self, y, y_pred):
        return accuracy_score(y, y_pred)
    
    def feature_voting(self, left_feature, right_feature):
        return (self.df[left_feature] - self.df[right_feature] > 0).astype(int)

    def features_model(self):
        results = []
        for feature in self.feature_names:
            left_feature = f"Left_{feature}"
            right_feature = f"Right_{feature}"
            y_pred = self.feature_voting(left_feature, right_feature)
            results.append((feature, self.accuracy(self.y, y_pred)))
        results.sort(key=lambda x: x[1])
            
        for result in results:
            print(f"{result[0]} Feature - {result[1]:.2%}")
