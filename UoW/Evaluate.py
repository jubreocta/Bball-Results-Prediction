import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

class Model:
    def __init__(self, df):
        self.df = df
        self.left_score = df.LeftScore
        self.right_score = df.RightScore
        self.y = (df.LeftScore - df.RightScore > 0).astype(int)

    def scaler(self):
        return MinMaxScaler()    
    
    def features(self):
        features = self.df.iloc[:, -50:].fillna(0)
        return pd.DataFrame(
             self.scaler().fit_transform(features),
             columns=features.columns
        )
    
    def feature_names(self):
        return {name.split("_", 1)[1] for name in self.features.columns}

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

    def train_test_split(self):
        train = self.df[~self.df.Season.isin(['2018-2019', '2019-2020'])].index.values.astype(int)
        test = self.df[self.df.Season.isin(['2018-2019', '2019-2020'])].index.values.astype(int)
        return train, test
    
    def lstm_model(self):
        train_index, test_index = self.train_test_split()
        X_train = self.features()[self.features().index.isin(train_index)].values
        y_train = self.y[self.y.index.isin(train_index)].values
        X_test = self.features()[self.features().index.isin(test_index)].values
        y_test = self.y[self.y.index.isin(test_index)].values
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(16))
        model.add(Dense(1))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        model.fit(
            X_train, y_train,
            batch_size=100, epochs=15000,
            verbose=1, shuffle=False
        )
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        print(self.accuracy(y_test, y_pred))