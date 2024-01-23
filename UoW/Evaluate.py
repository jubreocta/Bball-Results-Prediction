import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

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
        train = self.df[~self.df.Season.isin(["2018-2019", "2019-2020"])].index.values.astype(int)
        test = self.df[self.df.Season.isin(["2018-2019", "2019-2020"])].index.values.astype(int)
        return train, test
    
    def lookback_array(self, X, lookback):
        result = []
        for ind in range(len(X)):
            if ind < lookback - 1:
                zeros_array = np.zeros((lookback - ind - 1, len(X[0])))
                entry = np.concatenate((zeros_array, X[0: ind + 1]), axis=0)
            else:
                entry = X[max(0, ind - lookback + 1): ind + 1]
            print(entry.shape)
            print(entry)
            print(X[ind])
            #print(np.concatenate((zeros_array, X[ind]), axis=0))
            input()
        print([len(i) for i in result])
        exit()
        return np.array(result)

    def lstm_model(self, lookback):
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Bidirectional, BatchNormalization, Dense, Dropout, LSTM 
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.regularizers import l2
        train_index, test_index = self.train_test_split()
        X_train = self.features()[self.features().index.isin(train_index)].values
        X_train = self.lookback_array(X_train, lookback)
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        
        y_train = self.y[self.y.index.isin(train_index)].values
        
        X_test = self.features()[self.features().index.isin(test_index)].values
        X_test = self.lookback_array(X_test, lookback)
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        y_test = self.y[self.y.index.isin(test_index)].values

        model = Sequential()
        model.add(Bidirectional(LSTM(64, return_sequences=True, activation="relu"), input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())

        model.add(Bidirectional(LSTM(32, return_sequences=True, activation="relu")))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())

        model.add(Bidirectional(LSTM(16, activation="relu")))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())

        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())

        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(lr=0.001),
            metrics=["binary_accuracy"]
        )
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        model.fit(
            X_train, y_train.reshape(-1, 1),
            validation_data=(X_test, y_test.reshape(-1, 1)),
            batch_size=64, epochs=100,
            verbose=1, shuffle=False,
            callbacks=[early_stopping]
        )
        y_pred = (model.predict(X_test) > 0.5).astype("int32").squeeze()
        print(self.accuracy(y_test, y_pred))

    def lr_model(self):
        train_index, test_index = self.train_test_split()
        X_train = self.features()[self.features().index.isin(train_index)].values
        y_train = self.y[self.y.index.isin(train_index)].values
        X_test = self.features()[self.features().index.isin(test_index)].values
        y_test = self.y[self.y.index.isin(test_index)].values
        model = LogisticRegression(max_iter = 10000)
        model.fit(
            X_train, y_train,
        )
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        print(self.accuracy(y_test, y_pred))