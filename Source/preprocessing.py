from sklearn.experimental import enable_iterative_imputer
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self, df, target_col):
        self.df = df.copy()
        self.target_col = target_col

        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.imputer = IterativeImputer(estimator=LinearRegression(n_jobs=-1), max_iter=10)
        self.scaler = StandardScaler()

    def split(self, test_size=0.2, random_state=42):
        X = self.df.drop(self.target_col, axis=1)
        y = self.df[self.target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    def drop_columns(self):
        drop_cols = ['agent', 'company', 'reservation_status']
        self.X_train = self.X_train.drop(columns=drop_cols, errors='ignore')
        self.X_test = self.X_test.drop(columns=drop_cols, errors='ignore')

    def encoding(self):
        cat_cols = self.X_train.select_dtypes(include='object').columns

        self.X_train[cat_cols] = self.encoder.fit_transform(self.X_train[cat_cols])
        self.X_test[cat_cols] = self.encoder.transform(self.X_test[cat_cols])

    def advanced_imputation(self):
        self.X_train[:] = self.imputer.fit_transform(self.X_train)
        self.X_test[:] = self.imputer.transform(self.X_test)

    def scale_data(self):
        self.X_train[:] = self.scaler.fit_transform(self.X_train)
        self.X_test[:] = self.scaler.transform(self.X_test)

    def get_full_dataframe(self):
        train_df = self.X_train.copy()
        train_df[self.target_col] = self.y_train

        test_df = self.X_test.copy()
        test_df[self.target_col] = self.y_test

        full_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
        return full_df

