import os
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier

class FeatureSelector:
    def __init__(self, df, target_col, save_path):
        self.df = df
        self.target_col = target_col
        self.save_path = save_path

    def lasso_selection(self):
        X = self.df.drop(self.target_col, axis=1)
        y = self.df[self.target_col]
        model = LassoCV(cv=5)
        model.fit(X, y)
        selected_cols = X.columns[model.coef_ > 0]
        selected_df = self.df[selected_cols.tolist() + [self.target_col]]
        os.makedirs(self.save_path, exist_ok=True)
        selected_df.to_csv(os.path.join(self.save_path, "lasso_selected_dataset.csv"),index=False)

    def rf_selection(self):
        X = self.df.drop(self.target_col, axis=1)
        y = self.df[self.target_col]
        model = RandomForestClassifier(n_estimators=100,random_state=42)
        model.fit(X, y)
        selected_cols = X.columns[model.feature_importances_ > 0.2]
        selected_df = self.df[selected_cols.tolist() + [self.target_col]]
        os.makedirs(self.save_path, exist_ok=True)
        selected_df.to_csv(os.path.join(self.save_path, "rf_selected_dataset.csv"),index=False)
