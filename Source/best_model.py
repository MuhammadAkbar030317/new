import os
import joblib
import optuna

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class RandomForestOptunaTrainer:
    def __init__(self,df,target_col="is_canceled",test_size=0.2,random_state=42):
        self.df = df
        self.target_col = target_col
        self.random_state = random_state

        X = df.drop(target_col, axis=1)
        y = df[target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,test_size=test_size,random_state=random_state,stratify=y)

        self.best_model = None
        self.best_score = None

    def tune(self, n_trials=15):

        def objective(trial):
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_int("max_depth", 5, 30),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
                max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                random_state=self.random_state,
                n_jobs=-1
            )

            model.fit(self.X_train, self.y_train)
            preds = model.predict(self.X_test)
            return accuracy_score(self.y_test, preds)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.best_model = RandomForestClassifier(
            **study.best_params,
            random_state=self.random_state,
            n_jobs=-1
        )

        self.best_model.fit(self.X_train, self.y_train)
        self.best_score = accuracy_score(
            self.y_test,
            self.best_model.predict(self.X_test)
        )

        print(f" Best Accuracy: {self.best_score:.4f}")
        print(f" Best Params: {study.best_params}")

        return self.best_model

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, "random_forest_optuna.joblib")

        joblib.dump(self.best_model, model_path)

        print(f" Best model saqlandi → {model_path}")
