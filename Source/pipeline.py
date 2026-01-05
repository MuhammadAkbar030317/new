import pandas as pd
import numpy as np
import joblib
import optuna

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class HotelCancellationTrainer:
    def __init__(self):
        self.data_path = r"C:\Users\User\Desktop\Project_09\Data\Raw\hotel_bookings_updated_2024.csv"
        self.model_path = r"C:\Users\User\Desktop\Project_09\Models\hotel_cancellation_pipeline.joblib"

        self.target = "is_canceled"

        # ❌ leakage bo‘ladigan ustunlar
        self.drop_cols = [
            "reservation_status",
            "reservation_status_date"
        ]

    # -------------------------------
    # 1️⃣ Data load + split
    # -------------------------------
    def load_and_split(self):
        df = pd.read_csv(self.data_path)

        df = df.drop(columns=self.drop_cols, errors="ignore")

        X = df.drop(columns=[self.target])
        y = df[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        self.num_features = self.X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.cat_features = self.X_train.select_dtypes(include=["object"]).columns.tolist()

    # -------------------------------
    # 2️⃣ Preprocessor (Leakage-free)
    # -------------------------------
    def build_preprocessor(self):
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # ❗ OrdinalEncoder → ustunlar ko‘paymaydi
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ])

        self.preprocessor = ColumnTransformer([
            ("num", num_pipeline, self.num_features),
            ("cat", cat_pipeline, self.cat_features)
        ])

    # -------------------------------
    # 3️⃣ Feature Selection (Decision Tree)
    # -------------------------------
    def select_important_features(self):
        X_train_processed = self.preprocessor.fit_transform(self.X_train)

        tree = DecisionTreeClassifier(random_state=42)
        tree.fit(X_train_processed, self.y_train)

        importances = tree.feature_importances_

        threshold = np.mean(importances)
        self.selected_features_idx = np.where(importances > threshold)[0]

        print(f"✅ Selected features: {len(self.selected_features_idx)}")

    # -------------------------------
    # 4️⃣ Optuna + RandomForest
    # -------------------------------
    def optimize_and_train(self):

        X_train_processed = self.preprocessor.transform(self.X_train)
        X_test_processed = self.preprocessor.transform(self.X_test)

        X_train_sel = X_train_processed[:, self.selected_features_idx]
        X_test_sel = X_test_processed[:, self.selected_features_idx]

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "random_state": 42,
                "n_jobs": -1
            }

            model = RandomForestClassifier(**params)

            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []

            for train_idx, val_idx in cv.split(X_train_sel, self.y_train):
                X_tr, X_val = X_train_sel[train_idx], X_train_sel[val_idx]
                y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                scores.append(f1_score(y_val, preds))

            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)

        print("✅ Best params:", study.best_params)

        self.model = RandomForestClassifier(
            **study.best_params,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train_sel, self.y_train)

        self.evaluate(X_test_sel)

        self.save_pipeline()

    # -------------------------------
    # 5️⃣ Evaluation
    # -------------------------------
    def evaluate(self, X_test):
        preds = self.model.predict(X_test)

        print("\n📊 MODEL PERFORMANCE")
        print("Accuracy :", accuracy_score(self.y_test, preds))
        print("Precision:", precision_score(self.y_test, preds))
        print("Recall   :", recall_score(self.y_test, preds))
        print("F1-score :", f1_score(self.y_test, preds))

    # -------------------------------
    # 6️⃣ Save full pipeline
    # -------------------------------
    def save_pipeline(self):
        final_object = {
            "preprocessor": self.preprocessor,
            "selected_features_idx": self.selected_features_idx,
            "model": self.model
        }

        joblib.dump(final_object, self.model_path)
        print(f"\n💾 Model saved to: {self.model_path}")

    # -------------------------------
    # 7️⃣ Run all
    # -------------------------------
    def run(self):
        self.load_and_split()
        self.build_preprocessor()
        self.select_important_features()
        self.optimize_and_train()



