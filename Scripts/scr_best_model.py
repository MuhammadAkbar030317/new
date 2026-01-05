import pandas as pd
import sys
sys.path.append(r"C:\Users\User\Desktop\Project_09\Source")
from best_model import RandomForestOptunaTrainer
df = pd.read_csv(r"C:\Users\User\Desktop\Project_09\Data\Preprocessed\preprocessed.csv")

rf_trainer = RandomForestOptunaTrainer(df,target_col="is_canceled")

rf_trainer.tune(n_trials=40)

rf_trainer.save_model(r"C:\Users\User\Desktop\Project_09\Models")
