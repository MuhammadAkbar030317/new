import sys
import pandas as pd
sys.path.append(r"C:\Users\User\Desktop\AI_Projects\Project_09\Source")
from preprocessing import Preprocessor

df=pd.read_csv(r"C:\Users\User\Desktop\AI_Projects\Project_09\Data\Raw\hotel_bookings_updated_2024.csv")

pre=Preprocessor(df=df,target_col="is_canceled")
path = r"C:\Users\User\Desktop\AI_Projects\Project_09\Data\Preprocessed\preprocessed.csv"

pre.split()
pre.drop_columns()
pre.encoding()
pre.advanced_imputation()
pre.scale_data()
final_df = pre.get_full_dataframe()

final_df.to_csv(path, index=False)
print("Preprocessing tugadi, dataset saqlandi")