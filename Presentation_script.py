import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils.helper_models as helper_models
from modelling import Modelling
from xgboost import plot_tree

#path = "E:\PlusDental_Task\presentation\Churn_Modelling.csv"
data_directory_path = "E:\PlusDental_Task\presentation"
features_target_csv_file_name = "Churn_Modelling.csv"

#data_raw = pd.read_csv(path)
#print(data_raw.shape)

# Checking missing values Just for confirmation
# data_check = helper_models.missing_values_table(data_raw)
# print('Missing values in a column with the percentage', data_check)

modelling = Modelling(data_directory_path, features_target_csv_file_name)
modelling.modelling(data_directory_path, features_target_csv_file_name)

