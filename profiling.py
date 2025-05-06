import pandas_profiling as pdp
import pandas as pd
import time

if __name__ == "__main__":

    print('Start Profiling')
    data_raw = pd.read_csv("E:\PlusDental_Task\presentation\Churn_Modelling.csv", index_col=False)
    start = time.time()
    sample_for_profiling = data_raw.iloc[:5000]
    profile_target = pdp.ProfileReport(sample_for_profiling)
    print("Profile", time.time() - start, "s")
    profile_target.to_file("presentation.html")
