import time
import os
import pandas as pd

# from this project
import utils.helper_models as helper_models
import utils.common as common

class DataPrepare:

    def __init__(self,file_path, merge_file_name):
        self.file_path = file_path
        self.merge_file_name = merge_file_name

    def dataPrepare(self, file_path, merge_file_name):
        print('Start Data prepration:')
        try:
            start = time.time()

            data_raw = pd.read_csv(os.path.join(file_path, merge_file_name))
            print(data_raw.head())
            print(data_raw.shape)
            #print(data_raw.columns)
            #classDistribution = data_raw['ARR_DEL15'].value_counts()
            #print('Class imbalance:')
            #print(classDistribution)

            # Checking missing values Just for confirmation
            #data_check = helper_models.missing_values_table(data_raw)
            #print('Missing values in a column with the percentage', data_check)

            # Remove sparse columns that has more than 70% of the missing values
            cols_to_drop = ['Unnamed: 44', 'CANCELLATION_CODE','LATE_AIRCRAFT_DELAY',
                            'SECURITY_DELAY', 'NAS_DELAY', 'WEATHER_DELAY', 'CARRIER_DELAY']
            df_clean = common.remove_cols(data_raw, cols_to_drop)
            print(df_clean.columns)

            # Below Columns are made for data analysis and data explorations

            # Transforming month column into 5 weeks for analysis
            def Transform_month(cols):
                if cols in [1, 2, 3, 4, 5, 6, 7]:
                    return "week1"

                elif cols in [8, 9, 10, 11, 12, 13, 14]:
                    return "week2"

                elif cols in [15, 16, 17, 18, 19, 20, 21]:
                    return "week3"

                elif cols in [22, 23, 24, 25, 26, 27, 28]:
                    return "week4"

                else:
                    return "week5"

            df_clean['MONTH'] = df_clean["DAY_OF_MONTH"].apply(Transform_month)
            df_clean.drop("DAY_OF_MONTH", axis=1, inplace=True)

            # print(df_clean.head())

            # Transforming week column into weekend and weekdays for analysis
            def Transform_week(cols):
                if cols in [1, 7]:
                    return "Weekend"

                else:
                    return "Weekdays"

            df_clean['WEEK'] = df_clean["DAY_OF_WEEK"].apply(Transform_week)
            df_clean.drop("DAY_OF_WEEK", axis=1, inplace=True)
            # print(df_clean.head())

            df_clean["Flight_route"] = df_clean["ORIGIN"] + " to " + df_clean["DEST"]

            # Transforming Departure time and Arrival time columns into 3 categories : Morning, Afternoon, Evening
            def Transform_time(cols):
                if cols >= 600 and cols < 1200:
                    return "Morning"
                elif cols >= 1200 and cols < 1600:
                    return "Afternoon"
                else:
                    return "Evening"

            df_clean['ARRIVAL_TIME'] = df_clean["CRS_ARR_TIME"].apply(Transform_time)
            df_clean.drop("CRS_ARR_TIME", axis=1, inplace=True)

            df_clean['Departure_Time'] = df_clean["CRS_DEP_TIME"].apply(Transform_time)
            df_clean.drop("CRS_DEP_TIME", axis=1, inplace=True)

            print(df_clean.columns)


            # save locally for future Analysis and Modelling
            df_clean.to_csv(os.path.join(file_path, 'data_prepared.csv'), index=False)
            print('Prepared data Saved Succesfully...')


            end = time.time()
            print(end - start)

        except FileNotFoundError as ex:
            print('File can not be found %s', ex)
            raise ex