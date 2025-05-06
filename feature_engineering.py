import os
import pandas as pd

# from this project
import utils.common as common



class FeatureEngineering:

    def __init__(self,file_path, prepared_file_name):
        self.file_path = file_path
        self.prepared_file_name = prepared_file_name

    def featureEngineering(self, file_path, prepared_file_name):
        print('Start Feature Engineering:')

        try:
            data_prepared = pd.read_csv(os.path.join(file_path, prepared_file_name))
            print(data_prepared.head())
            print(data_prepared.shape)

            # Remove columns that are not useful to build our prediction model
            # Profile report is used to decide which columns are useful and which are not
            # Columns that are highly correlated are removed
            # Columns that has redundunt information are removed
            # Columns that has non useful information are removed

            cols_to_drop = ['ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'ARR_DELAY',
                            'ARR_DELAY_GROUP', 'ARR_DELAY_NEW', 'ARR_TIME',
                            'CANCELLED', 'CRS_ELAPSED_TIME', 'DEP_DEL15', 'DEP_DELAY',
                            'DEP_DELAY_GROUP','DEP_DELAY_NEW', 'DEP_TIME', 'DEST_AIRPORT_ID',
                            'DISTANCE_GROUP','DIVERTED','FL_DATE', 'FL_NUM', 'Flight_route',
                            'ORIGIN_AIRPORT_ID', 'ORIGIN_STATE_ABR', 'TAIL_NUM', 'WHEELS_OFF',
                            'WHEELS_ON', 'YEAR', 'Departure_Time', 'DEST', 'DEST_STATE_ABR', 'FLIGHTS']
            df_clean = common.remove_cols(data_prepared, cols_to_drop)
            print(df_clean.columns)

            # save features and target locally for Modelling
            df_clean.to_csv(os.path.join(file_path, 'features_target.csv'), index=False)
            print('Extract features and Saved Succesfully...')


        except FileNotFoundError as ex:
            print('File can not be found %s', ex)
            raise ex