# includes for main
import argparse
import os
import pandas as pd

from sklearn.metrics import classification_report

# from this project
#from data_read_merge import DataReadAndMerge
from data_prepare import DataPrepare
from data_exploration import DataExploration
from feature_engineering import FeatureEngineering
from modelling import Modelling
import utils.helper_models as helper_models


class TestPipeline:

    def __init__(self, input_csv_directory_path, input_csv_file_name,
                 prepared_csv_file_name, features_target_csv_file_name):
        self.input_csv_directory_path=input_csv_directory_path
        self.input_csv_file_name=input_csv_file_name
        self.prepared_csv_file_name = prepared_csv_file_name
        self.features_target_csv_file_name = features_target_csv_file_name

    def fit_pipeline(self, input_csv_directory_path, input_csv_file_name,
                     prepared_csv_file_name, features_target_csv_file_name):
        print('Start Testing pipeline')

        # model settings
        target_name = 'ARR_DEL15'

        try:

            data_test = pd.read_csv(os.path.join(input_csv_directory_path, input_csv_file_name))
            print(data_test.head())
            print(data_test.shape)

            data_prepare = DataPrepare(input_csv_directory_path, input_csv_file_name)
            data_prepare.dataPrepare(input_csv_directory_path, input_csv_file_name)

            feature_engineering = FeatureEngineering(input_csv_directory_path, prepared_csv_file_name)
            feature_engineering.featureEngineering(input_csv_directory_path,prepared_csv_file_name)

            features_target_test = pd.read_csv(os.path.join(input_csv_directory_path,
                                                 features_target_csv_file_name))

            # Dropping missing values if any
            features_target_test.dropna(axis=0, inplace=True)

            model_pipeline = helper_models.load_pipeline()

            prediction = model_pipeline.predict(features_target_test.drop(columns=target_name))
            probability = model_pipeline.predict_proba(features_target_test.drop(columns=target_name))

            print("Classification report: \n ", classification_report(features_target_test[target_name],
                                                                 prediction))

            helper_models.fill_confusion_matrix_and_save(features_target_test[target_name],prediction,
                                                         f_name='Test Confusion matrix',
                                                         out_dir=input_csv_directory_path)

            helper_models.plot_roc_curve_and_save(features_target_test[target_name],
                                                  probability, f_name='Test Roc Curve',
                                                  out_dir=input_csv_directory_path)

            print('Pipeline completed successfully and results are stored in data directory')

        except Exception as ex:
            print('Something went wrong with the Pipeline %s', ex)
            raise ex




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read the raw data')
    parser.add_argument('--input-csv-directory-path', type=str, default="./",
                        help='Full path to hidden test data ')
    parser.add_argument('--input-csv-file-name', type=str, default='test_data.csv',
                        help='Name of the test file')
    parser.add_argument('--prepared-csv-file-name', type=str, default='data_prepared.csv',
                        help='Name of the test prepared file')
    parser.add_argument('--features-target-csv-file-name', type=str, default='features_target.csv',
                        help='Name of the test features with target file')

    opt = parser.parse_args()

    test_pipeline = TestPipeline(opt.input_csv_directory_path,opt.input_csv_file_name,
                                 opt.prepared_csv_file_name, opt.features_target_csv_file_name)
    test_pipeline.fit_pipeline(opt.input_csv_directory_path,opt.input_csv_file_name,
                               opt.prepared_csv_file_name, opt.features_target_csv_file_name)