import time
import argparse

# import from this project
from data_read_merge import DataReadAndMerge
from data_prepare import DataPrepare
from data_exploration import DataExploration
from feature_engineering import FeatureEngineering
from modelling import Modelling
from model_pipeline import ModelPipeline


def main(data_directory_path,merge_csv_file_name,
                                 prepared_csv_file_name, features_target_csv_file_name):

    print("Model Process starts")

    #path = "E:\PlusDental_Task\sample_data"
    #merge_file_name = "data_merged.csv"
    #prepared_file_name = "data_prepared.csv"
    #feature_target_file_name = "features_target.csv"

    start = time.time()

    data_read_and_merge = DataReadAndMerge(data_directory_path,merge_csv_file_name)
    # data_read_and_merge.readAndMerge(path,merge_file_name)

    data_prepare = DataPrepare(data_directory_path, merge_csv_file_name)
    #data_prepare.dataPrepare(path, merge_file_name)

    #data_prepared = pd.read_csv(os.path.join(data_directory_path, prepared_csv_file_name))
    #print(data_prepared.head())
    #print(data_prepared.shape)

    #data_explore = DataExploration(data_prepared)
    #data_explore.dataExploration(data_prepared)

    feature_engineering = FeatureEngineering(data_directory_path, prepared_csv_file_name)
    #feature_engineering.featureEngineering(path,prepared_file_name)

    modelling = Modelling(data_directory_path, features_target_csv_file_name)
    #modelling.modelling(data_directory_path, features_target_csv_file_name)

    model_pipeline = ModelPipeline(data_read_and_merge, data_prepare,
                                       feature_engineering, modelling)
    model_pipeline.fit(data_directory_path, merge_csv_file_name,
                           prepared_csv_file_name, features_target_csv_file_name)

    print("Model Process ends", time.time() - start, "s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read the raw data')
    parser.add_argument('--data-directory-path', type=str, default="./data",
                        help='Full path to raw data ')
    parser.add_argument('--merge-csv-file-name', type=str, default='data_merged.csv',
                        help='Name of the test file')
    parser.add_argument('--prepared-csv-file-name', type=str, default='data_prepared.csv',
                        help='Name of the test prepared file')
    parser.add_argument('--features-target-csv-file-name', type=str, default='features_target.csv',
                        help='Name of the test features with target file')

    opt = parser.parse_args()
    print(opt)

    main(opt.data_directory_path,opt.merge_csv_file_name,
                                 opt.prepared_csv_file_name, opt.features_target_csv_file_name)



