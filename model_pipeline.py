
class ModelPipeline:

    def __init__(self,data_read_and_merge, data_prepare, feature_engineering, modelling):
        self.data_read_and_merge = data_read_and_merge
        self.data_prepare = data_prepare
        self.feature_engineering = feature_engineering
        self.modelling = modelling

    def fit(self, directory_path, merge_file_name,
            prepared_file_name, feature_target_file_name):

        print('Start model pipeline')

        try:

            self.data_read_and_merge.readAndMerge(directory_path,merge_file_name)
            self.data_prepare.dataPrepare(directory_path, merge_file_name)
            self.feature_engineering.featureEngineering(directory_path,prepared_file_name)
            self.modelling.modelling(directory_path, feature_target_file_name)


            print('Model Pipeline has been completed Successfully')

        except Exception as ex:
            print('Something went wrong with the Pipeline %s', ex)
            raise ex