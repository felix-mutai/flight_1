import shutil
import glob
import os


class DataReadAndMerge:

    def __init__(self,directory_path,merge_file_name):
        self.directory_path = directory_path
        self.merge_file_name = merge_file_name

    def readAndMerge(self, directory_path, merge_file_name):
        print('Start Reading Raw data from:', directory_path )
        print('Final Merged file will be saved at:', directory_path)
        try:
            os.chdir(directory_path)
            with open(merge_file_name, 'wb') as merge_data:
                for i, filenames in enumerate(glob.glob('*.{}'.format('csv'))):
                    if filenames == merge_file_name:
                        continue
                    with open(filenames, 'rb') as readfile:
                        if i != 0:
                            readfile.readline()
                        shutil.copyfileobj(readfile, merge_data)
            print('Successfully read and merge all files and save final dataframe to the directory')

        except Exception as ex:
            print('Files can not be read and merge%s', ex)
            raise ex