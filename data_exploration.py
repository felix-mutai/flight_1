import time
import os
import pandas as pd
import seaborn as sns
import utils.helper_exploration as exploration


class DataExploration:

    def __init__(self, prepared_data):
        self.prepared_data = prepared_data

    def dataExploration(self, prepared_data):
        print('Start Data Exploration:')

        exploration.check_target_distribution(prepared_data)
        exploration.explore_flight_date_vs_delay(prepared_data)
        exploration.explore_carrier_vs_delay(prepared_data)
        exploration.explore_time_vs_delay(prepared_data)
        exploration.explore_origin_vs_delay(prepared_data)
        exploration.explore_fligh_route_vs_delay(prepared_data)
        exploration.explore_time_of_day_vs_delay(prepared_data)

        print('Done Data Exploration and results are stored to the directory:')



