import os

from utils.helper_methods import get_subdirectories


def rename_files(directory_path):
    FLIGHT_ROUTES = get_subdirectories(directory_path)
    for flight_route in FLIGHT_ROUTES:
        flight_dir = os.path.join(directory_path, flight_route)
        attacks = get_subdirectories(flight_dir)
        for attack in attacks:
            attack_dir = os.path.join(flight_dir,attack)
            for flight_csv in os.listdir(f'{directory_path}/{flight_route}/{attack}'):
                origin_file_dir = os.path.join(attack_dir,flight_csv)
                new_file_dir = os.path.join(attack_dir,"sensors_"+flight_csv)
                os.rename(origin_file_dir, new_file_dir)


path ="C:\\Users\\Yehuda Pashay\\Desktop\\fligth_data\\check"
rename_files(path)