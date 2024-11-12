from pydantic import BaseModel
from typing import List
import json

conf_file_path = 'src/fem_saitec_demosath/config_files/saitec_demosath.json'

class DemosathConfig(BaseModel):
    """Uses pydantic validations to ensure that the configuration file is correctly formatted."""
    db_path: str # Path to the database file with Simulation and Sensor data
    start_date: str # Start date for the data analysis
    sensor_status_file_path: str # Path to the file containing the sensor status
    select_weather_channel: List[str] # List of weather channels to be displayed in weather tab
    selected_channels_ds_overview: List[str] # List of channels to be displayed in the data set overview tab
    list_envir_for_pair_plot: List[str] # List of channels to be displayed in the pair plot


def get_config(conf_file_path:str) -> DemosathConfig:
    """
    Reads the configuration file and returns a DemosathConfig object.
    

    Parameters:
    - conf_file_path (str): Path to the configuration file.

    Returns:
    - config (DemosathConfig): The configuration object.
    """
    with open(conf_file_path, "r") as conf_file:
        conf_dict = json.load(conf_file)
        config = DemosathConfig(**conf_dict)
    return config

app_config = get_config(conf_file_path)