# :package: Package description 

This  is a visualisation tool prototype which facilitates the exploration of Demosath simulation and measurement data.<br>
It enables users to visualize simulation accuracy statistics and detect outliers within the data. The visualization leverages a netCDF database, curated by France Energies Marines, encompassing both simulation and measurement data.

## :fast_forward: Model Operational runs
Operational runs of the Demosath SIMA model, provided by Saitec, are conducted by France Energies Marine.<br>
These runs are forced by measurements from environmental sensors, employing redundancies and selecting sensors based on accuracy.
Additionally, the model integrates weather forecasts to compensate for sensor unavailability or to fulfil forecasting needs.


## :wrench: Setup 

To set up `fem_saitec_demosath` on your local machine, clone the repository with the following command:

```bash
git clone https://gitlab.france-energies-marines.org/Romain/saitec_demosath.git
cd saitec_demosath
```

After cloning, you may want to create a dedicated environment and install the dependencies:

```bash
conda create -n fem_visu python=3.12.0
conda activate fem_visu
pip install -r requirements.txt
pip install -e .
```

VS Code specific setup :
If you are a VS code user and whish to adapt the GUI, to your needs you can use the Shiny (Posit) extension which will facilitate to run and debug the app.

<img src="src/fem_saitec_demosath/shiny_extension.png" alt="shiny extension" width="25%">

![run/debug shiny app with vscode extension](src/fem_saitec_demosath/www/shiny_debug_vscode.png)

## :clipboard: Prerequisites 
A valid netCDF database, which can be provided by FEM, is required to utilize this visualization package.


## :gear: Configuration 
The app can be configured using the config file located in the src\fem_saitec_demosath\config_files folder.
In this first version of the app, you choose the configuration file path in the config.py file.
For Saitec users, the two parameter to adapt should be the paths to the netcdf database and to the sensor status file, both defined in saitec_demosath.json file:
-  "db_path" .
-  "sensor_status_file_path" .

:warning: make sure data can be accessed at good speed between the server and the database, ideally the database is located on the same machine as the server.

Ensure the paths are correctly set-up for your case, for example :
"db_path" : "2024-07-02_merged_simu_sensors_db_saved.nc".

The config file is located here :
src\fem_saitec_demosath\config_files\saitec_demosath.json

## :rocket: Deployment 
To launch the visualization tool for development and testing, execute a unicorn server with the following command:

```bash
python.exe -m shiny run --host 0.0.0.0 --port 63603 src\fem_saitec_demosath\app.py
```

The --host 0.0.0.0 option allows server access from external machines, while --port 63603 specifies the server's port (modifiable as needed).

Access the visualization tool with a web browser within your private network at: this_PC_IP:63603


For production ready deployment (more users with multiple python process per app) options, consult the shiny documentation :
https://shiny.posit.co/py/docs/deploy.html


# :construction_worker: For Developers 

### Netcdf database
In the netcdf database, each variable (or channel) is mapped on two dimensions, The 'time' and the 'time_sensor' dimension.
'time' dimension map the variables with a date and an hour and correspond to one simulation or one hour of measurement.
'time_sensor' is in seconds and provide the actual logged timesteps from the simulations.

All the variables with simu_ prefix are from the simulation, the others are from the measurements.
Note the size of the database is not yet optimized. However the xarray library is used to load the data, so the data is not loaded in memory until it is needed.

To structure the current project we dissociated the data processing part from the visualisation part :
### :outbox_tray: Data processing
Located in src\fem_saitec_demosath\app.py, this segment processes data for visualization, utilizing functions from src\fem_saitec_demosath\components.

### :chart_with_upwards_trend: Visualisation
Defined in src\fem_saitec_demosath\app_ui.py, with each page's layout in separate files within src\fem_saitec_demosath\views. 
To add a page, create a new file in the views directory.
Then you can import the new page in the app_ui.py file and add it to the app layout.

# Acknowledgments
This package is developed by France Energies Marines, a research and development center dedicated to renewable marine energies.

The Dionysos project receives funding from France Energies Marines and its members and partners, as well as French State funding managed by the French National Research Agency under the France 2030 investment plan. This project is financially supported by Pôle Mer Bretagne Atlantique.

https://www.france-energies-marines.org/en/projects/dionysos/


## License
This project is licensed under GNU AGPLv3 License - see the LICENSE file for details.
## Author
Romain Ribault - romain.ribault@france-energies-marines.org
## Citing this Work
If you use this software in your research or if it becomes a part of your workflows, please consider citing it as follows:
Ribault, Romain. (2024). DemosathViz (Version 0.1.0) [Software]. Available at https://gitlab.france-energies-marines.org/Romain/saitec_demosath (Accessed [Date])

Please replace [Date] with the date you accessed the software.