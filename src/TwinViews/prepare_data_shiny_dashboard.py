import xarray as xr
import glob
import datetime

def load_simu_sensor_xr_dataset(year='2024', month='05', day='*', hour='*'):

    ds = xr.open_mfdataset(rf"X:\DIONYSOS\03 Documents Internes\Lot_5\DEMOSATH_monitoring\preprod\storage\{year}\{month}\{day}\Simulations\{hour}\*.nc", combine='nested')
    ds = ds.drop_dims(['Frequency_psd', 'time_psd'])
    # Create a dictionary that maps old names to new names
    rename_dict = {var: f'simu_{var}' for var in ds.data_vars}
    ds = ds.rename(rename_dict)
    #ds = ds.mean(dim='time_sensor')

    ds_scada = xr.open_mfdataset(rf"X:\DIONYSOS\03 Documents Internes\Lot_5\DEMOSATH_monitoring\preprod\storage\{year}\{month}\{day}\Sensors\*\*.nc", combine='nested')
    ds_scada = ds_scada.drop_dims(['Frequency_psd', 'time_psd'])
    #ds_scada_mean = ds_scada.mean(dim='time_sensor')

    return xr.merge([ds, ds_scada])


def get_all_ds_from_folder():
    ds = xr.open_mfdataset(rf"*.nc", combine='nested')
    ds.to_netcdf('merged_simu_sensors_db.nc', encoding={'time':{'units':'days since 1900-01-01', 'dtype': 'float64'}}, engine = "netcdf4")

def add_one_nc_file_in_db(db_path, new_nc_file):
    ds = xr.open_dataset(db_path)
    ds_new = xr.open_dataset(new_nc_file)
    ds = xr.merge([ds, ds_new])
    date= datetime.datetime.now().strftime("%Y-%m-%d")
    ds.to_netcdf(date+db_path, encoding={'time':{'units':'days since 1900-01-01', 'dtype': 'float64'}}, engine = "netcdf4")


def get_all_ds_from_date_list(save_path, date_list:list = ['2024-04-02', '2024-04-10']) -> xr.Dataset:

    def get_nc_files_list(folder='Simulations'):
        path_list = [rf"{save_path}\{date.split('-')[0]}\{date.split('-')[1]}\{date.split('-')[2]}\{folder}\*\*.nc" for date in date_list]
        
        # get all netcdf files from the path list
        nc_files_path_list = [glob.glob(path) for path in path_list]
        nc_files_path_list = [item for sublist in nc_files_path_list for item in sublist]

        return nc_files_path_list

    path_list= get_nc_files_list('Simulations')
    ds = xr.open_mfdataset(path_list, combine='nested')
    rename_dict = {var: f'simu_{var}' for var in ds.data_vars}
    ds = ds.rename(rename_dict)
    #ds = ds.mean(dim='time_sensor')

    path_list = get_nc_files_list('Sensors')
    ds_scada = xr.open_mfdataset(path_list, combine='nested')
    ds_scada = ds_scada.drop_dims(['Frequency_psd', 'time_psd'])
    #ds_scada_mean = ds_scada.mean(dim='time_sensor')

    return xr.merge([ds, ds_scada])

if __name__ == '__main__':

    save_path = r"X:\DIONYSOS\03 Documents Internes\Lot_5\DEMOSATH_monitoring\preprod\storage"
    date_list = [ '2024-06-01']
    
    #['2023-12-04', '2024-05-04', '2024-05-05', '2024-05-20', '2024-04-19', '2024-04-21']
    
    #['2024-04-02', '2024-04-06', '2024-04-07', '2024-04-08', '2024-04-09', '2024-04-10', '2024-04-11']



    ds = get_all_ds_from_date_list(save_path, date_list=date_list)
    ds.to_netcdf('2024-06-01.nc', encoding={'time':{'units':'days since 1900-01-01', 'dtype': 'float64'}}, engine = "netcdf4")
    # add_one_nc_file_in_db('merged_simu_sensors_db_saved.nc', 'new_datasets.nc')

    # ds_sensor_simu = load_simu_sensor_xr_dataset(year ="2024", month='04', day='02', hour='*')
    # ds_sensor_simu.to_netcdf('04122023_merged_simu_sensors_db.nc', encoding={'time':{'units':'days since 1900-01-01'}}, engine = "netcdf4")
