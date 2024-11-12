import os
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta, time
from pathlib import Path

from shiny import App, render
from shiny.types import ImgData

from shiny import App, render, ui, reactive
from shinywidgets import render_widget 
from prepare_data_shiny_dashboard import load_simu_sensor_xr_dataset

import plotly.graph_objects as go
from plotly.subplots import make_subplots as px_make_subplots
import plotly.express as px

# app UI is defined in app_ui.py
from app_ui import get_app_ui
from components.weather_display.dataframe_to_htmltable import dataframe_to_htmltable,  current_wind_to_htmlpicture
from components.stats_dataset.model_measure_functions import ScatterStats, EnhancedBoxPlot, XarrStats
from components.weather_display.display_hourly_envir import _current_wind_html, _current_wind_source, _current_waves_source, _current_waves_html, _current_current_html, _current_current_source
from components.stats_dataset.plotly_model_measure_funcs import plot_long_term_time_series_stats, plot_hour_positionning_plotly, plot_hour_time_series_plotly, get_stats_long_term, get_ds_box_plot_hourly, pair_plot_wind_missalignment_plot, compute_misalignment
from components.stats_dataset.envir_stats_overall import plot_envir_histograms_matplotlib

#####
# Import user configuration parameters from config file
from config import app_config
#####

####
# Load the simulation and sensor data from the database
db_path = os.path.join(app_config.db_path)
ds_sensor_simu = xr.open_dataset(db_path)

if 'time_psd' in ds_sensor_simu.dims:
    ds_sensor_simu = ds_sensor_simu.drop_dims('time_psd').drop_dims('Frequency_psd')

# Load sensors status file
if os.path.exists(app_config.sensor_status_file_path):
    sensor_status = pd.read_excel(app_config.sensor_status_file_path, sheet_name='shiny_flags', header=0)
    sensor_status.index = pd.to_datetime(sensor_status['date'])
    sensor_status= sensor_status.resample('h').ffill()
else : 
    print(app_config.sensor_status_file_path + ' not found')

channel_list = [ var.replace('simu_','') for var in list(ds_sensor_simu.data_vars) if (('simu_AI_' in var) or ('simu_V_' in var))]
channel_dict = {channel: channel for channel in channel_list}
channel_dict_axis =  channel_dict.copy()
channel_dict_axis['time_sensor'] = 'time_sensor'

#####
# Prepare all the display arrangement of the app, page by page
app_ui = get_app_ui(channel_dict, 
                    channel_dict_axis, 
                    start_date=pd.to_datetime(app_config.start_date)) 
#####

####
# Define the server function, with all the reactive functions
####
def server(input, output, session):

    ##########################
    # Long term comparison
    ##########################
    @reactive.calc
    def get_long_term_ds() -> xr.Dataset:
        slice_date = slice(input.daterange()[0], input.daterange()[1]+ timedelta(days=1))
        ds_plot = ds_sensor_simu.sel(time=slice_date)

        prod_iddle_select = input.prod_iddle_select()
        
        try:
            if prod_iddle_select == 'Prod':
                ds_plot = ds_plot.where(ds_plot['V_RotorRpm'].mean(dim='time_sensor') > 3, drop=True)
            elif prod_iddle_select == 'Iddle':
                ds_plot = ds_plot.where(ds_plot['V_RotorRpm'].mean(dim='time_sensor') < 3, drop=True)
        
        except ValueError:
            raise ValueError('No data available for the selected period')

        ds_plot = ds_plot.where(ds_plot['AI_WindSpeed'].mean(dim='time_sensor') > input.min_wind_speed(), drop=True)

        if input.filter_wind_misalignment_switch():
            ds_missalign = compute_misalignment(ds_plot, comp_channel='V_ST_TrueNacelleDir', simu=False, stat_metric='mean')
            # create a mask to filter the data
            mask = (ds_missalign > input.meas_min_misalignment()) & (ds_missalign< input.meas_max_misalignment())
            ds_plot = ds_plot.where(mask, drop=True)

            ds_missalign = compute_misalignment(ds_plot, comp_channel='V_ST_TrueNacelleDir', simu=True, stat_metric='mean')
            # create a mask to filter the data
            mask = (ds_missalign > input.simu_min_misalignment()) & (ds_missalign < input.simu_max_misalignment())
            ds_plot = ds_plot.where(mask, drop=True)

        if input.wave_sensor() == 'Bimep' :
            def check_Bimep_sensor_path(value):
                if 'Bimep' in value:
                    return True
                else:
                    return False
            
            mask = xr.apply_ufunc(check_Bimep_sensor_path, ds_plot['simu_Sensor_Path_Wave'], vectorize=True)
            ds_plot = ds_plot.where(mask, drop=True)

        return ds_plot

    def get_display_simu_sensor_channels(user_channel_list):
        all_channel_list = list(ds_sensor_simu.data_vars)
        display_simu_sensor_channels = []
        for channel in user_channel_list :
            for item in [var for var in all_channel_list if channel in var]:
                display_simu_sensor_channels.append(item)
        return display_simu_sensor_channels
    
    @render.text
    def get_all_hours() :
        ds = ds_sensor_simu
        ds = ds['AI_WindSpeed'].dropna(dim='time_sensor', how='all')
        # count the number of time where AI_windspeed is not all nan
        scada_hours = len(ds.dropna(dim='time', how='all'))

        ds = ds_sensor_simu['simu_AI_WindSpeed'].dropna(dim='time_sensor', how='all')
        simu_hours = len(ds.dropna(dim='time', how='all'))

        return f'Scada hours: {scada_hours} - Simu hours: {simu_hours}'

    @render.text
    def get_hours_scada():
        ds = get_long_term_ds()
        # remove nans due upsampling and merge with simulation
        ds = ds['AI_WindSpeed'].dropna(dim='time_sensor', how='all')
        # count the number of time where AI_windspeed is not all nan
        nb_hours_scada = len(ds.dropna(dim='time', how='all'))
        # nb_hours_simu = len(ds_sensor_simu['simu_AI_WindSpeed'].dropna(dim='time', how='all')['time'])
        return nb_hours_scada

    @render.text
    def get_hours_simu():
        ds = get_long_term_ds()
        # remove nans due upsampling and merge with simulation
        # count the number of time where AI_windspeed is not all nan
        nb_hours_simu = len(ds['simu_AI_WindSpeed'].dropna(dim='time', how='all')['time'])
        return nb_hours_simu
    
    @render_widget
    def sparkline_scada():
        ds = ds_sensor_simu
        # remove nans due upsampling and merge with simulation
        ds = ds['AI_WindSpeed'].dropna(dim='time_sensor', how='all')
        # count the number of time where AI_windspeed is not all nan
        ds = ds.dropna(dim='time', how='all')
        # count the number of present time for each month between the start and end date
        ds = ds.resample(time='1ME').count(dim='time').sel(time_sensor=1)
        ds = ds.to_dataframe()
        ds = ds.reset_index()
        # create a sparkline for each month
        fig = px.line(ds, x='time', y='AI_WindSpeed', title='Scada WindSpeed')
        fig.update_traces(
            line_color="orange",
            line_width=1,
            mode="markers+lines",
            fill="tozeroy",
            fillcolor="rgba(255,165,0,0.2)",  # RGBA for orange with transparency
            hoverinfo="y",
        )

        ds = ds_sensor_simu
        # remove nans due upsampling and merge with simulation
        ds = ds['simu_AI_WindSpeed'].dropna(dim='time_sensor', how='all')
        # count the number of time where AI_windspeed is not all nan
        ds = ds.dropna(dim='time', how='all')
        # count the number of present time for each month between the start and end date
        ds = ds.resample(time='1ME').count(dim='time').sel(time_sensor=1)
        ds = ds.to_dataframe()
        ds = ds.reset_index()

        fig.add_trace(
            go.Scatter(
                x=ds["time"],
                y=ds["simu_AI_WindSpeed"],
                mode="markers+lines",
                line=dict(color="navy", width=1),
                fill="tozeroy",
                fillcolor="rgba(0,0,128,0.2)",
                hoverinfo="y",
                showlegend=False,
            )
        )

        # nb_hours_simu = len(ds_sensor_simu['simu_AI_WindSpeed'].dropna(dim='time', how='all')['time'])
        fig.update_xaxes(visible=False, showgrid=False)
        fig.update_yaxes(visible=False, showgrid=False)
        fig.update_layout(
            height=100,
            hovermode="x",
            margin=dict(t=0, r=0, l=0, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        return fig
    
    
    def transform_sensor_path(value):
        if 'Scada' in value:
            return 'Sca'
        elif 'Bimep' in value:
            return 'Bim'
        elif '' == value:
            return '-'
        else:
            return 'For'

    @render.ui
    def weather_tab():
        ds_plot = get_long_term_ds()
        ds_plot = ds_plot.mean(dim='time_sensor')
        
        weather_channel = app_config.select_weather_channel + [ 'simu_Sensor_Path_Wind','simu_Sensor_Path_Wave', 'simu_Sensor_Path_Current']
        ds_plot = ds_plot[weather_channel]

        ds_plot = ds_plot.assign(wind_sensor=lambda ds_plot: xr.apply_ufunc(transform_sensor_path, ds_plot['simu_Sensor_Path_Wind'], vectorize=True))
        ds_plot = ds_plot.assign(wave_sensor=lambda ds_plot: xr.apply_ufunc(transform_sensor_path, ds_plot['simu_Sensor_Path_Wave'], vectorize=True))
        ds_plot = ds_plot.assign(curr_sensor=lambda ds_plot: xr.apply_ufunc(transform_sensor_path, ds_plot['simu_Sensor_Path_Current'], vectorize=True))

        ds_plot = ds_plot.drop_vars(['simu_Sensor_Path_Wind', 'simu_Sensor_Path_Wave', 'simu_Sensor_Path_Current'])

        ds_plot = ds_plot.to_dataframe()

        return ui.HTML(dataframe_to_htmltable(ds_plot))
    
    @render_widget
    def pair_plot_weather():
        if len(input.channels())>1:
            return 'Please select only one channel'
        else :
            ds_plot = get_long_term_ds()
            all_channels = list(ds_plot.keys())
            
            ds_sensor = ds_plot[[channel for channel in all_channels if ('simu_' not in channel) and ('V_' in channel) and ('LOAD_Pin' not in channel)]]
            ds_simu = ds_plot[[channel for channel in all_channels if ('simu_V_' in channel) and ('LOAD_Pin' not in channel) ]]
            rename_dict = {var: var.replace('simu_', '') for var in ds_simu.data_vars}
            ds_simu = ds_simu.rename(rename_dict)
            stat_to_apply = input.precision_metrics_stat_selection()
           
            ds_stat = getattr(ds_sensor[input.channels()[0]],stat_to_apply)(dim='time_sensor') - getattr(ds_simu[input.channels()[0]],stat_to_apply)(dim='time_sensor')
            
            list_envir = list(input.pair_plot_weather_variables())

            ds_envir = ds_plot[list_envir]

            if 'time_sensor' in ds_envir.dims:
                ds_envir = ds_envir.mean(dim='time_sensor')
            
            ds_stat = xr.merge([ds_stat, ds_envir])

            ds_stat = np.round(ds_stat.to_dataframe(), 3)
            ds_stat['time'] = ds_stat.index
            ds_stat = ds_stat.dropna()
            fig = px.scatter_matrix(ds_stat, 
                                    dimensions=list_envir, 
                                    color=input.channels()[0],
                                    hover_name='time',
                                    )
            return fig
        
    @render_widget
    def pair_plot_wind_missalignment_simu():
        
        inputs_stat = input.precision_metrics_stat_misalignment()
        weather_variables = list(input.misalignment_weather_variables())
        comp_channel = input.misalignment_comp_channel()

        ds_plot = get_long_term_ds()
        comp_channel = 'V_ST_TrueNacelleDir'
        fig = pair_plot_wind_missalignment_plot(ds_plot, comp_channel, simu=True, stat_metric=inputs_stat, weather_variables=weather_variables )

        return fig
    
    @render_widget
    def pair_plot_wind_missalignment_sensor():

        inputs_stat = input.precision_metrics_stat_misalignment()
        weather_variables = list(input.misalignment_weather_variables())
        comp_channel = input.misalignment_comp_channel()

        ds_plot = get_long_term_ds()
        comp_channel = 'V_ST_TrueNacelleDir'
        fig = pair_plot_wind_missalignment_plot(ds_plot, comp_channel, simu=False, stat_metric=inputs_stat, weather_variables=weather_variables )

        return fig

    def get_box_plot(ds_plot: xr.Dataset, display_channels: list, dim_stat='time_sensor'):
        mean, std, min, max = get_stats_long_term(ds_plot, dim_stat)
        fig = px_make_subplots(rows=2, cols=2, subplot_titles=('mean', 'std', 'min', 'max'))
        for var in display_channels:
            fig.add_trace(go.Box(y=mean[var].values, name = 'mean'+var ), row=1, col=1)
            fig.add_trace(go.Box(y=std[var].values, name= 'std '+var ), row=1, col=2)
            fig.add_trace(go.Box(y=min[var].values, name = 'min '+var), row=2, col=1)
            fig.add_trace(go.Box(y=max[var].values, name= 'max '+ var), row=2, col=2)
        fig.update_layout(height=600)

        return fig

    @render_widget
    def box_plot_variables_long_term():
        ds_plot = get_long_term_ds()
        
        # display only inputs channels as simu channels are fusionned with sensor channels
        display_channels= list(input.channels())
        fig = get_box_plot(ds_plot, display_channels)
        return fig

    @render_widget
    def channel_comparison():
        ds_plot = get_long_term_ds()
        stat_to_compute = input.precision_metrics_stat_selection_channel()
        ds_plot = ds_plot.dropna(dim='time', how='all')
        display_simu_sensor_channels = get_display_simu_sensor_channels(list(input.channels()))
        ds_plot = ds_plot[display_simu_sensor_channels]

        fig = plot_long_term_time_series_stats(ds_plot, display_simu_sensor_channels, stat_to_compute)
        
        return fig
    
    def ds_overview_gen():
        display_simu_sensor_channels= get_display_simu_sensor_channels([input.channels_stats_identifiers()])
        stat_to_apply = input.stat_selection()
        if input.dataset_overview_all() == False:
            ds = get_long_term_ds()[app_config.selected_channels_ds_overview + display_simu_sensor_channels]
        else :
            ds = ds_sensor_simu[app_config.selected_channels_ds_overview + display_simu_sensor_channels]
        
        ds_overview = ds.mean(dim='time_sensor').to_dataframe()
        ds_overview.drop(columns=display_simu_sensor_channels, inplace=True)
        
        ds_stat = getattr(ds[display_simu_sensor_channels],stat_to_apply)(dim='time_sensor').to_dataframe()
        
        sensor_channel = [var for var in display_simu_sensor_channels if 'simu_' not in var][0]
        simu_channel = [var for var in display_simu_sensor_channels if 'simu_' in var][0]
        ds_stat[f'{stat_to_apply}: Sensor - Simu'] = ds_stat[sensor_channel] - ds_stat[simu_channel]
        ds_stat = np.round(ds_stat, 2)  
        
        ds = pd.merge(ds_overview, sensor_status, how='left', left_index=True, right_index=True)
        ds = pd.merge(ds, ds_stat, how='left', left_index=True, right_index=True)
        if len(ds)>0:
            ds = ds[~(ds['V_RotorRpm'].isna() & ds['simu_V_RotorRpm'].isna())]
            ds['Turbine_status'] = ['Prod' if stat > 3 else 'Iddle' for stat in ds['V_RotorRpm']]
            ds['Simu_status'] = ['Prod' if stat > 3 else 'Iddle' for stat in ds['simu_V_RotorRpm']]
            # round values on 0 decimals for selected channels
            ds[['AI_WindSpeed']] = ds[['AI_WindSpeed']].round(0)
            ds['hm0'] = ds['hm0'].round(1)
            ds['date'] = ds.index.strftime('%Y-%m-%d %H:%M:%S')
            
            new_col_order = list(sensor_status.keys())
            new_col_order.remove('date')

            ds_overview = ds[['date', 'Turbine_status', 'Simu_status', 'AI_WindSpeed', 'hm0']+ display_simu_sensor_channels + [f'{stat_to_apply}: Sensor - Simu'] + new_col_order]

        return ds_overview
    
    @render.data_frame
    def dataset_overview():
        return render.DataGrid(ds_overview_gen(), filters=True)
    
    ##########################
    #1H Time series layout starts
    ##########################


    @reactive.calc
    def get_hourly_ds_plot() -> xr.Dataset:
        hour_start = time(input.hour_Tseries())
        date =  datetime.combine(input.date_Tseries(), hour_start)
        ds_plot = ds_sensor_simu.sel(time=date.strftime('%Y-%m-%d %H:%M:%S'))
        return ds_plot

    @render.ui
    def current_wind_html():
        ds_plot = get_hourly_ds_plot()
        return _current_wind_html(ds_plot)
    
    @render.ui
    def current_wind_source():
        ds_plot = get_hourly_ds_plot()
        return _current_wind_source(ds_plot=ds_plot)
    
    @render.ui
    def current_waves_source():
        ds_plot = get_hourly_ds_plot()
        return _current_waves_source(ds_plot=ds_plot)
    
    @render.ui
    def current_waves_html():
        ds_plot = get_hourly_ds_plot()
        return _current_waves_html(ds_plot=ds_plot)
    

    @render.ui
    def current_current_html():
        ds_plot = get_hourly_ds_plot()
        return _current_current_html(ds_plot=ds_plot)
    
    @render.ui
    def current_current_source():
        ds_plot = get_hourly_ds_plot()
        return _current_current_source(ds_plot=ds_plot)
    
    def get_box_plot_hourly(ds_plot: xr.Dataset, display_channels: list):
        ds_sensor, ds_simu = get_ds_box_plot_hourly(ds_plot)
        fig = go.Figure()
        for var in display_channels:
            fig.add_trace(go.Box(y=ds_sensor[var].values, name = 'sensor '+var, legendgroup=var, marker_color='darkorange'))
            fig.add_trace(go.Box(y=ds_simu[var].values, name= 'simu '+var, legendgroup=var, marker_color='navy'))
        fig.update_layout(height=500)

        return fig
 
    @render_widget
    def box_plot_variables_1H_time_series():
        ds_plot = get_hourly_ds_plot()
        fig = get_box_plot_hourly(ds_plot, input.channels())
        return fig
    
    @render_widget
    def plot_hour_positioning():
        ds_plot = get_hourly_ds_plot()
        display_simu_sensor_channels = ['V_MRU_Longitude_rel', 'V_MRU_Latitude_rel', 'simu_V_MRU_Longitude_rel', 'simu_V_MRU_Latitude_rel', 
                                        'V_MRU_Heading', 'simu_V_MRU_Heading',
                                        'V_ST_TrueNacelleDir', 'simu_V_ST_TrueNacelleDir',
                                        ]
        ds_plot = ds_plot[display_simu_sensor_channels]
        ds_plot=ds_plot.to_dataframe()
        ds_plot.index = pd.to_datetime(ds_plot.index, unit='s')
        resampling = input.resample_positioning()

        ds_plot = ds_plot.resample(resampling).mean().dropna()

        fig = plot_hour_positionning_plotly(ds_plot)

        return fig

    @render_widget
    def plot_hour_timeseries():

        ds_plot = get_hourly_ds_plot()
        display_simu_sensor_channels = get_display_simu_sensor_channels(list(input.channels()))
       
        ds_plot = ds_plot[display_simu_sensor_channels]

        ds_plot = ds_plot.interpolate_na(dim='time_sensor', method='linear', fill_value='extrapolate')
        ds_plot = ds_plot.ffill(dim='time_sensor')
        ds_plot = ds_plot.bfill(dim='time_sensor')
        ds_plot = ds_plot.fillna(0)
        
        

        color_list = ['navy' if 'simu_' in var else 'darkorange' for var in display_simu_sensor_channels]
        all_channel_list = list(ds_plot.data_vars)
        x_axis_list = [var for var in all_channel_list if input.x_axis() in var]

        if input.x_axis() == 'time_sensor':
            x_axis_list = ['time_sensor', 'time_sensor']

        fig= plot_hour_time_series_plotly(ds_plot, display_simu_sensor_channels, x_axis_list, color_list)
        
        return fig
    
    ##########################
    #1H Time series layout ends
    ##########################
    
    @render.plot
    def freq_stats():
        resample = input.resample()
        if len(input.channels())>1:
            return "Please select only one channel"
        else:
            display_simu_sensor_channels = get_display_simu_sensor_channels(list(input.channels()))
            ds_plot = get_long_term_ds()
            ds_plot = ds_plot[display_simu_sensor_channels]
        
            stats = XarrStats(ds_plot, input.channels()[0], lf= input.lf(), hf= input.hf(), sampling= resample)
            fig, ax = EnhancedBoxPlot(stats, input.channels()[0])

        return fig
    
    @render.plot
    def power_curve():
        resample = input.resample_aero()
        ds = get_long_term_ds()
        fig = ScatterStats(ds, 'V_GridRealPowerLog', sampling = resample)
        return fig
    
    @render.plot
    def envir_power_stats():
        ds = get_long_term_ds()
        fig = plot_envir_histograms_matplotlib(ds)
        return fig
    
    @render.image
    def fem_logo():
        dir = Path(__file__).resolve().parent
        logo_path = os.path.join(dir,'www', 'fem-couleur-logo.png')
        img: ImgData = {"src": str(logo_path), "width": "200px"}
        return img
    
    @render.image
    def saitec_logo():
        dir = Path(__file__).resolve().parent
        logo_path = os.path.join(dir,'www', 'saitec-logo.png')
        img: ImgData = {"src": str(logo_path), "width": "200px"}
        return img
    
####
# Run the shiny server   
www_dir = Path(__file__).parent / "www"
app = App(app_ui, server, static_assets=www_dir)
####