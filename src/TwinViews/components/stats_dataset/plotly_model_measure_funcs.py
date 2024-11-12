import xarray as xr
import plotly.graph_objects as go
from plotly.subplots import make_subplots as px_make_subplots
import pandas as pd
import numpy as np
import plotly.express as px

def plot_long_term_time_series_stats(ds_plot: xr.Dataset, display_simu_sensor_channels:list, stat_to_compute:str) -> go.Figure:
    # plot simulation results
    fig = px_make_subplots(rows=2, cols=1, shared_xaxes=True)

    user_channels_list = [var for var in display_simu_sensor_channels if 'simu_' not in var]
    
    for channel in user_channels_list :

        plot_channel = [var for var in display_simu_sensor_channels if channel in var]

        var_to_plot = [var for var in plot_channel if 'simu_' in var][0]

        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(ds_plot["time"].values),
                y=ds_plot[var_to_plot].mean(dim='time_sensor'),
                mode='markers+lines',
                hoverlabel= dict(namelength=-1),
                name='mean simu '+channel,
                line=dict(color='navy', width=2),   
                ),
            row=1, col=1)
        
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(ds_plot["time"].values),
                y=ds_plot[var_to_plot].max(dim='time_sensor'),
                line=dict(color='navy', width=2, dash='dash'),
                hoverlabel= dict(namelength=-1),
                visible='legendonly',
                name='max simu '+channel,
                    ),
        row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(ds_plot["time"].values),
                y=ds_plot[var_to_plot].min(dim='time_sensor'),
                hoverlabel= dict(namelength=-1),
                line=dict(color='navy', width=2, dash='dot'),
                visible='legendonly',
                name='min simu '+channel,
                    ),
        row=1, col=1)

        # plot sensor measurements
        var_to_plot = [var for var in plot_channel if 'simu_' not in var][0]

        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(ds_plot["time"].values),
                y=ds_plot[var_to_plot].mean(dim='time_sensor'),
                hoverlabel= dict(namelength=-1),
                mode='markers+lines',
                line=dict(color='darkorange', width=2),
                name='mean sensor '+channel,  
            ),
            row=1, col=1)
        
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(ds_plot["time"].values),
                y=ds_plot[var_to_plot].max(dim='time_sensor'),
                hoverlabel= dict(namelength=-1),
                line=dict(color='darkorange', width=2, dash='dash'),
                name='max sensor '+channel,
                visible='legendonly',  
            ),
        row=1, col=1)
        
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(ds_plot["time"].values),
                y=ds_plot[var_to_plot].min(dim='time_sensor'),
                hoverlabel= dict(namelength=-1),
                line=dict(color='darkorange', width=2, dash='dot'),
                name='min sensor '+channel,
                visible='legendonly',
            ),
            row=1, col=1
            )

        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(ds_plot["time"].values),
                y=getattr(ds_plot[var_to_plot], stat_to_compute)(dim='time_sensor') - getattr(ds_plot['simu_'+var_to_plot], stat_to_compute)(dim='time_sensor'),
                hoverlabel= dict(namelength=-1),
                mode='markers',
                line=dict(color='black', width=2),
                name= stat_to_compute + ': Sensor - Simu '+channel,
            ),  
            row=2, col=1
            )

    
    fig.update_layout(
        # title= input.channel() + " simu vs. sensor comparison",
        # yaxis_title= input.channel()+ " - unit : '" +  ds_sensor_simu[input.channel()].unit+ "'",
        xaxis_title="Time",
        # autosize=False,
        # width=41*len(ds_plot['time'])+200,
        # xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(t=0, r=0, l=200, b=0),
        hovermode="x unified",
        legend=dict(
            x=-300,  # 1 is the far right of the plot
            y=0.5,  # 0.5 is the middle of the plot
            orientation="v"  # "v" is vertical
        )
        )
    fig.update_xaxes(rangeslider= {'visible':True}, 
                        row=2, col=1)
    fig.update_xaxes(rangeslider= {'visible':False}, row=1, col=1)
    # fig.update_yaxes(visible=False, showgrid=False)

    return fig


def plot_hour_positionning_plotly(ds_plot: pd.DataFrame):

    fig = go.Figure()

    if len(ds_plot)>0:

        for i, (index, row) in enumerate(ds_plot.iterrows()):
            x_sensor_float = [row['V_MRU_Longitude_rel'], 30*np.sin(np.deg2rad(row['V_MRU_Heading']))+row['V_MRU_Longitude_rel']]
            y_sensor_float = [row['V_MRU_Latitude_rel'], 30*np.cos(np.deg2rad(row['V_MRU_Heading']))+row['V_MRU_Latitude_rel']]
            
            x_sensor_nacelle = [row['V_MRU_Longitude_rel']-41*np.sin(np.deg2rad(row['V_ST_TrueNacelleDir']-90)), +row['V_MRU_Longitude_rel']+41*np.sin(np.deg2rad(row['V_ST_TrueNacelleDir']-90))]
            y_sensor_nacelle = [row['V_MRU_Latitude_rel']-41*np.cos(np.deg2rad(row['V_ST_TrueNacelleDir']-90)), +row['V_MRU_Latitude_rel']+41*np.cos(np.deg2rad(row['V_ST_TrueNacelleDir']-90))]
            

            
            fig.add_trace(
                    go.Scatter(
                        x=x_sensor_float,
                        y=y_sensor_float,
                        mode='lines+markers',
                        name=f'sensor {i*10}min',
                        line=dict(color= 'darkorange', width= 0.5, dash='solid'),
                    )
                )
            fig.add_trace(
                    go.Scatter(
                        x=x_sensor_nacelle,
                        y=y_sensor_nacelle,
                        mode='lines+markers',
                        name=f'sensor nacelle {i*10}min',
                        line=dict(color= 'orange', width= 0.5, dash='dot'),
                        marker=dict(
                            # size=10,  # Marker size
                            # color='red',  # Marker color
                            line=dict(
                                width=2,  # Border width of the markers
                                color='orange'  # Border color of the markers
                            ),
                            symbol='triangle-up',  # Marker symbol shape
                            angle= np.deg2rad(row['V_ST_TrueNacelleDir']-90)
                        )
                    )  
                )

            x_simu = [row['simu_V_MRU_Longitude_rel'], 30*np.sin(np.deg2rad(row['simu_V_MRU_Heading']))+row['simu_V_MRU_Longitude_rel']]
            y_simu = [row['simu_V_MRU_Latitude_rel'], 30*np.cos(np.deg2rad(row['simu_V_MRU_Heading']))+row['simu_V_MRU_Latitude_rel']]

            x_simu_nacelle = [row['simu_V_MRU_Longitude_rel']-41*np.sin(np.deg2rad(row['simu_V_ST_TrueNacelleDir']-90)), +row['simu_V_MRU_Longitude_rel']+41*np.sin(np.deg2rad(row['simu_V_ST_TrueNacelleDir']-90))]
            y_simu_nacelle = [row['simu_V_MRU_Latitude_rel']-41*np.cos(np.deg2rad(row['simu_V_ST_TrueNacelleDir']-90)), +row['simu_V_MRU_Latitude_rel']+41*np.cos(np.deg2rad(row['simu_V_ST_TrueNacelleDir']-90))]

            fig.add_trace(
                go.Scatter(
                    x=x_simu,
                    y=y_simu,
                    mode='lines+markers',
                    name=f'simu {i*10}min',
                    line=dict(color= 'navy', width= 0.5, dash='solid'),
                )
            )

            fig.add_trace(
                    go.Scatter(
                        x=x_simu_nacelle,
                        y=y_simu_nacelle,
                        mode='lines+markers',
                        name=f'simu nacelle {i*10}min',
                        line=dict(color= 'blue', width= 0.5, dash='dot'),
                        marker=dict(
                            size=10,  # Marker size
                            # color='red',  # Marker color
                            line=dict(
                                width=2,  # Border width of the markers
                                color='blue'  # Border color of the markers
                            ),
                            symbol='triangle-up',  # Marker symbol shape
                            angle= np.deg2rad(row['simu_V_ST_TrueNacelleDir']-90)
                        )
                    )
            )

    fig.update_layout(
        # title="Hourly positioning",
        xaxis_title="Longitude (m)",
        yaxis_title="Latitude (m)",
        # xaxis_rangeslider_visible=True
    )
    return fig

def plot_hour_time_series_plotly(ds_plot: xr.Dataset, display_simu_sensor_channels: tuple, x_axis_list: list, color_list: list, ) -> go.Figure:
    fig = go.Figure()
    user_channels_list = [var for var in display_simu_sensor_channels if 'simu_' not in var]
    
    for channel in user_channels_list :
        plot_channel = [var for var in display_simu_sensor_channels if channel in var]
        for var, color, x_axis in zip(plot_channel, color_list, x_axis_list):
            
            fig.add_trace(
                go.Scatter(
                    x=ds_plot[x_axis].values,
                    y=ds_plot[var].values,
                    mode='lines',
                    name=var,
                    line=dict(color= color, width= 1.5),
                )
            )
    fig.update_layout(
        # title="Hourly timeseries",
        xaxis_title=x_axis_list[0] + " - unit : '" +  ds_plot[x_axis_list[0]].unit+ "'",
        # yaxis_title=input.channels()+ " - unit : '" +  ds_sensor_simu[input.channels()].unit+ "'",
        xaxis_rangeslider_visible=True
    )
    return fig

def get_stats_long_term(ds: xr.Dataset, dim_stat='time_sensor'):

    # split simu and sensor data
    all_channels = list(ds.keys())
    ds_sensor = ds[[channel for channel in all_channels if ('simu_' not in channel) and ('V_' in channel) and ('LOAD_Pin' not in channel)]]
    ds_simu = ds[[channel for channel in all_channels if ('simu_V_' in channel) and ('LOAD_Pin' not in channel) ]]
    rename_dict = {var: var.replace('simu_', '') for var in ds_simu.data_vars}
    ds_simu = ds_simu.rename(rename_dict)

    # get stats
    mean = ds_sensor.mean(dim=dim_stat) - ds_simu.mean(dim=dim_stat)
    std = ds_sensor.std(dim=dim_stat) - ds_simu.std(dim=dim_stat)
    min = ds_sensor.min(dim=dim_stat) - ds_simu.min(dim=dim_stat)
    max = ds_sensor.max(dim=dim_stat) - ds_simu.max(dim=dim_stat)

    return mean, std, min, max

def get_ds_box_plot_hourly(ds: xr.Dataset, dim_stat='time_sensor'):

    # split simu and sensor data
    all_channels = list(ds.keys())
    ds_sensor = ds[[channel for channel in all_channels if ('simu_' not in channel) and ('V_' in channel) and ('LOAD_Pin' not in channel)]]
    
    ds_simu = ds[[channel for channel in all_channels if ('simu_V_' in channel) and ('LOAD_Pin' not in channel) ]]
    rename_dict = {var: var.replace('simu_', '') for var in ds_simu.data_vars}
    ds_simu = ds_simu.rename(rename_dict)

    return ds_sensor, ds_simu


def compute_misalignment(ds_plot: xr.Dataset, comp_channel:str, simu:bool, stat_metric:str) -> xr.Dataset:
        ref_channel = 'V_ST_TrueWindDir'
        if simu:
            ref_channel = 'simu_V_ST_TrueWindDir'
            comp_channel = 'simu_'+comp_channel
        
        
        ref_ds = ds_plot[ref_channel]
        comp_ds = ds_plot[comp_channel]
        comp_ds = comp_ds.rename(ref_channel)

        ds_missalign = ref_ds - comp_ds
        ds_missalign = getattr(ds_missalign, stat_metric)(dim='time_sensor')
        ds_missalign = ds_missalign.rename('Misalignment °')

        return ds_missalign


def pair_plot_wind_missalignment_plot(ds_plot: xr.Dataset, comp_channel:str, simu:bool, stat_metric:str, weather_variables:list) -> go.Figure:

        ds_missalign = compute_misalignment(ds_plot, comp_channel, simu, stat_metric)

        list_envir = weather_variables
        ds_envir = ds_plot[list_envir]

        if 'time_sensor' in ds_envir.dims:
            ds_envir = ds_envir.mean(dim='time_sensor')

        ds_missalign = xr.merge([ds_missalign, ds_envir])
        

        ds_missalign = np.round(ds_missalign.to_dataframe(), 3)
        ds_missalign['time'] = ds_missalign.index
        ds_missalign = ds_missalign.dropna()
        fig = px.scatter_matrix(ds_missalign, 
                                dimensions=list_envir, 
                                color='Misalignment °',
                                hover_name='time',
                                )
    
        return fig


