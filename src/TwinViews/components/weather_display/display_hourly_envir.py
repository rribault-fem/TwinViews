import numpy as np
from datetime import datetime, time
from shiny import ui
import xarray as xr
from faicons import icon_svg


def convert_direction_goingto_conv_to_commingfrom(direction:float)->float:
    if direction>=180:
        direction = direction - 180
    else:
        direction = direction + 180
    return direction

def arrow_dir_html(direction:float, type='wind'):

    html_style ='''
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        #windIcon {      transform: rotate(45deg);    } 
    </style>
    </head>
    <body>
    '''.replace('45deg', str(direction-225) + 'deg')

    html_style = html_style.replace('windIcon', type+'Icon')

    return html_style + icon_svg('location-arrow')._repr_html_().replace('<svg', f'<svg id="{type}Icon"')+ "</body></html>"




def _current_wind_html(ds_plot: xr.Dataset):
    wind_speed = ds_plot['AI_WindSpeed'].mean(dim='time_sensor').values
    
    wind_dir = ds_plot['V_ST_TrueWindDir'].mean(dim='time_sensor').values

    html = '<div style="font-size: 15px;">' + str(np.round(wind_speed,0)) + ' m/s; dir: ' + str(wind_dir.astype(int))+ '°'+ arrow_dir_html(wind_dir)+'</div>' 
    return ui.HTML(html)

def _current_wind_source(ds_plot: xr.Dataset):
    wind_source = 'Scada' if 'Scada' in str(ds_plot['simu_Sensor_Path_Wind'].values) else 'Bimep' if 'Bimep' in str(ds_plot['simu_Sensor_Path_Wind'].values) else 'Forecast'
    return ui.HTML(wind_source)

def _current_waves_source(ds_plot: xr.Dataset):
    wave_source = 'Bimep' if 'Bimep' in str(ds_plot['simu_Sensor_Path_Wave'].values) else 'Forecast'
    return ui.HTML(wave_source)

def _current_waves_html(ds_plot: xr.Dataset):
    hs = ds_plot['simu_hs'].values
    tp = ds_plot['simu_tp'].values
    direction = ds_plot['simu_dp'].values

    return ui.HTML(f'<div style="font-size: 15px;"> <p>hs: {np.round(hs,1)}m, tp: {np.round(tp,1)}s, dir : {direction.astype(int)}°'+ arrow_dir_html(direction, type='wave')+'</p></div>')

def _current_current_html(ds_plot: xr.Dataset):
    current_speed = ds_plot['simu_cur'].values
    current_dir = ds_plot['simu_cur_dir'].values

    current_dir = convert_direction_goingto_conv_to_commingfrom(current_dir)

    return ui.HTML(f'<div style="font-size: 15px;"> <p>{np.round(current_speed,1)} cm/s; dir: {current_dir.astype(int)}°'+arrow_dir_html(current_dir, type='current')+' </p></div>') # {deg_to_arrow(current_dir)}

def _current_current_source(ds_plot: xr.Dataset):
    current_source = 'Bimep (conv: comming from)' if 'Bimep' in str(ds_plot['simu_Sensor_Path_Current'].values) else 'Forecast'
    return ui.HTML(current_source)