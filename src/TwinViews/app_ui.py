from shiny import ui, render
from shinywidgets import output_widget
from faicons import icon_svg
from pathlib import Path
import datetime

from views.long_term_comparisons import get_ui_long_term_comparison
from views.h_time_series import get_ui_1h_time_series
from views.Stats import get_ui_freq_stats
from views.aero import get_ui_aero
from views.wind_misalignement import get_ui_wind_misalignment
from views.Statistics_scatter import get_ui_statistics_scatter

def get_app_ui(channel_dict: dict, channel_dict_axis: dict, start_date: datetime.datetime):
    """
    Generates the user interface for the application.

    Parameters:
    - channel_dict (dict): A dictionary containing channel names as keys and their corresponding values.
    - channel_dict_axis (dict): A dictionary containing channel names as keys and their corresponding x-axis values.

    Returns:
    - app_ui (object): The generated user interface object.
    """
    app_ui = \
    ui.page_navbar(
        
        get_ui_long_term_comparison(channel_dict),

        get_ui_statistics_scatter(),

        get_ui_1h_time_series(channel_dict_axis, start_date),
       
        get_ui_freq_stats(),

        get_ui_wind_misalignment(),

        get_ui_aero(),

        ui.nav_spacer(

        ),

        ui.nav_control(
                ui.tags.a(
                    ui.tags.img(
                        src="img-fem-logo-blanc.png", height="45px"
                    ),
                    href="https://www.france-energies-marines.org/",
                    target="_blank",
                    id = "fem_a"
                ),),
        ui.nav_control(        
                ui.tags.a(
                    ui.tags.img(
                        src="saitec-logo.png", height="45px"
                    ),
                    href="https://www.saitec-offshore.com/",
                    target="_blank",
                    id = "saitec_a"
            )
        ),
        ui.nav_control(        
                ui.tags.a(
                    ui.tags.img(
                        src="demosath.jpg", height="60px"
                    ),
                    href="https://saitec-offshore.com/en/projects/demosath/",
                    target="_blank",
                    id = "demosath_a"
            )
        ),
        

        sidebar= ui.sidebar(

            ui.input_select(  
                "channels",  
                "Channels to compare :",  
                channel_dict,
                multiple=True,
                selected='V_MRU_Heave',
                size= 8
                ),

            ui.input_date_range("daterange", "Date range :", start=start_date.strftime('%Y-%m-%d'), end=start_date.strftime('%Y-%m-%d')),
            ui.input_radio_buttons(id="prod_iddle_select", label="Production iddle :", choices=['Prod', 'Iddle', 'All'], selected='All', inline=True),

            ui.input_numeric("min_wind_speed", "Min Wind Speed", value=0, min=0, max=50, step=1),

            ui.h6('Wind Misalignment :'),
            ui.input_switch("filter_wind_misalignment_switch", "Filter misalignment", value=False),
            ui.layout_column_wrap(
                ui.input_numeric("simu_min_misalignment", "Simu min", value=-180, min=-180, max=0, step=1),
                ui.input_numeric("simu_max_misalignment", "Simu max", value=180, min=0, max=180, step=1),
                width=1 / 2
            ),
            ui.layout_column_wrap(
                ui.input_numeric("meas_min_misalignment", "Measure min", value=-180, min=-180, max=0, step=1),
                ui.input_numeric("meas_max_misalignment", "Measure max", value=180, min=0, max=180, step=1),
                width=1 / 2
            ),

            ui.input_radio_buttons(id="wave_sensor", label="Wave sensor :", choices=['Bimep', 'All'], selected='All', inline=True),
            width = 300
        ),
        id = "tabs",
        title = ui.HTML('''<div id="logo_title">
                        <img src="logo-flowtom.png" alt="Image" width="50" height="50"><span>Dionysos APP</span>
                        </div> '''),
        window_title = "Dionysos proto : visualisation tool",
        footer = ui.output_image('saitec_logo', width=200, height=100)
    )

    return app_ui 