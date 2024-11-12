    
from shiny import ui
from shinywidgets import output_widget
from faicons import icon_svg
import datetime

def get_ui_1h_time_series(channel_dict_axis: list, start_date: datetime.datetime) :

    return ui.nav_panel("1H Time Series",
                ui.layout_column_wrap(            
                        ui.input_date("date_Tseries", "Date:", value=start_date.strftime('%Y-%m-%d'), width='auto'),
                        ui.input_numeric("hour_Tseries", "Hour", 0, min=0, max=24, width='auto'),
                    ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Current Weather"), 
                        ui.value_box(
                            "WIND ",
                            ui.output_ui("current_wind_html"), # function to return wind speed and source
                            ui.output_ui("current_wind_source"),
                            showcase=[icon_svg("wind")],
                            id='value_box_wind',
                        ),
                        ui.value_box(
                            "WAVES ",
                            ui.output_ui("current_waves_html"),
                            ui.output_ui("current_waves_source"),
                            showcase=[icon_svg("water")],
                            id='value_box_wave',
                            
                        ),
                        ui.value_box(
                            "CURRENT",
                            ui.output_ui("current_current_html"),
                            ui.output_ui("current_current_source"),
                            showcase=[icon_svg("location-arrow")],
                            id='value_box_current',
                        ),
                        max_height=400,
                    ),
                    ui.card(
                        ui.card_header("Demosath Positionning"),
                        ui.input_radio_buttons('resample_positioning', 'Select resampling period', ['5min', '10min', '30min'], inline=True, selected='30min'),
                        output_widget("plot_hour_positioning"),
                        max_height=400,
                    ),
                    col_widths=[3,9]
                ),
            
                ui.card(
                    ui.card_header("Compare Simulation and Model on 1H time series with max sampling frequency"),
          
                    output_widget("plot_hour_timeseries"),
                                ui.input_select(  
                        "x_axis",  
                        "Select a x_axis below:",  
                        channel_dict_axis,
                        selected='time_sensor'),
                ),

                ui.card(
                    ui.card_header("max frequency sampling stats : Sensor minus Simulations"),
                    output_widget("box_plot_variables_1H_time_series"),
                ),
        )
