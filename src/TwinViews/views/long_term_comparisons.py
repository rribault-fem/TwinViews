    
from shiny import ui
from shinywidgets import output_widget
from pathlib import Path
from faicons import icon_svg

def get_ui_long_term_comparison(channel_dict: dict):
    
    return ui.nav_panel("Long Term comparison",
                    
        ui.include_css(Path(__file__).parent.parent / "styles.css"),

        ui.card(
            ui.card_header("Data availability"),

            # ui.value_box(
            #     "SIMU HOURS",
            #     ui.output_text('get_hours_simu'),
            #     # ui.output_ui("current_waves"),
            #     showcase=[icon_svg("robot")],
            #     id='value_box_wave',
                
            # ),

        ui.layout_column_wrap( 
            ui.value_box(
                "Scada (orange) & Simu (navy) available hours on full dataset",
                ui.output_text('get_all_hours'),
                showcase=output_widget("sparkline_scada"),
                showcase_layout="bottom",
            ),
            ui.value_box(
                "SCADA HOURS within selected date range",
                ui.output_text('get_hours_scada'),
                # ui.output_ui("current_wind"),
                showcase=[icon_svg("ship")],
                id='value_box_wind',
                ),
            ui.value_box(
                "SIMU HOURS within selected date range",
                ui.output_text('get_hours_simu'),
                # ui.output_ui("current_waves"),
                showcase=[icon_svg("robot")],
                id='value_box_wave',
                
                ),
            ),        
        ),
        ui.card(
            ui.card_header("Weather conditions"),
            ui.output_ui("weather_tab"),
        ),
        
        ui.card(
            ui.card_header("Simu vs. Sensor time series & accuracy"),
            ui.input_radio_buttons('precision_metrics_stat_selection_channel', 'Select statistics', ['mean', 'std', 'min', 'max'], inline=True),
            output_widget("channel_comparison"),
        ),

        ui.card(
            ui.card_header("Hourly stats : Sensor minus Simulations"),
            output_widget("box_plot_variables_long_term"),
            height='auto'
        ),
        
        ui.card(
            ui.card_header("Dataset overview"),
            ui.input_switch("dataset_overview_all", "Show all dataset"),
            ui.layout_column_wrap( 
                ui.input_select(  
                    "channels_stats_identifiers",  
                    "Channel to analyse :",  
                    channel_dict,
                    multiple=False,
                    selected='V_MRU_Heave',
                    size= 5
                    ),
                ui.input_radio_buttons('stat_selection', 'Select statistics', ['mean', 'std', 'min', 'max']),
                # ui.download_button("download", "Download CSV"),
            ),
            ui.output_data_frame("dataset_overview"),
        ),
    ),