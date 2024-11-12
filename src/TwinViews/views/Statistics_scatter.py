    
from shiny import ui
from shinywidgets import output_widget
from pathlib import Path
from faicons import icon_svg
from config import app_config


def get_ui_statistics_scatter():
    
    return ui.nav_panel("Stats accuracy Scatter",
                    
        ui.include_css(Path(__file__).parent.parent / "styles.css"),
        
        ui.card(
            ui.card_header("Statistics over 1h"),
            ui.h4("Sensor - Simu accuracy stat vs. weather variables"),
            ui.layout_column_wrap(
                ui.input_radio_buttons('precision_metrics_stat_selection', 'Select statistics', ['mean', 'std', 'min', 'max'], inline=True),
                ui.input_select('pair_plot_weather_variables', 'Select weather variables', app_config.list_envir_for_pair_plot, selected=['simu_AI_WindSpeed', 'simu_tp'], multiple=True),                  
                ),
            output_widget('pair_plot_weather'),
        ),

        ), 