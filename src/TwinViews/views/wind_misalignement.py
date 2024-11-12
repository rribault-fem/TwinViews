    
from shiny import ui
from shinywidgets import output_widget
from pathlib import Path
from faicons import icon_svg
from config import app_config

def get_ui_wind_misalignment():
    
    return ui.nav_panel("Wind Misalignment",
                    
        ui.include_css(Path(__file__).parent.parent / "styles.css"),

        ui.card(
            ui.card_header("Wind missalignment"),
            ui.layout_column_wrap(
                ui.input_radio_buttons('precision_metrics_stat_misalignment', 'Select statistics', ['mean', 'std', 'min', 'max'], inline=True),
                ui.input_select('misalignment_weather_variables', 'Select weather variables', app_config.list_envir_for_pair_plot, selected=['simu_AI_WindSpeed', 'simu_tp'], multiple=True),
                ui.input_select('misalignment_comp_channel', 'Select channel to compare', ['V_ST_TrueNacelleDir', 'V_MRU_Heading'], selected='V_ST_TrueNacelleDir', multiple=False),
            ),
            ui.h4("Sensor :"),
            output_widget("pair_plot_wind_missalignment_sensor"),
            ui.h4("Simu :"),
            output_widget("pair_plot_wind_missalignment_simu"),
        ),

    )