 
from shiny import ui
from shinywidgets import output_widget

def get_ui_freq_stats():
    return  ui.nav_panel("Statistics",
                
                ui.card(
                    ui.card_header("Meteocean data histograms & power output"),
                    ui.output_plot('envir_power_stats'),
                    height='600px'
                ),

                # ui.card(
                #     ui.card_header("frequency-dependent statistics"),
                #     ui.layout_column_wrap(
                #         ui.input_radio_buttons('resample', 'Select resampling period', ['1min', '10min', '20min', '1h'],
                #                             selected='10min',
                #                             inline=True),

                #         ui.input_numeric("lf", "Low frequency", value=1/(16.5), min=0, max=1, step=0.001, width='auto'),
                #         ui.input_numeric("hf", "High frequency", value=1/(0.52), min=0, max=1, step=0.001, width='auto'),
                #     ),
                #     ui.output_plot('freq_stats'),
                # ),
            ),