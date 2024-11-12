from shiny import ui
from shinywidgets import output_widget
from pathlib import Path
from faicons import icon_svg

def get_ui_aero():
    return ui.nav_panel("Aero Analysis",
            ui.input_radio_buttons('resample_aero', 'Select resampling period', ['1s','10s','1min', '10min','1H'],
                            selected='10s',
                            inline=True),
            ui.card(
                    ui.card_header("stat"),
                    ui.output_plot('power_curve', height=600),
            ),
        )
