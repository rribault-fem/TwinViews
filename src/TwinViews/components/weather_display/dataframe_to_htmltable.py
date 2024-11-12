
import pandas as pd
from components.weather_display.color_maps_wind_wave_curr import cmapwind, cmapwave, cmaptp
from components.weather_display.format_variables import deg_to_arrow, formatnan, formatnanwave


dict_naming_wind_ms = {
    'WindSpeed': 'simu_AI_WindSpeed',
    'WindDir': 'simu_V_ST_TrueWindDir',
    'WindSource': 'wind_sensor',
}
vmin_wind=-5
vmax_wind=40

dict_naming_wave = {
    'Hs': 'simu_hs',
    'Tp': 'simu_tp',
    'WaveDir': 'simu_dp',
    'WaveSource': 'wave_sensor',
}

vmintp=-1
vmaxtp=20

dict_naming_curr = {
    'CurrSpeed': 'simu_cur',
    'CurrDir': 'simu_cur_dir',
    'CurrSource': 'curr_sensor',
}
vmincurr=-2
vmaxcurr=3

cmapcurr = cmapwind

html_style = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        .table-style {
            table-layout: fixed !important;
            height: fit-content;
            width: fit-content;
            font-size: 13px!important;
        }
        .table-style tbody {
            table-layout: fixed !important;
            height: fit-content;
            width: fit-content;
            font-size: 14px!important;
        }
        .table-style td, .table-style th {
            border: 1px solid #ffffff;
            text-align: center;
        }
        .table-style th {
            background-color: #f2f2f2 ;
        }
        .table-style th:first-child {
            width: 200px !important;
        }
        thead th:not(:first-child) {
            width: 31px!important;
        }
        .table-style tbody th {
            white-space: nowrap;
            height: 30px !important;
        }
        .table-style thead th {
            white-space: pre-wrap;
        }
    </style>
    </head>
    <body>
    """

def current_wind_to_htmlpicture(df: pd.DataFrame):
    styled_complete_dataframe = df.style.background_gradient(cmap=cmapwind,subset='Wind Speed', vmin=vmin_wind, vmax=vmax_wind)
    styled_complete_dataframe = styled_complete_dataframe.format(formatnan,subset='Wind Speed')
    styled_complete_dataframe = styled_complete_dataframe.format(deg_to_arrow, subset='Wind Direction')
    
    styled_complete_dataframe.set_table_attributes('class="table-style"')
    
    
    html_tab = styled_complete_dataframe.to_html(index=False)

    #puis je crées le block de code html complets
    complete_html_tab = html_style + html_tab + "</body></html>"

    # il est aussi possible de definir toute ces régles de style dans ton fichier css en utilisant le nom de class html crée plus tot seulement si elle ne sont pas variables.

    return complete_html_tab


def dataframe_to_htmltable(dataframe: pd.DataFrame):
    #dataframe classique indexer en fonction du temps qu'on transpose
    dataframe.index = dataframe.index.strftime('%a %d %Hh %M')
    dataframe = dataframe.T

    #j'applique mes modification de couleur et formats( nan value par exemple) en le passant sous forme d'un dataframes stylisé
    
    # color map for wind :
    styled_complete_dataframe = dataframe.style.background_gradient(cmap=cmapwind,subset=([dict_naming_wind_ms['WindSpeed'],], slice(None)), vmin=vmin_wind, vmax=vmax_wind)
    styled_complete_dataframe = styled_complete_dataframe.format(formatnan, subset=([dict_naming_wind_ms['WindSpeed'],], slice(None)))
    styled_complete_dataframe = styled_complete_dataframe.format(deg_to_arrow,subset=([dict_naming_wind_ms['WindDir']], slice(None)))
    # color map for waves :
    styled_complete_dataframe = styled_complete_dataframe.background_gradient(cmap=cmapwave,subset=([dict_naming_wave['Hs']], slice(None)), vmin=0, vmax=8)
    styled_complete_dataframe = styled_complete_dataframe.background_gradient(cmap=cmaptp,subset=([dict_naming_wave['Tp']], slice(None)), vmin=vmintp, vmax=vmaxtp)
    styled_complete_dataframe = styled_complete_dataframe.format(formatnanwave,subset=([dict_naming_wave['Hs']], slice(None)))
    styled_complete_dataframe = styled_complete_dataframe.format(formatnan,subset=([dict_naming_wave['WaveDir'],dict_naming_wave['Tp'],], slice(None)))
    styled_complete_dataframe = styled_complete_dataframe.format(deg_to_arrow,subset=([dict_naming_wave['WaveDir']], slice(None)))

    # color map for currents :
    styled_complete_dataframe = styled_complete_dataframe.background_gradient(cmap=cmapcurr,subset=([dict_naming_curr['CurrSpeed'],], slice(None)), vmin=vmincurr, vmax=vmaxcurr)
    styled_complete_dataframe = styled_complete_dataframe.format(formatnan,subset=([dict_naming_curr['CurrSpeed']], slice(None)))
    styled_complete_dataframe = styled_complete_dataframe.format(deg_to_arrow,subset=([dict_naming_curr['CurrDir']], slice(None)))


    #je definie le nom de class de mon dataframe stylisé puis le transforme en tableau html
    styled_complete_dataframe.set_table_attributes('class="table-style"')
    html_tab = styled_complete_dataframe.to_html()

    #je stock sous forme de str le block de code css que je vais appliqué à mon tableau


    
    #puis je crées le block de code html complets
    complete_html_tab = html_style + html_tab + "</body></html>"

    # il est aussi possible de definir toute ces régles de style dans ton fichier css en utilisant le nom de class html crée plus tot seulement si elle ne sont pas variables.

    return complete_html_tab