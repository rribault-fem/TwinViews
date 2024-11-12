#####
# Format fuctions for smoothing the display of the dataframe
#####

import math
import pandas as pd

def deg_to_arrow(v):

    if pd.isna(v):
        return '-'

    v2=(v+90)%360
    angle_radians = math.radians((v2-2*v2)%360)

    # Longueur du vecteur (diagonale du carré)
    vector_length = 20 * math.sqrt(2)


    # Coordonnées du centre du carré
    x = 50
    y = 50

    # Coordonnées du point d'arrivée
    x2 = x + vector_length * math.cos(angle_radians)
    y2 = y - vector_length * math.sin(angle_radians)

    # Coordonnées du point de départ
    x1 = x - vector_length * math.cos(angle_radians)
    y1 = y + vector_length * math.sin(angle_radians)



    html_arrow = '''
    <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <!-- A marker to be used as an arrowhead -->
            <marker
            id="arrow"
            viewBox="0 0 10 10"
            refX="5"
            refY="5"
            markerWidth="5"
            markerHeight="5"
            orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" />
            </marker>
        </defs>

        <!-- A line with a marker -->
        <line
            x1="'''+str(x1)+'''"
            y1="'''+str(y1)+'''"
            x2="'''+str(x2)+'''"
            y2="'''+str(y2)+'''"
            stroke="black"
            stroke-width="5"
            marker-end="url(#arrow)" />
        <!-- A curved path with markers -->
        
    </svg>
    '''

    return html_arrow

def formatnan(v):
    if pd.isna(v):
        return '-'
    return f"{v:.0f}"

def formatnanwave(v):
    if pd.isna(v):
        return '-'
    return f"{v:.1f}"