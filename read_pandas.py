# %%

# Paket für Bearbeitung von Tabellen
import pandas as pd
import numpy as np

# Paket
## zuvor !pip install plotly
## ggf. auch !pip install nbformat
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"


def read_my_csv():
    # Einlesen eines Dataframes
    ## "\t" steht für das Trennzeichen in der txt-Datei (Tabulator anstelle von Beistrich)
    ## header = None: es gibt keine Überschriften in der txt-Datei
    df = pd.read_csv("data/activities/activity.csv", sep=",", header=0)
    time = np.arange(0, len(df))
    df["Time"]=time
    
    # Gibt den geladen Dataframe zurück
    return df

def get_zone_limit(max_hr):
    zone_1 = [0.5*max_hr, 0.6*max_hr]
    zone_2 = [0.6*max_hr, 0.7*max_hr]
    zone_3 = [0.7*max_hr, 0.8*max_hr]
    zone_4 = [0.8*max_hr, 0.9*max_hr]
    zone_5 = [0.9*max_hr, 1.0*max_hr]

    zone_dict={"Zone_1": zone_1,
               "Zone_2": zone_2,
               "Zone_3": zone_3,
               "Zone_4": zone_4,
               "Zone_5": zone_5,}

    return zone_dict


def assign_zone(hr, zones):
    for zone, (low, high) in zones.items():
        if low <= hr < high:
            return zone
    return 'Zone_5'  # Falls hr == max_hr

def make_plot(df, zones):
    zone_colors = {
        'Zone_1': 'blue',
        'Zone_2': 'green', 
        'Zone_3': 'yellow',
        'Zone_4': 'orange',
        'Zone_5': 'red'
    }
    
    # Zone assignment should be done before calling this function
    df['Zone'] = df['HeartRate'].apply(lambda x: assign_zone(x, zones))
    
    fig = px.scatter(
        df, x='Time', y='HeartRate', color='Zone',
        color_discrete_map=zone_colors,
        labels={'HeartRate': 'Herzfrequenz [bpm], Power [W]', 'Time': 'Zeit [s]'},
        title='Herzfrequenz- und Leistungsanalyse'
    )
    
    fig.add_scatter(
        x=df['Time'], y=df['PowerOriginal'],
        mode='lines', name='Power', line=dict(color='black', width=2)
    )
    
    return fig

#%% Test - Nur ausführen wenn das Modul direkt gestartet wird
if __name__ == "__main__":
    df = read_my_csv()
    max_hr = df['HeartRate'].max()
    zones = get_zone_limit(max_hr)
    
    df['Zone'] = df['HeartRate'].apply(lambda x: assign_zone(x, zones))
    
    zone_counts = df['Zone'].value_counts().sort_index()
    zone_minutes = zone_counts / 60  # Umrechnung von Sekunden in Minuten

    print("Zeit in jeder Herzfrequenzzone (in Minuten):")
    for zone, minutes in zone_minutes.items():
        print(f"{zone}: {minutes:.1f} Minuten")

    fig = make_plot(df, zones)
    fig.show()

# %%
