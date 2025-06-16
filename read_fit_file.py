from fitparse import FitFile
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def read_fit_file(file):
    fitfile = FitFile(file)
    times = []
    all_records = []

    for record in fitfile.get_messages('record'):
        data = {}
        for field in record:
            data[field.name] = field.value
            if field.name == 'timestamp':
                times.append(field.value)
        all_records.append(data)

    # Zeit in Sekunden seit Beginn berechnen
    if times:
        start_time = times[0]
        times_seconds = [(t - start_time).total_seconds() for t in times]
        time = np.array(times_seconds)
    else:
        time = np.array([])

    df = pd.DataFrame(all_records)
    df['time_seconds'] = time
    return df

def plot_heart_rate(df):
    if 'heart_rate' in df and len(df['heart_rate']) > 0:
        hr = df['heart_rate'].to_numpy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=hr, mode='lines', name='Herzfrequenz'))
        fig.update_layout(title='Herzfrequenzverlauf', xaxis_title='Zeit (Index)', yaxis_title='Herzfrequenz (bpm)')
        return fig
    return None

def plot_altitude(df):
    if 'altitude' in df and len(df['altitude']) > 0:
        altitude = df['altitude'].to_numpy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=altitude, mode='lines+markers', name='Höhenmeter'))
        fig.update_layout(title='Höhenmeterverlauf', xaxis_title='Zeit (Index)', yaxis_title='Höhe (m)')
        return fig
    return None

def plot_gpx(df):
    if 'position_lat' in df and 'position_long' in df:
        latitude = df['position_lat'].to_numpy()
        longitude = df['position_long'].to_numpy()
        if len(latitude) > 0 and len(longitude) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scattergeo(
                lat=latitude,
                lon=longitude,
                mode='lines+markers',
                line=dict(width=2, color='blue'),
                marker=dict(size=5, color='red')
            ))
            fig.update_layout(
                title='GPX Route',
                geo=dict(
                    scope='europe',
                    projection_type='mercator',
                    showland=True,
                    landcolor='rgb(243, 243, 243)',
                    countrycolor='rgb(204, 204, 204)',
                )
            )
            return fig
    return None
