from fitparse import FitFile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gpxpy

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

def calculate_workout_duration_hours(df):
    if 'time_seconds' in df and len(df['time_seconds']) > 0:
        total_seconds = df['time_seconds'].max() - df['time_seconds'].min()
        duration_hours = total_seconds / 3600
        return duration_hours
    else:
        return 0

def plot_heart_rate(df, duration_hours):
    if 'heart_rate' in df and len(df['heart_rate']) > 0:
        hr = df['heart_rate'].to_numpy()
        time = df['time_seconds'].to_numpy() if 'time_seconds' in df else np.arange(len(hr))
        if duration_hours > 2:
            time_scaled = time / 3600
            xaxis_title = 'Zeit (Stunden)'
        else:
            time_scaled = time / 60
            xaxis_title = 'Zeit (Minuten)'
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_scaled, y=hr, mode='lines', name='Herzfrequenz'))
        fig.update_layout(title='Herzfrequenzverlauf', xaxis_title=xaxis_title, yaxis_title='Herzfrequenz (bpm)')
        return fig
    return None

def plot_altitude(df, duration_hours):
    if 'altitude' in df and len(df['altitude']) > 0:
        altitude = df['altitude'].to_numpy()
        time = df['time_seconds'].to_numpy() if 'time_seconds' in df else np.arange(len(altitude))
        if duration_hours > 2:
            time_scaled = time / 3600
            xaxis_title = 'Zeit (Stunden)'
        else:
            time_scaled = time / 60
            xaxis_title = 'Zeit (Minuten)'
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_scaled, y=altitude, mode='lines+markers', name='Höhenmeter'))
        fig.update_layout(title='Höhenmeterverlauf', xaxis_title=xaxis_title, yaxis_title='Höhe (m)')
        return fig
    return None


def plot_gpx(df):
    # Prüfen, ob die Spalten existieren
    if 'position_lat' not in df or 'position_long' not in df:
        return None

    # Nach Zeit sortieren, falls möglich
    if 'time_seconds' in df:
        df = df.sort_values('time_seconds')
    elif 'timestamp' in df:
        df = df.sort_values('timestamp')

    # Rohwerte in Grad umwandeln und ungültige Werte filtern
    latitude = df['position_lat'].to_numpy() / 1e7
    longitude = df['position_long'].to_numpy() / 1e7

    # Nur sinnvolle Koordinaten behalten (keine NaN, keine 0)
    mask = (
        (~pd.isnull(latitude)) & (~pd.isnull(longitude)) &
        (latitude != 0) & (longitude != 0)
    )
    latitude = latitude[mask]
    longitude = longitude[mask]

    if len(latitude) < 2 or len(longitude) < 2:
        return None

    # GPX-Objekt mit gpxpy erstellen (optional, für spätere Analysen)
    gpx = gpxpy.gpx.GPX()
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)
    for lat, lon in zip(latitude, longitude):
        gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(lat, lon))

    # Plotly-Plot erzeugen
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=latitude,
        lon=longitude,
        mode='lines+markers',
        line=dict(width=2, color='blue'),
        marker=dict(size=5, color='red')
    ))
    fig.update_layout(
        title='GPX Route (optimiert)',
        geo=dict(
            projection_type='mercator',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
        )
    )
    return fig



# Optional: Testlauf
if __name__ == "__main__":
    fit_file_path = 'data/fit_file/pillersee.fit'
    df = read_fit_file(fit_file_path)
    duration_hours = calculate_workout_duration_hours(df)
    fig_hr = plot_heart_rate(df, duration_hours)
    fig_alt = plot_altitude(df, duration_hours)
    fig_gpx = plot_gpx(df)
    print(df.head())
