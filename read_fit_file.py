from fitparse import FitFile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gpxpy
import folium

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

def get_lat_lon(df):
    # Nutze enhanced_* wenn vorhanden (bereits Dezimalgrad!)
    if 'enhanced_position_lat' in df and 'enhanced_position_long' in df:
        lat = df['enhanced_position_lat']
        lon = df['enhanced_position_long']
    elif 'position_lat' in df and 'position_long' in df:
        lat = df['position_lat'] / 1e7
        lon = df['position_long'] /1e7
    else:
        return None, None
    mask = (~pd.isnull(lat)) & (~pd.isnull(lon)) & (lat != 0) & (lon != 0)
    return lat[mask], lon[mask]

def plot_gpx(df):
    lat, lon = get_lat_lon(df)
    if lat is None or lon is None or len(lat) < 2:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=lat,
        lon=lon,
        mode='lines+markers',
        line=dict(width=2, color='blue'),
        marker=dict(size=5, color='red')
    ))
    fig.update_layout(
        title='GPX Route',
        geo=dict(
            projection_type='mercator',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
        )
    )
    return fig

def plot_gpx_folium(df):
    lat, lon = get_lat_lon(df)
    if lat is None or lon is None or len(lat) < 2:
        return None

    # Konvertiere zu Listen für Folium
    latitudes = list(lat)
    longitudes = list(lon)

    # Startpunkt der Karte (erster GPS-Punkt)
    start_coords = [latitudes[0], longitudes[0]]
    m = folium.Map(location=start_coords, zoom_start=13)

    # Route als PolyLine hinzufügen
    folium.PolyLine(list(zip(latitudes, longitudes)), color='blue', weight=5).add_to(m)

    return m

# Optional: Testlauf
if __name__ == "__main__":
    fit_file_path = 'data/fit_file/pillersee.fit'
    df = read_fit_file(fit_file_path)
    duration_hours = calculate_workout_duration_hours(df)
    fig_hr = plot_heart_rate(df, duration_hours)
    fig_alt = plot_altitude(df, duration_hours)
    fig_gpx = plot_gpx(df)
    m = plot_gpx_folium(df)
    print(get_lat_lon(df))
    print(df.head())