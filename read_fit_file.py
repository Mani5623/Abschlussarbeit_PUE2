from fitparse import FitFile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import folium
import matplotlib.cm as cm
import matplotlib.colors as colors
from functools import lru_cache

# Konstanten für bessere Performance
SEMICIRCLE_TO_DEGREE = 180 / 2**31
AVAILABLE_METRICS = {
    'altitude': 'Höhenmeter',
    'heart_rate': 'Herzfrequenz', 
    'speed': 'Geschwindigkeit',
    'power': 'Leistung'
}

def read_fit_file(file):
    """Optimierte FIT-File Einlesung mit besserer Performance"""
    fitfile = FitFile(file)
    all_records = []
    
    # Direkte Liste statt separater times Liste
    for record in fitfile.get_messages('record'):
        data = {field.name: field.value for field in record}
        all_records.append(data)
    
    if not all_records:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_records)
    
    # Zeit in Sekunden berechnen (falls timestamp vorhanden)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        start_time = df['timestamp'].iloc[0]
        df['time_seconds'] = (df['timestamp'] - start_time).dt.total_seconds()
    
    return df

def calculate_workout_duration_hours(df):
    """Effiziente Berechnung der Workout-Dauer"""
    if 'time_seconds' not in df or df.empty:
        return 0
    return (df['time_seconds'].iloc[-1] - df['time_seconds'].iloc[0]) / 3600

def create_time_plot(df, column, title, y_label, duration_hours):
    """Generische Funktion für Zeit-basierte Plots"""
    if column not in df or df[column].isna().all():
        return None
    
    time_data = df['time_seconds'] if 'time_seconds' in df else np.arange(len(df))
    
    # Zeitachse skalieren
    if duration_hours > 2:
        time_scaled = time_data / 3600
        xaxis_title = 'Zeit (Stunden)'
    else:
        time_scaled = time_data / 60
        xaxis_title = 'Zeit (Minuten)'
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_scaled, 
        y=df[column], 
        mode='lines', 
        name=y_label
    ))
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=y_label
    )
    return fig

def plot_heart_rate(df, duration_hours):
    """Herzfrequenz-Plot mit generischer Funktion"""
    return create_time_plot(df, 'heart_rate', 'Herzfrequenzverlauf', 'Herzfrequenz (bpm)', duration_hours)

def plot_altitude(df, duration_hours):
    """Höhen-Plot mit generischer Funktion"""
    return create_time_plot(df, 'altitude', 'Höhenmeterverlauf', 'Höhe (m)', duration_hours)

@lru_cache(maxsize=1)
def get_colormap():
    """Cached Colormap für bessere Performance"""
    return cm.get_cmap('viridis')

def get_lat_lon_optimized(df):
    """Optimierte GPS-Koordinaten Extraktion"""
    lat, lon = None, None
    
    # Prüfe enhanced position zuerst (häufiger und genauer)
    if 'enhanced_position_lat' in df and 'enhanced_position_long' in df:
        lat = df['enhanced_position_lat']
        lon = df['enhanced_position_long']
    elif 'position_lat' in df and 'position_long' in df:
        # Effiziente Vektorisierte Konvertierung
        lat = df['position_lat'] * SEMICIRCLE_TO_DEGREE
        lon = df['position_long'] * SEMICIRCLE_TO_DEGREE
    else:
        return None, None, None
    
    # Optimierte Maske mit numpy
    mask = (
        lat.notna() & lon.notna() & 
        (lat != 0) & (lon != 0) &
        (lat.abs() <= 90) & (lon.abs() <= 180)  # Validitätsprüfung
    )
    
    return lat[mask], lon[mask], mask

def get_available_metrics(df):
    """Effiziente Prüfung verfügbarer Metriken"""
    return {
        metric: label for metric, label in AVAILABLE_METRICS.items()
        if metric in df.columns and not df[metric].isna().all()
    }

def calculate_optimal_padding(lat, lon):
    """Berechnet optimales Padding basierend auf der Route"""
    lat_range = lat.max() - lat.min()
    lon_range = lon.max() - lon.min()
    
    # Dynamisches Padding basierend auf der Routengröße
    if lat_range < 0.01 and lon_range < 0.01:  # Sehr kleine Route
        return [50, 50]
    elif lat_range < 0.1 and lon_range < 0.1:  # Kleine Route
        return [30, 30]
    else:  # Große Route
        return [20, 20]

def plot_gpx_folium_colored(df, color_metric='altitude'):
    """Optimierte farbkodierte Folium-Karte mit Auto-Fit und ohne Zoom"""
    lat, lon, mask = get_lat_lon_optimized(df)
    if lat is None or len(lat) < 2:
        return None
    
    # Prüfe Metrik-Verfügbarkeit
    if color_metric not in df.columns or df[color_metric].isna().all():
        return plot_gpx_folium_simple(lat, lon)
    
    # Gefilterte Metrik-Daten
    metric_data = df[color_metric][mask].fillna(method='ffill').fillna(0)
    
    if len(lat) != len(metric_data):
        return plot_gpx_folium_simple(lat, lon)
    
    # Konvertiere zu numpy arrays für bessere Performance
    latitudes = lat.values
    longitudes = lon.values
    metric_values = metric_data.values
    
    # Berechne optimales Padding
    padding = calculate_optimal_padding(lat, lon)
    
    # Karte ohne Zoom-Kontrollen erstellen
    m = folium.Map(
        zoom_control=False,        # Zoom-Buttons entfernen
        scrollWheelZoom=False,     # Mausrad-Zoom deaktivieren
        doubleClickZoom=False,     # Doppelklick-Zoom deaktivieren
        touchZoom=False,           # Touch-Zoom deaktivieren
        boxZoom=False,             # Box-Zoom deaktivieren
        keyboard=False,            # Tastatur-Navigation deaktivieren
        dragging=True,             # Verschieben erlauben
        prefer_canvas=True         # Performance-Optimierung
    )
    
    # Farbkodierung nur wenn verschiedene Werte vorhanden
    if len(np.unique(metric_values)) > 1:
        # Normalisierung mit numpy für bessere Performance
        vmin, vmax = np.nanmin(metric_values), np.nanmax(metric_values)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        colormap = get_colormap()
        
        # Effiziente Segment-Erstellung
        coords = list(zip(latitudes, longitudes))
        for i in range(len(coords) - 1):
            avg_value = (metric_values[i] + metric_values[i + 1]) / 2
            color = colors.rgb2hex(colormap(norm(avg_value)))
            
            folium.PolyLine(
                locations=[coords[i], coords[i + 1]],
                color=color,
                weight=4,
                opacity=0.8
            ).add_to(m)
        
        # Kompakte Legende
        add_legend(m, color_metric, vmin, vmax)
    else:
        # Einfache Route
        folium.PolyLine(
            list(zip(latitudes, longitudes)),
            color='blue',
            weight=4
        ).add_to(m)
    
    # Start/End Marker
    add_start_end_markers(m, latitudes, longitudes)
    
    # ✅ Automatische Anpassung der Kartenansicht auf die gesamte Route
    bounds = [[lat.min(), lon.min()], [lat.max(), lon.max()]]
    m.fit_bounds(bounds, padding=padding)
    
    return m

def plot_gpx_folium_simple(lat, lon):
    """Einfache Folium-Karte ohne Farbkodierung mit Auto-Fit"""
    latitudes = lat.values
    longitudes = lon.values
    
    # Berechne optimales Padding
    padding = calculate_optimal_padding(lat, lon)
    
    # Karte ohne Zoom-Kontrollen
    m = folium.Map(
        zoom_control=False,
        scrollWheelZoom=False,
        doubleClickZoom=False,
        touchZoom=False,
        boxZoom=False,
        keyboard=False,
        dragging=True,
        prefer_canvas=True
    )
    
    folium.PolyLine(
        list(zip(latitudes, longitudes)),
        color='blue',
        weight=4
    ).add_to(m)
    
    add_start_end_markers(m, latitudes, longitudes)
    
    # ✅ Automatische Anpassung der Kartenansicht
    bounds = [[lat.min(), lon.min()], [lat.max(), lon.max()]]
    m.fit_bounds(bounds, padding=padding)
    
    return m

def add_legend(m, metric, vmin, vmax):
    """Fügt Legende zur Karte hinzu"""
    metric_label = AVAILABLE_METRICS.get(metric, metric)
    legend_html = f'''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 150px; height: 90px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px; border-radius: 5px;">
    <p><b>{metric_label}</b></p>
    <p>Min: {vmin:.1f}</p>
    <p>Max: {vmax:.1f}</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

def add_start_end_markers(m, latitudes, longitudes):
    """Fügt Start- und End-Marker hinzu"""
    folium.Marker(
        [latitudes[0], longitudes[0]],
        popup="Start",
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)
    
    folium.Marker(
        [latitudes[-1], longitudes[-1]],
        popup="Ende",
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)

# Backward compatibility
def plot_gpx_folium(df):
    """Backward compatibility wrapper"""
    lat, lon, _ = get_lat_lon_optimized(df)
    if lat is None or len(lat) < 2:
        return None
    return plot_gpx_folium_simple(lat, lon)

if __name__ == "__main__":
    import os
    fit_file_path = 'data/fit_file/pillersee.fit'
    
    if os.path.exists(fit_file_path):
        with open(fit_file_path, 'rb') as f:
            df = read_fit_file(f)
        
        # GPX-Plot erstellen
        m = plot_gpx_folium_colored(df, 'altitude')
        if m:
            m.save('gpx_test.html')
            print("✅ GPX-Plot erstellt: gpx_test.html")
        else:
            print("❌ Kein GPX-Plot möglich")
    else:
        print(f"❌ Datei nicht gefunden: {fit_file_path}")
