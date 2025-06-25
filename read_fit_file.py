from fitparse import FitFile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import folium
import matplotlib.cm as cm
import matplotlib.colors as colors
from functools import lru_cache
from datetime import timedelta

class FitFileAnalyzer:
    """Objektorientierte Klasse für FIT-File Analyse"""
    
    # Klassenkonstanten
    SEMICIRCLE_TO_DEGREE = 180 / 2**31
    AVAILABLE_METRICS = {
        'altitude': 'Höhenmeter',
        'heart_rate': 'Herzfrequenz', 
        'speed': 'Geschwindigkeit',
        'power': 'Leistung'
    }
    
    SPORT_METRICS = {
        "Radfahren": [
            ('power', 'W', 'Durchschnittliche Leistung'), 
            ('distance', 'km', 'Gefahrene Distanz', 1000)
        ],
        "Laufen": [
            ('distance', 'km', 'Gelaufene Distanz', 1000)
        ],
        "Schwimmen": [
            ('distance', 'm', 'Geschwommene Distanz')
        ],
        "Sonstiges": [
            ('distance', 'm', 'Distanz')
        ]
    }
    
    def __init__(self, fit_file):
        """Initialisiert den Analyzer mit einer FIT-Datei"""
        self.fit_file = fit_file
        self.df = None
        self.duration_hours = 0
        self.available_metrics = {}
        self._load_data()
    
    def _load_data(self):
        """Lädt und verarbeitet die FIT-Datei"""
        fitfile = FitFile(self.fit_file)
        all_records = []
        
        for record in fitfile.get_messages('record'):
            data = {field.name: field.value for field in record}
            all_records.append(data)
        
        if not all_records:
            self.df = pd.DataFrame()
            return
        
        self.df = pd.DataFrame(all_records)
        
        # Zeit in Sekunden berechnen
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            start_time = self.df['timestamp'].iloc[0]
            self.df['time_seconds'] = (self.df['timestamp'] - start_time).dt.total_seconds()
        
        self._calculate_duration()
        self._get_available_metrics()
    
    def _calculate_duration(self):
        """Berechnet die Workout-Dauer"""
        if 'time_seconds' not in self.df or self.df.empty:
            self.duration_hours = 0
            return
        self.duration_hours = (self.df['time_seconds'].iloc[-1] - self.df['time_seconds'].iloc[0]) / 3600
    
    def _get_available_metrics(self):
        """Ermittelt verfügbare Metriken"""
        self.available_metrics = {
            metric: label for metric, label in self.AVAILABLE_METRICS.items()
            if metric in self.df.columns and not self.df[metric].isna().all()
        }
    
    def format_duration(self):
        """Formatiert die Dauer in lesbares Format"""
        total_minutes = int(self.duration_hours * 60)
        hours_part = total_minutes // 60
        minutes_part = total_minutes % 60
        
        if hours_part > 0:
            return f"{hours_part}h {minutes_part}min"
        else:
            return f"{minutes_part}min"
    
    def get_sport_statistics(self, sport_type):
        """Berechnet sportartspezifische Statistiken"""
        stats = {}
        
        for metric_config in self.SPORT_METRICS.get(sport_type, []):
            metric = metric_config[0]
            unit = metric_config[1]
            label = metric_config[2]
            divisor = metric_config[3] if len(metric_config) > 3 else 1
            
            if metric in self.df and not self.df[metric].isna().all():
                value = self.df[metric].mean() if metric == 'power' else self.df[metric].max()
                value /= divisor
                stats[label] = {'value': value, 'unit': unit, 'metric': metric}
        
        return stats
    
    def calculate_speed_metrics(self, sport_type, distance_value, distance_unit):
        """Berechnet Geschwindigkeits- und Pace-Metriken"""
        if self.duration_hours <= 0:
            return None
        
        if distance_unit == 'km':
            speed = distance_value / self.duration_hours
            return {'type': 'speed', 'value': speed, 'unit': 'km/h', 'label': 'Durchschnittsgeschwindigkeit'}
        
        elif distance_unit == 'm':
            if sport_type == "Schwimmen":
                pace_per_100m = (self.duration_hours * 60) / (distance_value / 100)
                return {'type': 'pace', 'value': pace_per_100m, 'unit': 'min/100m', 'label': 'Pace'}
            
            elif sport_type == "Laufen":
                distance_km = distance_value / 1000
                pace_per_km = (self.duration_hours * 60) / distance_km
                pace_minutes = int(pace_per_km)
                pace_seconds = int((pace_per_km - pace_minutes) * 60)
                return {'type': 'pace', 'value': f"{pace_minutes}:{pace_seconds:02d}", 'unit': 'min/km', 'label': 'Pace'}
        
        return None
    
    def get_heart_rate_stats(self):
        """Berechnet Herzfrequenz-Statistiken"""
        if 'heart_rate' not in self.df or self.df['heart_rate'].isna().all():
            return None
        
        return {
            'avg': self.df['heart_rate'].mean(),
            'max': self.df['heart_rate'].max()
        }
    
    def get_elevation_gain(self):
        """Berechnet Höhenmeter bergauf"""
        if 'altitude' not in self.df or self.df['altitude'].isna().all():
            return None
        
        return (self.df['altitude'].diff().clip(lower=0)).sum()
    
    def create_heart_rate_plot(self):
        """Erstellt Herzfrequenz-Plot"""
        return self._create_time_plot('heart_rate', 'Herzfrequenzverlauf', 'Herzfrequenz (bpm)')
    
    def create_altitude_plot(self):
        """Erstellt Höhen-Plot"""
        return self._create_time_plot('altitude', 'Höhenmeterverlauf', 'Höhe (m)')
    
    def _create_time_plot(self, column, title, y_label):
        """Generische Funktion für Zeit-basierte Plots"""
        if column not in self.df or self.df[column].isna().all():
            return None
        
        time_data = self.df['time_seconds'] if 'time_seconds' in self.df else np.arange(len(self.df))
        
        # Zeitachse skalieren
        if self.duration_hours > 2:
            time_scaled = time_data / 3600
            xaxis_title = 'Zeit (Stunden)'
        else:
            time_scaled = time_data / 60
            xaxis_title = 'Zeit (Minuten)'
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_scaled, 
            y=self.df[column], 
            mode='lines', 
            name=y_label
        ))
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=y_label
        )
        return fig
    
    def create_gps_map(self, color_metric=None):
        """Erstellt GPS-Karte - standardmäßig ohne Farbkodierung"""
        lat, lon, mask = self._get_lat_lon_optimized()
        if lat is None or len(lat) < 2:
            return None
        
        # Standardmäßig ohne Farbe, nur wenn explizit gewählt
        if color_metric is None or color_metric not in self.df.columns or self.df[color_metric].isna().all():
            return self._create_simple_map(lat, lon)
        
        return self._create_colored_map(lat, lon, mask, color_metric)
    
    def _get_lat_lon_optimized(self):
        """Optimierte GPS-Koordinaten Extraktion"""
        lat, lon = None, None
        
        if 'enhanced_position_lat' in self.df and 'enhanced_position_long' in self.df:
            lat = self.df['enhanced_position_lat']
            lon = self.df['enhanced_position_long']
        elif 'position_lat' in self.df and 'position_long' in self.df:
            lat = self.df['position_lat'] * self.SEMICIRCLE_TO_DEGREE
            lon = self.df['position_long'] * self.SEMICIRCLE_TO_DEGREE
        else:
            return None, None, None
        
        mask = (
            lat.notna() & lon.notna() & 
            (lat != 0) & (lon != 0) &
            (lat.abs() <= 90) & (lon.abs() <= 180)
        )
        
        return lat[mask], lon[mask], mask
    
    def _create_colored_map(self, lat, lon, mask, color_metric):
        """Erstellt farbkodierte Karte mit Legende"""
        metric_data = self.df[color_metric][mask].fillna(method='ffill').fillna(0)
        
        if len(lat) != len(metric_data):
            return self._create_simple_map(lat, lon)
        
        latitudes = lat.values
        longitudes = lon.values
        metric_values = metric_data.values
        
        padding = self._calculate_optimal_padding(lat, lon)
        
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
        
        if len(np.unique(metric_values)) > 1:
            vmin, vmax = np.nanmin(metric_values), np.nanmax(metric_values)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            colormap = cm.get_cmap('viridis')
            
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
            
            # Verbesserte Legende mit Farbbalken
            self._add_color_legend(m, color_metric, vmin, vmax, colormap, norm)
        else:
            folium.PolyLine(
                list(zip(latitudes, longitudes)),
                color='blue',
                weight=4
            ).add_to(m)
        
        self._add_start_end_markers(m, latitudes, longitudes)
        
        bounds = [[lat.min(), lon.min()], [lat.max(), lon.max()]]
        m.fit_bounds(bounds, padding=padding)
        
        return m
    
    def _create_simple_map(self, lat, lon):
        """Erstellt einfache Karte ohne Farbkodierung"""
        latitudes = lat.values
        longitudes = lon.values
        
        padding = self._calculate_optimal_padding(lat, lon)
        
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
            weight=4,
            opacity=0.8
        ).add_to(m)
        
        self._add_start_end_markers(m, latitudes, longitudes)
        
        bounds = [[lat.min(), lon.min()], [lat.max(), lon.max()]]
        m.fit_bounds(bounds, padding=padding)
        
        return m
    
    def _calculate_optimal_padding(self, lat, lon):
        """Berechnet optimales Padding für die Karte"""
        lat_range = lat.max() - lat.min()
        lon_range = lon.max() - lon.min()
        
        if lat_range < 0.01 and lon_range < 0.01:
            return [50, 50]
        elif lat_range < 0.1 and lon_range < 0.1:
            return [30, 30]
        else:
            return [20, 20]
    
    def _add_color_legend(self, m, metric, vmin, vmax, colormap, norm):
        """Fügt erweiterte Farbbalken-Legende zur Karte hinzu"""
        metric_label = self.AVAILABLE_METRICS.get(metric, metric)
        
        # Erstelle Farbbalken-HTML
        gradient_colors = []
        for i in range(10):
            value = vmin + (vmax - vmin) * i / 9
            color = colors.rgb2hex(colormap(norm(value)))
            gradient_colors.append(color)
        
        gradient_string = ', '.join(gradient_colors)
        
        legend_html = f'''
        <div style="position: fixed; bottom: 50px; left: 50px; width: 180px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.3);">
        <p style="margin: 0 0 8px 0; font-weight: bold; text-align: center;">{metric_label}</p>
        <div style="height: 20px; background: linear-gradient(to right, {gradient_string}); 
                    border: 1px solid #ccc; margin: 5px 0;"></div>
        <div style="display: flex; justify-content: space-between; font-size: 10px; margin-top: 5px;">
            <span>{vmin:.1f}</span>
            <span>{((vmin + vmax) / 2):.1f}</span>
            <span>{vmax:.1f}</span>
        </div>
        <div style="text-align: center; font-size: 10px; color: #666; margin-top: 5px;">
            🟢 Start &nbsp;&nbsp;&nbsp; 🔴 Ende
        </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
    
    def _add_start_end_markers(self, m, latitudes, longitudes):
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
    
    def is_valid(self):
        """Prüft ob die Daten gültig sind"""
        return not self.df.empty


# Backward compatibility - alte Funktionen für bestehenden Code
def read_fit_file(file):
    """Wrapper-Funktion für Backward Compatibility"""
    analyzer = FitFileAnalyzer(file)
    return analyzer.df

def calculate_workout_duration_hours(df):
    """Wrapper-Funktion für Backward Compatibility"""
    if 'time_seconds' not in df or df.empty:
        return 0
    return (df['time_seconds'].iloc[-1] - df['time_seconds'].iloc[0]) / 3600
