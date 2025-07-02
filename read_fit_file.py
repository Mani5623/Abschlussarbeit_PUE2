from fitparse import FitFile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import folium
import matplotlib.cm as cm
import matplotlib.colors as colors
from datetime import timedelta
import os
import json
from geopy.distance import geodesic

class FitFileAnalyzer:
    SEMICIRCLE_TO_DEGREE = 180 / 2**31
    AVAILABLE_METRICS = {
        'altitude': 'Höhenmeter',
        'heart_rate': 'Herzfrequenz', 
        'speed': 'Geschwindigkeit',
        'power': 'Leistung'
    }

    def __init__(self, fit_file):
        self.fit_file = fit_file
        self.filename = getattr(fit_file, 'name', None)
        self.sportart = self._load_sportart()
        self.df = None
        self.duration_hours = 0
        self.available_metrics = {}
        self._load_data()

    def _load_sportart(self):
        if not self.filename or not self.filename.endswith(".fit"):
            return "Unbekannt"

        meta_path = self.filename.replace(".fit", ".json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    return json.load(f).get("sportart", "Unbekannt")
            except json.JSONDecodeError:
                return "Unbekannt"
        return "Unbekannt"

    def _load_data(self):
        fitfile = FitFile(self.fit_file)
        all_records = []

        for record in fitfile.get_messages('record'):
            data = {field.name: field.value for field in record}
            all_records.append(data)

        if not all_records:
            self.df = pd.DataFrame()
            return

        self.df = pd.DataFrame(all_records)

        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            start_time = self.df['timestamp'].iloc[0]
            self.df['time_seconds'] = (self.df['timestamp'] - start_time).dt.total_seconds()

        self._calculate_duration()
        self._add_gps_based_metrics()
        self._get_available_metrics()

    def _calculate_duration(self):
        if 'time_seconds' not in self.df or self.df.empty:
            self.duration_hours = 0
            return
        self.duration_hours = (self.df['time_seconds'].iloc[-1] - self.df['time_seconds'].iloc[0]) / 3600

    def _get_available_metrics(self):
        self.available_metrics = {
            metric: label for metric, label in self.AVAILABLE_METRICS.items()
            if metric in self.df.columns and not self.df[metric].isna().all()
        }
        if 'gps_speed' in self.df.columns and not self.df['gps_speed'].isna().all():
            self.available_metrics['gps_speed'] = 'GPS-Geschwindigkeit'

    def _add_gps_based_metrics(self):
        lat, lon, mask = self._get_lat_lon_optimized()
        if lat is None or len(lat) < 2:
            return

        if 'speed' not in self.df.columns or self.df['speed'].isna().all():
            print("⚠️ Geschwindigkeit wird aus GPS berechnet.")
            coords = list(zip(lat.values, lon.values))
            distances = [0] + [geodesic(coords[i], coords[i+1]).meters for i in range(len(coords)-1)]
            cum_dist = np.cumsum(distances)
            self.df.loc[mask, 'gps_distance'] = cum_dist

            if 'time_seconds' in self.df.columns:
                times = self.df.loc[mask, 'time_seconds'].values
                delta_t = np.diff(times)
                delta_s = np.diff(cum_dist)
                gps_speed = np.append([0], delta_s / np.maximum(delta_t, 1e-3))
                self.df.loc[mask, 'gps_speed'] = gps_speed
                self.df['gps_speed'] = self.df['gps_speed'].fillna(method='ffill').fillna(method='bfill')

        if ('altitude' not in self.df.columns or self.df['altitude'].isna().all()) and 'enhanced_altitude' in self.df.columns:
            print("⚠️ Höhenmeter werden aus 'enhanced_altitude' berechnet.")
            self.df['altitude'] = self.df['enhanced_altitude']

    def get_elevation_gain(self):
        if 'altitude' not in self.df.columns or self.df['altitude'].dropna().empty:
            return None

        gain = self.df['altitude'].fillna(method='ffill').diff().clip(lower=0).sum()
        return round(gain)

    def get_heart_rate_stats(self):
        if 'heart_rate' not in self.df.columns or self.df['heart_rate'].isna().all():
            return None

        return {
            'avg': self.df['heart_rate'].mean(),
            'max': self.df['heart_rate'].max()
        }

    def create_heart_rate_plot(self):
        return self._create_time_plot('heart_rate', 'Herzfrequenzverlauf', 'Herzfrequenz (bpm)')

    def create_altitude_plot(self):
        return self._create_time_plot('altitude', 'Höhenmeterverlauf', 'Höhe (m)')

    def _create_time_plot(self, column, title, y_label):
        if column not in self.df.columns or self.df[column].isna().all():
            return None

        time_data = self.df['time_seconds'] if 'time_seconds' in self.df.columns else np.arange(len(self.df))

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

    def _get_lat_lon_optimized(self):
        lat, lon = None, None

        if 'enhanced_position_lat' in self.df.columns and 'enhanced_position_long' in self.df.columns:
            lat = self.df['enhanced_position_lat']
            lon = self.df['enhanced_position_long']
        elif 'position_lat' in self.df.columns and 'position_long' in self.df.columns:
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

    def is_valid(self):
        return not self.df.empty