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

    def __init__(self, fit_file):
        """Initialisiert den Analyzer mit einer FIT-Datei"""
        self.fit_file = fit_file
        self.filename = getattr(fit_file, 'name', None)  # Name der Datei, falls verfügbar
        self.sportart = self._load_sportart()            # NEU: Automatisch Sportart laden
        self.df = None
        self.duration_hours = 0
        self.available_metrics = {}
        self._load_data()

    def _load_sportart(self):
        """Lädt die Sportart aus einer zugehörigen JSON-Datei, falls verfügbar"""
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
    
    def get_cycling_statistics(self):
        """Radfahren-spezifische Statistiken"""
        stats = []
        
        # Distanz
        if 'distance' in self.df.columns and not self.df['distance'].isna().all():
            dist = self.df['distance'].max() / 1000
            stats.append(f"🚴 **Distanz:** {dist:.2f} km")
        
        # Geschwindigkeit
        if 'speed' in self.df.columns and not self.df['speed'].isna().all():
            avg_speed = self.df['speed'].mean() * 3.6
            max_speed = self.df['speed'].max() * 3.6
            stats.append(f"⚡ **Durchschnittsgeschwindigkeit:** {avg_speed:.2f} km/h")
            stats.append(f"🏁 **Max. Geschwindigkeit:** {max_speed:.2f} km/h")
        
        # Leistung
        if 'power' in self.df.columns and not self.df['power'].isna().all():
            avg_power = self.df['power'].mean()
            max_power = self.df['power'].max()
            stats.append(f"🔋 **Durchschnittliche Leistung:** {avg_power:.0f} W")
            stats.append(f"🔋 **Maximale Leistung:** {max_power:.0f} W")
        
        # Kadenz (Trittfrequenz)
        if 'cadence' in self.df.columns and not self.df['cadence'].isna().all():
            avg_cad = self.df['cadence'].mean()
            stats.append(f"🔄 **Durchschnittliche Kadenz:** {avg_cad:.0f} rpm")
        
        # Kalorien
        if 'calories' in self.df.columns and not self.df['calories'].isna().all():
            calories = self.df['calories'].max()
            stats.append(f"🔥 **Kalorien:** {calories:.0f} kcal")
        
        return stats
    
    def get_running_statistics(self):
        """Laufen-spezifische Statistiken"""
        stats = []
        
        # Distanz und Pace
        if 'distance' in self.df.columns and not self.df['distance'].isna().all():
            dist = self.df['distance'].max() / 1000
            stats.append(f"🏃 **Distanz:** {dist:.2f} km")
            
            # Pace berechnen
            if dist > 0 and self.duration_hours > 0:
                pace = (self.duration_hours * 60) / dist
                pace_min = int(pace)
                pace_sec = int((pace - pace_min) * 60)
                stats.append(f"⏱️ **Pace:** {pace_min}:{pace_sec:02d} min/km")
        
        # Geschwindigkeit
        if 'speed' in self.df.columns and not self.df['speed'].isna().all():
            avg_speed = self.df['speed'].mean() * 3.6
            max_speed = self.df['speed'].max() * 3.6
            stats.append(f"⚡ **Durchschnittsgeschwindigkeit:** {avg_speed:.2f} km/h")
            stats.append(f"🏁 **Max. Geschwindigkeit:** {max_speed:.2f} km/h")
        
        # Schrittfrequenz
        if 'cadence' in self.df.columns and not self.df['cadence'].isna().all():
            avg_cad = self.df['cadence'].mean()
            stats.append(f"👟 **Durchschnittliche Schrittfrequenz:** {avg_cad:.0f} spm")
        
        # Running Power
        if 'power' in self.df.columns and not self.df['power'].isna().all():
            avg_power = self.df['power'].mean()
            stats.append(f"⚡ **Running Power:** {avg_power:.0f} W")
        
        # Running Dynamics
        if 'vertical_oscillation' in self.df.columns and not self.df['vertical_oscillation'].isna().all():
            vo = self.df['vertical_oscillation'].mean()
            stats.append(f"📈 **Vertikale Oszillation:** {vo:.2f} mm")
        
        if 'ground_contact_time' in self.df.columns and not self.df['ground_contact_time'].isna().all():
            gct = self.df['ground_contact_time'].mean()
            stats.append(f"📈 **Bodenkontaktzeit:** {gct:.2f} ms")
        
        if 'stride_length' in self.df.columns and not self.df['stride_length'].isna().all():
            sl = self.df['stride_length'].mean()
            stats.append(f"📈 **Schrittlänge:** {sl:.2f} m")
        
        # Kalorien
        if 'calories' in self.df.columns and not self.df['calories'].isna().all():
            calories = self.df['calories'].max()
            stats.append(f"🔥 **Kalorien:** {calories:.0f} kcal")
        
        return stats
    
    def get_swimming_statistics(self):
        """Schwimmen-spezifische Statistiken"""
        stats = []
        
        # Distanz und Pace
        if 'distance' in self.df.columns and not self.df['distance'].isna().all():
            dist = self.df['distance'].max()
            stats.append(f"🏊 **Distanz:** {dist:.0f} m")
            
            # Schwimm-Pace
            if dist > 0 and self.duration_hours > 0:
                pace = (self.duration_hours * 60) / (dist / 100)
                stats.append(f"⏱️ **Pace:** {pace:.2f} min/100m")
        
        # Schwimmzüge
        if 'total_strokes' in self.df.columns and not self.df['total_strokes'].isna().all():
            avg_strokes = self.df['total_strokes'].mean()
            stats.append(f"💦 **Durchschnittliche Züge:** {avg_strokes:.1f}")
        
        # SWOLF (Schwimmeffizienz)
        if 'swolf' in self.df.columns and not self.df['swolf'].isna().all():
            avg_swolf = self.df['swolf'].mean()
            stats.append(f"🔢 **Durchschnittlicher SWOLF:** {avg_swolf:.1f}")
        
        # Schwimmstil
        if 'swim_stroke' in self.df.columns and not self.df['swim_stroke'].isna().all():
            unique_strokes = self.df['swim_stroke'].unique()
            stroke_names = {
                0: "Freistil",
                1: "Rückenschwimmen", 
                2: "Brustschwimmen",
                3: "Schmetterling",
                4: "Drill",
                5: "Mixed"
            }
            strokes = [stroke_names.get(s, f"Stil {s}") for s in unique_strokes if not pd.isna(s)]
            if strokes:
                stats.append(f"🏊‍♂️ **Schwimmstil(e):** {', '.join(strokes)}")
        
        # Kalorien
        if 'calories' in self.df.columns and not self.df['calories'].isna().all():
            calories = self.df['calories'].max()
            stats.append(f"🔥 **Kalorien:** {calories:.0f} kcal")
        
        return stats
    
    def get_sport_statistics(self, sport_type):
        """Berechnet sportartspezifische Statistiken"""
        if sport_type == "Radfahren":
            return self.get_cycling_statistics()
        elif sport_type == "Laufen":
            return self.get_running_statistics()
        elif sport_type == "Schwimmen":
            return self.get_swimming_statistics()
        else:
            # Fallback für "Sonstiges"
            stats = []
            if 'distance' in self.df.columns and not self.df['distance'].isna().all():
                dist = self.df['distance'].max()
                stats.append(f"📊 **Distanz:** {dist:.0f} m")
            return stats
    
    def get_heart_rate_stats(self):
        """Berechnet Herzfrequenz-Statistiken"""
        if 'heart_rate' not in self.df.columns or self.df['heart_rate'].isna().all():
            return None
        
        return {
            'avg': self.df['heart_rate'].mean(),
            'max': self.df['heart_rate'].max()
        }
    
    def get_elevation_gain(self):
        """Berechnet Höhenmeter bergauf"""
        if 'altitude' not in self.df.columns or self.df['altitude'].isna().all():
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
        if column not in self.df.columns or self.df[column].isna().all():
            return None
        
        time_data = self.df['time_seconds'] if 'time_seconds' in self.df.columns else np.arange(len(self.df))
        
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


if __name__ == "__main__":
    import os
    fit_file_path = 'data/fit_file/test.fit'
    
    if os.path.exists(fit_file_path):
        with open(fit_file_path, 'rb') as f:
            analyzer = FitFileAnalyzer(f)
        
        print("✅ FitFileAnalyzer Test:")
        print(f"Dauer: {analyzer.format_duration()}")
        print(f"Verfügbare Metriken: {list(analyzer.available_metrics.keys())}")
        
        # Test sportartspezifische Statistiken
        cycling_stats = analyzer.get_cycling_statistics()
        print(f"Radfahren-Statistiken: {len(cycling_stats)} Einträge")
        
        # GPS-Plot erstellen
        m = analyzer.create_gps_map('altitude')
        if m:
            m.save('gpx_test.html')
            print("✅ GPX-Plot erstellt: gpx_test.html")
        else:
            print("❌ Kein GPX-Plot möglich")
    else:
        print(f"❌ Datei nicht gefunden: {fit_file_path}")
