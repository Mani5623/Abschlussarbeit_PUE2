import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.signal import find_peaks
from typing import Dict, List, Optional, Tuple

class EKGAnalyzer:
    """Zentrale Klasse für EKG-Analysen und Visualisierungen"""
    
    def __init__(self):
        self.current_ekg = None
        self.comparison_ekgs = []
    
    def create_dashboard_metrics(self, ekg: 'EKGdata', person_max_hr: int) -> Dict:
        """Erstellt alle Metriken für das Dashboard"""
        ekg.find_peaks(max_puls=person_max_hr)
        instant_hr = ekg.get_instant_hr()
        
        return {
            'avg_hr': ekg.estimate_hr(),
            'max_hr': instant_hr.max() if len(instant_hr) > 0 else 0,
            'min_hr': instant_hr.min() if len(instant_hr) > 0 else 0,
            'hrv': ekg.hr_variability(),
            'rr_avg': ekg.rr_interval_avg(),
            'irregularities': ekg.detect_irregularities(),
            'heart_rate_zone': ekg.get_heart_rate_zone(person_max_hr)
        }
    
    def create_comparison_table(self, ekg_list: List['EKGdata'], max_hr_list: List[int]) -> pd.DataFrame:
        """Erstellt Vergleichstabelle für mehrere EKGs"""
        comparison_data = []
        
        for i, (ekg, max_hr) in enumerate(zip(ekg_list, max_hr_list)):
            metrics = self.create_dashboard_metrics(ekg, max_hr)
            comparison_data.append({
                'Test': f"Test {i+1}",
                'Datum': ekg.date,
                'Avg HR (bpm)': metrics['avg_hr'],
                'Max HR (bpm)': f"{metrics['max_hr']:.1f}",
                'Min HR (bpm)': f"{metrics['min_hr']:.1f}",
                'HRV (ms)': metrics['hrv'],
                'Zone': metrics['heart_rate_zone']
            })
        
        return pd.DataFrame(comparison_data)
    
    def create_trend_plot(self, person_tests: List['EKGdata'], max_hr: int) -> go.Figure:
        """Erstellt Trend-Plot über mehrere Tests"""
        dates = []
        hr_values = []
        hrv_values = []
        
        for test in person_tests:
            test.find_peaks(max_puls=max_hr)
            dates.append(test.date)
            hr_values.append(test.estimate_hr())
            hrv_values.append(test.hr_variability())
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=hr_values, mode='lines+markers',
            name='Herzfrequenz (bpm)', line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=dates, y=hrv_values, mode='lines+markers',
            name='HRV (ms)', yaxis='y2', line=dict(color='blue')
        ))
        
        fig.update_layout(
            title="Herzfrequenz und HRV Trend",
            xaxis_title="Datum",
            yaxis=dict(title="Herzfrequenz (bpm)", side="left"),
            yaxis2=dict(title="HRV (ms)", overlaying="y", side="right"),
            height=400
        )
        
        return fig

class EKGdata:
    def __init__(self, ekg_dict, max_puls=220):
        self.id = ekg_dict["id"]
        self.date = ekg_dict["date"]
        self.data_path = ekg_dict["result_link"]
        self.df = pd.read_csv(self.data_path, sep='\t', header=None, names=['Messwerte in mV', 'Zeit in ms'])
        self.peaks = None
        self.max_puls = max_puls
        
        # Sampling Rate berechnen
        time = self.df["Zeit in ms"].values
        sampling_interval = np.median(np.diff(time))
        self.sampling_rate = 1000 / sampling_interval
    
    @classmethod
    def from_uploaded_data(cls, df: pd.DataFrame, max_puls: int = 220):
        """Erstellt EKGdata-Objekt aus hochgeladenen Daten"""
        instance = cls.__new__(cls)
        instance.id = "uploaded"
        instance.date = "heute"
        instance.data_path = None
        instance.df = df
        instance.peaks = None
        instance.max_puls = max_puls
        
        # Sampling Rate berechnen
        time = df["Zeit in ms"].values
        sampling_interval = np.median(np.diff(time))
        instance.sampling_rate = 1000 / sampling_interval
        
        return instance
    
    def get_heart_rate_zone(self, max_hr: int) -> str:
        """Bestimmt Herzfrequenz-Zone"""
        current_hr = self.estimate_hr()
        
        if current_hr < 0.6 * max_hr:
            return "Ruhe"
        elif current_hr < 0.7 * max_hr:
            return "Fettverbrennung"
        elif current_hr < 0.8 * max_hr:
            return "Aerob"
        elif current_hr < 0.9 * max_hr:
            return "Anaerob"
        else:
            return "Maximum"
    
    def create_interactive_plot(self, plot_type: str = "EKG + Herzfrequenz", 
                              time_window_min: float = 0.2) -> go.Figure:
        """Erstellt interaktiven Plot basierend auf Typ"""
        if self.peaks is None:
            self.find_peaks()
        
        df = self.df
        zeit_min = df["Zeit in ms"] / 60000
        instant_hr = self.get_instant_hr()
        
        fig = go.Figure()
        
        # EKG Signal hinzufügen
        if plot_type in ["EKG + Herzfrequenz", "Nur EKG"]:
            fig.add_trace(go.Scatter(
                x=zeit_min,
                y=df["Messwerte in mV"],
                mode='lines',
                name='EKG Signal',
                line=dict(color='blue', width=1)
            ))
            
            # Peaks hinzufügen
            peaks_df = df[df["Peak"] == 1]
            fig.add_trace(go.Scatter(
                x=peaks_df["Zeit in ms"] / 60000,
                y=peaks_df["Messwerte in mV"],
                mode='markers',
                name='R-Peaks',
                marker=dict(color='red', size=6)
            ))
        
        # Herzfrequenz hinzufügen
        if plot_type in ["EKG + Herzfrequenz", "Nur Herzfrequenz"] and len(instant_hr) > 0:
            peak_times_ms = df.loc[df["Peak"] == 1, "Zeit in ms"].values
            hr_times_min = (peak_times_ms[:-1] + np.diff(peak_times_ms) / 2) / 60000
            fig.add_trace(go.Scatter(
                x=hr_times_min,
                y=instant_hr,
                mode='lines+markers',
                name='Herzfrequenz',
                yaxis='y2',
                line=dict(color='green', width=2)
            ))
        
        # Layout konfigurieren
        layout_config = self._get_plot_layout(plot_type, zeit_min, time_window_min)
        fig.update_layout(layout_config)
        
        return fig
    
    def _get_plot_layout(self, plot_type: str, zeit_min: pd.Series, 
                        time_window_min: float) -> Dict:
        """Konfiguriert Plot-Layout"""
        layout = {
            'title': f'EKG Analyse - {plot_type}',
            'xaxis_title': 'Zeit (Minuten)',
            'height': 600,
            'xaxis': {
                'range': [zeit_min.min(), zeit_min.min() + time_window_min],
                'rangeslider': {'visible': True}
            }
        }
        
        if plot_type == "Nur Herzfrequenz":
            layout['yaxis'] = {'title': 'Herzfrequenz (bpm)'}
        else:
            layout['yaxis'] = {'title': 'EKG Amplitude (mV)', 'side': 'left'}
        
        if plot_type in ["EKG + Herzfrequenz", "Nur Herzfrequenz"]:
            layout['yaxis2'] = {
                'title': 'Herzfrequenz (bpm)',
                'overlaying': 'y',
                'side': 'right'
            }
        
        return layout
    
    def plot_time_series(self):
        fig = px.line(self.df.head(2000), x="Zeit in ms", y="Messwerte in mV", title="EKG Zeitreihe")
        return fig
    
    def find_peaks(self, max_puls=None, height=None):
        if max_puls is None:
            max_puls = self.max_puls
        
        signal = self.df["Messwerte in mV"]
        sampling_interval = 1000 / self.sampling_rate
        min_distance_ms = 60000 / max_puls
        distance_samples = int(min_distance_ms / sampling_interval)
        
        if height is None:
            height = np.percentile(signal, 90)
        
        peaks, _ = find_peaks(signal, distance=distance_samples, height=height)
        
        self.peaks = peaks
        self.df["Peak"] = 0
        self.df.loc[peaks, "Peak"] = 1
        
        return peaks
    
    def estimate_hr(self):
        if self.peaks is None:
            self.find_peaks()
        
        time = self.df["Zeit in ms"].values
        peak_times = time[self.peaks] / 1000
        rr_intervals = np.diff(peak_times)
        
        if len(rr_intervals) == 0:
            return 0
        
        avg_rr = np.mean(rr_intervals)
        heart_rate = 60 / avg_rr
        return round(heart_rate)
    
    def get_instant_hr(self):
        if self.peaks is None:
            self.find_peaks()
        
        time = self.df["Zeit in ms"].values
        peak_times = time[self.peaks] / 1000
        rr_intervals = np.diff(peak_times)
        
        if len(rr_intervals) == 0:
            return np.array([])
        
        instant_hr = 60 / rr_intervals
        return instant_hr
    
    def plot_with_peaks(self, window_ms=5000):
        if self.peaks is None:
            self.find_peaks()
        
        df_plot = self.df
        fig = px.line(df_plot, x="Zeit in ms", y="Messwerte in mV", title="EKG mit Peaks")
        peak_points = df_plot[df_plot["Peak"] == 1]
        fig.add_scatter(x=peak_points["Zeit in ms"], y=peak_points["Messwerte in mV"],
                        mode="markers", name="Peaks")
        
        start_time = df_plot["Zeit in ms"].iloc[0]
        end_time = start_time + window_ms
        fig.update_layout(
            xaxis=dict(
                range=[start_time, end_time],
                rangeslider=dict(visible=False),
                type="linear"
            )
        )
        return fig
    
    def min_hr(self):
        instant_hr = self.get_instant_hr()
        if len(instant_hr) == 0:
            return 0
        return round(np.min(instant_hr))
    
    def hr_variability(self):
        if self.peaks is None:
            self.find_peaks()
        
        time = self.df["Zeit in ms"].values
        peak_times = time[self.peaks]
        rr_intervals = np.diff(peak_times)
        if len(rr_intervals) == 0:
            return 0
        return round(np.std(rr_intervals), 2)
    
    def rr_interval_avg(self):
        if self.peaks is None:
            self.find_peaks()
        
        time = self.df["Zeit in ms"].values
        peak_times = time[self.peaks]
        rr_intervals = np.diff(peak_times)
        if len(rr_intervals) == 0:
            return 0
        return round(np.mean(rr_intervals), 2)
    
    def pp_interval_avg(self):
        return self.rr_interval_avg()
    
    def detect_irregularities(self, tolerance=0.1):
        if self.peaks is None:
            self.find_peaks()
        
        time = self.df["Zeit in ms"].values
        peak_times = time[self.peaks]
        rr_intervals = np.diff(peak_times)
        
        if len(rr_intervals) < 2:
            return {"irregular_rr": False, "irregular_pp": False}
        
        avg_rr = np.mean(rr_intervals)
        deviations = np.abs(rr_intervals - avg_rr) / avg_rr
        irregular = deviations > tolerance
        
        return {
            "irregular_rr": np.any(irregular),
            "irregular_pp": np.any(irregular)
        }
    
    def qrs_analysis(self):
        if self.peaks is None:
            self.find_peaks()
        
        time = self.df["Zeit in ms"].values
        peak_times = time[self.peaks]
        rr_intervals = np.diff(peak_times)
        
        if len(rr_intervals) == 0:
            return {
                "message": "Keine QRS-Analyse möglich: zu wenige Peaks erkannt",
                "rr_avg_ms": None
            }
        
        rr_avg = np.mean(rr_intervals)
        
        return {
            "rr_avg_ms": round(rr_avg, 2),
            "rr_std_ms": round(np.std(rr_intervals), 2),
            "message": "Basis-QRS-Analyse durchgeführt (nur RR-Statistik)"
        }

if __name__ == "__main__":
    print("This is a module with some functions to read the EKG data")
    with open("data/person_db.json") as file:
        person_data = json.load(file)
    ekg_dict = person_data[0]["ekg_tests"][0]
    max_puls = person_data[0].get("max_puls", 220)
    ekg = EKGdata(ekg_dict, max_puls=max_puls)
    
    print("EKG-Daten:")
    print(ekg.df.head())
    
    ekg.find_peaks()
    print("Herzfrequenz (geschätzt):", ekg.estimate_hr(), "bpm")
