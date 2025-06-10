import json
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.signal import find_peaks
import plotly.io as pio
pio.renderers.default = "browser"

class EKGdata:

    def __init__(self, ekg_dict):
        self.id = ekg_dict["id"]
        self.date = ekg_dict["date"]
        self.data_path = ekg_dict["result_link"]
        self.df = pd.read_csv(self.data_path, sep='\t', header=None, names=['Messwerte in mV', 'Zeit in ms'])
        self.peaks = None

    def plot_time_series(self):
        fig = px.line(self.df.head(2000), x="Zeit in ms", y="Messwerte in mV", title="EKG Zeitreihe")
        return fig

    def find_peaks(self, max_puls, height=None):
        signal = self.df["Messwerte in mV"]
    
        time = self.df["Zeit in ms"]
        sampling_interval = np.median(np.diff(time))  # ms pro Messpunkt
        sampling_rate = 1000 / sampling_interval       # Hz
    
        # Mindestabstand in Samples basierend auf Maximalpuls
        min_distance_ms = 60000 / max_puls              # ms
        distance_samples = int(min_distance_ms / sampling_interval)
    
        if height is None:
            height = np.percentile(signal, 90)
    
        peaks, _ = find_peaks(signal, distance=distance_samples, height=height)
    
        self.peaks = peaks
        self.df["Peak"] = 0
        self.df.loc[peaks, "Peak"] = 1
    
        return peaks


    def estimate_hr(self, sampling_rate_hz=1000):
        if self.peaks is None:
            self.find_peaks()
        rr_intervals = np.diff(self.peaks) / sampling_rate_hz
        if len(rr_intervals) == 0:
            return 0
        avg_rr = np.mean(rr_intervals)
        heart_rate = 60 / avg_rr
        return round(heart_rate)

    def plot_with_peaks(self, window_ms=5000):
        if self.peaks is None:
            self.find_peaks()
        df_plot = self.df  # Alle Daten anzeigen
        fig = px.line(df_plot, x="Zeit in ms", y="Messwerte in mV", title="EKG mit Peaks")
        peak_points = df_plot[df_plot["Peak"] == 1]
        fig.add_scatter(x=peak_points["Zeit in ms"], y=peak_points["Messwerte in mV"],
                        mode="markers", name="Peaks")
    # Initial sichtbarer Bereich auf window_ms Breite setzen (Start bei erstem Zeitwert)
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




if __name__ == "__main__":
    print("This is a module with some functions to read the EKG data")
    with open("data/person_db.json") as file:
        person_data = json.load(file)
    ekg_dict = person_data[0]["ekg_tests"][0]
    ekg = EKGdata(ekg_dict)

    print("EKG-Daten:")
    print(ekg.df.head())

    ekg.find_peaks()
    print("Herzfrequenz:", ekg.estimate_hr(), "bpm")

    # Optional: Show Plot (nur für Entwicklung, nicht in Streamlit verwenden)
    # fig = ekg.plot_with_peaks()
    # fig.show()
