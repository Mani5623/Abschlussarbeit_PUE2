import json
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.signal import find_peaks

class EKGdata:

    def __init__(self, ekg_dict, max_puls=220):  ### NEU: max_puls übergeben
        self.id = ekg_dict["id"]
        self.date = ekg_dict["date"]
        self.data_path = ekg_dict["result_link"]
        self.df = pd.read_csv(self.data_path, sep='\t', header=None, names=['Messwerte in mV', 'Zeit in ms'])
        self.peaks = None

        time = self.df["Zeit in ms"].values
        sampling_interval = np.median(np.diff(time))
        self.sampling_rate = 1000 / sampling_interval

        self.max_puls = max_puls  ### NEU: Maximalpuls als Attribut speichern

    def plot_time_series(self):
        fig = px.line(self.df.head(2000), x="Zeit in ms", y="Messwerte in mV", title="EKG Zeitreihe")
        return fig

    def find_peaks(self, max_puls=None, height=None):
        if max_puls is None:
            max_puls = self.max_puls  ### NEU: Standard ist self.max_puls

        signal = self.df["Messwerte in mV"]
        time = self.df["Zeit in ms"]
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
    max_puls = person_data[0].get("max_puls", 220)  ### NEU: Maximalpuls holen
    ekg = EKGdata(ekg_dict, max_puls=max_puls)

    print("EKG-Daten:")
    print(ekg.df.head())

    ekg.find_peaks()
    print("Herzfrequenz (geschätzt):", ekg.estimate_hr(), "bpm")