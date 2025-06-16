# Paket für Bearbeitung von Tabellen
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"


def read_my_csv():
    df = pd.read_csv("data/activities/activity.csv", sep=",", header=0)
    time = np.arange(0, len(df))
    df["Time"] = time
    return df

def get_zone_limit(max_hr):
    zone_1 = [0.5 * max_hr, 0.6 * max_hr]
    zone_2 = [0.6 * max_hr, 0.7 * max_hr]
    zone_3 = [0.7 * max_hr, 0.8 * max_hr]
    zone_4 = [0.8 * max_hr, 0.9 * max_hr]
    zone_5 = [0.9 * max_hr, 1.0 * max_hr]

    zone_dict = {
        "Zone_1": zone_1,
        "Zone_2": zone_2,
        "Zone_3": zone_3,
        "Zone_4": zone_4,
        "Zone_5": zone_5,
    }

    return zone_dict

def assign_zone(hr, zones):
    for zone, (low, high) in zones.items():
        if low <= hr < high:
            return zone
    return 'Zone_5'  # Falls hr == max_hr

def make_plot(df, zones):
    zone_colors = {
        'Zone_1': 'blue',
        'Zone_2': 'green',
        'Zone_3': 'yellow',
        'Zone_4': 'orange',
        'Zone_5': 'red'
    }

    df['Zone'] = df['HeartRate'].apply(lambda x: assign_zone(x, zones))

    fig = px.scatter(
        df, x='Time', y='HeartRate', color='Zone',
        color_discrete_map=zone_colors,
        labels={'HeartRate': 'Herzfrequenz [bpm], Power [W]', 'Time': 'Zeit [s]'},
        title='Herzfrequenz- und Leistungsanalyse'
    )

    fig.add_scatter(
        x=df['Time'], y=df['PowerOriginal'],
        mode='lines', name='Power', line=dict(color='black', width=2)
    )

    return fig

def leistungsanalyse(df, weight_kg, age, resting_hr):
    results = {}
    results['avg_hr'] = df['HeartRate'].mean()
    results['max_hr'] = df['HeartRate'].max()
    results['min_hr'] = df['HeartRate'].min()
    results['avg_power'] = df['PowerOriginal'].mean()
    results['max_power'] = df['PowerOriginal'].max()
    results['total_time_min'] = len(df) / 60

    kcal = (results['avg_hr'] * 0.6309 + weight_kg * 0.1988 + age * 0.2017 - 55.0969) * results['total_time_min'] / 4.184
    results['calories'] = kcal

    # VO2max mit linearem HR-Leistung Zusammenhang schätzen
    results['vo2max_est'] = vo2max_from_hr_power(df, weight_kg, results['max_hr'])

    return results

def vo2max_from_hr_power(df, weight_kg, max_hr):
    zones = get_zone_limit(max_hr)

    zone_data = []
    for zone, (low, high) in zones.items():
        df_zone = df[(df['HeartRate'] >= low) & (df['HeartRate'] < high)]
        if len(df_zone) == 0:
            continue
        avg_hr = df_zone['HeartRate'].mean()
        avg_power = df_zone['PowerOriginal'].mean()
        zone_data.append((avg_hr, avg_power))

    if len(zone_data) < 2:
        print("Zu wenige Datenpunkte für VO2max-Schätzung.")
        return None

    hrs, powers = zip(*zone_data)
    a, b = np.polyfit(hrs, powers, 1)  # lineare Regression

    est_max_power = a * max_hr + b

    vo2max = (est_max_power * 10.8) / weight_kg + 7

    return vo2max

#%% Test - Nur ausführen wenn das Modul direkt gestartet wird
if __name__ == "__main__":
    df = read_my_csv()
    max_hr = df['HeartRate'].max()
    zones = get_zone_limit(max_hr)

    df['Zone'] = df['HeartRate'].apply(lambda x: assign_zone(x, zones))

    # Abfrage Gewicht, Alter und Ruhepuls für Kalorien & VO2max-Berechnung
    try:
        weight = float(input("Bitte Gewicht in kg eingeben: "))
        age = int(input("Bitte Alter in Jahren eingeben: "))
        resting_hr = int(input("Bitte Ruheherzfrequenz eingeben: "))
    except ValueError:
        print("Ungültige Eingabe, Standardwerte werden verwendet (70kg, 30 Jahre, 60 bpm).")
        weight = 70
        age = 30
        resting_hr = 60

    results = leistungsanalyse(df, weight, age, resting_hr)

    print("\nErgebnisse der Leistungsanalyse:")
    print(f"Durchschnittliche Herzfrequenz: {results['avg_hr']:.1f} bpm")
    print(f"Maximale Herzfrequenz: {results['max_hr']} bpm")
    print(f"Minimale Herzfrequenz: {results['min_hr']} bpm")
    print(f"Durchschnittliche Leistung: {results['avg_power']:.1f} Watt")
    print(f"Maximale Leistung: {results['max_power']} Watt")
    print(f"Gesamtzeit: {results['total_time_min']:.1f} Minuten")
    print(f"Verbrannte Kalorien: {results['calories']:.1f} kcal")
    if results['vo2max_est'] is not None:
        print(f"Geschätzte VO2max: {results['vo2max_est']:.1f} ml/kg/min")
    else:
        print("VO2max konnte nicht geschätzt werden.")

    fig = make_plot(df, zones)
    fig.show()