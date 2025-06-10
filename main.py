import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
import read_data
import read_pandas
from PIL import Image
from person import Person
from ekgdata import EKGdata

DEFAULT_IMAGE_PATH = "data/pictures/none.jpg"

# Personennamen laden
person_names = read_data.get_person_list()

# Person auswählen (Sidebar, damit Auswahl vor Tabs erfolgt)
selected_name = st.sidebar.selectbox("Name der Versuchsperson", options=person_names)

# Person-Objekt erzeugen
person_obj = Person.load_by_name(selected_name)

# Tabs definieren
tab1, tab2, tab3 = st.tabs(["👤 Versuchsperson", "🫀 EKG-Daten", "🚴 Leistungstest"])

with tab1:
    st.header("Versuchsperson auswählen")

    # Bild und Infos
    if person_obj:
        picture_path = person_obj.picture_path or DEFAULT_IMAGE_PATH
        try:
            image = Image.open(picture_path)
            st.image(image, caption=f"{person_obj.lastname}, {person_obj.firstname}", width=250)
        except FileNotFoundError:
            st.warning("Bilddatei nicht gefunden.")
        except Exception as e:
            st.error(f"Fehler beim Laden des Bilds: {e}")

        st.write("Personen-ID:", person_obj.id)
        gender = person_obj.gender or "Unbekannt"
        st.write("Geschlecht:", gender)
    else:
        st.warning("Keine Person ausgewählt oder Person nicht gefunden.")
    
    st.write("Geburtsjahr",person_obj.date_of_birth )

with tab2:
    st.header("🫀 EKG-Datenanalyse")

    if person_obj and person_obj.ekg_tests:
        ekg_tests = person_obj.ekg_tests

        # Dropdown: Auswahl des EKG-Tests nach Datum und ID
        ekg_options = [f"ID {test.id} - {test.date}" for test in ekg_tests]
        selected_ekg_str = st.selectbox("EKG-Test auswählen", options=ekg_options)

        # Ausgewähltes EKG-Objekt
        selected_index = ekg_options.index(selected_ekg_str)
        ekg = ekg_tests[selected_index]

        # Maximalpuls aus Person (mit Default fallback)
        max_hr = person_obj.calc_max_heart_rate(gender=person_obj.gender)

        # Peaks finden & Herzfrequenz schätzen
        ekg.find_peaks(max_puls=max_hr)
        estimated_hr = ekg.estimate_hr()
        age = person_obj.calc_age()

        # Anzeige der Infos
        st.write("Personen-ID:", person_obj.id)
        st.write(f"Alter: {age} Jahre")
        st.write(f"EKG-ID: {ekg.id}")
        st.write(f"Geschätzte Herzfrequenz (durchschnittlich): {estimated_hr} bpm")
        st.write(f"Geschätzter Maximalpuls: {max_hr} bpm")

        df = ekg.df

        # EKG-Signal mit Peaks plotten (wie bisher)
        fig_ekg = go.Figure()
        fig_ekg.add_trace(go.Scatter(x=df["Zeit in ms"], y=df["Messwerte in mV"], mode='lines', name='EKG Signal'))
        peaks_df = df[df["Peak"] == 1]
        fig_ekg.add_trace(go.Scatter(x=peaks_df["Zeit in ms"], y=peaks_df["Messwerte in mV"], mode='markers', name='Peaks'))

        start = df["Zeit in ms"].min()
        fig_ekg.update_layout(
            title="EKG mit Peaks",
            xaxis=dict(
                range=[start, start + 5000],
                rangeslider=dict(visible=True),
                type="linear",
                autorange=False
            ),
            yaxis_title="Messwerte in mV",
            xaxis_title="Zeit in ms",
            height=400
        )
        st.plotly_chart(fig_ekg, use_container_width=True)

        # --- Neuer Plot: Herzfrequenz über Zeit ---

        # Berechnung Herzfrequenz über Zeit:
        peak_times = df.loc[df["Peak"] == 1, "Zeit in ms"].values
        rr_intervals_s = np.diff(peak_times) / 1000.0
        hr_over_time = 60 / rr_intervals_s
        hr_times = peak_times[:-1] + np.diff(peak_times) / 2

        hr_df = pd.DataFrame({"Zeit in ms": hr_times, "Herzfrequenz (bpm)": hr_over_time})

        fig_hr = go.Figure()
        fig_hr.add_trace(go.Scatter(x=hr_df["Zeit in ms"], y=hr_df["Herzfrequenz (bpm)"], mode="lines+markers", name="HR über Zeit"))
        fig_hr.update_layout(
            title="Herzfrequenzverlauf über die Zeit",
            xaxis_title="Zeit in ms",
            yaxis_title="Herzfrequenz (bpm)",
            height=400
        )
        st.plotly_chart(fig_hr, use_container_width=True)

    else:
        st.info("Keine EKG-Daten für diese Person vorhanden.")

with tab3:
    st.header("🚴 Leistungstest-Auswertung")

    max_hr_input = st.number_input("Manuelle Eingabe: Max. Herzfrequenz (für Zonenanalyse)", min_value=0, max_value=250, step=1)

    if st.button("Absenden"):
        df = read_pandas.read_my_csv()
        zones = read_pandas.get_zone_limit(max_hr_input)

        # Zonen zuweisen
        df['Zone'] = df['HeartRate'].apply(lambda x: read_pandas.assign_zone(x, zones))

        fig = read_pandas.make_plot(df, zones)
        st.plotly_chart(fig)

        zone_counts = df['Zone'].value_counts().sort_index()
        zone_minutes = zone_counts / 60  # Sekunden → Minuten

        st.subheader("🕒 Verweildauer in Herzfrequenzzonen")
        for zone, minutes in zone_minutes.items():
            st.write(f"{zone}: {minutes:.1f} Minuten")

        avg_power_per_zone = df.groupby('Zone')['PowerOriginal'].mean()

        st.subheader("⚡ Durchschnittliche Leistung je Zone")
        for zone, avg_power in avg_power_per_zone.items():
            st.write(f"{zone}: {avg_power:.1f} Watt")