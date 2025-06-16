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
import read_fit_file

DEFAULT_IMAGE_PATH = "data/pictures/none.jpg"

# Tabs als Registerkarten oben
tab1, tab2, tab3, tab4 = st.tabs(["👤 Versuchsperson", "🫀 EKG-Daten", "🚴 Leistungstest", "🏋️ Fit File"])

with tab1:
    # Personenauswahl
    person_names = read_data.get_person_list()
    selected_name = st.selectbox("Name der Versuchsperson", options=person_names, key="tab1_select")
    person_obj = Person.load_by_name(selected_name)

    st.header("Versuchsperson auswählen")
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
        st.write("Geburtsjahr", person_obj.date_of_birth)
    else:
        st.warning("Keine Person ausgewählt oder Person nicht gefunden.")

with tab2:
    # Personenauswahl
    person_names = read_data.get_person_list()
    selected_name = st.selectbox("Name der Versuchsperson", options=person_names, key="tab2_select")
    person_obj = Person.load_by_name(selected_name)

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
        instant_hr = ekg.get_instant_hr() 

        max_instant_hr = instant_hr.max() if len(instant_hr) > 0 else 0
        age = person_obj.calc_age()

        # Anzeige der Infos
        st.write("Personen-ID:", person_obj.id)
        st.write(f"Alter: {age} Jahre")
        st.write(f"EKG-ID: {ekg.id}")
        st.write(f"Geschätzte Herzfrequenz (durchschnittlich): {estimated_hr} bpm")
        st.write(f"Geschätzter Maximalpuls: {max_hr} bpm")
        st.write(f"Maximale Herzfrequenz in EKG: {max_instant_hr:.1f} bpm")

        df = ekg.df

        # EKG-Signal mit Peaks plotten (in Minuten)
        fig_ekg = go.Figure()
        fig_ekg.add_trace(go.Scatter(x=df["Zeit in ms"]/60000, y=df["Messwerte in mV"], mode='lines', name='EKG Signal'))
        peaks_df = df[df["Peak"] == 1]
        fig_ekg.add_trace(go.Scatter(x=peaks_df["Zeit in ms"]/60000, y=peaks_df["Messwerte in mV"], mode='markers', name='Peaks'))

        start = df["Zeit in ms"].min() / 60000
        fig_ekg.update_layout(
            title="EKG mit Peaks",
            xaxis=dict(
                range=[start, start + 5000/60000],  # 5000ms = ~0.083 Minuten
                rangeslider=dict(visible=True),
                type="linear",
                autorange=False
            ),
            yaxis_title="Messwerte in mV",
            xaxis_title="Zeit in Minuten",  # Geändert von ms zu Minuten
            height=400
        )
        st.plotly_chart(fig_ekg, use_container_width=True)

        # --- Neuer Plot: Instantane Herzfrequenz über Zeit (in Minuten) ---

        # beat-to-beat Instant HR aus EKGdata-Klasse holen
        instant_hr = ekg.get_instant_hr()

        if len(instant_hr) == 0:
            st.write("Keine Herzfrequenz-Daten verfügbar.")
        else:
            peak_times = df.loc[df["Peak"] == 1, "Zeit in ms"].values
            hr_times = (peak_times[:-1] + np.diff(peak_times) / 2) / 60000  # In Minuten umrechnen

            hr_df = pd.DataFrame({"Zeit in Minuten": hr_times, "Herzfrequenz (bpm)": instant_hr})

            fig_hr = go.Figure()
            fig_hr.add_trace(go.Scatter(x=hr_df["Zeit in Minuten"], y=hr_df["Herzfrequenz (bpm)"], mode="lines+markers", name="Instant HR"))
            fig_hr.update_layout(
                title="Instantane Herzfrequenz (beat-to-beat) über die Zeit",
                xaxis_title="Zeit in Minuten",  # Geändert von ms zu Minuten
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
        if max_hr_input <= 0:
            st.warning("Bitte geben Sie eine gültige maximale Herzfrequenz ein.")
        else:
            try:
                df = read_pandas.read_my_csv()
                required_columns = ['HeartRate', 'PowerOriginal']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    st.error(f"Fehlende Spalten in der CSV-Datei: {missing_columns}")
                else:
                    zones = read_pandas.get_zone_limit(max_hr_input)
                    df['Zone'] = df['HeartRate'].apply(lambda x: read_pandas.assign_zone(x, zones))
                    fig = read_pandas.make_plot(df, zones)
                    st.plotly_chart(fig, use_container_width=True)
                    zone_counts = df['Zone'].value_counts().sort_index()
                    zone_minutes = zone_counts / 60
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("🕒 Verweildauer in Herzfrequenzzonen")
                        for zone, minutes in zone_minutes.items():
                            st.write(f"**{zone}**: {minutes:.1f} Minuten")
                    with col2:
                        st.subheader("⚡ Durchschnittliche Leistung je Zone")
                        avg_power_per_zone = df.groupby('Zone')['PowerOriginal'].mean()
                        for zone, avg_power in avg_power_per_zone.items():
                            st.write(f"**{zone}**: {avg_power:.1f} Watt")
            except FileNotFoundError:
                st.error("CSV-Datei nicht gefunden. Überprüfen Sie den Pfad 'data/activities/activity.csv'")
            except Exception as e:
                st.error(f"Fehler beim Verarbeiten der Daten: {e}")
with tab4:
    st.header("🏋️ Fit File Analyse")
    # Hier KEINE Personenauswahl!
    uploaded_fit_file = st.file_uploader("Lade ein FIT-File hoch", type=["fit"])
    sportarten = ["Radfahren", "Laufen", "Schwimmen", "Sonstiges"]
    selected_sport = st.selectbox("Sportart auswählen", options=sportarten)
    if uploaded_fit_file is not None:
        fit_df = read_fit_file.read_fit_file(uploaded_fit_file)

        # Basisdaten
        hr = np.array(fit_df['heart_rate']) if 'heart_rate' in fit_df else np.array([])
        distance = np.array(fit_df['distance']) if 'distance' in fit_df else None
        power = np.array(fit_df['power']) if 'power' in fit_df else None
        altitude = np.array(fit_df['altitude']) if 'altitude' in fit_df else None

        if len(hr) > 0:
            st.write(f"Durchschnittliche Herzfrequenz: {np.mean(hr):.2f} bpm")
            st.write(f"Maximale Herzfrequenz: {np.max(hr)} bpm")
        else:
            st.info("Keine Herzfrequenzdaten im FIT-File gefunden.")

        # Sportartspezifische Auswertung
        if selected_sport == "Radfahren":
            if power is not None and len(power) > 0:
                st.write(f"Durchschnittliche Leistung: {np.mean(power):.2f} W")
                st.write(f"Maximale Leistung: {np.max(power):.2f} W")
            else:
                st.info("Keine Leistungsdaten im FIT-File.")
            if distance is not None and len(distance) > 0:
                st.write(f"Gefahrene Distanz: {np.max(distance)/1000:.2f} km")
        elif selected_sport == "Laufen":
            if distance is not None and len(distance) > 0:
                st.write(f"Gelaufene Distanz: {np.max(distance)/1000:.2f} km")
            if altitude is not None and len(altitude) > 0:
                st.write(f"Maximale Höhe: {np.max(altitude):.2f} m")
        elif selected_sport == "Schwimmen":
            if distance is not None and len(distance) > 0:
                st.write(f"Geschwommene Distanz: {np.max(distance):.2f} m")
            else:
                st.info("Keine Distanzdaten im FIT-File.")
        else:
            st.write("Allgemeine Auswertung:")
            if distance is not None and len(distance) > 0:
                st.write(f"Distanz: {np.max(distance):.2f} m")
            else:
                st.info("Keine Distanzdaten im FIT-File.")

        # Plot Herzfrequenzverlauf (optional)
        if len(hr) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=hr, mode="lines", name="Herzfrequenz"))
            fig.update_layout(title="Herzfrequenzverlauf", xaxis_title="Zeit (Index)", yaxis_title="Herzfrequenz (bpm)")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Bitte lade zuerst ein FIT-File hoch.")
