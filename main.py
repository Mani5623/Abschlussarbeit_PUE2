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
from streamlit_folium import st_folium
import read_fit_file
import heartpy as hp

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

        ekg_options = [f"ID {test.id} - {test.date}" for test in ekg_tests]
        selected_ekg_str = st.selectbox("EKG-Test auswählen", options=ekg_options)
        selected_index = ekg_options.index(selected_ekg_str)
        ekg = ekg_tests[selected_index]

        max_hr = person_obj.calc_max_heart_rate(gender=person_obj.gender)
        ekg.find_peaks(max_puls=max_hr)
        estimated_hr = ekg.estimate_hr()
        instant_hr = ekg.get_instant_hr()

        max_instant_hr = instant_hr.max() if len(instant_hr) > 0 else 0
        min_instant_hr = instant_hr.min() if len(instant_hr) > 0 else 0
        hr_variability_ms = ekg.hr_variability()
        age = person_obj.calc_age()

        st.write("Personen-ID:", person_obj.id)
        st.write(f"Alter: {age} Jahre")
        st.write(f"EKG-ID: {ekg.id}")
        st.write(f"Geschätzte Herzfrequenz (durchschnittlich): {estimated_hr} bpm")
        st.write(f"Geschätzter Maximalpuls: {max_hr} bpm")
        st.write(f"Maximale Herzfrequenz in EKG: {max_instant_hr:.1f} bpm")
        st.write(f"Minimale Herzfrequenz in EKG: {min_instant_hr:.1f} bpm")
        st.write(f"Herzfrequenz-Variabilität: {hr_variability_ms} ms")

        # QRS-Komplex-Analyse mit HeartPy, stille Fehlerbehandlung
        try:
            signal = ekg.df["Messwerte in mV"].values
            wd, m = hp.process(signal, sample_rate=ekg.sampling_rate)

            rr_avg = m.get('rr_avg', None)
            if rr_avg is None:
                raise ValueError("Kein RR-Intervall von HeartPy")

            qrs_peak_count = len(wd['peaklist'])

        except Exception:
            peak_times_ms = ekg.df.loc[ekg.df["Peak"] == 1, "Zeit in ms"].values
            if len(peak_times_ms) >= 2:
                rr_intervals_s = np.diff(peak_times_ms) / 1000
                rr_avg = rr_intervals_s.mean()
                qrs_peak_count = len(peak_times_ms)
            else:
                rr_avg = None
                qrs_peak_count = 0

        if rr_avg is not None and qrs_peak_count > 0:
            st.subheader("QRS-Analyse")
            st.write(f"Anzahl der QRS-Komplexe: {qrs_peak_count}")
            st.write(f"Durchschnittliches RR-Intervall: {rr_avg:.3f} s")

            df = ekg.df
            peak_times_ms = df.loc[df["Peak"] == 1, "Zeit in ms"].values

            if len(peak_times_ms) >= 2:
                rr_intervals_s = np.diff(peak_times_ms) / 1000
                rr_avg = rr_intervals_s.mean()

                # PP-Intervalle: hier identisch mit RR, falls keine P-Peaks vorhanden
                pp_intervals_s = rr_intervals_s
                pp_avg = pp_intervals_s.mean()

                st.write(f"Durchschnittliches PP-Intervall: {pp_avg:.3f} s")

                rr_deviation = np.abs(rr_intervals_s / rr_avg - 1)
                irregular_indices = np.where(rr_deviation > 0.10)[0]

                if len(irregular_indices) > 0:
                    with st.expander(f"Unregelmäßige RR-Intervalle (>10 % Abweichung): {len(irregular_indices)} von {len(rr_intervals_s)} anzeigen"):
                        for idx in irregular_indices:
                            st.write(f"Intervall {idx} – Dauer: {rr_intervals_s[idx]:.3f} s, Abweichung: {rr_deviation[idx]*100:.1f}%")
                else:
                    st.write("Keine unregelmäßigen RR-Intervalle gefunden.")

                if len(irregular_indices) > 0:
                    with st.expander(f"Unregelmäßige PP-Intervalle (>10 % Abweichung): {len(irregular_indices)} von {len(pp_intervals_s)} anzeigen"):
                        for idx in irregular_indices:
                            st.write(f"Intervall {idx} – Dauer: {pp_intervals_s[idx]:.3f} s, Abweichung: {rr_deviation[idx]*100:.1f}%")
                else:
                    st.write("Keine unregelmäßigen PP-Intervalle gefunden.")
            else:
                st.write("Nicht genügend Peaks für RR-/PP-Analyse.")
        else:
            st.write("Nicht genügend Peaks für RR-/PP-Analyse.")

        # EKG-Plot mit Peaks

        fig_ekg = go.Figure()
        fig_ekg.add_trace(go.Scatter(x=df["Zeit in ms"] / 60000, y=df["Messwerte in mV"], mode='lines', name='EKG Signal'))
        peaks_df = df[df["Peak"] == 1]
        fig_ekg.add_trace(go.Scatter(x=peaks_df["Zeit in ms"] / 60000, y=peaks_df["Messwerte in mV"], mode='markers', name='Peaks'))

        start = df["Zeit in ms"].min() / 60000
        fig_ekg.update_layout(
            title="EKG mit Peaks",
            xaxis=dict(
                range=[start, start + 5000 / 60000],
                rangeslider=dict(visible=True),
                type="linear",
                autorange=False
            ),
            yaxis_title="Messwerte in mV",
            xaxis_title="Zeit in Minuten",
            height=400
        )
        st.plotly_chart(fig_ekg, use_container_width=True)

        # Instantane Herzfrequenz-Plot
        instant_hr = ekg.get_instant_hr()
        if len(instant_hr) == 0:
            st.write("Keine Herzfrequenz-Daten verfügbar.")
        else:
            peak_times = df.loc[df["Peak"] == 1, "Zeit in ms"].values
            hr_times = (peak_times[:-1] + np.diff(peak_times) / 2) / 60000
            hr_df = pd.DataFrame({"Zeit in Minuten": hr_times, "Herzfrequenz (bpm)": instant_hr})

            fig_hr = go.Figure()
            fig_hr.add_trace(go.Scatter(x=hr_df["Zeit in Minuten"], y=hr_df["Herzfrequenz (bpm)"], mode="lines+markers", name="Instant HR"))
            fig_hr.update_layout(
                title="Instantane Herzfrequenz (beat-to-beat) über die Zeit",
                xaxis_title="Zeit in Minuten",
                yaxis_title="Herzfrequenz (bpm)",
                height=400
            )
            st.plotly_chart(fig_hr, use_container_width=True)

    else:
        st.info("Keine EKG-Daten für diese Person vorhanden.")



with tab3:
    st.header("🚴 Leistungstest-Auswertung")

    weight = st.number_input("Gewicht (kg)", min_value=30, max_value=200, value=70)
    age = st.number_input("Alter (Jahre)", min_value=10, max_value=120, value=30)
    resting_hr = st.number_input("Ruhepuls (bpm)", min_value=30, max_value=120, value=60)
    max_hr_input = st.number_input("Maximale Herzfrequenz (bpm) für Zonenanalyse", min_value=50, max_value=220, value=180)

    if st.button("Auswertung starten"):
        try:
            df = read_pandas.read_my_csv()

            zones = read_pandas.get_zone_limit(max_hr_input)
            df['Zone'] = df['HeartRate'].apply(lambda x: read_pandas.assign_zone(x, zones))

            fig = read_pandas.make_plot(df, zones)
            st.plotly_chart(fig, use_container_width=True)

            # Leistungsanalyse mit Einzelparametern
            results = read_pandas.leistungsanalyse(df, weight, age, resting_hr)

            st.subheader("📊 Analyseergebnisse")
            st.write(f"Durchschnittliche Herzfrequenz: {results['avg_hr']:.1f} bpm")
            st.write(f"Maximale Herzfrequenz: {results['max_hr']} bpm")
            st.write(f"Minimale Herzfrequenz: {results['min_hr']} bpm")
            st.write(f"Durchschnittliche Leistung: {results['avg_power']:.1f} Watt")
            st.write(f"Maximale Leistung: {results['max_power']} Watt")
            st.write(f"Gesamtdauer: {results['total_time_min']:.1f} Minuten")
            st.write(f"Geschätzte verbrannte Kalorien: {results['calories']:.0f} kcal")

            if results['vo2max_est'] is not None:
                st.write(f"Geschätzter VO2max: {results['vo2max_est']:.1f} ml/kg/min")
            else:
                st.write("VO2max konnte nicht geschätzt werden.")

            zone_counts = df['Zone'].value_counts().sort_index()
            zone_minutes = zone_counts / 60
            st.subheader("🕒 Zeit in Herzfrequenzzonen (Minuten)")
            for zone, minutes in zone_minutes.items():
                st.write(f"{zone}: {minutes:.1f} min")

            avg_power_per_zone = df.groupby('Zone')['PowerOriginal'].mean()
            st.subheader("⚡ Durchschnittliche Leistung je Zone")
            for zone, avg_power in avg_power_per_zone.items():
                st.write(f"{zone}: {avg_power:.1f} Watt")

        except FileNotFoundError:
            st.error("Datei 'activity.csv' nicht gefunden.")
        except Exception as e:
            st.error(f"Fehler bei der Auswertung: {e}")


with tab4:
    st.header("🏋️ Fit File Analyse")

    # Session State Initialisierung
    for key, default in [
        ('fitfile_submitted', False),
        ('last_file', None),
        ('cached_df', None),
        ('cached_filename', None)
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    uploaded_fit_file = st.file_uploader("Lade ein FIT-File hoch", type=["fit"])
    sportarten = ["Radfahren", "Laufen", "Schwimmen", "Sonstiges"]
    selected_sport = st.selectbox("Sportart auswählen", options=sportarten)

    # Reset bei neuer Datei
    if uploaded_fit_file is not None and st.session_state['last_file'] != uploaded_fit_file:
        st.session_state.update({
            'fitfile_submitted': False,
            'last_file': uploaded_fit_file,
            'cached_df': None,
            'cached_filename': None
        })

    if st.button("Abschicken"):
        st.session_state['fitfile_submitted'] = True

    if uploaded_fit_file is not None and st.session_state['fitfile_submitted']:
        # Caching für bessere Performance
        current_filename = uploaded_fit_file.name
        if (st.session_state['cached_df'] is None or 
            st.session_state['cached_filename'] != current_filename):
            
            with st.spinner("FIT-Datei wird verarbeitet..."):
                df = read_fit_file.read_fit_file(uploaded_fit_file)
                st.session_state['cached_df'] = df
                st.session_state['cached_filename'] = current_filename
        else:
            df = st.session_state['cached_df']

        if df.empty:
            st.error("Keine Daten in der FIT-Datei gefunden.")
        else:
            duration_hours = read_fit_file.calculate_workout_duration_hours(df)

            # ✅ Zeit-Formatierung hinzufügen
            def format_duration(hours):
                """Formatiert Stunden in lesbares Format"""
                total_minutes = int(hours * 60)
                hours_part = total_minutes // 60
                minutes_part = total_minutes % 60
                
                if hours_part > 0:
                    return f"{hours_part}h {minutes_part}min"
                else:
                    return f"{minutes_part}min"

            # ✅ Workout-Zeit anzeigen
            st.write(f"⏱️ **Workout-Dauer:** {format_duration(duration_hours)}")

            # Sportartspezifische Auswertung mit Zeit
            sport_metrics = {
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

            # ✅ Sportart-spezifische Metriken mit Geschwindigkeitsberechnung
            metrics_found = False
            for metric, unit, label, *divisor in sport_metrics.get(selected_sport, []):
                if metric in df and not df[metric].isna().all():
                    value = df[metric].mean() if metric == 'power' else df[metric].max()
                    if divisor:
                        value /= divisor[0]
                    st.write(f"📊 **{label}:** {value:.2f} {unit}")
                    
                    # ✅ Geschwindigkeit berechnen (nur für Distanz-Metriken)
                    if metric == 'distance' and duration_hours > 0:
                        if unit == 'km':
                            speed = value / duration_hours
                            st.write(f"🚴 **Durchschnittsgeschwindigkeit:** {speed:.2f} km/h")
                        elif unit == 'm' and selected_sport == "Schwimmen":
                            # Schwimm-Pace in min/100m
                            pace_per_100m = (duration_hours * 60) / (value / 100)
                            st.write(f"🏊 **Pace:** {pace_per_100m:.2f} min/100m")
                        elif unit == 'm' and selected_sport == "Laufen":
                            # Lauf-Pace in min/km
                            distance_km = value / 1000
                            pace_per_km = (duration_hours * 60) / distance_km
                            pace_minutes = int(pace_per_km)
                            pace_seconds = int((pace_per_km - pace_minutes) * 60)
                            st.write(f"🏃 **Pace:** {pace_minutes}:{pace_seconds:02d} min/km")
                    
                    metrics_found = True

            # ✅ Zusätzliche Zeit-basierte Statistiken
            if 'heart_rate' in df and not df['heart_rate'].isna().all():
                avg_hr = df['heart_rate'].mean()
                max_hr = df['heart_rate'].max()
                st.write(f"❤️ **Durchschnittspuls:** {avg_hr:.0f} bpm (Max: {max_hr:.0f} bpm)")

            if 'altitude' in df and not df['altitude'].isna().all():
                elevation_gain = (df['altitude'].diff().clip(lower=0)).sum()
                st.write(f"⛰️ **Höhenmeter bergauf:** {elevation_gain:.0f} m")

            # Plots in Spalten
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hr = read_fit_file.plot_heart_rate(df, duration_hours)
                if fig_hr:
                    st.plotly_chart(fig_hr, use_container_width=True)

            with col2:
                fig_alt = read_fit_file.plot_altitude(df, duration_hours)
                if fig_alt:
                    st.plotly_chart(fig_alt, use_container_width=True)

            # GPS-Karte mit Loading-State
            st.subheader("📍 GPS-Route")

            # Loading-Indikator für GPS-Verarbeitung
            with st.spinner("GPS-Daten werden verarbeitet..."):
                available_metrics = read_fit_file.get_available_metrics(df)

            # Nur eine UI-Instanz nach dem Laden
            if available_metrics:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    selected_metric = st.selectbox(
                        "Farbkodierung nach:",
                        options=list(available_metrics.keys()),
                        format_func=lambda x: available_metrics[x],
                        key="color_metric",
                        index=0
                    )
                
                with col2:
                    # Conditional rendering um Duplikate zu vermeiden
                    if 'color_metric' in st.session_state:
                        metric_label = available_metrics[st.session_state['color_metric']]
                        st.info(f"Route eingefärbt nach: **{metric_label}**")
                
                # Karte nur einmal rendern
                if 'color_metric' in st.session_state:
                    with st.spinner("Karte wird erstellt..."):
                        m = read_fit_file.plot_gpx_folium_colored(df, st.session_state['color_metric'])
                else:
                    with st.spinner("Karte wird erstellt..."):
                        m = read_fit_file.plot_gpx_folium(df)
            else:
                st.warning("Keine Metriken für Farbkodierung verfügbar")
                with st.spinner("Karte wird erstellt..."):
                    m = read_fit_file.plot_gpx_folium(df)
            
            if m:
                from streamlit_folium import st_folium
                st_folium(m, width=700, height=500)
            else:
                st.warning("Keine GPS-Daten gefunden.")
                
    else:
        st.info("Bitte laden Sie ein FIT-File hoch und klicken Sie auf 'Abschicken'.")




