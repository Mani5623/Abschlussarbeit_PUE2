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
import neurokit2 as nk

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
    st.header("🫀 EKG-Datenanalyse")

    # Upload eigener EKG-Daten
    uploaded_file = st.file_uploader(
        "Oder eigene EKG-Daten hochladen (CSV, Spalten: 'Messwerte in mV', 'Zeit in ms')",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            # CSV einlesen
            df_uploaded = pd.read_csv(uploaded_file, sep=None, engine='python')

            # Spalten prüfen
            if not {'Messwerte in mV', 'Zeit in ms'}.issubset(df_uploaded.columns):
                st.error("Die CSV muss die Spalten 'Messwerte in mV' und 'Zeit in ms' enthalten.")
            else:
                # Sampling-Rate bestimmen
                time = df_uploaded["Zeit in ms"].values
                sampling_interval = np.median(np.diff(time))
                sampling_rate = 1000 / sampling_interval

                # EKGdata-Objekt für Upload erzeugen (ohne Konstruktor)
                ekg = EKGdata.__new__(EKGdata)
                ekg.df = df_uploaded
                ekg.sampling_rate = sampling_rate
                ekg.peaks = None
                ekg.max_puls = 220  # Default Max-Puls, kann man anpassen

                # Peaks finden, HR berechnen
                ekg.find_peaks()
                est_hr = ekg.estimate_hr()
                instant_hr = ekg.get_instant_hr()

                st.write(f"Geschätzte Herzfrequenz: {est_hr} bpm")

                # Plot mit Peaks
                fig = ekg.plot_with_peaks()
                st.plotly_chart(fig, use_container_width=True)

                # NeuroKit2 HRV Analyse
                import neurokit2 as nk
                try:
                    processed, info = nk.ecg_process(
                        ekg.df["Messwerte in mV"].values,
                        sampling_rate=ekg.sampling_rate
                    )
                    rpeaks = info["ECG_R_Peaks"]
                    hrv_time = nk.hrv_time(rpeaks, sampling_rate=ekg.sampling_rate, show=False)
                    hrv_freq = nk.hrv_frequency(rpeaks, sampling_rate=ekg.sampling_rate, show=False)

                    st.subheader("HRV - Zeitbereich")
                    st.write(hrv_time)

                    st.subheader("HRV - Frequenzbereich")
                    st.write(hrv_freq)

                except Exception as e:
                    st.warning(f"NeuroKit2 Analyse konnte nicht durchgeführt werden: {e}")

        except Exception as e:
            st.error(f"Fehler beim Einlesen der Datei: {e}")

    else:
        # Auswahl gespeicherter Personen und EKG-Tests

        person_names = read_data.get_person_list()
        selected_name = st.selectbox("Name der Versuchsperson", options=person_names, key="tab2_select")
        person_obj = Person.load_by_name(selected_name)

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
            st.write(f"Geschätzte Herzfrequenz (durchschnittlich): {estimated_hr:.1f} bpm")
            st.write(f"Geschätzter Maximalpuls: {max_hr} bpm")
            st.write(f"Maximale Herzfrequenz in EKG: {max_instant_hr:.1f} bpm")
            st.write(f"Minimale Herzfrequenz in EKG: {min_instant_hr:.1f} bpm")
            st.write(f"Herzfrequenz-Variabilität (SDNN): {hr_variability_ms} ms")

            # Interpretation mit Werten
            def interpret_hrv_with_values(hrv_time_dict, hrv_freq_dict):
                interpretations = []

                sdnn = hrv_time_dict.get('HRV_SDNN', 0)
                if sdnn > 50:
                    interpretations.append(f"✅ SDNN ({sdnn:.1f} ms) ist hoch – gute Gesamt-HRV, gesundes autonomes Nervensystem.")
                elif 30 <= sdnn <= 50:
                    interpretations.append(f"⚠️ SDNN ({sdnn:.1f} ms) ist mittel – HRV ist moderat, evtl. leichte Belastung vorhanden.")
                else:
                    interpretations.append(f"❌ SDNN ({sdnn:.1f} ms) ist niedrig – mögliche Belastung, Stress oder Überlastung.")

                rmssd = hrv_time_dict.get('HRV_RMSSD', 0)
                if rmssd > 40:
                    interpretations.append(f"✅ RMSSD ({rmssd:.1f} ms) ist hoch – gute parasympathische Aktivität, gute Erholung.")
                elif 20 <= rmssd <= 40:
                    interpretations.append(f"⚠️ RMSSD ({rmssd:.1f} ms) ist mittel – moderate Erholung, evtl. leichte Belastung.")
                else:
                    interpretations.append(f"❌ RMSSD ({rmssd:.1f} ms) ist niedrig – geringe Erholung, möglicher Stress.")

                pnn50 = hrv_time_dict.get('HRV_pNN50', 0)
                if pnn50 > 10:
                    interpretations.append(f"✅ pNN50 ({pnn50:.1f}%) ist hoch – gutes Erholungsniveau.")
                elif 5 <= pnn50 <= 10:
                    interpretations.append(f"⚠️ pNN50 ({pnn50:.1f}%) ist mittel – moderate Erholung.")
                else:
                    interpretations.append(f"❌ pNN50 ({pnn50:.1f}%) ist niedrig – geringes Erholungsniveau.")

                lf_hf = hrv_freq_dict.get('HRV_LFHF', 0)
                if lf_hf < 2:
                    interpretations.append(f"✅ LF/HF-Verhältnis ({lf_hf:.2f}) ist ausgewogen – sympathische und parasympathische Aktivität im Gleichgewicht.")
                elif 2 <= lf_hf <= 5:
                    interpretations.append(f"⚠️ LF/HF-Verhältnis ({lf_hf:.2f}) ist leicht sympathisch dominiert – erhöhter Stresslevel möglich.")
                else:
                    interpretations.append(f"❌ LF/HF-Verhältnis ({lf_hf:.2f}) ist stark sympathisch dominiert – hoher Stress oder Aktivierung.")

                return interpretations

            # NeuroKit2 Analyse
            import neurokit2 as nk
            try:
                processed, info = nk.ecg_process(ekg.df["Messwerte in mV"].values, sampling_rate=ekg.sampling_rate)
                rpeaks = info["ECG_R_Peaks"]
                hrv_time = nk.hrv_time(rpeaks, sampling_rate=ekg.sampling_rate, show=False)
                hrv_freq = nk.hrv_frequency(rpeaks, sampling_rate=ekg.sampling_rate, show=False)

                interpretations = interpret_hrv_with_values(hrv_time.iloc[0].to_dict(), hrv_freq.iloc[0].to_dict())
                st.subheader("📝 Interpretation der HRV-Werte")
                for text in interpretations:
                    st.write(text)

                # Plot
                fig_nk = nk.ecg_plot(processed)
                st.plotly_chart(fig_nk, use_container_width=True)

            except Exception as e:
                st.warning(f"NeuroKit2 Analyse konnte nicht durchgeführt werden: {e}")

            # Plot EKG + Herzfrequenz
            df = ekg.df
            zeit_min = df["Zeit in ms"] / 60000

            plot_option = st.radio(
                "Was soll angezeigt werden?",
                options=["EKG + Herzfrequenz", "Nur EKG", "Nur Herzfrequenz"],
                index=0
            )

            fig = go.Figure()

            if plot_option in ["EKG + Herzfrequenz", "Nur EKG"]:
                fig.add_trace(go.Scatter(
                    x=zeit_min,
                    y=df["Messwerte in mV"],
                    mode='lines',
                    name='EKG Signal'
                ))

                peaks_df = df[df["Peak"] == 1]
                fig.add_trace(go.Scatter(
                    x=peaks_df["Zeit in ms"] / 60000,
                    y=peaks_df["Messwerte in mV"],
                    mode='markers',
                    name='Peaks'
                ))

            if plot_option in ["EKG + Herzfrequenz", "Nur Herzfrequenz"]:
                if len(instant_hr) > 0:
                    peak_times_ms = df.loc[df["Peak"] == 1, "Zeit in ms"].values
                    hr_times_min = (peak_times_ms[:-1] + np.diff(peak_times_ms) / 2) / 60000
                    fig.add_trace(go.Scatter(
                        x=hr_times_min,
                        y=instant_hr,
                        mode='lines+markers',
                        name='Instantane Herzfrequenz (bpm)',
                        yaxis='y2'
                    ))

            layout = dict(
                title="EKG + Herzfrequenz",
                xaxis_title="Zeit in Minuten",
                height=500,
                xaxis=dict(
                    range=[zeit_min.min(), zeit_min.min() + 0.2],
                    rangeslider=dict(visible=True)
                )
            )

            if plot_option == "Nur Herzfrequenz":
                layout["yaxis"] = dict(title="Herzfrequenz (bpm)")
            else:
                layout["yaxis"] = dict(title="Messwerte in mV", side="left")

            if plot_option in ["EKG + Herzfrequenz", "Nur Herzfrequenz"]:
                layout["yaxis2"] = dict(
                    title="Herzfrequenz (bpm)",
                    overlaying="y",
                    side="right"
                )

            fig.update_layout(layout)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Keine Person ausgewählt oder keine EKG-Daten vorhanden.")


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




