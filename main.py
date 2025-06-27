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
import os
from datetime import datetime, date
import io
import os


DEFAULT_IMAGE_PATH = "data/pictures/none.jpg"

# Session State initialisieren
if 'show_add_form' not in st.session_state:
    st.session_state.show_add_form = False

# Erweiterte Tabs mit neuem "Person hinzufügen" Tab
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["👤 Versuchsperson", "🫀 EKG-Daten", "🚴 Leistungstest", "🏋️ Fit File", "➕ Person hinzufügen", "📤 Daten zuordnen & hochladen"])

# Tab 1: Versuchsperson auswählen (ohne "Person hinzufügen" Sektion)
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
        st.write("Geburtsdatum:", person_obj.date_of_birth.year)
        st.write("Alter:", person_obj.calc_age(), "Jahre")
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
            st.write(f"Durchschnittliche Herzfrequenz: {estimated_hr:.1f} bpm")
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
            try:
                processed, info = nk.ecg_process(ekg.df["Messwerte in mV"].values, sampling_rate=ekg.sampling_rate)
                rpeaks = info["ECG_R_Peaks"]
                hrv_time = nk.hrv_time(rpeaks, sampling_rate=ekg.sampling_rate, show=False)
                hrv_freq = nk.hrv_frequency(rpeaks, sampling_rate=ekg.sampling_rate, show=False)

                interpretations = interpret_hrv_with_values(hrv_time.iloc[0].to_dict(), hrv_freq.iloc[0].to_dict())
                st.subheader("📝 Interpretation der HRV-Werte")
                for text in interpretations:
                    st.write(text)


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
                        name='Instant Herzfrequenz (bpm)',
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
    st.header("🏋️ FIT-Datei Analyse")

    person_names = read_data.get_person_list()
    selected_person_name = st.selectbox("Wähle eine Person", options=person_names, key="tab4_select")

    if selected_person_name:
        person_obj = Person.load_by_name(selected_person_name)

        if person_obj and person_obj.fit_files:
            fit_filenames = [f["filename"] for f in person_obj.fit_files]
            selected_fit_file = st.selectbox("Wähle eine FIT-Datei", options=fit_filenames, key="tab4_fitfile_select")

            if selected_fit_file:
                fit_path = os.path.join("data", "uploads", selected_fit_file)

                try:
                    with open(fit_path, "rb") as f:
                        uploaded_fit_file = io.BytesIO(f.read())

                    # Optional: Sportart wählen
                    selected_sport = st.selectbox("Sportart", ["Radfahren", "Laufen", "Schwimmen", "Sonstiges"], key="tab4_sportart")

                    # FIT-Datei analysieren
                    from read_fit_file import FitFileAnalyzer
                    analyzer = FitFileAnalyzer(uploaded_fit_file)

                    if not analyzer.is_valid():
                        st.error("Die FIT-Datei enthält keine verwertbaren Daten.")
                    else:

                        # Workout-Übersicht
                        st.subheader("📊 Workout-Übersicht")
            
                        col1, col2, col3, col4 = st.columns(4)
            
                        with col1:
                            st.metric("⏱️ Dauer", analyzer.format_duration())
            
                        with col2:
                            if 'distance' in analyzer.df.columns and not analyzer.df['distance'].isna().all():
                                dist = analyzer.df['distance'].max() / 1000
                                st.metric("📏 Distanz", f"{dist:.2f} km")
                            else:
                                st.metric("📏 Distanz", "N/A")
            
                        with col3:
                            hr_stats = analyzer.get_heart_rate_stats()
                            if hr_stats:
                                st.metric("❤️ Ø Puls", f"{hr_stats['avg']:.0f} bpm", f"Max: {hr_stats['max']:.0f}")
                            else:
                                st.metric("❤️ Puls", "N/A")
            
                        with col4:
                            if 'calories' in analyzer.df.columns and not analyzer.df['calories'].isna().all():
                                calories = analyzer.df['calories'].max()
                                st.metric("🔥 Kalorien", f"{calories:.0f} kcal")
                            else:
                                st.metric("🔥 Kalorien", "N/A")

                        st.divider()

                        # Sportartspezifische Dashboards
                        if selected_sport == "Radfahren":
                            st.subheader("🚴 Radfahren-Dashboard")
                            
                            # Hauptmetriken für Radfahren
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                if 'speed' in analyzer.df.columns and not analyzer.df['speed'].isna().all():
                                    avg_speed = analyzer.df['speed'].mean() * 3.6
                                    max_speed = analyzer.df['speed'].max() * 3.6
                                    st.metric("⚡ Geschwindigkeit", f"{avg_speed:.1f} km/h", f"Max: {max_speed:.1f}")
                                else:
                                    st.metric("⚡ Geschwindigkeit", "N/A")
                            
                            with col2:
                                if 'power' in analyzer.df.columns and not analyzer.df['power'].isna().all():
                                    avg_power = analyzer.df['power'].mean()
                                    max_power = analyzer.df['power'].max()
                                    st.metric("🔋 Leistung", f"{avg_power:.0f} W", f"Max: {max_power:.0f}")
                                else:
                                    st.metric("🔋 Leistung", "N/A")
                            
                            with col3:
                                if 'cadence' in analyzer.df.columns and not analyzer.df['cadence'].isna().all():
                                    avg_cad = analyzer.df['cadence'].mean()
                                    st.metric("🔄 Kadenz", f"{avg_cad:.0f} rpm")
                                else:
                                    st.metric("🔄 Kadenz", "N/A")
                            
                            with col4:
                                if selected_sport in ["Radfahren", "Laufen"]:
                                    elevation = analyzer.get_elevation_gain()
                                    if elevation:
                                        st.metric("⛰️ Höhenmeter", f"{elevation:.0f} m")
                                    else:
                                        st.metric("⛰️ Höhenmeter", "N/A")

                        elif selected_sport == "Laufen":
                            st.subheader("🏃 Laufen-Dashboard")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                if 'speed' in analyzer.df.columns and not analyzer.df['speed'].isna().all():
                                    avg_speed = analyzer.df['speed'].mean() * 3.6
                                    st.metric("⚡ Geschwindigkeit", f"{avg_speed:.1f} km/h")
                                else:
                                    st.metric("⚡ Geschwindigkeit", "N/A")
                            
                            with col2:
                                if 'distance' in analyzer.df.columns and not analyzer.df['distance'].isna().all():
                                    dist = analyzer.df['distance'].max() / 1000
                                    if dist > 0 and analyzer.duration_hours > 0:
                                        pace = (analyzer.duration_hours * 60) / dist
                                        pace_min = int(pace)
                                        pace_sec = int((pace - pace_min) * 60)
                                        st.metric("⏱️ Pace", f"{pace_min}:{pace_sec:02d} min/km")
                                    else:
                                        st.metric("⏱️ Pace", "N/A")
                                else:
                                    st.metric("⏱️ Pace", "N/A")
                            
                            with col3:
                                if 'cadence' in analyzer.df.columns and not analyzer.df['cadence'].isna().all():
                                    avg_cad = analyzer.df['cadence'].mean()
                                    st.metric("👟 Schrittfrequenz", f"{avg_cad:.0f} spm")
                                else:
                                    st.metric("👟 Schrittfrequenz", "N/A")
                            
                            with col4:
                                elevation = analyzer.get_elevation_gain()
                                if elevation:
                                    st.metric("⛰️ Höhenmeter", f"{elevation:.0f} m")
                                else:
                                    st.metric("⛰️ Höhenmeter", "N/A")

                        elif selected_sport == "Schwimmen":
                            st.subheader("🏊 Schwimmen-Dashboard")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                if 'distance' in analyzer.df.columns and not analyzer.df['distance'].isna().all():
                                    dist = analyzer.df['distance'].max()
                                    st.metric("🏊 Distanz", f"{dist:.0f} m")
                                else:
                                    st.metric("🏊 Distanz", "N/A")
                            
                            with col2:
                                if 'distance' in analyzer.df.columns and not analyzer.df['distance'].isna().all():
                                    dist = analyzer.df['distance'].max()
                                    if dist > 0 and analyzer.duration_hours > 0:
                                        pace = (analyzer.duration_hours * 60) / (dist / 100)
                                        st.metric("⏱️ Pace", f"{pace:.2f} min/100m")
                                    else:
                                        st.metric("⏱️ Pace", "N/A")
                                else:
                                    st.metric("⏱️ Pace", "N/A")
                            
                            with col3:
                                if 'total_strokes' in analyzer.df.columns and not analyzer.df['total_strokes'].isna().all():
                                    avg_strokes = analyzer.df['total_strokes'].mean()
                                    st.metric("💦 Züge", f"{avg_strokes:.1f}")
                                else:
                                    st.metric("💦 Züge", "N/A")
                            
                            with col4:
                                if 'swolf' in analyzer.df.columns and not analyzer.df['swolf'].isna().all():
                                    avg_swolf = analyzer.df['swolf'].mean()
                                    st.metric("🔢 SWOLF", f"{avg_swolf:.1f}")
                                else:
                                    st.metric("🔢 SWOLF", "N/A")

                        # Herzfrequenz-Zonen Analyse
                        if hr_stats:
                            st.subheader("❤️ Herzfrequenz-Zonen")
                            
                            # Beispiel für 30-jährige Person - könnte aus Personendaten kommen
                            max_hr_theoretical = 220 - 30
                            
                            zones = {
                                "Zone 1 (Regeneration)": (0.5 * max_hr_theoretical, 0.6 * max_hr_theoretical, "#4CAF50"),
                                "Zone 2 (Grundlagenausdauer)": (0.6 * max_hr_theoretical, 0.7 * max_hr_theoretical, "#FFEB3B"),
                                "Zone 3 (Aerobe Schwelle)": (0.7 * max_hr_theoretical, 0.8 * max_hr_theoretical, "#FF9800"),
                                "Zone 4 (Anaerobe Schwelle)": (0.8 * max_hr_theoretical, 0.9 * max_hr_theoretical, "#F44336"),
                                "Zone 5 (Neuromuskuläre Leistung)": (0.9 * max_hr_theoretical, max_hr_theoretical, "#9C27B0")
                            }
                            
                            for zone_name, (min_hr, max_hr, color) in zones.items():
                                if 'heart_rate' in analyzer.df.columns:
                                    time_in_zone = len(analyzer.df[
                                        (analyzer.df['heart_rate'] >= min_hr) & 
                                        (analyzer.df['heart_rate'] <= max_hr)
                                    ]) / len(analyzer.df) * 100
                                    
                                    st.progress(time_in_zone / 100, text=f"{zone_name}: {time_in_zone:.1f}%")

                        st.divider()

                        # Detaillierte Statistiken in Expandern
                        with st.expander("📈 Detaillierte Statistiken", expanded=False):
                            sport_stats = analyzer.get_sport_statistics(selected_sport)
                            
                            # Zeige Statistiken in Spalten
                            if sport_stats:
                                cols = st.columns(2)
                                for i, stat in enumerate(sport_stats):
                                    with cols[i % 2]:
                                        st.write(stat)

                        # Running Dynamics für Laufen
                        if selected_sport == "Laufen":
                            with st.expander("🏃 Running Dynamics", expanded=False):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    if 'vertical_oscillation' in analyzer.df.columns and not analyzer.df['vertical_oscillation'].isna().all():
                                        vo = analyzer.df['vertical_oscillation'].mean()
                                        st.metric("📈 Vertikale Oszillation", f"{vo:.2f} mm")
                                
                                with col2:
                                    if 'ground_contact_time' in analyzer.df.columns and not analyzer.df['ground_contact_time'].isna().all():
                                        gct = analyzer.df['ground_contact_time'].mean()
                                        st.metric("📈 Bodenkontaktzeit", f"{gct:.2f} ms")
                                
                                with col3:
                                    if 'stride_length' in analyzer.df.columns and not analyzer.df['stride_length'].isna().all():
                                        sl = analyzer.df['stride_length'].mean()
                                        st.metric("📈 Schrittlänge", f"{sl:.2f} m")

                        # Plots in Spalten
                        st.subheader("📊 Verlaufsdiagramme")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_hr = analyzer.create_heart_rate_plot()
                            if fig_hr:
                                st.plotly_chart(fig_hr, use_container_width=True)
                            else:
                                st.info("Keine Herzfrequenzdaten verfügbar")

                        with col2:
                            fig_alt = analyzer.create_altitude_plot()
                            if fig_alt:
                                st.plotly_chart(fig_alt, use_container_width=True)
                            else:
                                st.info("Keine Höhendaten verfügbar")

                        # GPS-Karte nur für Outdoor-Sportarten
                        if selected_sport in ["Radfahren", "Laufen"]:
                            st.subheader("📍 GPS-Route")

                            # Farbauswahl mit "Keine Farbe" als Standard
                            color_options = ["Keine Farbe"] + list(analyzer.available_metrics.keys())
                            
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                selected_option = st.selectbox(
                                    "Farbkodierung nach:",
                                    options=color_options,
                                    format_func=lambda x: x if x == "Keine Farbe" else analyzer.available_metrics[x],
                                    key="color_metric",
                                    index=0  # "Keine Farbe" ist Standard
                                )
                            
                            with col2:
                                if selected_option != "Keine Farbe":
                                    metric_label = analyzer.available_metrics[selected_option]
                                    st.info(f"🎨 Route eingefärbt nach: **{metric_label}**")
                                else:
                                    st.info("🔵 Route wird in einfacher blauer Farbe angezeigt")
                            
                            # Karte erstellen
                            if selected_option != "Keine Farbe":
                                with st.spinner("Farbkodierte Karte wird erstellt..."):
                                    m = analyzer.create_gps_map(selected_option)
                            else:
                                with st.spinner("Karte wird erstellt..."):
                                    m = analyzer.create_gps_map()  # Ohne color_metric = einfache Karte
                            
                            if m:
                                from streamlit_folium import st_folium
                                st_folium(m, width=700, height=500)
                            else:
                                st.warning("Keine GPS-Daten gefunden.")

                        # Workout-Bewertung
                        st.subheader("🏆 Workout-Zusammenfassung")
                        
                        # Berechne eine einfache Bewertung
                        score = 0
                        factors = []
                        
                        if 'power' in analyzer.df.columns and not analyzer.df['power'].isna().all():
                            power_score = min(analyzer.df['power'].mean() / 200 * 25, 25)
                            score += power_score
                            factors.append(f"Leistung: {power_score:.0f}/25")
                        
                        if hr_stats:
                            hr_score = min(hr_stats['avg'] / 150 * 25, 25)
                            score += hr_score
                            factors.append(f"Herzfrequenz: {hr_score:.0f}/25")
                        
                        if 'distance' in analyzer.df.columns and not analyzer.df['distance'].isna().all():
                            distance_score = min(analyzer.df['distance'].max() / 50000 * 25, 25)
                            score += distance_score
                            factors.append(f"Distanz: {distance_score:.0f}/25")
                        
                        duration_score = min(analyzer.duration_hours * 25, 25)
                        score += duration_score
                        factors.append(f"Dauer: {duration_score:.0f}/25")

                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("🎯 Workout-Score", f"{score:.0f}/100")
                        
                        with col2:
                            if score >= 90:
                                grade = "A+"
                                grade_color = "🟢"
                            elif score >= 80:
                                grade = "A"
                                grade_color = "🟢"
                            elif score >= 70:
                                grade = "B"
                                grade_color = "🟡"
                            elif score >= 60:
                                grade = "C"
                                grade_color = "🟠"
                            else:
                                grade = "D"
                                grade_color = "🔴"
                            
                            st.metric("📝 Bewertung", f"{grade_color} {grade}")
                        
                        with col3:
                            st.metric("⏱️ Trainingszeit", analyzer.format_duration())
                        
                        # Bewertungsdetails
                        with st.expander("🔍 Bewertungsdetails"):
                            for factor in factors:
                                st.write(f"• {factor}")
                        
                
                except FileNotFoundError:
                    st.error("FIT-Datei nicht gefunden im Upload-Ordner.")
            else:
                st.warning("Keine passende FIT-Datei gefunden.")
        else:
            st.info("Diese Person hat noch keine FIT-Dateien hochgeladen.")
    else:
        st.info("Bitte wähle eine Person aus.")

        
# Tab 5: Neue Person hinzufügen
with tab5:
    st.header("➕ Neue Person hinzufügen")
    st.write("Hier können Sie eine neue Versuchsperson zur Datenbank hinzufügen.")
    
    with st.form("add_person_form", clear_on_submit=True):
        st.subheader("Persönliche Daten")
        
        # Layout in zwei Spalten
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Grunddaten**")
            firstname = st.text_input("Vorname*", 
                                    placeholder="z.B. Max",
                                    help="Bitte geben Sie den Vornamen ein")
            
            lastname = st.text_input("Nachname*", 
                                   placeholder="z.B. Mustermann",
                                   help="Bitte geben Sie den Nachnamen ein")
            
            gender = st.selectbox("Geschlecht*", 
                                options=["male", "female"], 
                                format_func=lambda x: "👨 Männlich" if x == "male" else "👩 Weiblich",
                                help="Wählen Sie das Geschlecht aus")
        
        with col2:
            st.markdown("**Geburtsdatum & Bild**")
            
            # Kalender für Geburtsdatum
            birth_date = st.date_input(
                "Geburtsdatum*",
                value=date(2000, 1, 1),
                min_value=date(1900, 1, 1),
                max_value=date.today(),
                help="Wählen Sie das Geburtsdatum aus dem Kalender"
            )
            
            # Bildupload
            uploaded_file = st.file_uploader(
                "Profilbild (optional)", 
                type=['png', 'jpg', 'jpeg'],
                help="Laden Sie ein Profilbild hoch (PNG, JPG oder JPEG)"
            )
            
            # Vorschau des hochgeladenen Bildes
            if uploaded_file is not None:
                try:
                    preview_image = Image.open(uploaded_file)
                    st.image(preview_image, caption="Bildvorschau", width=200)
                except Exception as e:
                    st.error(f"Fehler beim Anzeigen der Bildvorschau: {e}")
        
        # Zusätzliche Informationen
        st.markdown("---")
        st.markdown("**Zusätzliche Informationen**")
        
        # Berechne Alter basierend auf ausgewähltem Datum
        if birth_date:
            calculated_age = date.today().year - birth_date.year
            if date.today() < date(date.today().year, birth_date.month, birth_date.day):
                calculated_age -= 1
            st.info(f"📅 Berechnetes Alter: {calculated_age} Jahre")
        
        # Submit-Buttons
        st.markdown("---")
        col_submit, col_reset = st.columns([1, 1])
        
        with col_submit:
            submitted = st.form_submit_button(
                "👤 Person speichern", 
                type="primary",
                use_container_width=True
            )
        
        with col_reset:
            reset_form = st.form_submit_button(
                "🔄 Formular zurücksetzen",
                use_container_width=True
            )
        
        # Formular-Verarbeitung
        if submitted:
            # Validierung
            if not firstname or not lastname:
                st.error("❌ Bitte füllen Sie alle Pflichtfelder (*) aus.")
            elif not birth_date:
                st.error("❌ Bitte wählen Sie ein gültiges Geburtsdatum aus.")
            else:
                try:
                    # Prüfe auf Duplikate
                    if Person.person_exists(firstname, lastname):
                        st.error(f"❌ Person {firstname} {lastname} existiert bereits in der Datenbank!")
                    else:
                        # Neue Person erstellen
                        success = Person.add_new_person(
                            firstname=firstname,
                            lastname=lastname,
                            birth_date=birth_date,  # Übergebe das date-Objekt
                            gender=gender,
                            uploaded_file=uploaded_file
                        )
                        
                        if success:
                            st.success(f"✅ Person {firstname} {lastname} wurde erfolgreich hinzugefügt!")
                            st.balloons()  # Feier-Animation
                            
                            # Zeige Zusammenfassung
                            with st.expander("📋 Zusammenfassung der hinzugefügten Person"):
                                st.write(f"**Name:** {firstname} {lastname}")
                                st.write(f"**Geschlecht:** {'Männlich' if gender == 'male' else 'Weiblich'}")
                                st.write(f"**Geburtsdatum:** {birth_date.strftime('%d.%m.%Y')}")
                                st.write(f"**Alter:** {calculated_age} Jahre")
                                if uploaded_file:
                                    st.write("**Profilbild:** ✅ Hochgeladen")
                                else:
                                    st.write("**Profilbild:** ❌ Nicht vorhanden")
                        else:
                            st.error("❌ Fehler beim Speichern der Person. Bitte versuchen Sie es erneut.")
                
                except Exception as e:
                    st.error(f"❌ Unerwarteter Fehler: {str(e)}")
        
        if reset_form:
            st.info("🔄 Formular wurde zurückgesetzt.")

with tab6:
    st.header("📤 Daten zuordnen & verwalten")

    all_person_names = read_data.get_person_list()
    selected_name_upload = st.selectbox("Wähle eine Person", all_person_names, key="upload_select")

    if selected_name_upload:
        selected_person = Person.load_by_name(selected_name_upload)
        st.subheader(f"Daten für: {selected_person.firstname} {selected_person.lastname}")

        # --- Upload ---
        st.markdown("### 🆕 Datei hochladen")
        uploaded_ekg = st.file_uploader("EKG-Datei (CSV)", type=["csv"], key="upload_ekg_file")
        uploaded_fit = st.file_uploader("FIT-Datei (.fit)", type=["fit"], key="upload_fit_file")

        if st.button("✅ Hochladen"):
            results = []

            if uploaded_ekg:
                success_ekg = selected_person.add_uploaded_file(uploaded_ekg, "ekg")
                results.append(("EKG", success_ekg))
            if uploaded_fit:
                success_fit = selected_person.add_uploaded_file(uploaded_fit, "fit")
                results.append(("FIT", success_fit))

            if not uploaded_ekg and not uploaded_fit:
                st.warning("Bitte mindestens eine Datei auswählen.")
            else:
                for filetype, success in results:
                    if success:
                        st.success(f"{filetype}-Datei erfolgreich hochgeladen.")
                    else:
                        st.error(f"{filetype}-Datei konnte nicht gespeichert werden.")

        # --- Bestehende Dateien löschen ---
        st.markdown("### 🗑️ Hochgeladene FIT-Dateien löschen")
        fit_files = selected_person.fit_files
        if fit_files:
            fit_filenames = [f["filename"] for f in fit_files]
            selected_file_to_delete = st.selectbox("Wähle eine FIT-Datei zum Löschen", fit_filenames)

            if st.button("🗑️ Datei löschen"):
                success = selected_person.remove_file(selected_file_to_delete, "fit")
                if success:
                    st.success(f"{selected_file_to_delete} wurde gelöscht.")
                    st.rerun()  # 🆕 streamlit.rerun statt deprecated experimental_rerun
                else:
                    st.error("Löschen fehlgeschlagen.")
        else:
            st.info("Keine FIT-Dateien vorhanden.")