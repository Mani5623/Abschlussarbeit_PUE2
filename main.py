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

DEFAULT_IMAGE_PATH = "data/pictures/none.jpg"

# Session State initialisieren
if 'show_add_form' not in st.session_state:
    st.session_state.show_add_form = False

# Erweiterte Tabs mit neuem "Person hinzufügen" Tab
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["👤 Versuchsperson", "🫀 EKG-Daten", "🚴 Leistungstest", "🏋️ Fit File", "➕ Person hinzufügen", "📤 Daten hochladen"])

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


import os

with tab4:
    st.header("🏋️ Fit File Analyse")

    # Personenauswahl für FIT-Dateien
    person_names = read_data.get_person_list()
    selected_person_name = st.selectbox("👤 Wähle eine Person", person_names, key="tab4_person_select")
    selected_person = Person.load_by_name(selected_person_name) if selected_person_name else None

    if selected_person and selected_person.fit_files:
        file_options = {
            (entry.get("original_name") or entry["filename"]): entry["filename"]
            for entry in selected_person.fit_files
        }

        selected_label = st.selectbox("📂 Wähle eine FIT-Datei", options=list(file_options.keys()))
        selected_filename = file_options[selected_label]
        selected_fit_entry = next(
            (entry for entry in selected_person.fit_files if entry["filename"] == selected_filename), None
        )

        if selected_fit_entry and st.button("📊 Analyse starten"):
            file_path = os.path.join("data/uploads", selected_fit_entry["filename"])
            try:
                with open(file_path, "rb") as f:
                    from read_fit_file import FitFileAnalyzer
                    analyzer = FitFileAnalyzer(f)

                if not analyzer.is_valid():
                    st.error("Keine Daten in der FIT-Datei gefunden.")
                else:
                    st.write(f"⏱️ **Workout-Dauer:** {analyzer.format_duration()}")

                    sportarten = ["Radfahren", "Laufen", "Schwimmen", "Sonstiges"]
                    selected_sport = st.selectbox("Sportart auswählen", options=sportarten)

                    stats = analyzer.get_sport_statistics(selected_sport)
                    for label, data in stats.items():
                        st.write(f"📊 **{label}:** {data['value']:.2f} {data['unit']}")

                        if data['metric'] == 'distance':
                            speed_metric = analyzer.calculate_speed_metrics(
                                selected_sport, data['value'], data['unit']
                            )
                            if speed_metric:
                                icon = "🚴" if selected_sport == "Radfahren" else "🏊" if selected_sport == "Schwimmen" else "🏃"
                                st.write(f"{icon} **{speed_metric['label']}:** {speed_metric['value']} {speed_metric['unit']}")

                    hr_stats = analyzer.get_heart_rate_stats()
                    if hr_stats:
                        st.write(f"❤️ **Durchschnittspuls:** {hr_stats['avg']:.0f} bpm (Max: {hr_stats['max']:.0f} bpm)")

                    elevation = analyzer.get_elevation_gain()
                    if elevation:
                        st.write(f"⛰️ **Höhenmeter bergauf:** {elevation:.0f} m")

                    col1, col2 = st.columns(2)

                    with col1:
                        fig_hr = analyzer.create_heart_rate_plot()
                        if fig_hr:
                            st.plotly_chart(fig_hr, use_container_width=True)

                    with col2:
                        fig_alt = analyzer.create_altitude_plot()
                        if fig_alt:
                            st.plotly_chart(fig_alt, use_container_width=True)

                    st.subheader("📍 GPS-Route")
                    color_options = ["Keine Farbe"] + list(analyzer.available_metrics.keys())

                    col1, col2 = st.columns([1, 2])

                    with col1:
                        selected_option = st.selectbox(
                            "Farbkodierung nach:",
                            options=color_options,
                            format_func=lambda x: x if x == "Keine Farbe" else analyzer.available_metrics[x],
                            key="color_metric",
                            index=0
                        )

                    with col2:
                        if selected_option != "Keine Farbe":
                            metric_label = analyzer.available_metrics[selected_option]
                            st.info(f"🎨 Route eingefärbt nach: **{metric_label}**")
                        else:
                            st.info("🔵 Route wird in einfacher blauer Farbe angezeigt")

                    if selected_option != "Keine Farbe":
                        with st.spinner("Farbkodierte Karte wird erstellt..."):
                            m = analyzer.create_gps_map(selected_option)
                    else:
                        with st.spinner("Karte wird erstellt..."):
                            m = analyzer.create_gps_map()

                    if m:
                        from streamlit_folium import st_folium
                        st_folium(m, width=700, height=500)
                    else:
                        st.warning("Keine GPS-Daten gefunden.")
            except Exception as e:
                st.error(f"Fehler beim Laden/Analysieren der FIT-Datei: {e}")
    else:
        st.info("Bitte wähle eine Person mit FIT-Dateien.")


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
    st.header("📤 Daten zuordnen & hochladen")

    all_person_names = read_data.get_person_list()
    selected_name_upload = st.selectbox("Wähle eine Person", all_person_names, key="upload_select")

    if selected_name_upload:
        selected_person = Person.load_by_name(selected_name_upload)

        st.subheader(f"Daten für: {selected_person.firstname} {selected_person.lastname}")

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
