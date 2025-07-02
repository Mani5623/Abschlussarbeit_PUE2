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
from datetime import datetime
import json
from ekgdata import EKGdata, EKGAnalyzer


DEFAULT_IMAGE_PATH = "data/pictures/none.jpg"

# Session State initialisieren
if 'show_add_form' not in st.session_state:
    st.session_state.show_add_form = False

# Erweiterte Tabs mit neuem "Person hinzufügen" Tab
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["👤 Versuchsperson","➕ Person hinzufügen", "🚴 Leistungstest", "🫀 EKG-Daten", "🏋️ Fit File Analyse", "📤 Fit File zuordnen & hochladen"])

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

# Streamlit Konfiguration
st.set_page_config(
    page_title="EKG Analyse Dashboard",
    page_icon="🫀",
    layout="wide"
)

with tab2:
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

with tab3:
    st.header("🚴 Leistungstest-Auswertung")
    
    # Eingabebereich in ansprechenden Spalten
    st.subheader("📝 Persönliche Daten eingeben")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        weight = st.number_input(
            "⚖️ Gewicht (kg)", 
            min_value=30, 
            max_value=200, 
            value=70,
            help="Ihr Körpergewicht für die Kalorienberechnung"
        )
    
    with col2:
        age = st.number_input(
            "🎂 Alter (Jahre)", 
            min_value=10, 
            max_value=120, 
            value=30,
            help="Ihr Alter für die VO2max-Schätzung"
        )
    
    with col3:
        resting_hr = st.number_input(
            "💤 Ruhepuls (bpm)", 
            min_value=30, 
            max_value=120, 
            value=60,
            help="Ihre Herzfrequenz in Ruhe"
        )
    
    with col4:
        max_hr_input = st.number_input(
            "🔥 Max. Herzfrequenz (bpm)", 
            min_value=50, 
            max_value=220, 
            value=180,
            help="Maximale Herzfrequenz für Zonenanalyse"
        )

    # Zentrierter Start-Button
    col_button1, col_button2, col_button3 = st.columns([1, 2, 1])
    with col_button2:
        start_analysis = st.button(
            "🚀 Auswertung starten", 
            type="primary",
            use_container_width=True
        )

    if start_analysis:
        try:
            with st.spinner("📊 Daten werden analysiert..."):
                df = read_pandas.read_my_csv()

                zones = read_pandas.get_zone_limit(max_hr_input)
                df['Zone'] = df['HeartRate'].apply(lambda x: read_pandas.assign_zone(x, zones))

                # Plot mit verbessertem Titel
                fig = read_pandas.make_plot(df, zones)
                fig.update_layout(
                    title="🚴 Leistungstest-Verlauf mit Herzfrequenzzonen",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)

                # Leistungsanalyse
                results = read_pandas.leistungsanalyse(df, weight, age, resting_hr)

                # Hauptmetriken in ansprechenden Cards
                st.subheader("📊 Hauptmetriken")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "💓 Ø Herzfrequenz", 
                        f"{results['avg_hr']:.1f} bpm",
                        delta=f"Max: {results['max_hr']} bpm",
                        border=True
                    )
                
                with col2:
                    st.metric(
                        "⚡ Ø Leistung", 
                        f"{results['avg_power']:.1f} W",
                        delta=f"Max: {results['max_power']} W",
                        border=True
                    )
                
                with col3:
                    st.metric(
                        "⏱️ Gesamtdauer", 
                        f"{results['total_time_min']:.1f} min",
                        border=True
                    )
                
                with col4:
                    st.metric(
                        "🔥 Kalorien", 
                        f"{results['calories']:.0f} kcal",
                        border=True
                    )

                # VO2max in separater Metrik falls verfügbar
                if results['vo2max_est'] is not None:
                    col_vo2_1, col_vo2_2, col_vo2_3 = st.columns([1, 2, 1])
                    with col_vo2_2:
                        st.metric(
                            "🫁 Geschätzter VO2max", 
                            f"{results['vo2max_est']:.1f} ml/kg/min",
                            help="Maximale Sauerstoffaufnahme - Indikator für Ausdauerleistungsfähigkeit",
                            border=True
                        )
                else:
                    st.info("ℹ️ VO2max konnte nicht geschätzt werden - möglicherweise unzureichende Daten")

                # Detailanalyse in Expandern
                with st.expander("🎯 Herzfrequenzzonen-Analyse", expanded=True):
                    zone_counts = df['Zone'].value_counts().sort_index()
                    zone_minutes = zone_counts / 60
                    
                    st.subheader("🕒 Zeit in den verschiedenen Herzfrequenzzonen")
                    
                    # Zone-Definitionen für besseres Verständnis
                    zone_descriptions = {
                        "Zone 1": "Aktive Erholung (sehr leicht)",
                        "Zone 2": "Grundlagenausdauer (leicht)", 
                        "Zone 3": "Aerobe Schwelle (moderat)",
                        "Zone 4": "Anaerobe Schwelle (hart)",
                        "Zone 5": "Neuromuskuläre Leistung (maximal)"
                    }
                    
                    # Zeige Zonen in Spalten
                    zone_cols = st.columns(len(zone_minutes))
                    for i, (zone, minutes) in enumerate(zone_minutes.items()):
                        with zone_cols[i]:
                            description = zone_descriptions.get(zone, "")
                            st.metric(
                                f"{zone}",
                                f"{minutes:.1f} min",
                                delta=description,
                                border=True
                            )

                with st.expander("⚡ Leistungsanalyse nach Zonen", expanded=False):
                    avg_power_per_zone = df.groupby('Zone')['PowerOriginal'].mean()
                    
                    st.subheader("💪 Durchschnittliche Leistung je Herzfrequenzzone")
                    
                    # Leistung pro Zone in Spalten
                    power_cols = st.columns(len(avg_power_per_zone))
                    for i, (zone, avg_power) in enumerate(avg_power_per_zone.items()):
                        with power_cols[i]:
                            st.metric(
                                f"{zone}",
                                f"{avg_power:.1f} W",
                                border=True
                            )

                # Zusätzliche Insights
                with st.expander("📈 Weitere Erkenntnisse", expanded=False):
                    col_insight1, col_insight2 = st.columns(2)
                    
                    with col_insight1:
                        st.subheader("🎯 Trainingsempfehlungen")
                        
                        # Einfache Trainingsempfehlungen basierend auf Zonen
                        total_time = results['total_time_min']
                        zone2_time = zone_minutes.get('Zone 2', 0)
                        zone4_time = zone_minutes.get('Zone 4', 0)
                        
                        if zone2_time / total_time > 0.6:
                            st.success("✅ Gute Grundlagenausdauer-Belastung")
                        elif zone4_time / total_time > 0.3:
                            st.warning("⚠️ Intensive Belastung - achten Sie auf ausreichende Erholung")
                        else:
                            st.info("ℹ️ Ausgewogene Belastungsverteilung")
                    
                    with col_insight2:
                        st.subheader("📊 Leistungskennzahlen")
                        
                        # Berechne zusätzliche Kennzahlen
                        hr_range = results['max_hr'] - results['min_hr']
                        power_variability = (results['max_power'] - results['avg_power']) / results['avg_power'] * 100
                        
                        st.write(f"**Herzfrequenz-Spanne:** {hr_range} bpm")
                        st.write(f"**Leistungsvariabilität:** {power_variability:.1f}%")
                        st.write(f"**Kalorien pro Minute:** {results['calories']/results['total_time_min']:.1f} kcal/min")

        except FileNotFoundError:
            st.error("❌ Datei 'activity.csv' wurde nicht gefunden. Bitte stellen Sie sicher, dass die Datei im richtigen Verzeichnis liegt.")
        except Exception as e:
            st.error(f"❌ Fehler bei der Auswertung: {e}")
            st.info("💡 Tipp: Überprüfen Sie das Datenformat und stellen Sie sicher, dass alle erforderlichen Spalten vorhanden sind.")

# Analyzer initialisieren
analyzer = EKGAnalyzer()

with tab4:
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
                st.error("❌ Die CSV muss die Spalten 'Messwerte in mV' und 'Zeit in ms' enthalten.")
            else:
                st.success("✅ Datei erfolgreich hochgeladen!")
                
                # Sampling-Rate bestimmen
                time = df_uploaded["Zeit in ms"].values
                sampling_interval = np.median(np.diff(time))
                sampling_rate = 1000 / sampling_interval

                # EKGdata-Objekt für Upload erzeugen
                ekg = EKGdata.__new__(EKGdata)
                ekg.df = df_uploaded
                ekg.sampling_rate = sampling_rate
                ekg.peaks = None
                ekg.max_puls = 220

                # Peaks finden, HR berechnen
                ekg.find_peaks()
                est_hr = ekg.estimate_hr()
                instant_hr = ekg.get_instant_hr()

                # Metriken in Spalten anzeigen
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("💓 Herzfrequenz", f"{est_hr} bpm", border=True)
                
                with col2:
                    max_hr = instant_hr.max() if len(instant_hr) > 0 else 0
                    st.metric("📈 Maximum", f"{max_hr:.1f} bpm", border=True)
                
                with col3:
                    min_hr = instant_hr.min() if len(instant_hr) > 0 else 0
                    st.metric("📉 Minimum", f"{min_hr:.1f} bpm", border=True)
                
                with col4:
                    hrv = ekg.hr_variability()
                    st.metric("📊 HRV", f"{hrv} ms", border=True)

                # Plot mit Peaks
                fig = ekg.plot_with_peaks()
                fig.update_layout(
                    title="🫀 EKG-Analyse mit R-Peak Erkennung",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)

                # NeuroKit2 HRV Analyse
                try:
                    import neurokit2 as nk
                    processed, info = nk.ecg_process(
                        ekg.df["Messwerte in mV"].values,
                        sampling_rate=ekg.sampling_rate
                    )
                    rpeaks = info["ECG_R_Peaks"]
                    hrv_time = nk.hrv_time(rpeaks, sampling_rate=ekg.sampling_rate, show=False)
                    hrv_freq = nk.hrv_frequency(rpeaks, sampling_rate=ekg.sampling_rate, show=False)

                    # HRV-Analyse in Expander
                    with st.expander("🔬 Erweiterte HRV-Analyse", expanded=True):
                        col_hrv1, col_hrv2 = st.columns(2)
                        
                        with col_hrv1:
                            st.subheader("⏱️ Zeitbereich-Analyse")
                            hrv_dict = hrv_time.iloc[0].to_dict()
                            
                            # Metriken für Zeitbereich
                            st.metric("SDNN", f"{hrv_dict.get('HRV_SDNN', 0):.1f} ms", border=True)
                            st.metric("RMSSD", f"{hrv_dict.get('HRV_RMSSD', 0):.1f} ms", border=True)
                            st.metric("pNN50", f"{hrv_dict.get('HRV_pNN50', 0):.1f}%", border=True)
                        
                        with col_hrv2:
                            st.subheader("📊 Frequenzbereich-Analyse")
                            freq_dict = hrv_freq.iloc[0].to_dict()
                            
                            # Metriken für Frequenzbereich
                            st.metric("LF Power", f"{freq_dict.get('HRV_LF', 0):.1f} ms²", border=True)
                            st.metric("HF Power", f"{freq_dict.get('HRV_HF', 0):.1f} ms²", border=True)
                            st.metric("LF/HF Ratio", f"{freq_dict.get('HRV_LFHF', 0):.2f}", border=True)

                except Exception as e:
                    st.warning(f"⚠️ NeuroKit2 Analyse konnte nicht durchgeführt werden: {e}")

        except Exception as e:
            st.error(f"❌ Fehler beim Einlesen der Datei: {e}")

    else:
        # Container für gespeicherte Personen
        with st.container():
            st.subheader("👥 Gespeicherte EKG-Daten analysieren")
            
            # Auswahl gespeicherter Personen und EKG-Tests
            person_names = read_data.get_person_list()
            selected_name = st.selectbox("👤 Name der Versuchsperson", options=person_names, key="tab3_select")
            person_obj = Person.load_by_name(selected_name)

            if person_obj and person_obj.ekg_tests:
                ekg_tests = person_obj.ekg_tests

                ekg_options = [f"📅 {test.date} - ID {test.id}" for test in ekg_tests]
                selected_ekg_str = st.selectbox("🫀 EKG-Test auswählen", options=ekg_options)
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

                # Basis-Informationen in Metriken
                st.subheader("📊 Basis-Informationen")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("👤 Alter", f"{age} Jahre", border=True)
                
                with col2:
                    st.metric("💓 Ø HR", f"{estimated_hr:.1f} bpm", border=True)
                
                with col3:
                    st.metric("📈 Max HR", f"{max_instant_hr:.1f} bpm", border=True)
                
                with col4:
                    st.metric("📉 Min HR", f"{min_instant_hr:.1f} bpm", border=True)
                
                with col5:
                    st.metric("📊 HRV", f"{hr_variability_ms} ms", border=True)

                # HRV-Interpretation Funktion
                def interpret_hrv_with_values(hrv_time_dict, hrv_freq_dict):
                    interpretations = []
                    
                    sdnn = hrv_time_dict.get('HRV_SDNN', 0)
                    if sdnn > 50:
                        interpretations.append(("success", f"✅ SDNN ({sdnn:.1f} ms) ist hoch – gute Gesamt-HRV, gesundes autonomes Nervensystem."))
                    elif 30 <= sdnn <= 50:
                        interpretations.append(("warning", f"⚠️ SDNN ({sdnn:.1f} ms) ist mittel – HRV ist moderat, evtl. leichte Belastung vorhanden."))
                    else:
                        interpretations.append(("error", f"❌ SDNN ({sdnn:.1f} ms) ist niedrig – mögliche Belastung, Stress oder Überlastung."))

                    rmssd = hrv_time_dict.get('HRV_RMSSD', 0)
                    if rmssd > 40:
                        interpretations.append(("success", f"✅ RMSSD ({rmssd:.1f} ms) ist hoch – gute parasympathische Aktivität, gute Erholung."))
                    elif 20 <= rmssd <= 40:
                        interpretations.append(("warning", f"⚠️ RMSSD ({rmssd:.1f} ms) ist mittel – moderate Erholung, evtl. leichte Belastung."))
                    else:
                        interpretations.append(("error", f"❌ RMSSD ({rmssd:.1f} ms) ist niedrig – geringe Erholung, möglicher Stress."))

                    pnn50 = hrv_time_dict.get('HRV_pNN50', 0)
                    if pnn50 > 10:
                        interpretations.append(("success", f"✅ pNN50 ({pnn50:.1f}%) ist hoch – gutes Erholungsniveau."))
                    elif 5 <= pnn50 <= 10:
                        interpretations.append(("warning", f"⚠️ pNN50 ({pnn50:.1f}%) ist mittel – moderate Erholung."))
                    else:
                        interpretations.append(("error", f"❌ pNN50 ({pnn50:.1f}%) ist niedrig – geringes Erholungsniveau."))

                    lf_hf = hrv_freq_dict.get('HRV_LFHF', 0)
                    if lf_hf < 2:
                        interpretations.append(("success", f"✅ LF/HF-Verhältnis ({lf_hf:.2f}) ist ausgewogen – sympathische und parasympathische Aktivität im Gleichgewicht."))
                    elif 2 <= lf_hf <= 5:
                        interpretations.append(("warning", f"⚠️ LF/HF-Verhältnis ({lf_hf:.2f}) ist leicht sympathisch dominiert – erhöhter Stresslevel möglich."))
                    else:
                        interpretations.append(("error", f"❌ LF/HF-Verhältnis ({lf_hf:.2f}) ist stark sympathisch dominiert – hoher Stress oder Aktivierung."))

                    return interpretations

                # NeuroKit2 Analyse
                try:
                    import neurokit2 as nk
                    processed, info = nk.ecg_process(ekg.df["Messwerte in mV"].values, sampling_rate=ekg.sampling_rate)
                    rpeaks = info["ECG_R_Peaks"]
                    hrv_time = nk.hrv_time(rpeaks, sampling_rate=ekg.sampling_rate, show=False)
                    hrv_freq = nk.hrv_frequency(rpeaks, sampling_rate=ekg.sampling_rate, show=False)

                    # HRV-Bewertung in Expander mit klickbaren Buttons
                    with st.expander("📝 HRV-Bewertung", expanded=True):
                        interpretations = interpret_hrv_with_values(hrv_time.iloc[0].to_dict(), hrv_freq.iloc[0].to_dict())
                        
                        # Detaillierte HRV-Erklärungen
                        hrv_info = {
                            "SDNN": {
                                "title": "SDNN - Gesamtvariabilität",
                                "description": "**Standard Deviation of NN intervals**\n\nMisst die Gesamtvariabilität der Herzfrequenz über einen bestimmten Zeitraum. Reflektiert die Aktivität des gesamten autonomen Nervensystems.\n\n📊 **Bewertung:**\n• >50ms = Ausgezeichnet (gesundes autonomes System)\n• 30-50ms = Moderat (leichte Belastung möglich)\n• <30ms = Niedrig (Stress oder Überlastung)"
                            },
                            "RMSSD": {
                                "title": "RMSSD - Parasympathische Aktivität",
                                "description": "**Root Mean Square of Successive Differences**\n\nMisst die kurzfristige Herzfrequenzvariabilität und spiegelt hauptsächlich die parasympathische (vagale) Aktivität wider.\n\n📊 **Bewertung:**\n• >40ms = Sehr gut (hohe Erholungsfähigkeit)\n• 20-40ms = Moderat (durchschnittliche Erholung)\n• <20ms = Niedrig (geringe Erholung, möglicher Stress)"
                            },
                            "pNN50": {
                                "title": "pNN50 - Erholungsindikator",
                                "description": "**Percentage of NN intervals > 50ms**\n\nProzentsatz der aufeinanderfolgenden RR-Intervalle, die sich um mehr als 50ms unterscheiden. Starker Indikator für parasympathische Aktivität.\n\n📊 **Bewertung:**\n• >10% = Hoch (gutes Erholungsniveau)\n• 5-10% = Mittel (moderate Erholung)\n• <5% = Niedrig (geringes Erholungsniveau)"
                            },
                            "LF/HF": {
                                "title": "LF/HF Ratio - Autonome Balance",
                                "description": "**Low Frequency/High Frequency Ratio**\n\nVerhältnis zwischen niederfrequenten (0.04-0.15 Hz) und hochfrequenten (0.15-0.4 Hz) Komponenten der HRV. Zeigt das Gleichgewicht zwischen sympathischer und parasympathischer Aktivität.\n\n📊 **Bewertung:**\n• <2 = Ausgewogen (gesunde Balance)\n• 2-5 = Leicht sympathisch (erhöhter Stress möglich)\n• >5 = Stark sympathisch (hoher Stress/Aktivierung)"
                            }
                        }
                        
                        # Session State für Info-Anzeige initialisieren
                        if 'show_hrv_info' not in st.session_state:
                            st.session_state.show_hrv_info = {}
                        
                        for i, (status, text) in enumerate(interpretations):
                            # Finde den entsprechenden HRV-Parameter
                            current_param = None
                            for param in hrv_info.keys():
                                if param in text:
                                    current_param = param
                                    break
                            
                            # Erstelle Spalten für Text und Info-Button
                            col_text, col_info = st.columns([10, 1])
                            
                            with col_text:
                                if status == "success":
                                    st.success(text)
                                elif status == "warning":
                                    st.warning(text)
                                else:
                                    st.error(text)
                            
                            with col_info:
                                if current_param and current_param in hrv_info:
                                    # Klickbarer Info-Button
                                    button_key = f"hrv_info_btn_{current_param}_{i}"
                                    if st.button("ℹ️", key=button_key, use_container_width=True):
                                        # Toggle Info-Anzeige
                                        if button_key in st.session_state.show_hrv_info:
                                            st.session_state.show_hrv_info[button_key] = not st.session_state.show_hrv_info[button_key]
                                        else:
                                            st.session_state.show_hrv_info[button_key] = True
                            
                            # Zeige Info wenn Button geklickt wurde
                            if current_param and f"hrv_info_btn_{current_param}_{i}" in st.session_state.show_hrv_info:
                                if st.session_state.show_hrv_info[f"hrv_info_btn_{current_param}_{i}"]:
                                    st.info(f"**{hrv_info[current_param]['title']}**\n\n{hrv_info[current_param]['description']}")

                    # Zusätzlicher Expander mit allen HRV-Erklärungen
                    with st.expander("📚 Alle HRV-Parameter Erklärungen", expanded=False):
                        tab_sdnn, tab_rmssd, tab_pnn50, tab_lfhf = st.tabs(["SDNN", "RMSSD", "pNN50", "LF/HF"])
                        
                        with tab_sdnn:
                            st.markdown(f"### {hrv_info['SDNN']['title']}")
                            st.markdown(hrv_info['SDNN']['description'])
                        
                        with tab_rmssd:
                            st.markdown(f"### {hrv_info['RMSSD']['title']}")
                            st.markdown(hrv_info['RMSSD']['description'])
                        
                        with tab_pnn50:
                            st.markdown(f"### {hrv_info['pNN50']['title']}")
                            st.markdown(hrv_info['pNN50']['description'])
                        
                        with tab_lfhf:
                            st.markdown(f"### {hrv_info['LF/HF']['title']}")
                            st.markdown(hrv_info['LF/HF']['description'])

                    # HRV-Daten in Expander
                    with st.expander("🔬 Detaillierte HRV-Analyse", expanded=False):
                        col_hrv1, col_hrv2 = st.columns(2)
                        
                        with col_hrv1:
                            st.subheader("⏱️ Zeitbereich-Analyse")
                            hrv_dict = hrv_time.iloc[0].to_dict()
                            
                            # Wichtigste Zeitbereich-Metriken
                            col_time1, col_time2 = st.columns(2)
                            with col_time1:
                                st.metric("SDNN", f"{hrv_dict.get('HRV_SDNN', 0):.1f} ms", border=True)
                                st.metric("RMSSD", f"{hrv_dict.get('HRV_RMSSD', 0):.1f} ms", border=True)
                            with col_time2:
                                st.metric("pNN50", f"{hrv_dict.get('HRV_pNN50', 0):.1f}%", border=True)
                                st.metric("Mean NN", f"{hrv_dict.get('HRV_MeanNN', 0):.1f} ms", border=True)
                        
                        with col_hrv2:
                            st.subheader("📊 Frequenzbereich-Analyse")
                            freq_dict = hrv_freq.iloc[0].to_dict()
                            
                            # Wichtigste Frequenzbereich-Metriken
                            col_freq1, col_freq2 = st.columns(2)
                            with col_freq1:
                                st.metric("LF Power", f"{freq_dict.get('HRV_LF', 0):.1f} ms²", border=True)
                                st.metric("HF Power", f"{freq_dict.get('HRV_HF', 0):.1f} ms²", border=True)
                            with col_freq2:
                                st.metric("LF/HF Ratio", f"{freq_dict.get('HRV_LFHF', 0):.2f}", border=True)
                                st.metric("Total Power", f"{freq_dict.get('HRV_TP', 0):.1f} ms²", border=True)

                except Exception as e:
                    st.warning(f"⚠️ NeuroKit2 Analyse konnte nicht durchgeführt werden: {e}")

                # Plot-Konfiguration
                st.subheader("📈 EKG-Visualisierung")
                
                col_plot1, col_plot2 = st.columns([3, 1])
                with col_plot1:
                    plot_option = st.radio(
                        "📊 Anzeige-Modus:",
                        options=["EKG + Herzfrequenz", "Nur EKG", "Nur Herzfrequenz"],
                        index=0,
                        horizontal=True
                    )
                with col_plot2:
                    time_range = st.slider("⏱️ Zeitbereich (Min):", 0.1, 2.0, 0.2, 0.1)

                # Plot EKG + Herzfrequenz
                df = ekg.df
                zeit_min = df["Zeit in ms"] / 60000

                fig = go.Figure()

                # Dynamische Titel basierend auf Auswahl
                plot_titles = {
                    "EKG + Herzfrequenz": f"🫀 EKG & Herzfrequenz - {person_obj.firstname} {person_obj.lastname}",
                    "Nur EKG": f"📈 EKG-Signal - {person_obj.firstname} {person_obj.lastname}",
                    "Nur Herzfrequenz": f"💓 Herzfrequenz-Verlauf - {person_obj.firstname} {person_obj.lastname}"
                }

                if plot_option in ["EKG + Herzfrequenz", "Nur EKG"]:
                    fig.add_trace(go.Scatter(
                        x=zeit_min,
                        y=df["Messwerte in mV"],
                        mode='lines',
                        name='EKG Signal',
                        line=dict(color='#3498db', width=1.5)
                    ))

                    peaks_df = df[df["Peak"] == 1]
                    fig.add_trace(go.Scatter(
                        x=peaks_df["Zeit in ms"] / 60000,
                        y=peaks_df["Messwerte in mV"],
                        mode='markers',
                        name='R-Peaks',
                        marker=dict(color='#e74c3c', size=8, symbol='diamond')
                    ))

                if plot_option in ["EKG + Herzfrequenz", "Nur Herzfrequenz"]:
                    if len(instant_hr) > 0:
                        peak_times_ms = df.loc[df["Peak"] == 1, "Zeit in ms"].values
                        hr_times_min = (peak_times_ms[:-1] + np.diff(peak_times_ms) / 2) / 60000
                        fig.add_trace(go.Scatter(
                            x=hr_times_min,
                            y=instant_hr,
                            mode='lines+markers',
                            name='Herzfrequenz',
                            yaxis='y2',
                            line=dict(color='#2ecc71', width=3),
                            marker=dict(size=6)
                        ))

                layout = dict(
                    title=dict(
                        text=plot_titles[plot_option],
                        font=dict(size=20)
                    ),
                    xaxis_title="Zeit in Minuten",
                    height=600,
                    xaxis=dict(
                        range=[zeit_min.min(), zeit_min.min() + time_range],
                        rangeslider=dict(visible=False)
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
                st.info("⚠️ Keine Person ausgewählt oder keine EKG-Daten vorhanden.")



with tab5:
    def safe_mean(series):
        return series.mean() if series is not None and not series.dropna().empty else None

    def safe_max(series):
        return series.max() if series is not None and not series.dropna().empty else None

    st.header("🏋️ FIT-Datei Analyse")

    person_names = read_data.get_person_list()
    selected_person_name = st.selectbox("Wähle eine Person", options=person_names, key="tab5_select")

    if selected_person_name:
        person_obj = Person.load_by_name(selected_person_name)

        if person_obj:
            # Verwende die neue Methode
            fit_files = person_obj.get_fit_files_from_directory()
            
            if fit_files:
                fit_filenames = [f"{f['filename']} ({f['sportart']})" for f in fit_files]
                selected_fit_file = st.selectbox("Wähle eine FIT-Datei", options=fit_filenames, key="tab5_fitfile_select")

                if selected_fit_file:
                    # Finde die ausgewählte Datei
                    selected_filename = selected_fit_file.split(" (")[0]  # Entferne Sportart-Zusatz
                    selected_file_info = next((f for f in fit_files if f["filename"] == selected_filename), None)
                    
                    if selected_file_info:
                        fit_path = selected_file_info["filepath"]
                        sportart = selected_file_info["sportart"]

                        try:
                            with open(fit_path, "rb") as f:
                                uploaded_fit_file = io.BytesIO(f.read())

                            # Sportart anzeigen
                            st.info(f"📋 Sportart: {sportart}")

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
                                    dist_col = 'distance' if 'distance' in analyzer.df.columns and not analyzer.df['distance'].isna().all() else (
                                                'gps_distance' if 'gps_distance' in analyzer.df.columns and not analyzer.df['gps_distance'].isna().all() else None)

                                    if dist_col:
                                        dist = analyzer.df[dist_col].safe_max() / 1000

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
                                        calories = analyzer.df['calories'].safe_max()
                                        st.metric("🔥 Kalorien", f"{calories:.0f} kcal")
                                    else:
                                        st.metric("🔥 Kalorien", "N/A")

                                st.divider()

                                # Sportartspezifische Dashboards
                                if sportart == "Radfahren":
                                    st.subheader("🚴 Radfahren-Dashboard")
                                    
                                    # Hauptmetriken für Radfahren
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        speed_col = 'speed' if 'speed' in analyzer.df.columns and not analyzer.df['speed'].isna().all() else (
                                                    'gps_speed' if 'gps_speed' in analyzer.df.columns and not analyzer.df['gps_speed'].isna().all() else None)

                                        if speed_col:
                                            avg_speed = analyzer.df[speed_col].safe_mean() * 3.6
                                            max_speed = analyzer.df[speed_col].safe_max() * 3.6
                                            st.metric("⚡ Geschwindigkeit", f"{avg_speed:.1f} km/h", f"Max: {max_speed:.1f}")
                                        else:
                                            st.metric("⚡ Geschwindigkeit", "N/A")

                                    
                                    with col2:
                                        if 'power' in analyzer.df.columns and not analyzer.df['power'].isna().all():
                                            avg_power = analyzer.df['power'].safe_mean()
                                            max_power = analyzer.df['power'].safe_max()
                                            st.metric("🔋 Leistung", f"{avg_power:.0f} W", f"Max: {max_power:.0f}")
                                        else:
                                            st.metric("🔋 Leistung", "N/A")
                                    
                                    with col3:
                                        if 'cadence' in analyzer.df.columns and not analyzer.df['cadence'].isna().all():
                                            avg_cad = analyzer.df['cadence'].safe_mean()
                                            st.metric("🔄 Kadenz", f"{avg_cad:.0f} rpm")
                                        else:
                                            st.metric("🔄 Kadenz", "N/A")
                                    
                                    with col4:
                                        elevation = analyzer.get_elevation_gain()
                                        if elevation:
                                            st.metric("⛰️ Höhenmeter", f"{elevation:.0f} m")
                                        else:
                                            st.metric("⛰️ Höhenmeter", "N/A")

                                elif sportart == "Laufen":
                                    st.subheader("🏃 Laufen-Dashboard")
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        speed_col = 'speed' if 'speed' in analyzer.df.columns and not analyzer.df['speed'].isna().all() else (
                                                    'gps_speed' if 'gps_speed' in analyzer.df.columns and not analyzer.df['gps_speed'].isna().all() else None)

                                        if speed_col:
                                            avg_speed = analyzer.df[speed_col].safe_mean() * 3.6
                                            max_speed = analyzer.df[speed_col].safe_max() * 3.6
                                            st.metric("⚡ Geschwindigkeit", f"{avg_speed:.1f} km/h", f"Max: {max_speed:.1f}")
                                        else:
                                            st.metric("⚡ Geschwindigkeit", "N/A")
                                    
                                    with col2:
                                        dist_col = 'distance' if 'distance' in analyzer.df.columns and not analyzer.df['distance'].isna().all() else (
                                                    'gps_distance' if 'gps_distance' in analyzer.df.columns and not analyzer.df['gps_distance'].isna().all() else None)

                                        if dist_col:
                                            dist = analyzer.df[dist_col].safe_max() / 1000
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
                                            avg_cad = analyzer.df['cadence'].safe_mean()
                                            st.metric("👟 Schrittfrequenz", f"{avg_cad:.0f} spm")
                                        else:
                                            st.metric("👟 Schrittfrequenz", "N/A")
                                    
                                    with col4:
                                        elevation = analyzer.get_elevation_gain()
                                        if elevation:
                                            st.metric("⛰️ Höhenmeter", f"{elevation:.0f} m")
                                        else:
                                            st.metric("⛰️ Höhenmeter", "N/A")

                                elif sportart == "Schwimmen":
                                    st.subheader("🏊 Schwimmen-Dashboard")
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        if 'distance' in analyzer.df.columns and not analyzer.df['distance'].isna().all():
                                            dist = analyzer.df['distance'].safe_max()
                                            st.metric("🏊 Distanz", f"{dist:.0f} m")
                                        else:
                                            st.metric("🏊 Distanz", "N/A")
                                    
                                    with col2:
                                        if 'distance' in analyzer.df.columns and not analyzer.df['distance'].isna().all():
                                            dist = analyzer.df['distance'].safe_max()
                                            if dist > 0 and analyzer.duration_hours > 0:
                                                pace = (analyzer.duration_hours * 60) / (dist / 100)
                                                st.metric("⏱️ Pace", f"{pace:.2f} min/100m")
                                            else:
                                                st.metric("⏱️ Pace", "N/A")
                                        else:
                                            st.metric("⏱️ Pace", "N/A")
                                    
                                    with col3:
                                        if 'total_strokes' in analyzer.df.columns and not analyzer.df['total_strokes'].isna().all():
                                            avg_strokes = analyzer.df['total_strokes'].safe_mean()
                                            st.metric("💦 Züge", f"{avg_strokes:.1f}")
                                        else:
                                            st.metric("💦 Züge", "N/A")
                                    
                                    with col4:
                                        if 'swolf' in analyzer.df.columns and not analyzer.df['swolf'].isna().all():
                                            avg_swolf = analyzer.df['swolf'].safe_mean()
                                            st.metric("🔢 SWOLF", f"{avg_swolf:.1f}")
                                        else:
                                            st.metric("🔢 SWOLF", "N/A")

                                # Herzfrequenz-Zonen Analyse
                                if hr_stats:
                                    st.subheader("❤️ Herzfrequenz-Zonen")
                                    
                                    # Verwende Alter der Person für bessere Berechnung
                                    max_hr_theoretical = person_obj.calc_max_heart_rate()
                                    
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
                                if sportart in ["Radfahren", "Laufen"]:
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
                            st.error("FIT-Datei nicht gefunden.")
                        except Exception as e:
                            st.error(f"Fehler beim Laden der FIT-Datei: {e}")
            else:
                st.info("Diese Person hat noch keine FIT-Dateien hochgeladen.")
        else:
            st.warning("Person nicht gefunden.")
    else:
        st.info("Bitte wähle eine Person aus.")

with tab6:
    st.header("📤 FIT-Datei hochladen und verwalten")

    # Personenauswahl
    person_names = read_data.get_person_list()
    selected_name = st.selectbox("Wähle eine Person aus", person_names, key="tab6_person_select")
    person_obj = Person.load_by_name(selected_name)

    # Sportartauswahl
    sportart = st.selectbox("Sportart auswählen", ["Laufen", "Radfahren", "Schwimmen", "Andere"], key="sport_select")

    # Uploadbereich
    uploaded_file = st.file_uploader("Wähle eine FIT-Datei aus", type=["fit"], key="fit_upload")

    # Upload in session_state speichern & file_saved verwalten
    if uploaded_file is not None:
        # Falls eine neue Datei hochgeladen wurde, file_saved zurücksetzen
        if ('uploaded_file' not in st.session_state or 
            st.session_state['uploaded_file'].name != uploaded_file.name):
            st.session_state['file_saved'] = False
        st.session_state['uploaded_file'] = uploaded_file

    if 'uploaded_file' in st.session_state:
        if not st.session_state.get('file_saved', False):
            fit_dir = os.path.join("data", "fit_file", str(person_obj.id))
            os.makedirs(fit_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = st.session_state['uploaded_file'].name.replace(".fit", f"_{timestamp}.fit")
            filepath = os.path.join(fit_dir, filename)

            with open(filepath, "wb") as f:
                f.write(st.session_state['uploaded_file'].getbuffer())

            meta_path = filepath.replace(".fit", ".json")
            with open(meta_path, "w") as meta_file:
                json.dump({"sportart": sportart}, meta_file)

            st.success(f"FIT-Datei gespeichert für {selected_name} mit Sportart: {sportart}")

            st.session_state['file_saved'] = True

    st.divider()

    st.subheader("📁 Bereits hochgeladene FIT-Dateien")

    fit_dir = os.path.join("data", "fit_file", str(person_obj.id))
    if not os.path.exists(fit_dir):
        st.info("Keine FIT-Dateien für diese Person vorhanden.")
    else:
        fit_files = [f for f in os.listdir(fit_dir) if f.endswith(".fit")]

        if not fit_files:
            st.info("Keine FIT-Dateien für diese Person vorhanden.")
        else:
            # Statistik sammeln
            stats = {}

            for file in fit_files:
                file_path = os.path.join(fit_dir, file)
                with open(file_path, "rb") as f:
                    analyzer = read_fit_file.FitFileAnalyzer(f)

                sport = analyzer.sportart or "Unbekannt"

                dist_m = 0
                if "distance" in analyzer.df.columns and not analyzer.df["distance"].isna().all():
                    dist_m = analyzer.df["distance"].max()

                dur_h = analyzer.duration_hours

                ascent_m = 0
                if sport in ["Laufen", "Radfahren"]:
                    elevation = analyzer.get_elevation_gain()
                    if elevation is not None:
                        ascent_m = elevation

                if sport not in stats:
                    stats[sport] = {
                        "distance": 0,
                        "duration": 0,
                        "count": 0,
                        "ascent": 0
                    }

                stats[sport]["distance"] += dist_m
                stats[sport]["duration"] += dur_h
                stats[sport]["count"] += 1
                if sport in ["Laufen", "Radfahren"]:
                    stats[sport]["ascent"] += ascent_m

            # Statistik anzeigen
            st.subheader("📊 Aggregierte Statistik nach Sportart")
            for sport, data in stats.items():
                dist_km = data["distance"] / 1000
                dur_h = data["duration"]
                count = data["count"]
                ascent_m = data["ascent"]

                st.markdown(f"### {sport}")
                st.markdown(f"- Anzahl Aufzeichnungen: **{count}**")
                st.markdown(f"- Gesamtdistanz: **{dist_km:.2f} km**")
                st.markdown(f"- Gesamtdauer: **{dur_h:.2f} h**")
                if sport in ["Laufen", "Radfahren"]:
                    st.markdown(f"- Gesamthöhenmeter: **{ascent_m:.0f} m**")

            # Dateien mit Löschen-Button anzeigen
            st.subheader("📄 FIT-Dateien im Detail")
            for file in sorted(fit_files):
                file_path = os.path.join(fit_dir, file)
                meta_path = file_path.replace(".fit", ".json")

                # Sportart aus JSON lesen
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r") as f:
                            sportart_file = json.load(f).get("sportart", "Unbekannt")
                    except json.JSONDecodeError:
                        sportart_file = "Ungültige JSON"
                else:
                    sportart_file = "Keine Angabe"

                cols = st.columns([4, 2, 1])
                cols[0].markdown(f"📄 **{file}**  \n🧩 *Sportart:* {sportart_file}")
                if cols[2].button("❌ Löschen", key=f"delete_{file}"):
                    os.remove(file_path)
                    if os.path.exists(meta_path):
                        os.remove(meta_path)
                    st.success(f"'{file}' und zugehörige Metadaten wurden gelöscht.")
                    st.rerun()