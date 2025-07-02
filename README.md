"""
🏋️ FIT-File Analyse – Streamlit App
====================================

Diese Anwendung analysiert sportliche **FIT-Dateien** (z. B. von Garmin-Geräten) im Kontext individueller Versuchspersonen.
Sie ist in **Python** mit **Streamlit** umgesetzt und modular nach objektorientierten Prinzipien strukturiert.

---------------------------------------------------
🚀 Installation
---------------------------------------------------

🔧 Voraussetzungen:
- Installiere [PDM](https://pdm.fming.dev/latest/), falls noch nicht vorhanden:
    pip install pdm

📦 Notwendige Abhängigkeiten (automatisch installiert mit `pdm install`):
- streamlit
- pandas
- plotly
- numpy
- fitparse
- geopy
- neurokit2
- streamlit-folium
- matplotlib (optional)

🛠️ Setup:
1. Projektordner klonen oder entpacken
2. Abhängigkeiten installieren:
    pdm install
3. Anwendung starten:
    pdm run streamlit run main.py
   oder
    streamlit run main.py

---------------------------------------------------
🗂️ Projektstruktur und Dateibeschreibung
---------------------------------------------------

.
├── main.py                → Streamlit-Hauptanwendung mit Tabs und UI
├── person.py              → Person-Objekte mit Altersberechnung, Datei-Zugriff etc.
├── ekgdata.py             → EKG-Auswertung mit NeuroKit2, Visualisierungen & HRV
├── read_fit_file.py       → Analyse von FIT-Dateien (Distanz, Dauer, Plots, GPS etc.)
├── read_data.py           → Zugriff auf JSON-Personendaten
├── data/
│   ├── person_db.json     → Personenstammdaten
│   ├── pictures/          → (optionale) Profilbilder
│   └── fit_files/         → FIT-Dateien nach Person sortiert

---------------------------------------------------
📁 Beispiel-Datenstruktur:
---------------------------------------------------
data/
├── person_db.json
├── fit_files/
│   └── 1/
│       └── Lauf_01.fit
└── pictures/
    └── 1.jpg

---------------------------------------------------
📊 Features der Anwendung (gesamte App)
---------------------------------------------------

🔹 **Versuchspersonen-Verwaltung (Tab 1)**
   • Auswahl und Anzeige von Personen aus der `person_db.json`
   • Altersberechnung basierend auf Geburtsdatum
   • Profilbildanzeige
   
🔹 **Person Hinzufügen (Tab 2)**
   • Anlegen neuer Versuchspersonen

🔹 **Leistungstests aus CSV-Dateien (Tab 3)**
   • Analyse von Aktivitätsdaten (z. B. Ergometer-/Laktattests)
   • Darstellung von Leistung und Puls über Zeit
   • Berechnung von Mittelwerten und Maximalwerten

🔹 **EKG-Datenanalyse (Tab 4)**
   • Hochladen und Auswertung eigener EKG-Daten (CSV)
   • Detektion von R-Peaks, Herzfrequenz, Herzfrequenzvariabilität (HRV)
   • Interaktive Plots und Metriken (bpm, SDNN, RMSSD, LF/HF etc.)
   • Automatische Interpretation anhand gesundheitsrelevanter Schwellen

🔹 **FIT-Datei Analyse (Tab 5)**
   • Analyse sportartspezifischer Daten (Radfahren, Laufen, Schwimmen)
   • Berechnung von Distanz, Dauer, Geschwindigkeit, Pace, Höhenmetern, Kalorien
   • Plots: Herzfrequenz, Höhe, Karte mit GPS-Route (ggf. farbkodiert)
   • Herzfrequenzzonen mit Prozentanteilen
   • Dynamische Workout-Bewertung (Score + Bewertungsskala)

🔹 **FIT-Datei Upload & Zuordnung (Tab 6)**
   • Upload von `.fit`-Dateien direkt über die Oberfläche
   • Automatisches Umbenennen inkl. Timestamp
   • Zuordnung zu einer Person und Auswahl der Sportart
   • Anzeige aller vorhandenen Dateien pro Person
   • Möglichkeit zur Analyse und Löschung direkt aus der Oberfläche
