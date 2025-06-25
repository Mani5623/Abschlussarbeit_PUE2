import json
from ekgdata import EKGdata
from datetime import datetime, date
import os

class Person:
    def __init__(self, person_dict):
        self.id = person_dict["id"]
        self.firstname = person_dict["firstname"]
        self.lastname = person_dict["lastname"]
        
        # Verbesserte Datumsverarbeitung
        birth_data = person_dict["date_of_birth"]
        if isinstance(birth_data, str):
            # Falls als String gespeichert, versuche zu parsen
            try:
                self.date_of_birth = datetime.strptime(birth_data, "%Y-%m-%d").date()
            except ValueError:
                # Fallback für alte Daten (nur Jahr)
                try:
                    self.date_of_birth = date(int(birth_data), 1, 1)
                except ValueError:
                    self.date_of_birth = date(2000, 1, 1)  # Default-Datum
        elif isinstance(birth_data, int):
            # Alte Daten (nur Jahr)
            self.date_of_birth = date(birth_data, 1, 1)
        else:
            self.date_of_birth = birth_data
            
        self.picture_path = person_dict.get("picture_path")
        self.gender = person_dict.get("gender")
        self.ekg_tests_raw = person_dict.get("ekg_tests", [])
        self.ekg_tests = [EKGdata(test) for test in self.ekg_tests_raw]

    def calc_age(self):
        """Berechnet das genaue Alter basierend auf dem Geburtsdatum"""
        today = date.today()
        age = today.year - self.date_of_birth.year
        
        # Prüfe ob Geburtstag dieses Jahr schon war
        if today < date(today.year, self.date_of_birth.month, self.date_of_birth.day):
            age -= 1
            
        return age

    def calc_max_heart_rate(self, gender=None):
        """Berechnet die maximale Herzfrequenz"""
        age = self.calc_age()
        used_gender = gender or self.gender
        
        if used_gender and used_gender.lower() == "male":
            return round(223 - 0.9 * age)
        elif used_gender and used_gender.lower() == "female":
            return round(226 - 0.9 * age)
        else:
            # Fallback: Allgemeine Formel
            return round(220 - age)

    @staticmethod
    def load_person_data():
        """Lädt alle Personen als Dictionary-Liste"""
        try:
            with open("data/person_db.json", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            print("Warnung: person_db.json nicht gefunden. Erstelle leere Liste.")
            return []
        except json.JSONDecodeError:
            print("Fehler: person_db.json ist beschädigt. Erstelle leere Liste.")
            return []

    @staticmethod
    def get_person_list(person_data=None):
        """Erstellt Liste aller Namen im Format 'Nachname, Vorname'"""
        if person_data is None:
            person_data = Person.load_person_data()
        return [f"{p['lastname']}, {p['firstname']}" for p in person_data]

    @staticmethod
    def find_person_data_by_name(suchstring):
        """Findet Personendatensatz per 'Nachname, Vorname'-String"""
        if suchstring == "None" or not suchstring:
            return {}
        
        person_data = Person.load_person_data()
        try:
            lastname, firstname = suchstring.split(", ")
        except ValueError:
            return {}
        
        for p in person_data:
            if p["lastname"] == lastname and p["firstname"] == firstname:
                return p
        return {}

    @classmethod
    def load_by_name(cls, name_str):
        """Instanziiert eine Person anhand des Namens"""
        person_dict = cls.find_person_data_by_name(name_str)
        if person_dict:
            return cls(person_dict)
        return None

    @staticmethod
    def get_next_id():
        """Generiert eine neue eindeutige ID"""
        person_data = Person.load_person_data()
        if not person_data:
            return 1
        max_id = max(p.get("id", 0) for p in person_data)
        return max_id + 1

    @staticmethod
    def save_uploaded_image(uploaded_file, firstname, lastname):
        """Speichert hochgeladenes Bild und gibt Pfad zurück"""
        if uploaded_file is None:
            return None
        
        # Erstelle Ordner falls nicht vorhanden
        image_dir = "data/pictures"
        os.makedirs(image_dir, exist_ok=True)
        
        # Generiere Dateinamen
        file_extension = uploaded_file.name.split('.')[-1].lower()
        filename = f"{lastname}_{firstname}.{file_extension}"
        file_path = os.path.join(image_dir, filename)
        
        try:
            # Speichere Datei
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            return file_path
        except Exception as e:
            print(f"Fehler beim Speichern des Bildes: {e}")
            return None

    @staticmethod
    def add_new_person(firstname, lastname, birth_date, gender, uploaded_file=None):
        """Fügt eine neue Person zur Datenbank hinzu"""
        try:
            # Lade bestehende Daten
            person_data = Person.load_person_data()
            
            # Prüfe auf Duplikate
            for person in person_data:
                if (person["firstname"].lower() == firstname.lower() and 
                    person["lastname"].lower() == lastname.lower()):
                    raise ValueError(f"Person {firstname} {lastname} existiert bereits!")
            
            # Speichere Bild falls vorhanden
            picture_path = Person.save_uploaded_image(uploaded_file, firstname, lastname)
            
            # Konvertiere Datum zu String für JSON-Speicherung
            if isinstance(birth_date, date):
                birth_date_str = birth_date.strftime("%Y-%m-%d")
            else:
                birth_date_str = str(birth_date)
            
            # Erstelle neue Person
            new_person = {
                "id": Person.get_next_id(),
                "firstname": firstname,
                "lastname": lastname,
                "date_of_birth": birth_date_str,
                "picture_path": picture_path,
                "gender": gender,
                "ekg_tests": []
            }
            
            # Füge zur Liste hinzu
            person_data.append(new_person)
            
            # Erstelle Backup-Ordner falls nicht vorhanden
            os.makedirs("data", exist_ok=True)
            
            # Speichere zurück in JSON
            with open("data/person_db.json", "w", encoding="utf-8") as file:
                json.dump(person_data, file, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"Fehler beim Hinzufügen der Person: {e}")
            return False

    @staticmethod
    def person_exists(firstname, lastname):
        """Prüft ob Person bereits existiert"""
        try:
            person_data = Person.load_person_data()
            for person in person_data:
                if (person["firstname"].lower() == firstname.lower() and 
                    person["lastname"].lower() == lastname.lower()):
                    return True
            return False
        except Exception as e:
            print(f"Fehler bei der Duplikatsprüfung: {e}")
            return False

    @staticmethod
    def delete_person(person_id):
        """Löscht eine Person aus der Datenbank"""
        try:
            person_data = Person.load_person_data()
            
            # Finde Person zum Löschen
            person_to_delete = None
            for i, person in enumerate(person_data):
                if person["id"] == person_id:
                    person_to_delete = person
                    del person_data[i]
                    break
            
            if person_to_delete:
                # Lösche auch das Bild falls vorhanden
                if person_to_delete.get("picture_path"):
                    try:
                        os.remove(person_to_delete["picture_path"])
                    except FileNotFoundError:
                        pass
                
                # Speichere aktualisierte Daten
                with open("data/person_db.json", "w", encoding="utf-8") as file:
                    json.dump(person_data, file, indent=2, ensure_ascii=False)
                
                return True
            return False
            
        except Exception as e:
            print(f"Fehler beim Löschen der Person: {e}")
            return False

    def update_person_data(self, **kwargs):
        """Aktualisiert Personendaten"""
        try:
            person_data = Person.load_person_data()
            
            # Finde die Person in den Daten
            for person in person_data:
                if person["id"] == self.id:
                    # Aktualisiere die Felder
                    for key, value in kwargs.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
                            if key == "date_of_birth" and isinstance(value, date):
                                person[key] = value.strftime("%Y-%m-%d")
                            else:
                                person[key] = value
                    break
            
            # Speichere aktualisierte Daten
            with open("data/person_db.json", "w", encoding="utf-8") as file:
                json.dump(person_data, file, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"Fehler beim Aktualisieren der Person: {e}")
            return False

    def __str__(self):
        """String-Repräsentation der Person"""
        return f"{self.firstname} {self.lastname} (ID: {self.id})"

    def __repr__(self):
        """Repräsentation für Debugging"""
        return f"Person(id={self.id}, name='{self.firstname} {self.lastname}', age={self.calc_age()})"


if __name__ == "__main__":
    print("This is a module with some functions to read the person data")

    # Test: Alle Namen anzeigen
    try:
        persons = Person.load_person_data()
        person_names = Person.get_person_list(persons)
        print("Alle Personen:", person_names)

        # Test: Eine Person laden und ihre Daten anzeigen
        if person_names:
            person = Person.load_by_name(person_names[0])
            if person:
                print(f"\nTest-Person: {person}")
                print("Alter:", person.calc_age())
                print("Max. Herzfrequenz:", person.calc_max_heart_rate())
                print("Anzahl EKG-Tests:", len(person.ekg_tests))
    except Exception as e:
        print(f"Fehler beim Testen: {e}")
