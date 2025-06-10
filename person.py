import json
from ekgdata import EKGdata
from datetime import datetime

class Person:

    def __init__(self, person_dict):
        self.id = person_dict["id"]
        self.firstname = person_dict["firstname"]
        self.lastname = person_dict["lastname"]
        self.date_of_birth = person_dict["date_of_birth"]
        self.picture_path = person_dict["picture_path"]
        self.gender = person_dict["gender"]
        self.ekg_tests_raw = person_dict.get("ekg_tests", [])
        self.ekg_tests = [EKGdata(test) for test in self.ekg_tests_raw]  # Liste von EKGdata-Objekten

    def calc_age(self):
        current_year = datetime.now().year
        return current_year - self.date_of_birth

    def calc_max_heart_rate(self, gender="male"):
        age = self.calc_age()
        if gender.lower() == "male":
            return round(223 - 0.9 * age)
        elif gender.lower() == "female":
            return round(226 - 0.9 * age)
        else:
            raise ValueError("Ungültiges Geschlecht – bitte 'male' oder 'female' angeben.")


    @staticmethod
    def load_person_data():
        """Lädt alle Personen als Dictionary-Liste"""
        with open("data/person_db.json") as file:
            return json.load(file)

    @staticmethod
    def get_person_list(person_data):
        """Erstellt Liste aller Namen im Format 'Nachname, Vorname'"""
        return [f"{p['lastname']}, {p['firstname']}" for p in person_data]

    @staticmethod
    def find_person_data_by_name(suchstring):
        """Findet Personendatensatz per 'Nachname, Vorname'-String"""
        if suchstring == "None":
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


if __name__ == "__main__":
    print("This is a module with some functions to read the person data")

    # Test: Alle Namen anzeigen
    persons = Person.load_person_data()
    person_names = Person.get_person_list(persons)
    print("Alle Personen:", person_names)

    # Test: Eine Person laden und ihre EKGs anzeigen
    person = Person.load_by_name("Huber, Julian")
    print(f"Name: {person.firstname} {person.lastname}")
    print("Alter:", person.calc_age())
    print("Anzahl EKG-Tests:", len(person.ekg_tests))