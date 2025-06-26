import json
from ekgdata import EKGdata
from datetime import datetime, date
import os

class Person:
    def __init__(self, person_dict):
        self.id = person_dict["id"]
        self.firstname = person_dict["firstname"]
        self.lastname = person_dict["lastname"]

        birth_data = person_dict["date_of_birth"]
        if isinstance(birth_data, str):
            try:
                self.date_of_birth = datetime.strptime(birth_data, "%Y-%m-%d").date()
            except ValueError:
                try:
                    self.date_of_birth = date(int(birth_data), 1, 1)
                except ValueError:
                    self.date_of_birth = date(2000, 1, 1)
        elif isinstance(birth_data, int):
            self.date_of_birth = date(birth_data, 1, 1)
        else:
            self.date_of_birth = birth_data

        self.picture_path = person_dict.get("picture_path")
        self.gender = person_dict.get("gender")
        self.ekg_tests_raw = person_dict.get("ekg_tests", [])
        self.fit_files_raw = person_dict.get("fit_files", [])
        self.ekg_tests = [EKGdata(test) for test in self.ekg_tests_raw]
        self.fit_files = self.fit_files_raw

    def calc_age(self):
        today = date.today()
        age = today.year - self.date_of_birth.year
        if today < date(today.year, self.date_of_birth.month, self.date_of_birth.day):
            age -= 1
        return age

    def calc_max_heart_rate(self, gender=None):
        age = self.calc_age()
        used_gender = gender or self.gender

        if used_gender and used_gender.lower() == "male":
            return round(223 - 0.9 * age)
        elif used_gender and used_gender.lower() == "female":
            return round(226 - 0.9 * age)
        else:
            return round(220 - age)

    @staticmethod
    def load_person_data():
        try:
            with open("data/person_db.json", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            print("Warnung: person_db.json nicht gefunden.")
            return []
        except json.JSONDecodeError:
            print("Fehler: person_db.json ist beschädigt.")
            return []

    @staticmethod
    def get_person_list(person_data=None):
        if person_data is None:
            person_data = Person.load_person_data()
        return [f"{p['lastname']}, {p['firstname']}" for p in person_data]

    @staticmethod
    def find_person_data_by_name(suchstring):
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
        person_dict = cls.find_person_data_by_name(name_str)
        if person_dict:
            return cls(person_dict)
        return None

    @staticmethod
    def get_next_id():
        person_data = Person.load_person_data()
        if not person_data:
            return 1
        max_id = max(p.get("id", 0) for p in person_data)
        return max_id + 1

    @staticmethod
    def save_uploaded_image(uploaded_file, firstname, lastname):
        if uploaded_file is None:
            return None

        image_dir = "data/pictures"
        os.makedirs(image_dir, exist_ok=True)

        file_extension = uploaded_file.name.split('.')[-1].lower()
        filename = f"{lastname}_{firstname}.{file_extension}"
        file_path = os.path.join(image_dir, filename)

        try:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            return file_path
        except Exception as e:
            print(f"Fehler beim Speichern des Bildes: {e}")
            return None

    @staticmethod
    def add_new_person(firstname, lastname, birth_date, gender, uploaded_file=None):
        try:
            person_data = Person.load_person_data()
            for person in person_data:
                if (person["firstname"].lower() == firstname.lower() and 
                    person["lastname"].lower() == lastname.lower()):
                    raise ValueError(f"Person {firstname} {lastname} existiert bereits!")

            picture_path = Person.save_uploaded_image(uploaded_file, firstname, lastname)

            if isinstance(birth_date, date):
                birth_date_str = birth_date.strftime("%Y-%m-%d")
            else:
                birth_date_str = str(birth_date)

            new_person = {
                "id": Person.get_next_id(),
                "firstname": firstname,
                "lastname": lastname,
                "date_of_birth": birth_date_str,
                "picture_path": picture_path,
                "gender": gender,
                "ekg_tests": [],
                "fit_files": []
            }

            person_data.append(new_person)
            os.makedirs("data", exist_ok=True)

            with open("data/person_db.json", "w", encoding="utf-8") as file:
                json.dump(person_data, file, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            print(f"Fehler beim Hinzufügen der Person: {e}")
            return False

    def add_uploaded_file(self, file, filetype):
        if file is None or filetype not in ["ekg", "fit"]:
            return False

        os.makedirs("data/uploads", exist_ok=True)

        extension = file.name.split(".")[-1].lower()
        timestamp = datetime.now().isoformat(timespec="seconds")
        safe_filename = f"{self.lastname}_{self.firstname}_{filetype}_{timestamp.replace(':', '-')}.{extension}"
        file_path = os.path.join("data/uploads", safe_filename)

        try:
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            entry = {
                "filename": safe_filename,
                "original_name": file.name,  # ⬅️ NEU: Ursprünglicher Dateiname
                "source": "upload",
                "timestamp": timestamp
            }

            person_data = Person.load_person_data()
            for p in person_data:
                if p["id"] == self.id:
                    if filetype == "ekg":
                        p.setdefault("ekg_tests", []).append(entry)
                    elif filetype == "fit":
                        p.setdefault("fit_files", []).append(entry)
                    break

            with open("data/person_db.json", "w", encoding="utf-8") as file:
                json.dump(person_data, file, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            print(f"Fehler beim Speichern der Datei: {e}")
            return False

    def update_person_data(self, **kwargs):
        try:
            person_data = Person.load_person_data()
            for person in person_data:
                if person["id"] == self.id:
                    for key, value in kwargs.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
                            if key == "date_of_birth" and isinstance(value, date):
                                person[key] = value.strftime("%Y-%m-%d")
                            else:
                                person[key] = value
                    break

            with open("data/person_db.json", "w", encoding="utf-8") as file:
                json.dump(person_data, file, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            print(f"Fehler beim Aktualisieren der Person: {e}")
            return False

    def __str__(self):
        return f"{self.firstname} {self.lastname} (ID: {self.id})"

    def __repr__(self):
        return f"Person(id={self.id}, name='{self.firstname} {self.lastname}', age={self.calc_age()})"


if __name__ == "__main__":
    try:
        persons = Person.load_person_data()
        person_names = Person.get_person_list(persons)
        print("Alle Personen:", person_names)

        if person_names:
            person = Person.load_by_name(person_names[0])
            if person:
                print(f"\nTest-Person: {person}")
                print("Alter:", person.calc_age())
                print("Max. Herzfrequenz:", person.calc_max_heart_rate())
                print("Anzahl EKG-Tests:", len(person.ekg_tests))
    except Exception as e:
        print(f"Fehler beim Testen: {e}")