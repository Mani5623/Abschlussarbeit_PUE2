import json

def load_person_data():
    """A Function that knows where the person database is and returns a dictionary with the persons"""
    # Opening JSON file
    file = open("data/person_db.json")

    # Loading the JSON File in a dictionary
    person_data = json.load(file)

    return person_data

def get_person_list():
    """
    Gibt eine Liste mit Strings der Form 'Nachname, Vorname' für jede Person zurück.
    """
    data = load_person_data()
    person_list = []

    for eintrag in data:
        vorname = eintrag["firstname"]
        nachname = eintrag["lastname"]
        person_list.append(f"{nachname}, {vorname}")
    
    return person_list

def find_person_data_by_name(suchstring):
    """Findet die Person in der Datenbank basierend auf 'Nachname, Vorname'."""
    person_data = load_person_data()

    if suchstring == "None":
        return {}

    try:
        nachname, vorname = [s.strip() for s in suchstring.split(", ")]
    except ValueError:
        return {}

    for eintrag in person_data:
        if eintrag["lastname"] == nachname and eintrag["firstname"] == vorname:
            return eintrag

    return {}  # keine Person gefunden


if __name__ == "__main__":
     print(load_person_data())
     print(get_person_list())
