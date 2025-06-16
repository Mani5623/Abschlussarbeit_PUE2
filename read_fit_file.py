from fitparse import FitFile
import numpy as np
import pandas as pd

def read_fit_file(file):
    fitfile = FitFile(file)
    times = []
    all_records = []

    for record in fitfile.get_messages('record'):
        data = {}
        for field in record:
            data[field.name] = field.value
            if field.name == 'timestamp':
                times.append(field.value)
        all_records.append(data)

    # Zeit in Sekunden seit Beginn berechnen
    if times:
        start_time = times[0]
        times_seconds = [(t - start_time).total_seconds() for t in times]
        time = np.array(times_seconds)
    else:
        time = np.array([])

    # DataFrame bauen
    df = pd.DataFrame(all_records)
    df['time_seconds'] = time  # Zeitspalte hinzufügen

    return df
