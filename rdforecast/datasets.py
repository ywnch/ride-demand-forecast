import numpy as np
import pandas as pd

def load_training_data(filepath=None):
    if filepath is None:
        filepath = 'https://s3-ap-southeast-1.amazonaws.com/grab-aiforsea-dataset/traffic-management.zip'
        print("'filepath' not given, download data from:", filepath)
    data = pd.read_csv(filepath, compression='infer')
    print("Data loaded.")
    print("N:", len(data))
    print(data.head(3))
    return data

def check_sanity():
    pass

def convert_datetime(data):
    from datetime import datetime, timedelta
    first_day = datetime(2018, 1, 1)  # TODO: temporary
    df = data.copy(deep=True)
    df['date'] = np.array(list(map(timedelta, df['day']))) + first_day
    df['time'] = df['timestamp'].apply(lambda x: datetime.strptime(x, '%H:%M').time())
    df['datetime'] = [pd.datetime.combine(d, t) for d, t in zip(df['date'], df['time'])]
    return df

def preprocess():
    pass

def split():
    pass