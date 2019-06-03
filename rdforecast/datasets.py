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

def preprocess():
    pass

def split():
    pass