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


def check_sanity(df):
    if df.isna().sum().sum() != 0:
        n0 = len(df)
        df.dropna(inplace=True)
        n1 = len(df)
        diff = n0 - n1
        print('Dropped {} ({:.2f}%) records with missing values.'.format(diff, diff / n0 * 100))
    else:
        print('No missing values found.')

    assert df['demand'].min() >= 0
    assert df['demand'].max() <= 1
    print('First day in sequence:', df['day'].min())
    print('Last day in sequence:', df['day'].max())

    return df


def split_train_test(df, n_days=14, path=None):
    split_day = df['day'].max() - n_days
    df_train = df[df['day'] <= split_day]
    df_test = df[df['day'] > split_day]

    print('Train data size: {} ({} days)'.format(len(df_train),
                                                 len(df_train['day'].unique())))
    print('Shape:', df_train.shape)
    print('\nTest data size: {} ({} days)'.format(len(df_test),
                                                  len(df_test['day'].unique())))
    print('Shape:', df_test.shape)

    if path is not None:
        print('\nSaving split datasets to', path)
        df_train.to_csv(path + 'train.csv', index=None)
        df_test.to_csv(path + 'test.csv', index=None)
        print('Done.')

    return df_train, df_test


def convert_datetime(df):
    # from datetime import datetime, timedelta
    # print('Converting datetime features...')
    # df['datetime'] = df['timestamp'].apply(lambda x: datetime.strptime(x, '%H:%M').time())
    # df['date'] = df['datetime'].dt.date
    # df['time'] = df['datetime'].dt.time
    # print('Done.')
    # return df


def preprocess():
    pass


def split():
    pass