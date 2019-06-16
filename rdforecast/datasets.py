import numpy as np
import pandas as pd


def load_training_data(filepath=None):
    if filepath is None:
        filepath = 'https://s3-ap-southeast-1.amazonaws.com/grab-aiforsea-dataset/traffic-management.zip'
        print("'filepath' not given, download data from:", filepath)
    data = pd.read_csv(filepath, compression='infer')
    print("Data loaded.")
    print('TAZ:', len(set(data['geohash6'])))
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

    assert df['demand'].min() >= 0, 'Demand < 0 found in data.'
    assert df['demand'].max() <= 1, 'Demand > 1 found in data.'
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


def process_timestamp(data, add_time=False):
    """Extract time featrues from timestamp.

    Params
    ------
    add_time (bool): include a time-formatted column

    Returns
    -------
    df (dataframe): with time features (all sequential features starting from 0)
        - timestep: quarters throughout the whole period
        - weekly: timestep repeated in weekly cycle
        - quarter: quarter sequence in a day
        - hour: hour sequence in a day (may not align with actual hour in the timezone)
        - dow: day sequence in a week (may not align with actual weekday/end)
        - time: timestamp in datetime.time data type
    """
    df = data.copy(deep=True)
    ts = df['timestamp'].unique()
    h, m = zip(*[t.split(':') for t in ts])
    h = np.array([int(i) for i in h])
    m = np.array([int(i) for i in m])
    ts_num = (h * 60 + m) / 15
    ts_to_num = dict(zip(ts, ts_num))
    df['timestep'] = ((df['day'] - 1) * 96 + data['timestamp'].map(ts_to_num))
    df['timestep'] = df['timestep'].astype(int)
    df['weekly'] = df['timestep'] % 672
    df['quarter'] = df['timestep'] % 96
    df['hour'] = [int(t.split(':')[0]) for t in df['timestamp']]
    df['dow'] = df['day'] % 7

    if add_time:
        from datetime import timedelta
        try:
            tmp = df['quarter'] * timedelta(minutes=15)
            df['time'] = pd.to_datetime(tmp).dt.time
        except TypeError:
            print('`time` not added, try using pandas 0.24.1.')

    return df
