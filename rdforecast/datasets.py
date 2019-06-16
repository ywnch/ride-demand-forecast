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

    n_dup = df.duplicated(['geohash6', 'day', 'timestamp']).sum()
    if n_dup != 0:
        print('Number of duplicates in data:', n_dup)
    else:
        print('No duplicates found.')

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


def process_timestamp(data, add_time=False, fix=False):
    """Extract time featrues from timestamp.

    Params
    ------
    data (dataframe / list): containing array-like timestamps (HH:MM)
    add_time (bool): include a time-formatted column
    fix (bool): fix mode, timestep already provided

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
    if type(data) is list:
        # useful for retrieving info for missing timestamps
        df = pd.DataFrame(data, columns=['timestep'])
        df['day'] = df['timestep'] // 96 + 1
    elif fix:
        df = data.copy(deep=True)
        df['day'] = df['timestep'] // 96 + 1
    else:
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
    df['hour'] = df['quarter'] // 4
    df['dow'] = df['day'] % 7

    if add_time:
        from datetime import timedelta
        try:
            tmp = df['quarter'] * timedelta(minutes=15)
            df['time'] = pd.to_datetime(tmp).dt.time
        except TypeError:
            print('`time` not added, try using pandas 0.24.1.')

    return df


def expand_timestep(df, test_data):
    """Expand data to include full timesteps for all TAZs, filled with zeros.

    Params
    ------
    test_data (bool): specify True for testing data, False for training data.
                      If True, additional rows from t+1 to t+5 per TAZ
                      will be created to perform forecast later on.
    """
    # expand all TAZs by full timesteps
    min_ts = int(df['timestep'].min())
    max_ts = int(df['timestep'].max())
    if test_data:
        print('Expanding testing data and fill NaNs with '
              '0 demands for all timesteps per TAZ; '
              'also generating T+1 to T+5 slots for forecasting...')
        timesteps = list(range(min_ts, max_ts + 6))
    else:
        print('Expanding training data and fill NaNs with '
              '0 demands for all timesteps per TAZ...')
        timesteps = list(range(min_ts, max_ts + 1))
    print('Might take a couple of minutes... :)')
    full_df = pd.concat([pd.DataFrame({'geohash6': taz,
                                       'timestep': timesteps})
                         for taz in df['geohash6'].unique()],
                        ignore_index=True,
                        sort=False)

    # merge back fixed features: TAZ-based, timestep-based
    taz_info = ['geohash6', 'label_weekly_raw', 'label_weekly',
                'label_daily', 'label_quarterly', 'active_rate', 'lon', 'lat']
    ts_info = ['day', 'timestep', 'weekly', 'quarter', 'hour', 'dow']
    demand_info = ['geohash6', 'timestep', 'demand']

    full_df = full_df.merge(df[taz_info].drop_duplicates(),
                            how='left', on=['geohash6'])
    full_df = full_df.merge(df[ts_info].drop_duplicates(),
                            how='left', on=['timestep'])

    # NOTE: there are 9 missing timesteps:
    #       1671, 1672, 1673, 1678, 1679, 1680, 1681, 1682, 1683
    #       also, the new t+1 to t+5 slots in test data will miss out ts_info
    # a = set(df['timestep'].unique())
    # b = set(timesteps)
    # print(a.difference(b))
    # print(b.difference(a))

    # fix missing timestep-based information:
    missing = full_df[full_df['day'].isna()]
    patch = process_timestamp(missing, fix=True)
    full_df.fillna(patch, inplace=True)

    # merge row-dependent feature: demand
    full_df = full_df.merge(df[demand_info].drop_duplicates(),
                            how='left', on=['geohash6', 'timestep'])
    full_df['demand'].fillna(0, inplace=True)

    print('Done.')
    print('Missing values:')
    print(full_df.isna().sum())

    return full_df
