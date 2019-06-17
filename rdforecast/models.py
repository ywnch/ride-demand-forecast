# Author: Yuwen Chang


import numpy as np
import pandas as pd
import geohash
from . import datasets


# helper functions
def decode_geohash(df):
    print('Decoding geohash...')
    df['lon'], df['lat'] = zip(*[(latlon[1], latlon[0]) for latlon
                                 in df['geohash6'].map(geohash.decode)])
    return df


def cap(old):
    """Caps predicted values to [0, 1]"""
    new = [min(1, y) for y in old]
    new = [max(0, y) for y in new]
    return np.array(new)


# core functions
def expand_timestep(df, test_data):
    """Expand data to include full timesteps for all TAZs, filled with zeros.

    Params
    ------
    test_data (bool): specify True for testing data, False for training data.
                      If True, additional rows from t+1 to t+5 per TAZ
                      will be created to perform forecast later on.
    """
    # extract coordinates
    df = decode_geohash(df)

    # expand all TAZs by full timesteps
    min_ts = int(df['timestep'].min())
    max_ts = int(df['timestep'].max())
    if test_data:
        print('Expanding testing data and fill NaNs with '
              '0 demands for all timesteps per TAZ; '
              'also generating T+1 to T+5 slots for forecasting...')
        timesteps = list(range(min_ts, max_ts + 7))  # predicting T+1 to T+6
    else:
        print('Expanding training data and fill NaNs with '
              '0 demands for all timesteps per TAZ...')
        timesteps = list(range(min_ts, max_ts + 1))
    print('Might take a moment depending on machines...')

    # create full df skeleton
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
    patch = datasets.process_timestamp(missing, fix=True)
    full_df.fillna(patch, inplace=True)

    # merge row-dependent feature: demand
    full_df = full_df.merge(df[demand_info].drop_duplicates(),
                            how='left', on=['geohash6', 'timestep'])
    full_df['demand'].fillna(0, inplace=True)

    if test_data:
        full_df.loc[full_df['timestep'] > max_ts, 'demand'] = -1

    print('Done.')
    print('Missing values:')
    print(full_df.isna().sum())

    return full_df


def get_history(df, periods):
    """
    Append historical demands of TAZs as a new feature
    from `periods` of timesteps (15-min) before.
    """
    # create diff_zone indicator (curr TAZ != prev TAZ (up to periods) row-wise)
    shft = pd.DataFrame.shift(df[['geohash6', 'demand']], periods=periods)
    diff_zone = df['geohash6'] != shft['geohash6']
    shft.loc[diff_zone, 'demand'] = -1  # set -1 if different TAZ
    df['demand_t-%s' % periods] = shft['demand']
    df['demand_t-%s' % periods].fillna(-1, inplace=True)  # set NaNs to -1
    return df


def generate_features(df, history):
    """"""
    if history is not None:
        print('Retrieving historical demands...')
        [get_history(df, h) for h in history]
    print('Generating features...')
    # NOTE: be aware of timezones (see explore_function segmentation.ipynb)
    # df['am_peak'] = ((df['hour'] >= 22) | (df['hour'] <= 2)).astype(int)
    # df['midnight'] = ((df['hour'] >= 17) & (df['hour'] < 22)).astype(int)
    df['weekend'] = (df['dow'] > 4).astype(int)
    df['st_trend'] = df['demand_t-1'] - df['demand_t-2']
    df['mt_trend'] = df['demand_t-1'] - df['demand_t-5']
    df['st_trend_1d'] = df['demand_t-96'] - df['demand_t-97']
    df['mt_trend_1d'] = df['demand_t-96'] - df['demand_t-101']
    df['st_trend_1w'] = df['demand_t-672'] - df['demand_t-673']
    df['mt_trend_1w'] = df['demand_t-672'] - df['demand_t-677']
    df['lt_trend_1d'] = df['demand_t-96'] - df['demand_t-672']
    print('Done.')
    return df


def get_train_validate(full_df, features, split):
    """Generate training and validation sets with features."""
    X = full_df[features + ['demand']]
    print('[dtypes of features (including demand):]')
    print(X.dtypes.value_counts())

    print('\nSplit train and validation sets on day', split)
    X_train = X[X['day'] <= split]
    X_val = X[X['day'] > split]

    y_train = X_train.pop('demand')
    y_val = X_val.pop('demand')

    days_train = len(X_train['day'].unique())
    days_val = len(X_val['day'].unique())
    print('')
    print(days_train, 'days in train set.')
    print('X_train:', X_train.shape)
    print('y_train:', y_train.shape)
    print('')
    print(days_val, 'days in validation set.')
    print('X_val:', X_val.shape)
    print('y_val:', y_val.shape)

    return X_train, X_val, y_train, y_val


def get_test_forecast(full_df, features):
    """Generate testing and forecasting sets with features."""
    # TODO: same functionality, merge with get_train_validate
    X = full_df[features + ['demand']]
    print('[dtypes of features (including demand):]')
    print(X.dtypes.value_counts())

    # get the horizons for final forecasting
    print('\nSplit test and forecast sets')
    split = X['timestep'].max() - 6
    X_test = X[X['timestep'] <= split]
    X_forecast = X[X['timestep'] > split]

    y_test = X_test.pop('demand')
    y_forecast = X_forecast.pop('demand')

    print('X_test:', X_test.shape)
    print('y_test:', y_test.shape)
    print('X_forecast:', X_forecast.shape)
    print('y_forecast:', y_forecast.shape)

    return X_test, X_forecast, y_test, y_forecast


def get_forecast_output(full_df, y_forecast, shift=False, path=None):
    """Generate the forecast output following the training data format.
    Params
    ------
    full_df (dataframe): as generated from `models.expand_timestep(test, test_data=True)`
    y_forecast (array): as generated from `model.predict(X_forecast)`
    shift (bool): if True, all forecast results will be shifted 1 timestep ahead,
                  i.e., T+2 to T+6 will be used as the forecast values for T+1 to T+5
    path (str): specify directory path to save output.csv
    Returns
    -------
    X_forecast (dataframe): the final output dataframe containing forecast values for
                            all TAZs from T+1 to T+5 following the final T in test data,
                            in the format of input data.
    """
    X = full_df[['geohash6', 'day', 'timestep']]

    # get the horizons for final forecasting
    split = X['timestep'].max() - 6
    X_forecast = X[X['timestep'] > split].sort_values(['geohash6', 'timestep'])
    
    # formatting and convert timestep back to timestamp
    X_forecast['timestamp'] = datasets.tstep_to_tstamp(X_forecast.pop('timestep'))
    X_forecast['day'] = X_forecast['day'].astype(int)
    
    # append forecast results
    y_forecast = cap(y_forecast)  # calibrate results beyond boundaries [0, 1]
    X_forecast['demand'] = y_forecast
    
    # drop additional T+6 horizon, after shifting if specified
    shft = pd.DataFrame.shift(X_forecast[['geohash6', 'demand']], periods=-1)
    keep = X_forecast['geohash6'] == shft['geohash6']
    if shift:
        print('\n[INFO] Forecast output values shifted (1T).')
        X_forecast['demand'] = shft['demand']
        filename = 'output_shift.csv'
    else:
        filename = 'output.csv'
    X_forecast = X_forecast[keep].reset_index(drop=True)  # drop out the extra T+6
    
    if path is not None:
        print('Saving forecast output to', path)
        X_forecast.to_csv(path + filename, index=None)
        print('Done.')

    return X_forecast


def mitigate_history(X_train, X_val):
    """Prevents overfitting on features that are not always available."""
    X_train.loc[X_train.sample(frac=0.4).index, ['demand_t-1', 'st_trend', 'mt_trend']] = -1
    X_train.loc[X_train.sample(frac=0.3).index, ['demand_t-2']] = -1
    X_val.loc[X_val.sample(frac=0.4).index, ['demand_t-1', 'st_trend', 'mt_trend']] = -1
    X_val.loc[X_val.sample(frac=0.3).index, ['demand_t-2']] = -1
    return X_train, X_val


# evaluation functions
def get_baselines(full_df):
    """Generate some simple baselines to compare with."""
    df = full_df[(full_df['demand_t-1'] != -1) &
                 (full_df['demand_t-672'] != -1)]
    print('[Simple Baselines]')

    # historical timesteps
    y_true = df['demand']
    print('Naive T-1:')
    eval_RMSE(y_true, df['demand_t-1'])
    print('Naive T-96 (1 day):')
    eval_RMSE(y_true, df['demand_t-96'])
    print('Naive T-672 (1 week):')
    eval_RMSE(y_true, df['demand_t-672'])
    print('T-1 + (T-96 - T-97):')
    eval_RMSE(y_true, (df['demand_t-1'] + df['st_trend_1d']))
    print('T-1 + (T-672 - T-673):')
    eval_RMSE(y_true, (df['demand_t-1'] + df['st_trend_1w']))

    # historical average (weekly better than dow)
    # dow_hist_avg = train_full[['geohash6', 'dow', 'demand']].groupby(['geohash6', 'dow']).mean().reset_index()
    # dow_hist_avg.columns = ['geohash6', 'dow', 'hist_avg_dow']
    # tmp = train_full.merge(dow_hist_avg, how='left', on=['geohash6', 'dow'])
    # tmp['hist_avg_dow'].fillna(0, inplace=True)
    # eval_RMSE(tmp['demand'], tmp['hist_avg_dow'])

    weekly_hist_avg = df[['geohash6', 'weekly', 'demand']].groupby(['geohash6', 'weekly']).mean().reset_index()
    weekly_hist_avg.columns = ['geohash6', 'weekly', 'hist_avg_weekly']
    tmp = df.merge(weekly_hist_avg, how='left', on=['geohash6', 'weekly'])
    tmp['hist_avg_weekly'].fillna(0, inplace=True)

    print('Historical average (by 672 timesteps weekly cycle per TAZ):')
    eval_RMSE(tmp['demand'], tmp['hist_avg_weekly'])


def eval_RMSE(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    assert len(y_true) == len(y_pred), 'Lengths mismatch.'
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(rmse)


def eval_MAPE(y_true, y_pred):
    assert len(y_true) == len(y_pred), 'Lengths mismatch.'
    mask = (y_true != 0)
    mape = (np.abs(y_true - y_pred) / y_true)[mask].mean()
    print(mape)
    return mape


# visualization functions
def plot_forecast(df, taz, start=None, end=None):
    import matplotlib.pyplot as plt
    sample = df[df['geohash6'] == taz].sort_values('timestep')
    fig = plt.figure(figsize=(24, 6))
    ax = fig.add_subplot(111)
    ax.plot(sample['timestep'][start:end],
            sample['y_test'][start:end],
            color='steelblue', label='True')
    ax.plot(sample['timestep'][start:end],
            sample['y_pred'][start:end],
            color='indianred', label='Prediction')
    try:
        ax.plot(sample['timestep'][start:end],
                sample['y_pred_shift'][start:end],
                color='pink', label='Prediction_shifted')
    except KeyError:
        pass
    plt.legend(fontsize=18)
    plt.grid()
    plt.show()
