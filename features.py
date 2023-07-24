from joblib import Parallel, delayed  # parallel computing to save time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import preprocesing

DATA_DIR = './dataOptiver/'

def make_features(train, train_ids, test, test_ids):
    df_train = compute_book_trade_features(list_stock_ids=train_ids, is_train=True, data_dir=DATA_DIR)
    df_train = train.merge(df_train, on=['row_id'], how='left')
    df_test = compute_book_trade_features(list_stock_ids=test_ids, is_train=False, data_dir=DATA_DIR)
    df_test = test.merge(df_test, on=['row_id'], how='left')

    df_train, df_test = make_target_encoding(df_train, df_test)
    df_train['stock_id'] = df_train['stock_id'].astype(int)
    df_test['stock_id'] = df_test['stock_id'].astype(int)

    return df_train, df_test


def calc_wap(df):
    wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (
            df['bid_size1'] + df['ask_size1'])
    return wap


def calc_wap2(df):
    wap = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (
            df['bid_size2'] + df['ask_size2'])
    return wap


def log_return(list_stock_prices):
    return pd.Series(np.log(list_stock_prices).diff())


def realized_volatility(series):
    return pd.Series(np.sqrt(np.sum(series ** 2)))


def count_unique(series):
    return len(np.unique(series))


def compute_book_features(df, stock_id):
    df['wap'] = calc_wap(df)
    df['log_return'] = df.groupby('time_id')['wap'].apply(log_return).reset_index(level=0, drop=True)
    df['wap2'] = calc_wap2(df)
    df['log_return2'] = df.groupby('time_id')['wap2'].apply(log_return).reset_index(level=0, drop=True)
    df['wap_balance'] = abs(df['wap'] - df['wap2'])
    df['price_spread'] = (df['ask_price1'] - df['bid_price1']) / ((df['ask_price1'] + df['bid_price1']) / 2)
    df['bid_spread'] = df['bid_price1'] - df['bid_price2']
    df['ask_spread'] = df['ask_price1'] - df['ask_price2']
    df['total_volume'] = (df['ask_size1'] + df['ask_size2']) + (df['bid_size1'] + df['bid_size2'])
    df['volume_imbalance'] = abs((df['ask_size1'] + df['ask_size2']) - (df['bid_size1'] + df['bid_size2']))

    create_feature_dict = {
        'log_return': [realized_volatility],
        'log_return2': [realized_volatility],
        'wap_balance': [np.mean],
        'price_spread': [np.mean],
        'bid_spread': [np.mean],
        'ask_spread': [np.mean],
        'volume_imbalance': [np.mean],
        'total_volume': [np.mean],
        'wap': [np.mean],
    }
    df_feature = pd.DataFrame(df.groupby(['time_id']).agg(create_feature_dict)).reset_index()
    df_feature.columns = ['_'.join(col) for col in df_feature.columns]  # time_id is changed to time_id_

    # groupby / last XX seconds
    last_seconds = [300]
    for second in last_seconds:
        second = 600 - second
        df_feature_sec = pd.DataFrame(
            df.query(f'seconds_in_bucket >= {second}').groupby(['time_id']).agg(create_feature_dict)).reset_index()
        df_feature_sec.columns = ['_'.join(col) for col in df_feature_sec.columns]  # time_id is changed to time_id_
        df_feature_sec = df_feature_sec.add_suffix('_' + str(second))
        df_feature = pd.merge(df_feature, df_feature_sec, how='left', left_on='time_id_', right_on=f'time_id__{second}')
        df_feature = df_feature.drop([f'time_id__{second}'], axis=1)
    df_feature['row_id'] = df_feature['time_id_'].apply(lambda x: f'{stock_id}-{x}')
    df_feature = df_feature.drop(['time_id_'], axis=1)

    return df_feature


def compute_trade_features(df, stock_id):
    df['log_return'] = df.groupby('time_id')['price'].apply(log_return).reset_index(level=0, drop=True)

    aggregate_dictionary = {
        'log_return': [realized_volatility],
        'seconds_in_bucket': [count_unique],
        'size': [np.sum],
        'order_count': [np.mean],
    }
    df_feature = df.groupby('time_id').agg(aggregate_dictionary).reset_index()
    df_feature.columns = ['_'.join(col) for col in df_feature.columns]

    last_seconds = [300]
    for second in last_seconds:
        second = 600 - second
        df_feature_sec = df.query(f'seconds_in_bucket >= {second}').groupby('time_id').agg(aggregate_dictionary)
        df_feature_sec = df_feature_sec.reset_index()
        df_feature_sec.columns = ['_'.join(col) for col in df_feature_sec.columns]
        df_feature_sec = df_feature_sec.add_suffix('_' + str(second))
        df_feature = pd.merge(df_feature, df_feature_sec, how='left', left_on='time_id_', right_on=f'time_id__{second}')
        df_feature = df_feature.drop([f'time_id__{second}'], axis=1)

    df_feature = df_feature.add_prefix('trade_')
    df_feature['row_id'] = df_feature['trade_time_id_'].apply(lambda x: f'{stock_id}-{x}')
    df_feature = df_feature.drop(['trade_time_id_'], axis=1)

    return df_feature


def compute_book_trade_features(data_dir, list_stock_ids, is_train=True):
    df = pd.DataFrame()

    def for_joblib(stock_id):
        df_book, df_trade = preprocesing.load_data(data_dir=data_dir, stock_id=stock_id, is_train=is_train)
        df_tmp = pd.merge(compute_book_features(df_book, stock_id), compute_trade_features(df_trade, stock_id),
                          on='row_id',
                          how='left')
        return pd.concat([df, df_tmp])

    df = Parallel(n_jobs=-1, verbose=1)(
        delayed(for_joblib)(stock_id) for stock_id in list_stock_ids
    )
    df = pd.concat(df, ignore_index=True)
    return df


def make_target_encoding(df_train, df_test):
    df_train['stock_id'] = df_train['row_id'].apply(lambda x: x.split('-')[0])
    df_test['stock_id'] = df_test['row_id'].apply(lambda x: x.split('-')[0])
    stock_id_target_mean = df_train.groupby('stock_id')['target'].mean()
    df_test['stock_id'].map(stock_id_target_mean)
    tmp = np.repeat(np.nan, df_train.shape[0])
    kf = KFold(n_splits=2, shuffle=True, random_state=19911109)  # these K-fold steps are confusing.
    for idx_1, idx_2 in kf.split(df_train):
        target_mean = df_train.iloc[idx_1].groupby('stock_id')['target'].mean()
        tmp[idx_2] = df_train['stock_id'].iloc[idx_2].map(target_mean)
    df_train['stock_id_target_enc'] = tmp

    return df_train, df_test
