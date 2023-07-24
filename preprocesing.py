import pandas as pd


def load_data(data_dir, stock_id, is_train=True):
    if is_train:
        file_path_book = data_dir + "book_train.parquet/stock_id=" + str(stock_id)
        file_path_trade = data_dir + "trade_train.parquet/stock_id=" + str(stock_id)
    else:
        file_path_book = data_dir + "book_test.parquet/stock_id=" + str(stock_id)
        file_path_trade = data_dir + "trade_test.parquet/stock_id=" + str(stock_id)
    return pd.read_parquet(file_path_book), pd.read_parquet(file_path_trade)


def load_train_test_data(data_dir):
    train = pd.read_csv(data_dir + 'train.csv')
    train_ids = train.stock_id.unique()
    train['row_id'] = train['stock_id'].astype(str) + '-' + train['time_id'].astype(str)
    train = train[['row_id', 'target']]

    test = pd.read_csv(data_dir + 'test.csv')
    test_ids = test.stock_id.unique()

    return train, train_ids, test, test_ids
