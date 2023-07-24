import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import preprocesing
import features
import model
import model_diag

DATA_DIR = './dataOptiver/'

train, train_ids, test, test_ids = preprocesing.load_train_test_data(DATA_DIR)
df_train, df_test = features.make_features(train, train_ids, test, test_ids)

X = df_train.drop(['row_id', 'target'], axis=1)
y = df_train['target']

params = {
    "objective": "rmse",
    "metric": "rmse",
    "boosting_type": "gbdt",
    'early_stopping_rounds': 30,
    'learning_rate': 0.01,
    'lambda_l1': 1,
    'lambda_l2': 1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
}
kf = KFold(n_splits=5, random_state=19901028, shuffle=True)
oof = pd.DataFrame()  # out-of-fold result
models = []  # models
gain_importance_list = []
split_importance_list = []
score = model.training(X, y, params, kf, models, gain_importance_list, split_importance_list)

mean_gain_df = model_diag.calc_mean_importance(gain_importance_list)
model_diag.plot_importance(mean_gain_df, title='Model feature importance by gain')
mean_gain_df = mean_gain_df.reset_index().rename(columns={'index': 'feature_names'})
mean_gain_df.to_csv('gain_importance_mean.csv', index=False)

mean_split_df = model_diag.calc_mean_importance(split_importance_list)
model_diag.plot_importance(mean_split_df, title='Model feature importance by split')
mean_split_df = mean_split_df.reset_index().rename(columns={'index': 'feature_names'})
mean_split_df.to_csv('split_importance_mean.csv', index=False)

def testing():
    y_pred = df_test[['row_id']]
    X_test = df_test.drop(['time_id', 'row_id'], axis=1)
    target = np.zeros(len(X_test))

    # light gbm models
    for model in models:
        pred = model.predict(X_test[X.columns], num_iteration=model.best_iteration)
        target += pred / len(models)
    y_pred = y_pred.assign(target=target)
    return y_pred