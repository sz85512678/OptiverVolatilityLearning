import lightgbm as lgbm
import numpy as np
import model_diag


def training(X, y, lgb_params, kf, models, gain_importance_list, split_importance_list):
    scores = 0.0
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
        print("Fold :", fold + 1)
        X_train, y_train = X.loc[trn_idx], y[trn_idx]
        X_valid, y_valid = X.loc[val_idx], y[val_idx]
        lgbm_train = lgbm.Dataset(X_train, y_train, weight=1 / np.square(y_train))
        lgbm_valid = lgbm.Dataset(X_valid, y_valid, reference=lgbm_train, weight=1 / np.square(y_valid))

        model = lgbm.train(params=lgb_params,
                           train_set=lgbm_train,
                           valid_sets=[lgbm_train, lgbm_valid],
                           num_boost_round=5000,
                           feval=model_diag.feval_RMSPE,
                           categorical_feature=['stock_id']
                           )

        y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
        RMSPE = round(model_diag.rmspe(y_true=y_valid, y_pred=y_pred), 3)
        print(f'Performance of the　prediction: , RMSPE: {RMSPE}')
        scores += RMSPE / 5
        models.append(model)
        print("*" * 100)

        feature_names = X_train.columns.values.tolist()
        gain_importance_df = model_diag.calc_model_importance(
            model, feature_names=feature_names, importance_type='gain')
        gain_importance_list.append(gain_importance_df)

        split_importance_df = model_diag.calc_model_importance(
            model, feature_names=feature_names, importance_type='split')
        split_importance_list.append(split_importance_df)

    return scores
