"""
__author__: josep ferrandiz

"""
import sys
import os

import pandas as pd
import numpy as np

from config.logger import logger
from recommender.recommender import lfm as lfm
from joblib import Parallel, delayed

cwd = os.getcwd()
pdir = '/'.join(cwd.split('/')[:-1])
gpdir = '/'.join(cwd.split('/')[:-2])
sys.path = [gpdir] + [pdir] + [cwd] + sys.path


def overfit_check(lfm_obj, data, val_data=None):
    """
    checks that patk does not degrade too much on a new data set
    :param lfm_obj: LFM obj with optimal hyper-pars
    :param data: train data
    :param val_data: data to use to test performance
    :return: best pars for lfm and performance (precision)
    """
    lfm_pars = lfm_obj.get_params()
    new_lfm = lfm.LFM(data, test_data=val_data, **lfm_pars)

    if len(lfm_obj.fit_pars_list) == 0:   # patk = 0 for all tests!
        logger.error('could not find reco parameters. Defaulting to package defaults')
        lfm_obj.fit_pars_list = [(0.0, {'learning_schedule': 'adagrad', 'no_components': 10, 'learning_rate': 0.05, 'item_alpha': 0.0, 'user_alpha': 0.0, 'max_sampled': 10, 'random_state': None})]  # default

    fit_pars_list = sorted(lfm_obj.fit_pars_list, key=lambda x: -x[0])   # sort by decreasing patk
    best_pars_ = fit_pars_list[0]                                        # default
    patk = best_pars_[0]
    best_pars = best_pars_[1]
    for p_test, pars in fit_pars_list:
        user_features, item_features = new_lfm.fit(pars)
        p_val = new_lfm.score_(item_features, user_features, new_lfm.test_interactions_df)
        logger.info('test_patk: ' + str(np.round(p_test, 4)) + ' val_patk: ' + str(np.round(p_val, 4)) + ' pars: ' + str(pars))
        if p_val >= p_test / 2.0:
            best_pars = {k: v for k, v in pars.items()}
            patk = p_val
            break
    return best_pars, patk


def predict(data, val_data, mdl_pars, fit_pars, item_list=None, user_list=None, score_col='reco_col', num_threads=1):
    """
    predict reco scores
    :param data: train data DF
    :param val_data: if a non-empty DF, val_data is only used to ensure that no train/test split is performed internally.
                     Otherwise, it is not used.
                     If None, TBD????
                     -- this is a hack: there should be a cleaner way to do this!!!!!!!!!!!!!!!!!
    :param mdl_pars: pars used in LFM
    :param fit_pars: best fit pars for performance
    :param item_list: items to include. If None, all items in data
    :param user_list: users to include. If None, all users in data
    :param score_col: output col with reco score
    :param num_threads: number of threads
    :return: DF with columns user, item, reco score
    """
    lfm_obj = lfm.LFM(data, test_data=val_data, **mdl_pars)  # val_data not None ensure correct set up
    uid_map = lfm_obj.uid_map      # user to index
    uid_map = uid_map if user_list is None else {k: v for k, v in lfm_obj.uid_map.items() if k in user_list}  # user to index
    iid_map = lfm_obj.iid_map      # item to index
    iid_map = iid_map if item_list is None else {k: v for k, v in lfm_obj.iid_map.items() if k in item_list}  # user to index
    u_f = pd.DataFrame(pd.Series(uid_map)).reset_index()
    u_f.columns = ['user', 'uid']
    i_f = pd.DataFrame(pd.Series(iid_map)).reset_index()
    i_f.columns = ['item', 'iid']
    pred_df = u_f.merge(i_f, how='cross')
    user_features, item_features = lfm_obj.fit(fit_pars)
    pred_df[score_col] = lfm_obj.model.predict(user_ids=pred_df['uid'].astype(np.int32).values, item_ids=pred_df['iid'].astype(np.int32).values,
                                               user_features=user_features, item_features=item_features, num_threads=num_threads)
    logger.info('reco_df:: users: ' + str(pred_df['user'].nunique()) + ' items: ' + str(pred_df['item'].nunique()) + ' rows: ' + str(len(pred_df)))
    return pred_df[['user', 'item', score_col]].copy()


def set_h_pars(data, test_data=None,
               user_feature_cols=None, item_feature_cols=None,
               test_frac=0.2, n_calls=None, verbose=False,
               min_gain=0.01, top_n=3, with_features=True):
    """
    set hyper-parameters of lfm model
    :param data: train data DF
    :param test_data: test data DF. If None, test data is generated from data using LFM internal tools.
                      Otherwise, the data split is provided externally (e.g. it is time dependent): the data DF contains the train data and the test_data DF contains the test data.
    :param user_feature_cols: user feature cols in data
    :param item_feature_cols: item feature cols in data
    :param test_frac: test data fraction in the case tes_data is None
    :param n_calls: number of optimization calls
    :param verbose:
    :param min_gain: minimum gain from current best to adopt a new parameter set
    :param top_n: the k in precision @ k
    :param with_features: True: include features, False, no features
    :return: LFM obj with parameter sets tried
    """
    # set hyper-pars
    if user_feature_cols is None and item_feature_cols is None:
        with_features = False
    logger.info('hyper-params starts')
    lfm_obj = lfm.LFM(data, test_data=test_data, user_feature_cols=user_feature_cols, item_feature_cols=item_feature_cols,
                      test_frac=test_frac, n_calls=n_calls, verbose=verbose, min_gain=min_gain, top_n=top_n)
    logger.info('LFM obj created')
    lfm_obj.h_fit(with_features=with_features)         # fit hyper-parameters
    return lfm_obj
