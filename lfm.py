"""
__author__: josep ferrandiz
Note:
    1. Use lfm for sv
    2. Use new_lfm for FI
TODO: migrate sv to new_lfm
TODO: add overfit check to new_lfm
TODO: after all TODOs are completed. drop lfm and rename new lfm
"""
import numpy as np
import pandas as pd

# system skopt
from skopt import forest_minimize
from skopt.space import Real, Integer, Categorical

# local skopt
# from scikit_optimize.scikit_optimize.skopt import forest_minimize
# from scikit_optimize.scikit_optimize.skopt.space import Real, Integer, Categorical

from lightfm import LightFM
from lightfm.data import Dataset
from lightfm import cross_validation

from scipy import sparse
from config.logger import logger

N_JOBS = 6


class LFM(object):
    def __init__(self, data, test_data=None,
                 user_feature_cols=None, item_feature_cols=None,
                 test_frac=None, n_calls=None, verbose=False, min_gain=0.01, top_n=3):
        self.data = data
        self.test_frac = test_frac
        self.test_data = test_data

        self.user_feature_cols = list() if user_feature_cols is None else user_feature_cols
        self.item_feature_cols = list() if item_feature_cols is None else item_feature_cols

        if 'rating' not in self.data.columns:
            self.data['rating'] = 1.0

        uf_all, uf_data = self.build_features(user_feature_cols, 'user')
        if_all, if_data = self.build_features(item_feature_cols, 'item')

        self.dataset = Dataset()
        self.dataset.fit(self.data['user'], self.data['item'], item_features=if_all, user_features=uf_all)

        self.user_features = self.dataset.build_user_features(uf_data) if len(user_feature_cols) > 0 else None
        self.item_features = self.dataset.build_item_features(if_data) if len(item_feature_cols) > 0 else None

        # maps ###################################
        # uid_map = {..., user: user_idx, ...}
        # iid_map = {..., item: item_idx, ...}
        # ufeature_map = {..., 'ufeature_name:ufeature_value': user_feature_index, ...}
        # ifeature_map = {..., 'ifeature_name:ifeature_value': item_feature_index, ...}
        self.uid_map, self.ufeature_map, self.iid_map, self.ifeature_map = self.dataset.mapping()

        if self.test_data is not None:       # test_data may come from a time dependent cutoff
            if 'rating' not in self.test_data.columns:
                self.test_data['rating'] = 1.0

            # cold start check: drop users and items in test that are present in train
            # >>>>>>>>>>>>>> but we could still have a user that repeats items in test <<<<<<<<<<<<<<
            self.test_data = self.test_data[self.test_data['user'].isin(self.data['user'].unique())]  # only trained users in test: this should already be the case
            self.test_data = self.test_data[self.test_data['item'].isin(self.data['item'].unique())]  # only trained items in test: this should already be the case

        self.try_ctr = None
        self.n_calls = 10 if n_calls is None else max(n_calls, 10)
        self.verbose = verbose
        self.min_gain = min_gain                                 # min gains to consider improvement in hyper-pars
        self.patk = 0.0
        self.best_pars = None
        self.top_n = top_n
        self.fit_pars_list = list()
        self.train_interactions_df = None

        # set the params space
        self.user_alpha_exp = (-6, -1)   # if user_alpha is None else user_alpha
        self.item_alpha_exp = (-6, -1)   # if item_alpha is None else item_alpha
        self.space_ = [
            Integer(10, 200, name='no_components'),                                                                  # no_components
            Integer(1, 250, name='epochs'),                                                                          # epochs
            Real(10**-2, 10.0**0, name='learning_rate', prior='log-uniform'),                                        # learning_rate (adagrad is always better)
            Integer(5, 25, name='max_sampled'),                                                                      # max_sampled
            Real(10**self.item_alpha_exp[0], 10**self.item_alpha_exp[1], name='item_alpha', prior='log-uniform'),    # item_alpha
            Real(10**self.user_alpha_exp[0], 10**self.user_alpha_exp[1], name='user_alpha', prior='log-uniform'),    # user_alpha
            Categorical([0, 1], name='item_bias'),                                                                   # item_bias
            Categorical([0, 1], name='user_bias'),                                                                   # user_bias
            Categorical([0, 1, 2], name='rating'),                                                                   # rating type: 0: ratings = 1, 1 = linear time dependent ratings, 2 = geom time dependent ratings
            Categorical([0, 1], name='weight')                                                                       # popularity weight: 0: weight = 1, 1 = fw, 2 = geom time dependent ratings
        ]

    def get_params(self):
        return {
            'user_feature_cols': self.user_feature_cols,
            'item_feature_cols': self.item_feature_cols,
            'test_frac': self.test_frac, 'n_calls': self.n_calls,
            'verbose': self.verbose, 'min_gain': self.min_gain, 'top_n': self.top_n}

    def set_test_interactions_df(self, test_data):
        test_interactions_df = pd.pivot_table(test_data, index='user', columns='item', values='rating', aggfunc=sum)  # rating always 1 in test
        test_interactions_df.fillna(0.0, inplace=True)

        # set cols to LFM index id
        test_interactions_df.columns = [self.iid_map[c] for c in test_interactions_df.columns]

        # add missing cols to test_interactions DF
        new_cols = [c for c in self.train_interactions_df.columns if c not in test_interactions_df.columns]
        fc = pd.DataFrame(0, index=test_interactions_df.index, columns=new_cols)
        test_interactions_df = pd.concat([test_interactions_df, fc], axis=1)

        # add missing rows to test_interactions DF
        test_interactions_df.index = [self.uid_map[c] for c in test_interactions_df.index]
        new_idx = [c for c in self.train_interactions_df.index if c not in test_interactions_df.index]  # rows missing in test
        ft = pd.DataFrame(0, index=new_idx, columns=test_interactions_df.columns)
        fout = pd.concat([test_interactions_df, ft], axis=0)

        # set dtypes
        for c in fout.columns:
            fout[c] = fout[c].astype(np.int32)

        fout.sort_index(axis=0, inplace=True)
        fout.sort_index(axis=1, inplace=True)
        return fout

    def set_data(self, rating_idx=0, weight_idx=0):
        # set rating col for train data
        # compute train and test interactions
        if rating_idx == 1:
            rating_col = 'l_rating'
        elif rating_idx == 2:
            rating_col = 'g_rating'
        else:  # default: all 1
            rating_col = 'rating'
        if rating_col not in self.data.columns:
            logger.warning(rating_col + ' not available in data. Defaulting to rating')
            rating_col = 'rating'

        # set popularity penalty col
        if weight_idx == 1:
            weight_col = 's_weight'
        else:      # default: all 1
            weight_col = 'weight'
        if weight_col not in self.data.columns:
            logger.warning(weight_col + ' not available in data. Defaulting to weight')
            rating_col = 'weight'
        self.data['this_rating'] = self.data[rating_col] * self.data[weight_col]
        self.interactions, self.weights = self.dataset.build_interactions(self.data[['user', 'item', 'this_rating']].values)

        # set train and test interactions
        if self.test_data is not None:     # this may happen when the test/train split is time dependent
            # build test interactions
            self.train_interactions = sparse.coo_matrix(self.interactions)
            self.train_interactions_df = pd.DataFrame.sparse.from_spmatrix(self.interactions)
            self.train_interactions_df.sort_index(axis=0, inplace=True)
            self.train_interactions_df.sort_index(axis=1, inplace=True)
            self.test_interactions_df = self.set_test_interactions_df(self.test_data)
            self.test_interactions = sparse.coo_matrix(self.test_interactions_df.values) if len(self.test_interactions_df) > 0 else None

        # #################### NOT TESTED ##################################
        # ########################################################################
        # ########################################################################
        else:   # data contains test and train data. there may be cold start problem when building train and test interactions
            if 0.0 < self.test_frac < 1.0:
                self.train_interactions, self.test_interactions = cross_validation.random_train_test_split(self.interactions, test_percentage=self.test_frac)  # should ensure no overlaps
                self.test_interactions_df = pd.DataFrame.sparse.from_spmatrix(self.test_interactions)
                self.train_interactions.sort_index(axis=0, inplace=True)
                self.train_interactions.sort_index(axis=1, inplace=True)
                self.test_interactions.sort_index(axis=0, inplace=True)
                self.test_interactions.sort_index(axis=1, inplace=True)
            else:
                logger.error('LFM: test_frac must be between 0 and 1')
                self.train_interactions, self.test_interactions = self.interactions, None
        # ########################################################################
        # ########################################################################
        # ########################################################################

    def h_fit(self, ctr=0, with_features=True):
        self.try_ctr = ctr
        self.iter_ctr = 0
        max_ctr = 2

        # add features to space
        space = [x for x in self.space_]
        if with_features is True:
            if self.item_features is None:
                space = self.space_ + [Categorical([0], name='item_features')]
            else:
                space = self.space_ + [Categorical([0, 1], name='item_features')]
            if self.user_features is None:
                space += [Categorical([0], name='user_features')]
            else:
                space += [Categorical([0, 1], name='user_features')]

        self.par_names = [x.name for x in space]

        # find opt hyper-pars
        logger.info('lfm h-pars starts')
        if ctr < max_ctr:
            try:
                self.res = forest_minimize(self.score, space, n_calls=self.n_calls,
                                           random_state=1234 + ctr + 10 * int(with_features), verbose=self.verbose, n_jobs=N_JOBS)
                logger.info('LFM.h_fit:: skopt completed in ' + str(self.iter_ctr) +
                            ' iterations with patk: ' + str(self.patk))  # and parameters: \n' + str(self.best_pars))
            except ValueError as msg:
                logger.warning('LFM.h_fit:: skopt with_features: ' + str(with_features) + ' failed to converge after ' + str(self.iter_ctr) + ' iterations and ctr ' + str(ctr) +
                               ' with patk: ' + str(self.patk))
                logger.info(msg)
                self.h_fit(ctr=ctr + 1, with_features=with_features)  # try again
        logger.info('pars list: ' + str(self.fit_pars_list))

    def score(self, params):    # score to minimize
        # logger.info('h-pars: score compute for params: ' + str(params))
        fit_pars = {n: params[ix] for ix, n in enumerate(self.par_names)}
        u_features, i_features = self.fit(fit_pars)
        self.iter_ctr += 1
        patk = np.nan if self.test_interactions is None else self.score_(i_features, u_features, self.test_interactions_df)
        # logger.info('h-pars: score_ computed for params: ' + str(params))
        if np.isnan(patk):
            return 0.0
        else:        # record running best in case of failure in skopt. We only track improvements
            if self.patk == 0.0 or patk > self.patk * (1.0 + self.min_gain):
                self.patk = patk
                if self.patk > 0.0:
                    self.fit_pars_list.append((self.patk, fit_pars))           # list of increasingly better hyper-pars
                logger.info('LFM.score:: after ' + str(self.iter_ctr) + ' iterations: new patk: ' + str(self.patk))
            return -patk        # we are minimizing

    def score_(self, i_features, u_features, interactions_df):
        # use predict() to score so that we can include users that repeat items
        # interactions: index = user_index, cols = item_index
        # logger.info('h-pars: score_ compute starts')
        row_sums = interactions_df.sum(axis=1)
        f = interactions_df[row_sums > 0]                                    # f: only users that bought/used an item
        f.index = [c for c in interactions_df.index if row_sums.loc[c] > 0]  # only score user indices that are item buyers/users
        cols = f.columns                                                     # will score all items
        buys = f.apply(lambda x: np.array(x[x > 0].index), axis=1)           # list of columns with non-zero ratings for each index (user) shape(len(f), 1 --list of purchased items--)

        # predict for each user that bought/used an item his/her score for each item
        # np array of shape(len(f), 1 --list of items with top_n scores--)
        preds = f.apply(lambda x: np.argsort(-self.model.predict(np.array([x.name] * len(cols)), np.array(cols),
                                                                 item_features=i_features, user_features=u_features,
                                                                 num_threads=N_JOBS))[:self.top_n], axis=1)
        pf = pd.concat([buys, preds], axis=1)
        pf.columns = ['buy', 'buy_hat']  # index: user index
        pf['match'] = pf.apply(lambda x: len(set(x['buy']).intersection(set(x['buy_hat']))), axis=1)  # when match >= 1, a purchased item is in the top_n scores
        pk = pf['match'].sum() / len(pf)  # avg patk
        return pk

    def fit(self, fit_pars):
        self.set_data(rating_idx=fit_pars['rating'], weight_idx=fit_pars['weight'])
        self.model = LightFM(loss='warp', learning_schedule='adagrad',
                             learning_rate=fit_pars['learning_rate'],
                             no_components=fit_pars['no_components'],
                             user_alpha=fit_pars['user_alpha'],
                             item_alpha=fit_pars['item_alpha'],
                             max_sampled=fit_pars['max_sampled'],
                             random_state=1234
                             )

        u_features = None if fit_pars['user_features'] == 0 else self.user_features
        i_features = None if fit_pars['item_features'] == 0 else self.item_features
        self.model.fit(self.train_interactions,
                       sample_weight=self.weights,
                       item_features=i_features, user_features=u_features,
                       epochs=fit_pars['epochs'], num_threads=N_JOBS,
                       verbose=False)
        self.model.item_biases *= fit_pars['item_bias']
        self.model.user_biases *= fit_pars['user_bias']
        # logger.info('LFM obj data fit')
        return u_features, i_features

    def build_features(self, cols_, f_name_):
        """
        Build the data structures for user/item features
        :param cols_: cols with feature data
        :param f_name_: user or item
        :return: list of all unique user/item features and user/item input data for build_*_features
        """
        all_features_, feat_data = None, None
        if len(cols_) > 0:
            f_data = self.data[[f_name_] + cols_].drop_duplicates()
            dx = f_data.groupby(f_name_).apply(lambda x: [str(k) + ':' + str(v) for k, v in x[cols_].to_dict(orient='records')[0].items()])
            all_features_ = list(set([x for z in dx.to_list() for x in z]))
            feat_data = list(zip(dx.index, dx))
        return all_features_, feat_data





