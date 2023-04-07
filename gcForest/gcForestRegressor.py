#!usr/bin/env python
import copy
import itertools
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

from .estimators import regressive_estimators
from .kfold_wrapper import kfold_wrapper

class GCForestRegressor(regressive_estimators):

    def __init__(self, shape_1X=None, n_mgsRFtree=30, window=None, stride=1, 
                 cascade_test_size=0.2, n_cascadeRF=2, n_cascadeRFtree=101, cascade_layer=np.inf, 
                 mgs_criterion='mse', cascade_criterion='mse', 
                 min_samples_mgs=0.1, min_samples_cascade=0.05, tolerance=0.0, n_jobs=-1, 
                 cv_method='group', n_mgs_cv=10, n_cascade_cv=10, cv_mgs_valSize=0.2, cv_cascade_valSize=0.2, 
                 scoring=None):
        """ GCForest Regression.

        :param shape_1X: int or tuple list or np.array (default=None)
            Shape of a single sample element [n_lines, n_cols]. Required when calling mg_scanning!
            For sequence data a single int can be given.

        :param n_mgsRFtree: int (default=30)
            Number of trees in a Random Forest during Multi Grain Scanning.

        :param window: int (default=None)
            List of window sizes to use during Multi Grain Scanning.
            If 'None' no slicing will be done.

        :param stride: int (default=1)
            Step used when slicing the data.

        :param cascade_test_size: float or int (default=0.2)
            Split fraction or absolute number for cascade training set splitting.

        :param n_cascadeRF: int (default=2)
            Number of estimators in a cascade layer. 
            Only useful when estimators can use out-of-bag samples.
            For each pseudo Random Forest a complete Random Forest is created, hence
            the total numbe of Random Forests in a layer will be 2*n_cascadeRF.

        :param n_cascadeRFtree: int (default=101)
            Number of trees in a single Random Forest in a cascade layer.
        
        :param cascade_layer: int (default=np.inf)
            mMximum number of cascade layers allowed.
            Useful to limit the contruction of the cascade.
        
        :param min_samples_mgs: float or int (default=0.1)
            Minimum number of samples in a node to perform a split
            during the training of Multi-Grain Scanning Random Forest.
            If int number_of_samples = int.
            If float, min_samples represents the fraction of the initial n_samples to consider.

        :param min_samples_cascade: float or int (default=0.05)
            Minimum number of samples in a node to perform a split
            during the training of Cascade Random Forest.
            If int number_of_samples = int.
            If float, min_samples represents the fraction of the initial n_samples to consider.
            
        :param tolerance: float (default=0.0)
            Accuracy tolerance for the casacade growth.
            If the improvement in accuracy is not better than the tolerance the construction is
            stopped.
        
        :param cv_method: str (default='group')
            The method of cross-validation
            If 'group', CV will use LeaveOneGroupOut in sklearn.model_selection.
            If 'sequence', CV will be processed as time series.
            Orther methods will be added in new version.
        
        :param n_mgs_cv: int (default=10)
            Number of folds of CV in a estimator during Multi Grain Scanning.
            Only useful when the estimator can use out-of-bag samples.
        
        :param n_cascade_cv: int (default=10)
            Number of folds of CV in a estimator in a cascade layer.
            Only useful when the estimator can use out-of-bag samples.
        
        :param cv_mgs_valSize: float (default=0.2)
            During Multi Grain Scanning, the ratio of valication in member of serial n_folds.
            And the length of valicaiton is rounded off.
        
        :param cv_cascade_valSize: float (default=0.2)
            In a cascade layer, the ratio of valication in member of serial n_folds.
            And the length of valicaiton is rounded off.
        
        :param n_jobs: int (default=-1)
            The number of jobs to run in parallel for any Random Forest fit and predict.
            If -1, then the number of jobs is set to the number of cores.
        
        :param scoring: str (default=None)
            The method of score evaluation, 
            Here use classification metrics.
            Best value at 1 and worst score at 0 for All scores.
            In addition, you can input a score function that can be recognized by sklearn.
        """
        
        setattr(self, 'shape_1X', shape_1X)
        setattr(self, 'n_layer', 0)
        setattr(self, '_n_samples', 0)
        setattr(self, 'n_cascadeRF', int(n_cascadeRF))
        if isinstance(window, int):
            setattr(self, 'window', [window])
        elif isinstance(window, list):
            setattr(self, 'window', window)
        elif window is None:
            setattr(self, 'window', window)
        else:
            raise ValueError('Param window cannot get ', window)
        setattr(self, 'stride', stride)
        setattr(self, 'cascade_test_size', cascade_test_size)
        setattr(self, 'n_mgsRFtree', int(n_mgsRFtree))
        setattr(self, 'n_cascadeRFtree', int(n_cascadeRFtree))
        setattr(self, 'cascade_layer', cascade_layer)
        setattr(self, 'mgs_criterion', mgs_criterion)
        setattr(self, 'cascade_criterion', cascade_criterion)
        setattr(self, 'min_samples_mgs', min_samples_mgs)
        setattr(self, 'min_samples_cascade', min_samples_cascade)
        setattr(self, 'tolerance', tolerance)
        setattr(self, 'n_jobs', n_jobs)
        
        super(GCForestRegressor, self).__init__(n_mgsRFtree=self.n_mgsRFtree, 
             mgs_criterion=self.mgs_criterion, cascade_criterion=self.cascade_criterion,
             n_cascadeRFtree=self.n_cascadeRFtree, min_samples_mgs=self.min_samples_mgs, 
             min_samples_cascade=self.min_samples_cascade, n_jobs=self.n_jobs)
        
        setattr(self, 'kfold_wrapper', kfold_wrapper)
        setattr(self, 'cv_method', cv_method)
        setattr(self, 'n_mgs_cv', n_mgs_cv)
        setattr(self, 'n_cascade_cv', n_cascade_cv)
        setattr(self, 'cv_mgs_valSize', cv_mgs_valSize)
        setattr(self, 'cv_cascade_valSize', cv_cascade_valSize)
        
        if scoring is None or scoring == 'r2':
            setattr(self, 'scoring', r2_score)
        elif scoring == 'explained_variance':
            setattr(self, 'scoring', explained_variance_score)
        else:
            setattr(self, 'scoring', scoring)
        

    def fit(self, X, y):
        """ Training the gcForest on input data X and associated target y.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param y: np.array
            1D array containing the target values.
            Must be of shape [n_samples]
        """
        if np.shape(X)[0] != len(y):
            raise ValueError('Sizes of y and X do not match.')

        mgs_X = self.mg_scanning(X, y)
        _ = self.cascade_forest(mgs_X, y)

    def predict(self, X):
        """ Predict the regression value of unknown samples X.

        :param X: np.array
            Array containing the input samples.
            Must be of the same shape [n_samples, data] as the training inputs.

        :return: np.array
            1D array containing the predicted regression for each input sample.
        """
        mgs_X = self.mg_scanning(X)
        cascade_all_pred = self.cascade_forest(mgs_X)
        predict = np.mean(cascade_all_pred, axis=0)

        return predict

    def mg_scanning(self, X, y=None):
        """ Performs a Multi Grain Scanning on input data.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param y: np.array (default=None)

        :return: np.array
            Array of shape [n_samples, .. ] containing Multi Grain Scanning sliced data.
        """
        setattr(self, '_n_samples', np.shape(X)[0])
        shape_1X = getattr(self, 'shape_1X')
        if isinstance(shape_1X, int):
            shape_1X = [1,shape_1X]
        if not getattr(self, 'window'):
            setattr(self, 'window', [shape_1X[1]])

        mgs_pred = []

        for wdw_size in getattr(self, 'window'):
            wdw_pred = self.window_slicing_pred(X, wdw_size, shape_1X, y=y)
            mgs_pred.append(wdw_pred)

        return np.concatenate(mgs_pred, axis=1)

    def window_slicing_pred(self, X, window, shape_1X, y=None):
        """ Performs a window slicing of the input data and send them through Estimators.
        If target values 'y' are provided sliced data are then used to train the Estimators.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param window: int
            Size of the window to use for slicing.

        :param shape_1X: list or np.array
            Shape of a single sample.

        :param y: np.array (default=None)
            Target values. If 'None' no training is done.

        :return: np.array
            Array of size [n_samples, ..] containing the Estimators.
            prediction for each input sample.
        """
        stride = getattr(self, 'stride')

        if shape_1X[0] > 1:
            print('Slicing Images...')
            sliced_X, sliced_y = self._window_slicing_img(X, window, shape_1X, y=y, stride=stride)
        else:
            print('Slicing Sequence...')
            sliced_X, sliced_y = self._window_slicing_sequence(X, window, shape_1X, y=y, stride=stride)
            
        mgs_list, mgs_estimators, mgs_OOB = self.get_mgs_estimators()
        for i, mgs in enumerate(mgs_list):
            if y is not None:
                #the estimator must has sklearn API,and can work with function predict
                estimator = mgs_estimators[mgs]
                print('Training MGS Model: ', mgs)
                
                if mgs_OOB[mgs]:
                    estimator.fit(sliced_X, sliced_y)
                    setattr(self, '_mgs%s_%d'%(mgs, window), copy.deepcopy(estimator))
                    pred_est = estimator.oob_prediction_
                elif not mgs_OOB[mgs]:
                    estimator_cv = self.kfold_wrapper(estimator, est_fun='reg', n_folds=self.n_mgs_cv, 
                                                      fold_method=self.cv_method, val_size=self.cv_mgs_valSize)                    
                    estimator_cv.fit(sliced_X, sliced_y)
                    setattr(self, '_mgs%s_%d'%(mgs, window), copy.deepcopy(estimator_cv.estimator))
                    pred_est = estimator_cv.cv_pred_prob
            
            if hasattr(self, '_mgs%s_%d'%(mgs, window)) and y is None:
                estimator = getattr(self, '_mgs%s_%d'%(mgs, window))
                pred_est = estimator.predict(sliced_X)
            
            if i==0:
                pred = pred_est
            elif i>0:
                pred = np.c_[pred, pred_est]
                
        return pred.reshape([getattr(self, '_n_samples'), -1])

    def _window_slicing_img(self, X, window, shape_1X, y=None, stride=1):
        """ Slicing procedure for images

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param window: int
            Size of the window to use for slicing.

        :param shape_1X: list or np.array
            Shape of a single sample [n_lines, n_cols].

        :param y: np.array (default=None)
            Target values.

        :param stride: int (default=1)
            Step used when slicing the data.

        :return: np.array and np.array
            Arrays containing the sliced images and target values (empty if 'y' is None).
        """
        if any(s < window for s in shape_1X):
            raise ValueError('window must be smaller than both dimensions for an image')

        len_iter_x = np.floor_divide((shape_1X[1] - window), stride) + 1
        len_iter_y = np.floor_divide((shape_1X[0] - window), stride) + 1
        iterx_array = np.arange(0, stride*len_iter_x, stride)
        itery_array = np.arange(0, stride*len_iter_y, stride)

        ref_row = np.arange(0, window)
        ref_ind = np.ravel([ref_row + shape_1X[1] * i for i in range(window)])
        inds_to_take = [ref_ind + ix + shape_1X[1] * iy
                        for ix, iy in itertools.product(iterx_array, itery_array)]

        sliced_imgs = np.take(X, inds_to_take, axis=1).reshape(-1, window**2)

        if y is not None:
            sliced_target = np.repeat(y, len_iter_x * len_iter_y)
        elif y is None:
            sliced_target = None

        return sliced_imgs, sliced_target

    def _window_slicing_sequence(self, X, window, shape_1X, y=None, stride=1):
        """ Slicing procedure for sequences (aka shape_1X = [.., 1]).

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param window: int
            Size of the window to use for slicing.

        :param shape_1X: list or np.array
            Shape of a single sample [n_lines, n_col].

        :param y: np.array (default=None)
            Target values.

        :param stride: int (default=1)
            Step used when slicing the data.

        :return: np.array and np.array
            Arrays containing the sliced sequences and target values (empty if 'y' is None).
        """
        if shape_1X[1] < window:
            raise ValueError('window must be smaller than the sequence dimension')

        len_iter = np.floor_divide((shape_1X[1] - window), stride) + 1
        iter_array = np.arange(0, stride*len_iter, stride)

        ind_1X = np.arange(np.prod(shape_1X))
        inds_to_take = [ind_1X[i:i+window] for i in iter_array]
        sliced_sqce = np.take(X, inds_to_take, axis=1).reshape(-1, window)

        if y is not None:
            sliced_target = np.repeat(y, len_iter)
        elif y is None:
            sliced_target = None

        return sliced_sqce, sliced_target

    def cascade_forest(self, X, y=None):
        """ Perform (or train if 'y' is not None) a cascade forest estimator or other customer estimators.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param y: np.array (default=None)
            Target values. If 'None' perform training.

        :return: np.array
            1D array containing the predicted class for each input sample.
        """
        if y is not None:
            setattr(self, 'n_layer', 0)
            test_size = getattr(self, 'cascade_test_size')
            max_layers = getattr(self, 'cascade_layer')
            tol = getattr(self, 'tolerance')

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

            self.n_layer += 1
            print('n_layer: ', self.n_layer, '   ', 'X_train shape: ', X_train.shape)
            prf_crf_pred_ref = self._cascade_layer(X_train, y_train)
            print('n_layer: ', self.n_layer, '   ', 'prf_crf_pred_ref length: ', len(prf_crf_pred_ref))
            accuracy_ref = self._cascade_evaluation(X_test, y_test)
            feat_arr = self._create_feat_arr(X_train, prf_crf_pred_ref)
            print('n_layer: ', self.n_layer, '   ', 'feat_arr shape: ', feat_arr.shape)
            
            self.n_layer += 1
            prf_crf_pred_layer = self._cascade_layer(feat_arr, y_train)
            print('n_layer: ', self.n_layer, '   ', 'prf_crf_pred_layer length: ', len(prf_crf_pred_layer))
            accuracy_layer = self._cascade_evaluation(X_test, y_test)

            while accuracy_layer > (accuracy_ref + tol) and self.n_layer <= max_layers:
                accuracy_ref = accuracy_layer
                prf_crf_pred_ref = prf_crf_pred_layer
                feat_arr = self._create_feat_arr(X_train, prf_crf_pred_ref)
                self.n_layer += 1
                prf_crf_pred_layer = self._cascade_layer(feat_arr, y_train)
                accuracy_layer = self._cascade_evaluation(X_test, y_test)

            if accuracy_layer < accuracy_ref :
                n_cascadeRF = getattr(self, 'n_cascadeRF')
                cas_list, _, cas_OOB = self.get_cascade_estimators()
                for irf in range(n_cascadeRF):
                    for i, cas in enumerate(cas_list):
                        if cas_OOB[cas]:
                            delattr(self, '_cas%s_%d_%d'%(cas, self.n_layer, irf))
                        elif not cas_OOB[cas] and irf==0:
                            delattr(self, '_cas%s_%d_%d'%(cas, self.n_layer, irf))
                self.n_layer -= 1

        elif y is None:
            at_layer = 1
            prf_crf_pred_ref = self._cascade_layer(X, layer=at_layer)
            while at_layer < getattr(self, 'n_layer'):
                at_layer += 1
                feat_arr = self._create_feat_arr(X, prf_crf_pred_ref)
                prf_crf_pred_ref = self._cascade_layer(feat_arr, layer=at_layer)

        return prf_crf_pred_ref

    def _cascade_layer(self, X, y=None, layer=0):
        """ Cascade layer containing Random Forest or/and orther estimators .
        If y is not None the layer is trained.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param y: np.array (default=None)
            Target values. If 'None' perform training.

        :param layer: int (default=0)
            Layer indice. Used to call the previously trained layer.

        :return: list
            List containing the prediction probabilities for all samples.
        """
        
        n_cascadeRF = getattr(self, 'n_cascadeRF')
        
        cas_list, cas_estimators, cas_OOB = self.get_cascade_estimators()
        est_pred = []
        if y is not None:
            print('Adding/Training Layer, n_layer={}'.format(self.n_layer))
            
            for irf in range(n_cascadeRF):
                for i, cas in enumerate(cas_list):
                    estimator = cas_estimators[cas]
                    
                    if cas_OOB[cas]:
                        estimator.fit(X, y)
                        setattr(self, '_cas%s_%d_%d'%(cas, self.n_layer, irf), copy.deepcopy(estimator))
                        est_pred.append(estimator.oob_prediction_)
                    elif not cas_OOB[cas] and irf==0:
                        estimator_cv = self.kfold_wrapper(estimator, est_fun='class', n_folds=self.n_cascade_cv, 
                                                          fold_method=self.cv_method, val_size=self.cv_cascade_valSize)
                        estimator_cv.fit(X, y)
                        setattr(self, '_cas%s_%d_%d'%(cas, self.n_layer, irf), copy.deepcopy(estimator_cv.estimator))
                        est_pred.append(estimator_cv.cv_pred_prob)
        
        elif y is None:
            for irf in range(n_cascadeRF):
                for i, cas in enumerate(cas_list):
                    if cas_OOB[cas]:
                        estimator = getattr(self, '_cas%s_%d_%d'%(cas, layer, irf))
                        est_pred.append( estimator.predict(X) )
                    elif not cas_OOB[cas] and irf==0:
                        estimator = getattr(self, '_cas%s_%d_%d'%(cas, layer, irf))
                        est_pred.append( estimator.predict(X) )
        
        return est_pred

    def _cascade_evaluation(self, X_test, y_test):
        """ Evaluate the accuracy of the cascade using X and y.

        :param X_test: np.array
            Array containing the test input samples.
            Must be of the same shape as training data.

        :param y_test: np.array
            Test target values.

        :return: float
            the cascade accuracy.
        """
        casc_pred = np.mean(self.cascade_forest(X_test), axis=0)
        casc_score= self.scoring(y_test, casc_pred)
        print('Layer validation score = {}'.format(casc_score))

        return casc_score

    def _create_feat_arr(self, X, est_preds):
        """ Concatenate the original feature vector with the predicition probabilities
        of a cascade layer.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param est_preds: list
            Prediction probabilities by a cascade layer for X.

        :return: np.array
            Concatenation of X and the prediction.
            To be used for the next layer in a cascade forest.
        """
        swap_pred = np.swapaxes(est_preds, 0, 1)
        add_feat = swap_pred.reshape([np.shape(X)[0], -1])
        feat_arr = np.concatenate([add_feat, X], axis=1)

        return feat_arr
