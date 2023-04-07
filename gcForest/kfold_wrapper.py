#!usr/bin/env python
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

class kfold_wrapper(object):
    """
    K-Fold Wrapper
    """
    def __init__(self, estimator, est_fun, n_folds=None, fold_method=None, val_size=0.2):
        """
        Parameters
        ----------
        :param estimator: class
            Class of Estimator, must have Sklearn API
            
        :param est_fun: str
            Function of Estimator,
            If 'class', the estimator is treated as a classifier.
            If 'reg', the estimator is treated as a regressor.
        
        :param n_folds: int (default=None)
            Number of folds.
            If n_folds=1 or None, means no K-Fold
        
        :param fold_method: str (default=None)
            Method to Fold.
            If 'None', group cross_folds will be adopted.
            If 'group', the same method as 'None'.
            If 'sequence', serial n_folds will be adopted, 
                and param val_size will be useful.
        
        :param val_size: float (default=0.2)
            the ratio of valication in member of serial n_folds,
            and the length of valicaiton is rounded off.
        """
            
        setattr(self, 'estimator', estimator)
        setattr(self, 'est_fun', est_fun)
        if n_folds is None:
            setattr(self, 'n_folds', 1)
        else:
            setattr(self, 'n_folds', int(n_folds))
        if fold_method is None:
            setattr(self, 'fold_method', 'group')
        else:
            setattr(self, 'fold_method', fold_method)
        setattr(self, 'val_size', val_size)
        
        setattr(self, 'cv_pred_prob', None)
        
    def _data_group_foldindx(self, X, y):
        assert 2 <= len(X.shape) <= 3, "X.shape should be n x k or n x n2 x k"
        assert len(X.shape) == len(y.shape) + 1
        n_stratify = X.shape[0]
        fold_len = int(np.divide(n_stratify, self.n_folds))
        
        if self.n_folds==1:
            cv = [(np.arange(len(X)), np.arange(len(X)))]
        elif self.n_folds>1:
            groups = []
            for i in range(1, self.n_folds):
                groups = groups + list(i*np.ones(fold_len))
            groups = groups + list((i+1)*np.ones(n_stratify-i*fold_len))
            logo = LeaveOneGroupOut()
            cv = [(t, v) for (t, v) in logo.split(range(n_stratify), y, groups)]
        
        return cv
    
    def _data_series_foldindex(self, X):
        assert 2 <= len(X.shape) <= 3, "X.shape should be n x k or n x n2 x k"
        n_stratify = X.shape[0]
        test_len = n_stratify/(np.divide(1, self.val_size)+self.n_folds-1)
        test_len = int(test_len)
        
        if self.n_folds==1:
            cv = [(np.arange(len(X)), np.arange(len(X)))]
        elif self.n_folds>1:
            cv = []
            for i in range(self.n_folds-1):
                cv = cv + [(np.arange(n_stratify-test_len*(i+1)), np.arange(n_stratify-test_len*(i+1), n_stratify-test_len*i))]
                start_loc = n_stratify-test_len*(i+1)
            cv = cv + [( np.arange(start_loc, n_stratify), np.arange(start_loc) )]
        
        return cv.reverse()
    
    def fit(self, X, y, **kwargs):
        print('CV training ...')
        cv_index = [(np.arange(len(X)), np.arange(len(X)))]
        if self.fold_method == 'group':
            cv_index = self._data_group_foldindx(X, y)
        elif self.fold_method == 'sequence':
            cv_index = self._data_series_foldindex(X)
        
        cv_pred_prob = np.array([])
        for i in range(len(cv_index)):
            each_index = cv_index[i]
            X_train = X[each_index[0]]
            y_train = y[each_index[0]]
            X_test = X[each_index[1]]
            
            self.estimator.fit(X_train, y_train, **kwargs)
            if self.est_fun == 'class':
                each_prob = self.estimator.predict_proba(X_test)
            elif self.est_fun == 'reg':
                each_prob = self.estimator.predict(X_test)
            
            if i==0:
                cv_pred_prob = each_prob
            else:
                cv_pred_prob = np.concatenate( [cv_pred_prob, each_prob], axis=0 )
        self.cv_pred_prob = cv_pred_prob
        
        if i>0:
            print('Whole dataset training ...')
            self.estimator.fit(X, y, **kwargs)
        setattr(self, 'estimator', self.estimator)
        
    def predict_proba(self, X=None):
        if self.est_fun == 'class':
            return self.cv_pred_prob
        elif self.est_fun == 'reg':
            raise TypeError('Regressor has not predict_proba')
    
    def predict(self, X=None):
        if self.est_fun == 'class':
            pred_proba = self.predict_proba(X=X)
            return np.argmax(pred_proba, axis=1)
        elif self.est_fun == 'reg':
            return self.cv_pred_prob
    