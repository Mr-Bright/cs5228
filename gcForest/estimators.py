#!usr/bin/env python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor

class estimators(object):
    def __init__(self, is_customer=None):
        
        setattr(self, 'estimator_mgs_list', [])
        setattr(self, 'estimator_cascade_list', [])
        setattr(self, 'estimator_mgs_map', {})
        setattr(self, 'estimator_cascade_map', {})
        setattr(self, 'estimator_mgs_OOB', {})
        setattr(self, 'estimator_cascade_OOB', {})
        
    def make_estimator(self, estimator, name, est_type, is_OOB=False, is_add=True):
        """
        the estimator must have sciki-learn API
        """
        if not isinstance(name, str):
            raise ValueError('name must be an int, got %s' % type(name))
        
        if not is_add:
            if est_type == 'mgs':
                setattr(self, 'estimator_mgs_list', [])
                setattr(self, 'estimator_mgs_map', {})
                setattr(self, 'estimator_mgs_OOB', {})
            elif est_type == 'cascade':
                setattr(self, 'estimator_cascade_list', [])
                setattr(self, 'estimator_cascade_map', {})
                setattr(self, 'estimator_cascade_OOB', {})
                
        if est_type == 'mgs':
            self.estimator_mgs_list.append(name)
            self.estimator_mgs_map[name] = estimator
            self.estimator_mgs_OOB[name] = is_OOB
        elif est_type == 'cascade':
            self.estimator_cascade_list.append(name)
            self.estimator_cascade_map[name] = estimator
            self.estimator_cascade_OOB[name] = is_OOB
            
        
    def get_mgs_estimators(self):
        return self.estimator_mgs_list, self.estimator_mgs_map, self.estimator_mgs_OOB
    
    def get_cascade_estimators(self):
        return self.estimator_cascade_list, self.estimator_cascade_map, self.estimator_cascade_OOB



class classified_estimators(estimators):
    def __init__(self, n_mgsRFtree=100, n_cascadeRFtree=101, 
                 min_samples_mgs=0.1, min_samples_cascade=0.05, n_jobs=-1):
        
        super().__init__(estimators)
        
        prf_mgs = RandomForestClassifier(n_estimators=n_mgsRFtree, max_features='sqrt',
                                         min_samples_split=min_samples_mgs, bootstrap=True, oob_score=True, n_jobs=n_jobs)
        crf_mgs = RandomForestClassifier(n_estimators=n_mgsRFtree, max_features=1,
                                         min_samples_split=min_samples_mgs, bootstrap=True, oob_score=True, n_jobs=n_jobs)
        per_mgs = ExtraTreesClassifier(n_estimators=n_mgsRFtree, max_features='sqrt',
                                       min_samples_split=min_samples_mgs, bootstrap=True, oob_score=True, n_jobs=n_jobs)
        cer_mgs = ExtraTreesClassifier(n_estimators=n_mgsRFtree, max_features=0.5,
                                       min_samples_split=min_samples_mgs, bootstrap=True, oob_score=True, n_jobs=n_jobs)
        self.estimator_mgs_list = ['prf', 'crf', 'per', 'cer']
        self.estimator_mgs_map['prf'] = prf_mgs
        self.estimator_mgs_map['crf'] = crf_mgs
        self.estimator_mgs_map['per'] = per_mgs
        self.estimator_mgs_map['cer'] = cer_mgs
        self.estimator_mgs_OOB['prf'] = True
        self.estimator_mgs_OOB['crf'] = True
        self.estimator_mgs_OOB['per'] = True
        self.estimator_mgs_OOB['cer'] = True
        
        prf_cascade = RandomForestClassifier(n_estimators=n_cascadeRFtree, max_features='sqrt', min_samples_split=min_samples_cascade, 
                                             bootstrap=True, oob_score=True, n_jobs=n_jobs)
        crf_cascade = RandomForestClassifier(n_estimators=n_cascadeRFtree, max_features=1, min_samples_split=min_samples_cascade, 
                                             bootstrap=True, oob_score=True, n_jobs=n_jobs)
        per_cascade = ExtraTreesClassifier(n_estimators=n_cascadeRFtree, max_features='sqrt', min_samples_split=min_samples_cascade, 
                                           bootstrap=True, oob_score=True, n_jobs=n_jobs)
        cer_cascade = ExtraTreesClassifier(n_estimators=n_cascadeRFtree, max_features=0.5, min_samples_split=min_samples_cascade, 
                                           bootstrap=True, oob_score=True, n_jobs=n_jobs)
        self.estimator_cascade_list = ['prf', 'crf', 'per', 'cer']
        self.estimator_cascade_map['prf'] = prf_cascade
        self.estimator_cascade_map['crf'] = crf_cascade
        self.estimator_cascade_map['per'] = per_cascade
        self.estimator_cascade_map['cer'] = cer_cascade
        self.estimator_cascade_OOB['prf'] = True
        self.estimator_cascade_OOB['crf'] = True
        self.estimator_cascade_OOB['per'] = True
        self.estimator_cascade_OOB['cer'] = True


class regressive_estimators(estimators):
    def __init__(self, n_mgsRFtree=100, n_cascadeRFtree=101, mgs_criterion='mse',
                 cascade_criterion='mse', min_samples_mgs=0.01, min_samples_cascade=0.01, n_jobs=-1):
        
        super().__init__(estimators)
        #estimators.__init__(self)
        
        prf_mgs = RandomForestRegressor(n_estimators=n_mgsRFtree, max_features='sqrt', criterion=mgs_criterion,
                                        min_samples_split=min_samples_mgs, bootstrap=True, oob_score=True, n_jobs=n_jobs)
        crf_mgs = RandomForestRegressor(n_estimators=n_mgsRFtree, max_features='log2', criterion=mgs_criterion,
                                        min_samples_split=min_samples_mgs, bootstrap=True, oob_score=True, n_jobs=n_jobs)
        per_mgs = ExtraTreesRegressor(n_estimators=n_mgsRFtree, max_features='sqrt', criterion=mgs_criterion,
                                      min_samples_split=min_samples_mgs, bootstrap=True, oob_score=True, n_jobs=n_jobs)
        cer_mgs = ExtraTreesRegressor(n_estimators=n_mgsRFtree, max_features='log2', criterion=mgs_criterion,
                                      min_samples_split=min_samples_mgs, bootstrap=True, oob_score=True, n_jobs=n_jobs)
        self.estimator_mgs_list = ['prf', 'crf', 'per', 'cer']
        self.estimator_mgs_map['prf'] = prf_mgs
        self.estimator_mgs_map['crf'] = crf_mgs
        self.estimator_mgs_map['per'] = per_mgs
        self.estimator_mgs_map['cer'] = cer_mgs
        self.estimator_mgs_OOB['prf'] = True
        self.estimator_mgs_OOB['crf'] = True
        self.estimator_mgs_OOB['per'] = True
        self.estimator_mgs_OOB['cer'] = True
        
        prf_cascade = RandomForestRegressor(n_estimators=n_cascadeRFtree, max_features='sqrt', criterion=cascade_criterion,
                                            min_samples_split=min_samples_cascade, bootstrap=True, oob_score=True, n_jobs=n_jobs)
        crf_cascade = RandomForestRegressor(n_estimators=n_cascadeRFtree, max_features='log2', criterion=cascade_criterion,
                                            min_samples_split=min_samples_cascade, bootstrap=True, oob_score=True, n_jobs=n_jobs)
        per_cascade = ExtraTreesRegressor(n_estimators=n_cascadeRFtree, max_features='sqrt', criterion=cascade_criterion,
                                          min_samples_split=min_samples_cascade, bootstrap=True, oob_score=True, n_jobs=n_jobs)
        cer_cascade = ExtraTreesRegressor(n_estimators=n_cascadeRFtree, max_features='log2', criterion=cascade_criterion,
                                          min_samples_split=min_samples_cascade, bootstrap=True, oob_score=True, n_jobs=n_jobs)
        self.estimator_cascade_list = ['prf', 'crf', 'per', 'cer']
        self.estimator_cascade_map['prf'] = prf_cascade
        self.estimator_cascade_map['crf'] = crf_cascade
        self.estimator_cascade_map['per'] = per_cascade
        self.estimator_cascade_map['cer'] = cer_cascade
        self.estimator_cascade_OOB['prf'] = True
        self.estimator_cascade_OOB['crf'] = True
        self.estimator_cascade_OOB['per'] = True
        self.estimator_cascade_OOB['cer'] = True