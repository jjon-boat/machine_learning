#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:15:56 2019

@author: jiongwu

获取sklearn中所有的方法
"""

import sklearn

model_level_1 = getattr(sklearn,"__all__")

for each in model_level_1:
    func = locals()[each]
    print('>>>'+each+'\n'+'*-'*30)
    try:
        print(func.__all__,end='\n\n')
    except:
        print('\n')
    
    
    
"""
>>>calibration
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-


>>>cluster
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['AffinityPropagation', 'AgglomerativeClustering', 'Birch', 'DBSCAN', 'KMeans', 'FeatureAgglomeration', 'MeanShift', 'MiniBatchKMeans', 'SpectralClustering', 'affinity_propagation', 'dbscan', 'estimate_bandwidth', 'get_bin_seeds', 'k_means', 'linkage_tree', 'mean_shift', 'spectral_clustering', 'ward_tree', 'SpectralBiclustering', 'SpectralCoclustering']

>>>covariance
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['EllipticEnvelope', 'EmpiricalCovariance', 'GraphLasso', 'GraphLassoCV', 'LedoitWolf', 'MinCovDet', 'OAS', 'ShrunkCovariance', 'empirical_covariance', 'fast_mcd', 'graph_lasso', 'ledoit_wolf', 'ledoit_wolf_shrinkage', 'log_likelihood', 'oas', 'shrunk_covariance']

>>>cross_decomposition
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-


>>>cross_validation
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['KFold', 'LabelKFold', 'LeaveOneLabelOut', 'LeaveOneOut', 'LeavePLabelOut', 'LeavePOut', 'ShuffleSplit', 'StratifiedKFold', 'StratifiedShuffleSplit', 'PredefinedSplit', 'LabelShuffleSplit', 'check_cv', 'cross_val_score', 'cross_val_predict', 'permutation_test_score', 'train_test_split']

>>>datasets
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['clear_data_home', 'dump_svmlight_file', 'fetch_20newsgroups', 'fetch_20newsgroups_vectorized', 'fetch_lfw_pairs', 'fetch_lfw_people', 'fetch_mldata', 'fetch_olivetti_faces', 'fetch_species_distributions', 'fetch_california_housing', 'fetch_covtype', 'fetch_rcv1', 'fetch_kddcup99', 'get_data_home', 'load_boston', 'load_diabetes', 'load_digits', 'load_files', 'load_iris', 'load_breast_cancer', 'load_linnerud', 'load_mlcomp', 'load_sample_image', 'load_sample_images', 'load_svmlight_file', 'load_svmlight_files', 'load_wine', 'make_biclusters', 'make_blobs', 'make_circles', 'make_classification', 'make_checkerboard', 'make_friedman1', 'make_friedman2', 'make_friedman3', 'make_gaussian_quantiles', 'make_hastie_10_2', 'make_low_rank_matrix', 'make_moons', 'make_multilabel_classification', 'make_regression', 'make_s_curve', 'make_sparse_coded_signal', 'make_sparse_spd_matrix', 'make_sparse_uncorrelated', 'make_spd_matrix', 'make_swiss_roll', 'mldata_filename']

>>>decomposition
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['DictionaryLearning', 'FastICA', 'IncrementalPCA', 'KernelPCA', 'MiniBatchDictionaryLearning', 'MiniBatchSparsePCA', 'NMF', 'PCA', 'RandomizedPCA', 'SparseCoder', 'SparsePCA', 'dict_learning', 'dict_learning_online', 'fastica', 'non_negative_factorization', 'randomized_svd', 'sparse_encode', 'FactorAnalysis', 'TruncatedSVD', 'LatentDirichletAllocation']

>>>dummy
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-


>>>ensemble
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['BaseEnsemble', 'RandomForestClassifier', 'RandomForestRegressor', 'RandomTreesEmbedding', 'ExtraTreesClassifier', 'ExtraTreesRegressor', 'BaggingClassifier', 'BaggingRegressor', 'IsolationForest', 'GradientBoostingClassifier', 'GradientBoostingRegressor', 'AdaBoostClassifier', 'AdaBoostRegressor', 'VotingClassifier', 'bagging', 'forest', 'gradient_boosting', 'partial_dependence', 'weight_boosting']

>>>exceptions
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['NotFittedError', 'ChangedBehaviorWarning', 'ConvergenceWarning', 'DataConversionWarning', 'DataDimensionalityWarning', 'EfficiencyWarning', 'FitFailedWarning', 'NonBLASDotWarning', 'SkipTestWarning', 'UndefinedMetricWarning']

>>>externals
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-


>>>feature_extraction
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['DictVectorizer', 'image', 'img_to_graph', 'grid_to_graph', 'text', 'FeatureHasher']

>>>feature_selection
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['GenericUnivariateSelect', 'RFE', 'RFECV', 'SelectFdr', 'SelectFpr', 'SelectFwe', 'SelectKBest', 'SelectFromModel', 'SelectPercentile', 'VarianceThreshold', 'chi2', 'f_classif', 'f_oneway', 'f_regression', 'mutual_info_classif', 'mutual_info_regression']

>>>gaussian_process
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['GaussianProcess', 'correlation_models', 'regression_models', 'GaussianProcessRegressor', 'GaussianProcessClassifier', 'kernels']

>>>grid_search
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['GridSearchCV', 'ParameterGrid', 'fit_grid_point', 'ParameterSampler', 'RandomizedSearchCV']

>>>isotonic
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['check_increasing', 'isotonic_regression', 'IsotonicRegression']

>>>kernel_approximation
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-


>>>kernel_ridge
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-


>>>learning_curve
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['learning_curve', 'validation_curve']

>>>linear_model
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['ARDRegression', 'BayesianRidge', 'ElasticNet', 'ElasticNetCV', 'Hinge', 'Huber', 'HuberRegressor', 'Lars', 'LarsCV', 'Lasso', 'LassoCV', 'LassoLars', 'LassoLarsCV', 'LassoLarsIC', 'LinearRegression', 'Log', 'LogisticRegression', 'LogisticRegressionCV', 'ModifiedHuber', 'MultiTaskElasticNet', 'MultiTaskElasticNetCV', 'MultiTaskLasso', 'MultiTaskLassoCV', 'OrthogonalMatchingPursuit', 'OrthogonalMatchingPursuitCV', 'PassiveAggressiveClassifier', 'PassiveAggressiveRegressor', 'Perceptron', 'RandomizedLasso', 'RandomizedLogisticRegression', 'Ridge', 'RidgeCV', 'RidgeClassifier', 'RidgeClassifierCV', 'SGDClassifier', 'SGDRegressor', 'SquaredLoss', 'TheilSenRegressor', 'enet_path', 'lars_path', 'lasso_path', 'lasso_stability_path', 'logistic_regression_path', 'orthogonal_mp', 'orthogonal_mp_gram', 'ridge_regression', 'RANSACRegressor']

>>>manifold
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['locally_linear_embedding', 'LocallyLinearEmbedding', 'Isomap', 'MDS', 'smacof', 'SpectralEmbedding', 'spectral_embedding', 'TSNE']

>>>metrics
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['accuracy_score', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'auc', 'average_precision_score', 'calinski_harabaz_score', 'classification_report', 'cluster', 'cohen_kappa_score', 'completeness_score', 'confusion_matrix', 'consensus_score', 'coverage_error', 'euclidean_distances', 'explained_variance_score', 'f1_score', 'fbeta_score', 'fowlkes_mallows_score', 'get_scorer', 'hamming_loss', 'hinge_loss', 'homogeneity_completeness_v_measure', 'homogeneity_score', 'jaccard_similarity_score', 'label_ranking_average_precision_score', 'label_ranking_loss', 'log_loss', 'make_scorer', 'matthews_corrcoef', 'mean_absolute_error', 'mean_squared_error', 'mean_squared_log_error', 'median_absolute_error', 'mutual_info_score', 'normalized_mutual_info_score', 'pairwise_distances', 'pairwise_distances_argmin', 'pairwise_distances_argmin_min', 'pairwise_distances_argmin_min', 'pairwise_kernels', 'precision_recall_curve', 'precision_recall_fscore_support', 'precision_score', 'r2_score', 'recall_score', 'roc_auc_score', 'roc_curve', 'SCORERS', 'silhouette_samples', 'silhouette_score', 'v_measure_score', 'zero_one_loss', 'brier_score_loss']

>>>mixture
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['DPGMM', 'GMM', 'VBGMM', '_validate_covars', 'distribute_covar_matrix_to_match_covariance_type', 'log_multivariate_normal_density', 'sample_gaussian', 'GaussianMixture', 'BayesianGaussianMixture']

>>>model_selection
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
('BaseCrossValidator', 'GridSearchCV', 'TimeSeriesSplit', 'KFold', 'GroupKFold', 'GroupShuffleSplit', 'LeaveOneGroupOut', 'LeaveOneOut', 'LeavePGroupsOut', 'LeavePOut', 'RepeatedKFold', 'RepeatedStratifiedKFold', 'ParameterGrid', 'ParameterSampler', 'PredefinedSplit', 'RandomizedSearchCV', 'ShuffleSplit', 'StratifiedKFold', 'StratifiedShuffleSplit', 'check_cv', 'cross_val_predict', 'cross_val_score', 'cross_validate', 'fit_grid_point', 'learning_curve', 'permutation_test_score', 'train_test_split', 'validation_curve')

>>>multiclass
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['OneVsRestClassifier', 'OneVsOneClassifier', 'OutputCodeClassifier']

>>>multioutput
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['MultiOutputRegressor', 'MultiOutputClassifier', 'ClassifierChain']

>>>naive_bayes
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['BernoulliNB', 'GaussianNB', 'MultinomialNB']

>>>neighbors
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['BallTree', 'DistanceMetric', 'KDTree', 'KNeighborsClassifier', 'KNeighborsRegressor', 'NearestCentroid', 'NearestNeighbors', 'RadiusNeighborsClassifier', 'RadiusNeighborsRegressor', 'kneighbors_graph', 'radius_neighbors_graph', 'KernelDensity', 'LSHForest', 'LocalOutlierFactor']

>>>neural_network
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['BernoulliRBM', 'MLPClassifier', 'MLPRegressor']

>>>pipeline
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['Pipeline', 'FeatureUnion']

>>>preprocessing
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['Binarizer', 'FunctionTransformer', 'Imputer', 'KernelCenterer', 'LabelBinarizer', 'LabelEncoder', 'MultiLabelBinarizer', 'MinMaxScaler', 'MaxAbsScaler', 'QuantileTransformer', 'Normalizer', 'OneHotEncoder', 'RobustScaler', 'StandardScaler', 'add_dummy_feature', 'PolynomialFeatures', 'binarize', 'normalize', 'scale', 'robust_scale', 'maxabs_scale', 'minmax_scale', 'label_binarize', 'quantile_transform']

>>>random_projection
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['SparseRandomProjection', 'GaussianRandomProjection', 'johnson_lindenstrauss_min_dim']

>>>semi_supervised
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['LabelPropagation', 'LabelSpreading']

>>>svm
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['LinearSVC', 'LinearSVR', 'NuSVC', 'NuSVR', 'OneClassSVM', 'SVC', 'SVR', 'l1_min_c', 'liblinear', 'libsvm', 'libsvm_sparse']

>>>tree
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['DecisionTreeClassifier', 'DecisionTreeRegressor', 'ExtraTreeClassifier', 'ExtraTreeRegressor', 'export_graphviz']

>>>discriminant_analysis
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
['LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis']

>>>clone
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-


"""
    
    
    
    