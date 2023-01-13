#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 18:30:02 2022

@author: bmondal
"""

import numpy as np
from sklearn.svm import SVR, SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, cross_validate
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.metrics import r2_score, mean_squared_error, classification_report, \
    accuracy_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error,max_error
    
from ActiveLearningGeneralFunctions import GenerateBadSamples, GenerateBadSamples_SVRSVC
import time, os, glob
import inspect, pickle
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3 as sq
#%% ###########################################################################
def CalculateAccuracyFromLabels(y_prediction_whole_space,PredictProb=False):
    '''
    ==> accuracy_score(nature_prediction_mode_models,nature_prediction_model) # for each sample
    accuracy:=mean_model(1(y_prediction=mode_model(y_prediction)))
    '''
    if PredictProb:
        TMP = pd.DataFrame()
        for I in y_prediction_whole_space.columns.get_level_values('lv11').unique():
            TMP[I] = y_prediction_whole_space.loc[(slice(None),pd.IndexSlice[:,I])].mean(axis=1)
        return TMP.max(axis=1)
    else:  
        y_prediction_whole_space['MODE'] = y_prediction_whole_space.mode(axis=1).iloc[:, 0].astype(int) # Note: if mode is 50:50 the tag is 0 (==direct).
        return y_prediction_whole_space.apply(lambda a: np.mean([1 if val==a[-1] else 0 for val in a[:-1]]), axis=1)

def CheckVersions():
    # Check the versions of libraries
    print("The model was created using the following library versions. Please make\
        sure they matches, else there may be error.")
    # Python version
    import sys
    print('Python: 3.8.3; Your version is: {}'.format(sys.version))
    # scipy
    import scipy
    print('scipy: 1.6.3; Your version is: {}'.format(scipy.__version__))
    # numpy
    import numpy
    print('numpy: 1.20.3; Your version is: {}'.format(numpy.__version__))
    # matplotlib
    import matplotlib
    print('matplotlib: 3.4.3; Your version is: {}'.format(matplotlib.__version__))
    # pandas
    import pandas
    print('pandas: 1.2.3; Your version is: {}'.format(pandas.__version__))
    # scikit-learn
    import sklearn
    print('sklearn: 0.24.1; Your version is: {}'.format(sklearn.__version__))

def convert_conc2atomnumber(concsrray,natoms=216):
    N_atom = (concsrray[['INDIUM']]*(natoms/100)).round(0).astype(int)
    N_atom['GALLIUM'] = 216 - N_atom['INDIUM']
    N_atom[['PHOSPHORUS', 'ARSENIC']] = (concsrray[['PHOSPHORUS', 'ARSENIC']]*2.16).round(0).astype(int)
    N_atom['ANTIMONY'] = 216-N_atom[['PHOSPHORUS', 'ARSENIC']].sum(axis=1)
    N_atom['STRAIN'] = concsrray['STRAIN']
    return N_atom

#%%% ==========================================================================
def define_default_model(SVRModel=False,SVCmodel=False,PredictProb=False):
    # Define a Standard Scaler to normalize inputs
    scaler = StandardScaler()
    tol = 1e-5;cache_size=10000
    if SVRModel:
        svm_model = SVR(kernel="rbf",tol=tol,cache_size=cache_size)
    elif SVCmodel:
        svm_model = SVC(kernel="rbf",tol=tol,cache_size=cache_size,probability=PredictProb)
    else:
        exit
    # Set pipeline
    pipe = Pipeline(steps=[("scaler", scaler), ("svm", svm_model)])
    return pipe

def ModelParameterGrid():
    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    C_range = [1e0, 1e1, 50, 1e2, 500, 1e3]
    gamma_range = [1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 5.e+01, 1.e+02] #np.logspace(-2, 2, 5)
    param_grid={"svm__C": C_range, 
                "svm__gamma": gamma_range}
    
    return param_grid
#%%% ==========================================================================
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py
# def print_dataframe_v1(filtered_cv_results):
#     """Pretty print for filtered dataframe"""
#     for mean_precision, std_precision, mean_recall, std_recall, params in zip(
#         filtered_cv_results["mean_test_precision"],
#         filtered_cv_results["std_test_precision"],
#         filtered_cv_results["mean_test_recall"],
#         filtered_cv_results["std_test_recall"],
#         filtered_cv_results["params"],
#     ):
#         print(
#             f"precision: {mean_precision:0.3f} (±{std_precision:0.03f}),"
#             f" recall: {mean_recall:0.3f} (±{std_recall:0.03f}),"
#             f" for {params}"
#         )
#     print()

# def print_dataframe(filtered_cv_results):
#     """Pretty print for filtered dataframe"""
#     for mean_precision, std_precision, mean_recall, std_recall, params in zip(
#         filtered_cv_results["mean_test_r2"],
#         filtered_cv_results["std_test_r2"],
#         filtered_cv_results["mean_test_neg_mean_absolute_error"],
#         filtered_cv_results["std_test_neg_mean_absolute_error"],
#         filtered_cv_results["params"],
#     ):
#         print(
#             f"r2 score: {mean_precision:0.3f} (±{std_precision:0.03f}),"
#             f"mean_absolute_error: {mean_recall:0.3f} (±{std_recall:0.03f}),"
#             f" for {params}"
#         )
#     print()
    
def collect_best_m_models_dynamic(cv_results, scoringfn, precision_threshold_default=0.80, n_sigma=1, m_models_default=10):
    """Define the strategy to select the best estimator.

    The strategy defined here is to filter-out all results below a 1st scoringfn threshold, 
    rank the remaining by 2nd scoringfn and keep all models with one standard
    deviation of the best by 2nd scoringfn. Once these models are selected, we can select the
    fastest model to predict.

    Parameters
    ----------
    cv_results : dict of numpy (masked) ndarrays
        CV results as returned by the `GridSearchCV`.

    Returns
    -------
    best_index : int
        The index of the best estimator as it appears in `cv_results`.
    """
    ##### print the info about the grid-search for the different scores
    cv_results_ = pd.DataFrame(cv_results)
    # print("All grid-search results:")
    # print(cv_results)

    #### Filter-out all results below the threshold
    precision_threshold = min(precision_threshold_default, cv_results_["mean_test_"+scoringfn[0]].mean())
    high_precision_cv_results = cv_results_[cv_results_["mean_test_"+scoringfn[0]] > precision_threshold]
    
    # print(f"Models with a precision higher than {precision_threshold:.3f}::")
    # print(f"{high_precision_cv_results[['mean_test_'+scoringfn[0], 'mean_test_'+scoringfn[1]]]}")
    # high_precision_cv_results["mean_test_"+scoringfn[1]].plot.hist()
    # print_dataframe(high_precision_cv_results)

    high_precision_cv_results = high_precision_cv_results[
        [
            "mean_score_time",
            "mean_test_"+scoringfn[0],
            "std_test_"+scoringfn[0],
            "mean_test_"+scoringfn[1],
            "std_test_"+scoringfn[1],
            "rank_test_"+scoringfn[0],
            "rank_test_"+scoringfn[1],
            "params",
        ]
    ]

    ##### Select the most performant models in terms of 2nd scoring fn
    ##### (within n_sigma from the best)
    best_models_std = high_precision_cv_results["mean_test_"+scoringfn[1]].std()
    best_models = high_precision_cv_results["mean_test_"+scoringfn[1]].max()        
    best_models_threshold = best_models - n_sigma*best_models_std        
    # print(f"Threshold to choose models within n_sigma = {best_models_threshold}.")
    high_recall_cv_results = high_precision_cv_results[
        high_precision_cv_results["mean_test_"+scoringfn[1]] > best_models_threshold
    ]
    # print(
    #     "Out of the previously selected high precision models, we keep all the\n"
    #     "the models within one standard deviation of the highest mean_absolute_error model:"
    # )
    # print_dataframe(high_recall_cv_results)
    
    ##### From the best candidates, select the fastest m model to predict
    m_models = min(m_models_default, len(high_recall_cv_results))
    print(f'Number of independent models = {m_models}')
    # if m_models < m_models_default: 
    #     print(f'>> Final # of independent models = {m_models}')
        # print(f'>> Warning: # of independent models generated = {m_models}. This is less than requested (= {m_models_default}).' ) 
        # print('>> Note: This could be because of all the models perform similarly. Or not enough # of hyperparameters combinations.')
        # print(f'>> Total # of hyperparameters combinations found = {len(cv_results_["params"])}.')
    fastest_top_m_mean_absolute_error_high_precision_index = high_recall_cv_results.nsmallest(m_models,"mean_score_time").index.tolist()
    
    return fastest_top_m_mean_absolute_error_high_precision_index
    
    # # From the best candidates, select the fastest model to predict
    # fastest_top_mean_absolute_error_high_precision_index = high_recall_cv_results[
    #     "mean_score_time"
    # ].idxmin()

    # print(
    #     "\nThe selected final model is the fastest to predict out of the previously\n"
    #     "selected subset of best models based on r2_score and mean_absolute_error.\n"
    #     "Its scoring time is:\n\n"
    #     f"{high_recall_cv_results.loc[fastest_top_mean_absolute_error_high_precision_index]}"
    # )

    # return fastest_top_mean_absolute_error_high_precision_index
    
def collect_best_m_models_static(cv_results, scoringfn, m_models_default=10):
    """Define the strategy to select the best estimator.

    The strategy defined here is to filter best m models  to predict based on scoringfn.

    Parameters
    ----------
    cv_results : dict of numpy (masked) ndarrays
        CV results as returned by the `GridSearchCV`.

    Returns
    -------
    best_index : int
        The index of the best estimator as it appears in `cv_results`.
    """

    cv_results_ = pd.DataFrame(cv_results)
    m_models = min(m_models_default, len(cv_results_["params"]))
    if m_models < m_models_default: 
        print(f'>> Warning: # of independent models could be generated = {m_models}. This is less than requested (= {m_models_default}).' ) 
        print('>> Not enough # of hyperparameters combinations available.')

    best_m_model_indices = cv_results_.nlargest(m_models, "mean_test_"+scoringfn).index.tolist() 
    return best_m_model_indices

def m_bestmodel_parametersearch(X_train, X_test, y_train, y_test, X_predict,
                                random_state=None,m_models=10,
                                precision_threshold=0.98,Static_m_Models=False,
                                scoringfn=['r2','neg_mean_absolute_error'], njobs=-1,
                                SVRModel=False,SVCmodel=False,PredictProb=False):
    if SVCmodel: scoringfn= ['accuracy','balanced_accuracy'] #['precision','recall']
    assert len(scoringfn) == 2, 'The scoring function should be list of 2 metric. The models based on 2 metrics in order.'
    model_estimator = define_default_model(SVRModel=SVRModel,SVCmodel=SVCmodel)
    model_param_grid = ModelParameterGrid()
    svmgrid = GridSearchCV(estimator=model_estimator,
                           param_grid=model_param_grid,
                           cv = 5,
                           scoring = scoringfn,
                           n_jobs=njobs,
                           refit=False) #refit_strategy)
    # print(svmgrid)
    svmgrid.fit(X_train, y_train)
    if Static_m_Models:
        best_k_params = collect_best_m_models_static(svmgrid.cv_results_, scoringfn[0], m_models_default=m_models)
    else:
        best_k_params = collect_best_m_models_dynamic(svmgrid.cv_results_, scoringfn, precision_threshold_default=precision_threshold)
    y_prediction_models = pd.DataFrame()
    model_accuracy = {}
    best_model_parameters = {}
    for I, pm in enumerate(best_k_params):
        tmp_params = svmgrid.cv_results_['params'][pm]
        models_estimator = define_default_model(SVRModel=SVRModel,SVCmodel=SVCmodel,PredictProb=PredictProb)
        models_estimator.set_params(**tmp_params)
        models_estimator.fit(X_train, y_train)
        y_svm = models_estimator.predict(X_test) 
        
        ##### Collect model accuracies
        if SVRModel:
            model_accuracy['model-'+str(I)] = mean_absolute_error(y_test, y_svm)
        elif SVCmodel:
            model_accuracy['model-'+str(I)] = accuracy_score(y_test, y_svm)
        best_model_parameters['model-'+str(I)] = tmp_params
        # print(f'model-{I}: ',"best parameter (CV score=%0.3f):" % svmgrid.best_score_, svmgrid.best_params_)
        ##### Predict data s
        if PredictProb:
            TuplesList = [('model-'+str(I),ii) for ii in models_estimator.classes_]
            PP_dataframe = pd.DataFrame(models_estimator.predict_proba(X_predict),columns=pd.MultiIndex.from_tuples(TuplesList, names=['lv10','lv11']))
            y_prediction_models = pd.concat([y_prediction_models,PP_dataframe],axis=1)
        else:
            y_prediction_models['model-'+str(I)] = models_estimator.predict(X_predict)
        # y_prediction_models['model-'+str(I)] = models_estimator.predict_proba(X_predict) if PredictProb else models_estimator.predict(X_predict)
        # print(y_prediction_models)
        
    if len(best_k_params) < m_models:
        for I in range(len(best_k_params),m_models):
            model_accuracy['model-'+str(I)] = np.nan 
        
    return model_accuracy, y_prediction_models, best_model_parameters

def m_models_data_split(X, y, X_predict,
                        refit = True, random_state=None,m_models=10,
                        scoringfn='r2', njobs=-1):
    
    y_prediction_models = pd.DataFrame()
    model_accuracy = {}
    best_model_parameters = {}
    for I in range(m_models):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=0.2)
        model_estimator, model_param_grid = define_default_model()
        svrgrid = GridSearchCV(estimator=model_estimator,
                               param_grid=model_param_grid,
                               cv = 5,
                               scoring = scoringfn,
                               n_jobs=njobs,
                               refit=refit)

        svrgrid.fit(X_train, y_train)
        y_svr = svrgrid.predict(X_test)
        
        ##### COllect model accuracies
        model_accuracy['model-'+str(I)] = mean_absolute_error(y_test, y_svr)
        best_model_parameters['model-'+str(I)] = svrgrid.best_params_
        # print(f'model-{I}: ',"best parameter (CV score=%0.3f):" % svrgrid.best_score_, svrgrid.best_params_)
        ##### Predict data 
        y_prediction_models['model-'+str(I)] = svrgrid.predict(X_predict)
    
    return model_accuracy, y_prediction_models, best_model_parameters

def ALmodels(model_accuracy_last_k, model_err_last_k,
             conn, ML_df, PreservedTestSamples, PredictOvernSamples, X_predict,
             model_accuracy_cutoff, model_error_cutoff, 
             model_saturates_epsilon, nature_accuracy_cutoff, std_cutoff,
             Check_model_accuracy4last_k, ntry_max,
             XFEATURES, xfeatures, yfeatures, LoopIndex,
             random_state=None, m_models=10, Static_m_Models=False, natoms=216,
             TrainingReachedAccuracy=False, TrainingReachedAccuracy_mag = False,
             SVRModel=True,SVCmodel=False,SVRSVCmodel=False,SVRdependentSVC=False,
             PredictProb=False):
    if SVRModel==False and SVCmodel==False and SVRSVCmodel==False and SVRdependentSVC==False:
        print("No model training is mentioned. Are you sure?")
        exit
        
    if SVRModel or SVRSVCmodel:
        if SVRSVCmodel: yfeatures='BANDGAP'
        model_err_mean_last_k = np.nanmean(model_err_last_k)
        model_err, y_prediction_models, best_model_parameters = \
            m_bestmodel_parametersearch(ML_df[xfeatures], PreservedTestSamples[xfeatures],
                                        ML_df[yfeatures], PreservedTestSamples[yfeatures],
                                        X_predict, random_state=random_state,m_models=m_models,
                                        SVRModel=True,SVCmodel=False,Static_m_Models=Static_m_Models)
    
        ##### Check if models reached cut-off accuracy in bandgap magnitude
        model_err_mean = np.nanmean(list(model_err.values())) # mean_model(mean_sample(AbsoluteError_per_model))
        model_err_std = np.nanstd(list(model_err.values())) # std_model(mean_sample(AbsoluteError_per_model))
        model_accuracy_saturation_value = abs(model_err_mean - model_err_mean_last_k)
     
        print(f'Mean absolute error in out-of-sample bandgap magnitude prediction (average over model) = {model_err_mean:.3f}±{model_err_std:.3f} eV')
        if model_err_mean < model_error_cutoff: # Loop convergence condition 3.2
            if SVRSVCmodel and not SVRdependentSVC:
                TrainingReachedAccuracy_mag = True
                PickRandomConfigurationAtom = (pd.DataFrame(),) #pd.DataFrame(),pd.DataFrame()
                TestPickRandomConfiguration = pd.DataFrame()
                print(f' - SVR-Models reached to cutoff bandgap magnitude accuary (cutoff={model_error_cutoff} eV).')
            else: # SVRModel  or (SVRSVCmodel and SVRdependentSVC) = True
                print(f'\n* Models reached to cutoff bandgap magnitude accuary (cutoff={model_error_cutoff} eV).')
                print('* Training complete. Exiting loop.')
                return model_err, model_err_mean, best_model_parameters, True, None, None
        elif model_accuracy_saturation_value < model_saturates_epsilon: # Loop convergence condition 4
            if SVRSVCmodel and not SVRdependentSVC:
                TrainingReachedAccuracy_mag = True 
                PickRandomConfigurationAtom = (pd.DataFrame(),) #pd.DataFrame(),pd.DataFrame()
                TestPickRandomConfiguration = pd.DataFrame()
                print(f' - AL SVR-model reached to saturation accuary (={model_accuracy_saturation_value:.4f} eV, cutoff={model_saturates_epsilon:.4f} eV)')
            else:
                print(f'\n** AL model reached to saturation accuary (={model_accuracy_saturation_value:.4f} eV, cutoff={model_saturates_epsilon:.4f} eV)')
                print(f'\tSaturation: ΔMAE = abs[MAE(current step)-MAE(average over last {Check_model_accuracy4last_k} steps)]')
                print('** Terminating training. Model cannot perform better.')
                return model_err, model_err_mean, best_model_parameters, True, None, None
        else: 
            ##### Dynamic number of new data to add in re-train set.
            ##### Get X_predictions where predicted std values greater than mean std.
            y_prediction_models_STD = y_prediction_models.std(axis=1) 
            STD_mean = y_prediction_models_STD.mean()
            TrainingReachedAccuracy_mag, PickRandomConfigurationAtom, TestPickRandomConfiguration = \
                GenerateBadSamples(PredictOvernSamples,y_prediction_models_STD > STD_mean,conn,XFEATURES,
                                   ntry_max,LoopIndex,std_cutoff,natoms=natoms, SVRdependentSVC=SVRdependentSVC,
                                   LoopNonConvergenceCondition5 = STD_mean > std_cutoff, SVRSVCmodel=SVRSVCmodel,SVRModel=True)
            if SVRModel:
                return model_err, model_err_mean, best_model_parameters, TrainingReachedAccuracy_mag, PickRandomConfigurationAtom, TestPickRandomConfiguration
            elif SVRSVCmodel and SVRdependentSVC: 
                print('Picking bad samples for bandgap nature using SVR hyperparameters (SVRdependentSVC).')
                y_prediction_models_nature = pd.DataFrame()
                for key, tmp_params in best_model_parameters.items():
                    models_estimator = define_default_model(SVRModel=False,SVCmodel=True,PredictProb=PredictProb)
                    models_estimator.set_params(**tmp_params)
                    models_estimator.fit(ML_df[xfeatures], ML_df['NATURE'])
                    if PredictProb:
                        TuplesList = [(key,ii) for ii in models_estimator.classes_]
                        PP_dataframe = pd.DataFrame(models_estimator.predict_proba(X_predict),columns=pd.MultiIndex.from_tuples(TuplesList, names=['lv10','lv11']))
                        y_prediction_models_nature = pd.concat([y_prediction_models_nature,PP_dataframe],axis=1)
                    else:
                        y_prediction_models_nature[key] = models_estimator.predict(X_predict)
                y_prediction_models_nature_accuracy = CalculateAccuracyFromLabels(y_prediction_models_nature,PredictProb=PredictProb)
                Accuracy_mean = y_prediction_models_nature_accuracy.mean()
                if Accuracy_mean < nature_accuracy_cutoff:
                    _, PickRandomConfigurationAtom, TestPickRandomConfiguration = \
                        GenerateBadSamples_SVRSVC(PredictOvernSamples,y_prediction_models_nature_accuracy < Accuracy_mean,conn,XFEATURES,
                                                  PickRandomConfigurationAtom[0],TestPickRandomConfiguration,ntry_max,LoopIndex,natoms=natoms)
                else:
                    print(f' - Mean accuracy (over samples) of bandgap nature prediction (mode over models) reached cutoff = {nature_accuracy_cutoff}.')
                    print(' - No bad bandgap nature samples. No additional samples from bad bandgap nature prediction is added to feed-back batch.')
                return model_err, model_err_mean, best_model_parameters, TrainingReachedAccuracy_mag, PickRandomConfigurationAtom, TestPickRandomConfiguration
    
    if SVCmodel or SVRSVCmodel:
        if SVRSVCmodel: yfeatures='NATURE'
        model_accuracy_mean_last_k = np.nanmean(model_accuracy_last_k)
        model_accuracy, y_prediction_models_n, best_model_parameters_n = \
            m_bestmodel_parametersearch(ML_df[xfeatures], PreservedTestSamples[xfeatures],
                                        ML_df[yfeatures], PreservedTestSamples[yfeatures],
                                        X_predict, random_state=random_state,m_models=m_models,PredictProb=PredictProb,
                                        SVRModel=False,SVCmodel=True,Static_m_Models=Static_m_Models)
        
        ##### Check if models reached cut-off accuracy in bandgap magnitude
        model_accuracy_mean = np.nanmean(list(model_accuracy.values())) # mean_model(mean_sample(1(x_predict=x_true)_per_model))
        model_accuracy_std = np.nanstd(list(model_accuracy.values())) # std_model(mean_sample(1(x_predict=x_true)_per_model))
        model_accuracy_saturation_value = abs(model_accuracy_mean - model_accuracy_mean_last_k)
        
        # print(model_accuracy_last_k, model_accuracy_mean_last_k, model_accuracy_mean, model_accuracy_saturation_value)
     
        print(f'Accuracy in out-of-sample bandgap nature prediction (average over model) = {model_accuracy_mean:.3f}±{model_accuracy_std:.3f}')
        if model_accuracy_mean > model_accuracy_cutoff: # Loop convergence condition 3.1
            if SVCmodel:
                print(f'\n* Models reached to cutoff bandgap nature prediction accuary (cutoff={model_accuracy_cutoff}).')
                print('* Training complete. Exiting loop.')
                return model_accuracy, model_accuracy_mean, best_model_parameters_n, True, None, None
            else: # SVRSVCmodel and not SVRdependentSVC
                print(f' - SVC-Models reached to cutoff bandgap nature prediction accuary (cutoff={model_accuracy_cutoff}).')
                if TrainingReachedAccuracy_mag:
                    print('\n* Training complete. Exiting loop.')
                    return (model_err,model_accuracy), (model_err_mean,model_accuracy_mean), (best_model_parameters,best_model_parameters_n),\
                        True, None, None
                else:
                    return (model_err,model_accuracy), (model_err_mean,model_accuracy_mean), (best_model_parameters,best_model_parameters_n),\
                        False, PickRandomConfigurationAtom, TestPickRandomConfiguration
        elif model_accuracy_saturation_value < model_saturates_epsilon: # Loop convergence condition 4
            if SVCmodel:
                print(f'\n** AL model reached to saturation accuary ({model_accuracy_saturation_value:.4f}, cutoff={model_saturates_epsilon:.4f})')
                print(f'\tSaturation: Δaccuracy = abs[mean_accuracy_score(current step)-mean_accuracy_score(average over last {Check_model_accuracy4last_k} steps)]')
                print('** Terminating training. Model cannot perform better.')
                return model_accuracy, model_accuracy_mean, best_model_parameters_n, True, None, None
            else: # SVRSVCmodel and not SVRdependentSVC
                print(f' - AL SVC-model reached to saturation accuary ({model_accuracy_saturation_value:.4f}, cutoff={model_saturates_epsilon:.4f})')
                if TrainingReachedAccuracy_mag:
                    print('\n** Terminating training. Model cannot perform better.')
                    return (model_err,model_accuracy), (model_err_mean,model_accuracy_mean), (best_model_parameters,best_model_parameters_n),\
                        True, None, None
                else:
                    return (model_err,model_accuracy), (model_err_mean,model_accuracy_mean), (best_model_parameters,best_model_parameters_n),\
                        False, PickRandomConfigurationAtom, TestPickRandomConfiguration
        else: 
            ##### Dynamic number of new data to add in re-train set.
            ##### Get X_predictions where predicted accuracy values greater than mean accuracy.
            y_prediction_models_n_accuracy = CalculateAccuracyFromLabels(y_prediction_models_n,PredictProb=PredictProb)
            # plt.figure(); y_prediction_models_n_accuracy.plot.hist()
            Accuracy_mean = y_prediction_models_n_accuracy.mean()
            if SVRSVCmodel: # and not SVRdependentSVC
                # print(PickRandomConfigurationAtom[0])
                if Accuracy_mean < nature_accuracy_cutoff:
                    TrainingReachedAccuracy, PickRandomConfigurationAtom, TestPickRandomConfiguration = \
                        GenerateBadSamples_SVRSVC(PredictOvernSamples,y_prediction_models_n_accuracy < Accuracy_mean,conn,XFEATURES,
                                                  PickRandomConfigurationAtom[0],TestPickRandomConfiguration,ntry_max,LoopIndex,natoms=natoms)   
                else:
                    print(f' - Mean accuracy (over samples) of bandgap nature prediction (mode over models) reached cutoff = {nature_accuracy_cutoff}.')
                    # print(' - No bad bandgap nature samples. No additional samples from bad bandgap nature prediction is added to feed-back batch.')
                    TrainingReachedAccuracy = True
                        
                TrainingReachedAccuracy_f = TrainingReachedAccuracy
                if TrainingReachedAccuracy: TrainingReachedAccuracy_f = TrainingReachedAccuracy_mag
                if TrainingReachedAccuracy and TrainingReachedAccuracy_mag:
                    print('\n* Training complete. Exiting AL loop.')
                return (model_err,model_accuracy), (model_err_mean,model_accuracy_mean), (best_model_parameters,best_model_parameters_n),\
                    TrainingReachedAccuracy_f, PickRandomConfigurationAtom, TestPickRandomConfiguration
            else:
                TrainingReachedAccuracy, PickRandomConfigurationAtom, TestPickRandomConfiguration = \
                    GenerateBadSamples(PredictOvernSamples,y_prediction_models_n_accuracy < Accuracy_mean,conn,XFEATURES,
                                       ntry_max,LoopIndex,nature_accuracy_cutoff,natoms=natoms,
                                       LoopNonConvergenceCondition5 = Accuracy_mean < nature_accuracy_cutoff,SVRSVCmodel=SVRSVCmodel)
            
                return model_accuracy, model_accuracy_mean, best_model_parameters_n, TrainingReachedAccuracy, PickRandomConfigurationAtom, TestPickRandomConfiguration

#%%% ==========================================================================
def PredictFinal(C,gamma,SVRModel=False,SVCModel=False):
    scaler = StandardScaler()
    tol = 1e-5;cache_size=10000
    if SVRModel:
        svm = SVR(kernel="rbf",C=C, gamma=gamma,tol=tol,cache_size=cache_size)
    elif SVCModel:
        svm = SVC(kernel="rbf",C=C, gamma=gamma,tol=tol,cache_size=cache_size)
    else:
        exit
    # Set pipeline
    svmpipe = Pipeline(steps=[("scaler", scaler), ("svmm", svm)])
    return svmpipe

def PrintConfusionMatrix(X_predict,labels,save=False, savepath='.',figname='/ConfusionMatrix.png'):
    cm = confusion_matrix(X_predict['TrueBandgapNature'], X_predict['PredictBandgapNature'], labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)   
    disp.plot()
    disp.ax_.set_title('AL bandgap nature prediction\n0=direct, 1=indirect')
    if save:
        disp.figure_.savefig(savepath+figname,bbox_inches = 'tight',dpi=300)
        plt.close()
    else:
        plt.show()   
    # print('Classification report:\n',classification_report(X_predict['TrueBandgapNature'], X_predict['PredictBandgapNature']))
    return

def GenerateBandgapMagFigsData(X_predict,df,y_prediction_whole_space,yfeatures):
    '''
    # PredictMeanBandgap = mean_model(bandgap_predict) # for each sample
    # PredictSTDBandgap = std_model(bandgap_predict) # for each sample
    # BandgapError = PredictMeanBandgap - DFTbandgap # for each sample
    '''
    X_predict['TrueBandgap'] = df[yfeatures].copy()
    X_predict['PredictMeanBandgap'] = y_prediction_whole_space.mean(axis=1)
    X_predict['BandgapError'] = X_predict['PredictMeanBandgap'] - X_predict['TrueBandgap']
    X_predict['PredictSTDBandgap'] = y_prediction_whole_space.std(axis=1)
    #### MAE = mean_sample(abs(mean_model(y_predict) - y_true))
    print('\tBandgap magnitude prediction: (for each sample prediction is average over models)')
    print(f'\t\tMAE = {X_predict["BandgapError"].abs().mean():.3f}±{X_predict["BandgapError"].abs().std():.3f} eV, Max error = {X_predict["BandgapError"].abs().max():.3f} eV')
    print('\t\t[mae_per_sample:=abs(mean_model(y_predict)-y_true)')
    print('\t\t MAE:=mean_sample(mae_per_sample)±std_sample(mae_per_sample)')
    print('\t\t Max error:=max_sample(mae_per_sample)]')
    return X_predict

def CalculateAccuracyFromPredictionLabels(y_prediction_whole_space,TrueValues):
    TMP = y_prediction_whole_space.copy()
    TMP['TrueBandgapNature'] = TrueValues.copy()
    return TMP.apply(lambda a: np.mean([1 if val==a[-1] else 0 for val in a[:-1]]), axis=1)

def GenerateBandgapNatureFigsData(X_predict,df,y_prediction_whole_space,yfeatures):
    '''
    PredictBandgapNature is based on mode labels of models. Using predict_proba is possible but not recommended. 
    '''
    X_predict['TrueBandgapNature'] = df[yfeatures].copy()
    X_predict['PredictBandgapNature'] = y_prediction_whole_space.mode(axis=1).iloc[:, 0].astype(int) # Note: if mode is 50:50 the tag is 0 (==direct).
    X_predict['NatureAccuracyTag'] = CalculateAccuracyFromPredictionLabels(y_prediction_whole_space,X_predict['PredictBandgapNature'])
    X_predict['NatureAccuracyWRTtrue'] = CalculateAccuracyFromPredictionLabels(y_prediction_whole_space,X_predict['TrueBandgapNature'])    
    print("\tBandgap nature prediction: (for each sample prediction is mode over models)")
    print(f"\t\tAccuracy = {accuracy_score(X_predict['TrueBandgapNature'], X_predict['PredictBandgapNature']):.3f}")
    print('\t\t[Accuracy:=accuracy_score(y_true,mode_models(y_predict)); y_true=DFT_nature]')
    return X_predict

def GenerateBandgapFigsData(AL_dbname,df,xfeatures,yfeatures):   
    conn = sq.connect(AL_dbname)
    Models_ = pd.read_sql_query("SELECT * FROM BestModelsParameters", conn, index_col='HyperParameters')
    ML_df = pd.read_sql_query('SELECT * FROM ALLDATA', conn)
    conn.close()
    C_values = Models_.loc['svm__C'].values.tolist()
    gamma_values = Models_.loc['svm__gamma'].values.tolist()
    X_predict = df[xfeatures].copy()
    y_prediction_whole_space = pd.DataFrame()
    if yfeatures=='BANDGAP':
        SVRModel=True ; SVCModel=False
    elif yfeatures=='NATURE':
        SVRModel=False ; SVCModel=True
        
    for I in range(len(C_values)):
        svm_model = PredictFinal(C_values[I],gamma_values[I],SVRModel=SVRModel,SVCModel=SVCModel)
        svm_model.fit(ML_df[xfeatures], ML_df[yfeatures])
        y_prediction_whole_space['model-'+str(I)] = svm_model.predict(X_predict)

    if yfeatures=='BANDGAP':
        X_predict = GenerateBandgapMagFigsData(X_predict,df,y_prediction_whole_space,yfeatures)
        labels = None
    elif yfeatures=='NATURE':
        X_predict = GenerateBandgapNatureFigsData(X_predict,df,y_prediction_whole_space,yfeatures)
        labels=svm_model.classes_
    return X_predict, labels

def GenerateBandgapBothFigsData(AL_dbname,df,xfeatures):
    conn = sq.connect(AL_dbname)
    Models_svr = pd.read_sql_query("SELECT * FROM BestModelsParameters", conn, index_col='HyperParameters')
    Models_svc = pd.read_sql_query("SELECT * FROM NatureBestModelsParameters", conn, index_col='HyperParameters')
    ML_df = pd.read_sql_query('SELECT * FROM ALLDATA', conn)
    conn.close()
    C_values = [Models_svr.loc['svm__C'].values.tolist(),Models_svc.loc['svm__C'].values.tolist()]
    gamma_values = [Models_svr.loc['svm__gamma'].values.tolist(),Models_svc.loc['svm__gamma'].values.tolist()]
    X_predict = df[xfeatures].copy()
    y_prediction_whole_space = pd.DataFrame()
    y_prediction_nature_whole_space = pd.DataFrame()
    for I in range(len(C_values[0])):
        svm_model = PredictFinal(C_values[0][I],gamma_values[0][I],SVRModel=True,SVCModel=False)
        svm_model.fit(ML_df[xfeatures], ML_df['BANDGAP'])
        y_prediction_whole_space['model-'+str(I)] = svm_model.predict(X_predict)
    for I in range(len(C_values[1])):     
        svm_model = PredictFinal(C_values[1][I],gamma_values[1][I],SVRModel=False,SVCModel=True)
        svm_model.fit(ML_df[xfeatures], ML_df['NATURE'])
        y_prediction_nature_whole_space['model-'+str(I)] = svm_model.predict(X_predict)
        
    X_predict = GenerateBandgapMagFigsData(X_predict,df,y_prediction_whole_space,'BANDGAP')
    X_predict = GenerateBandgapNatureFigsData(X_predict,df,y_prediction_nature_whole_space,'NATURE')
    labels=svm_model.classes_
    return X_predict, labels