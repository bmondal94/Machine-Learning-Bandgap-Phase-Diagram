#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 12:22:42 2021

@author: bmondal
"""

import numpy as np
from sklearn.svm import SVR, SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, cross_validate,ShuffleSplit
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.metrics import r2_score, mean_squared_error, classification_report, make_scorer,\
    accuracy_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error,max_error
import time
import inspect, pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import pandas as pd
import sys
import os
import sqlite3 as sq
from datetime import datetime
from contextlib import redirect_stdout

#%%
params = {'legend.fontsize': 18,
          'figure.figsize': (8, 6),
         'axes.labelsize': 24,
         'axes.titlesize': 24,
         'xtick.labelsize':24,
         'ytick.labelsize': 24,
         'errorbar.capsize':2}
plt.rcParams.update(params)

#%% ###########################################################################
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
    
def CreateResultDirectories(dirpathSystem_,BinaryConversion, refitm,dont_restart_output_file,DumpData_,DataFolderPath=None):
    SaveFigPath = dirpathSystem_ +'/Figs/' 
    SaveMoviePath = dirpathSystem_ + '/MOVIE/' 
    SaveHTMLPath = dirpathSystem_ + '/' +'HTML'
    modelPATHS = dirpathSystem_+'/'+'MODELS'+'/'
    if DumpData_: 
        DataFolderPath = dirpathSystem_ +'/Data/'
        if isinstance(refitm, str):
            DataFolderPath += refitm + '/'
        os.makedirs(DataFolderPath,exist_ok=True)
        
    if isinstance(refitm, str):
        SaveFigPath += refitm + '/'
        SaveMoviePath += refitm + '/'
        SaveHTMLPath += refitm + '/'
        modelPATHS += refitm + '/'

    OutPutTxtFile = modelPATHS + 'output.txt'
    svr_bandgap = modelPATHS + 'svrmodel_bandgap'
    svc_EgNature = modelPATHS + 'svcmodel_EgNature'
    if BinaryConversion:
        svc_EgNature += '_binary'
    svr_bw = modelPATHS + 'svrmodel_bw'
    svr_bw_dm = modelPATHS + 'DirectMultioutput/'
    # os.makedirs(svr_bw_dm,exist_ok=True)
    svr_lp = modelPATHS + 'svrmodel_lp'
    
    os.makedirs(SaveFigPath,exist_ok=True) 
    os.makedirs(SaveMoviePath,exist_ok=True)
    os.makedirs(modelPATHS,exist_ok=True) 
    if not dont_restart_output_file: 
        open(OutPutTxtFile, 'w').close()
    
    return SaveFigPath, SaveMoviePath, SaveHTMLPath, modelPATHS, DataFolderPath, OutPutTxtFile, svr_bandgap,\
        svc_EgNature, svr_bw, svr_bw_dm, svr_lp
    
def PlotCV_results(svrgrid, paramss, scoringfn_tmp, SVRgridparameters_c, SVRgridparameters_gamma,
                   titletxt="Validation accuracy",save=False, savepath='.', figname='GridSearchTestScore.png'):
    #https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
    results = pd.DataFrame(svrgrid.cv_results_)
    # print(svrgrid.cv_results_)
    # print(results)
    
    for III in paramss:
        if '_gamma' in III: 
            indexname = 'param_' + III
        elif '_C' in III:
            colname = 'param_' + III
            
    if isinstance(scoringfn_tmp,str): scoringfn_tmp=('score',)
    for I in scoringfn_tmp:
        results[[colname,indexname]] = results[[colname,indexname]].astype(np.float64)
        scores_matrix = results.pivot(index=indexname, columns=colname, values=f"mean_test_{I}")
        
        
        plt.figure(figsize=(8, 8))
        im = plt.imshow(scores_matrix)
    
        plt.ylabel("$\gamma$")
        plt.xlabel("C")
        cb = plt.colorbar(label='mean test score')
        plt.yticks(np.arange(len(SVRgridparameters_gamma)), ["{:.0E}".format(x) for x in SVRgridparameters_gamma])
        plt.xticks(np.arange(len(SVRgridparameters_c)), ["{:.0E}".format(x) for x in SVRgridparameters_c], rotation=30 )
        plt.title(titletxt)
        plt.gca().tick_params(axis='both',which='major',length=10,width=2)
        plt.gca().tick_params(axis='both',which='minor',length=6,width=2)
        cb.ax.tick_params(labelsize=20,length=8,width=2)
        # plt.tight_layout()
        if save:
            # pass
            plt.savefig(savepath+'/'+f'{I}'+figname,bbox_inches = 'tight',dpi=300)
            plt.close()
        else:
            plt.show()

    return im

def Plot_Prediction_Actual_results(X, Y, tsxt=None, data_unit_label='eV',save=False, savepath='.', figname='TruePrediction.png'):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, aspect='equal')
    plt.xlabel(f"True values ({data_unit_label})")
    plt.ylabel(f"Predictions ({data_unit_label})")
    plt.scatter(X, Y,marker='.')

    # lim_min, lim_max = min(X), max(X)
    # plt.ylim(min(min(Y), lim_min), max(max(Y),lim_max))
    # plt.xlim(lim_min, lim_max)
    # # plt.ylim(lim_min, lim_max)
    # plt.plot([lim_min, lim_max],[lim_min, lim_max], color='k')
    
    lim_min, lim_max = min(min(Y), min(X)), max(max(Y),max(X))
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)
    plt.plot([lim_min, lim_max],[lim_min, lim_max], color='k')
    
    plt.title(tsxt)
    # DIFF_ = np.sqrt(np.sum((X - Y)**2)/len(X))
    # ax.plot([],[],' ',label=f"Max error = {max(DIFF_):.2f} eV \n {'MAE':>11} = {np.mean(DIFF_):.2f}±{np.std(DIFF_):.2f} eV")
    # plt.plot([],[],' ',label=f"RMSE = {DIFF_:.2f} eV")
    # plt.legend(handlelength=0)
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.4))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.4))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.gca().tick_params(axis='both',which='major',length=10,width=2)
    plt.gca().tick_params(axis='both',which='minor',length=6,width=2)
    plt.tight_layout()
    if save:
        plt.savefig(savepath+'/'+figname,bbox_inches = 'tight',dpi=300)
        plt.close()
    else:
        plt.show()

def plot_err_dist(XX, YY, text=None,data_unit_label='eV',save=False, savepath='.', figname='TruePredictErrorHist.png'):
    plt.figure()
    # Check error distribution
    plt.subplot()
    plt.title(text)
    error = YY - XX
    plt.hist(error, bins=25)
    plt.xlabel(f'Prediction error ({data_unit_label})')
    plt.ylabel('Count (arb.)')
    plt.gca().tick_params(axis='both',which='major',length=10,width=2)
    plt.gca().tick_params(axis='both',which='minor',length=6,width=2)
    if save:
        plt.savefig(savepath+'/'+figname,bbox_inches = 'tight',dpi=300)
        plt.close()
    else:
        plt.show()
    return 

def PlotConfusionMatrix(y_testt, y_predictt, display_labels,save=False,savepath='.',figname='TestSet.png'):
    cm = confusion_matrix(y_testt, y_predictt)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=display_labels)
    disp = UpdateConfuxionMatrixDisplay(disp)
    disp.ax_.tick_params(axis='both',which='major',length=10,width=2)
    disp.ax_.tick_params(axis='both',which='minor',length=6,width=2)
    if save:
        disp.figure_.savefig(savepath+figname,bbox_inches = 'tight',dpi=300)
        plt.close()   
    else:
        plt.show()
    return disp

def UpdateConfuxionMatrixDisplay(disp):
    disp.plot(colorbar=False)
    # disp.ax_.set_title('Bandgap nature prediction\n1=direct, 0=indirect')
    disp.plot(colorbar=False)
    plt.setp(disp.ax_.get_yticklabels(), rotation='vertical',va='center')
    # disp.text_[0,0].set_text(disp.text_[0,0].get_text()+'\nhi')
    return disp

def GenerateData(df, xfeatures=['ANTIMONY','STRAIN'], yfeatures=['BANDGAP', 'SO']):        
    return df[xfeatures], df[yfeatures]

def GenerateDataV2(df, xfeatures=['ANTIMONY','STRAIN'], yfeatures=['BANDGAP', 'SO']):    
    if isinstance(yfeatures, str): yfeatures = [yfeatures]
    if isinstance(xfeatures, str): xfeatures = [xfeatures]
    return df[xfeatures+yfeatures].copy()

def LabelConversionFn(lbs,rdict={1:'direct',0:'indirect'}):
    print('Mapping ==> ',rdict)
    return [rdict.get(int(i)) for i in lbs]

def CrosValidateDataAgain(npipe, X, Y, scoringfn,txtx="all samples",cvsplitt=10,njobs=-1):
    
    print(f"Evaluating {cvsplitt} cross validation over {txtx}:")
    kfold = KFold(n_splits=cvsplitt, random_state=1, shuffle=True)
    if isinstance(scoringfn, str):   
        cv_results_final = cross_val_score(npipe, X, Y, cv=kfold, scoring=scoringfn, n_jobs=njobs)
        print(f"Score over {txtx} ({scoringfn}): {cv_results_final.mean():.3f}±{cv_results_final.std():.3f}")
    else:
        cv_results_final = cross_validate(npipe, X, Y, cv=kfold, scoring=scoringfn, n_jobs=njobs)
        for I in scoringfn:
            print(f"Score over {txtx} ({I}): {cv_results_final['test_'+I].mean():.3f}±{cv_results_final['test_'+I].std():.3f}")
    return cv_results_final

def SVRdependentSVCmodel(model_name,):
    if isinstance(model_name, str) and model_name.endswith('.sav'):
        assert 'svrmodel' in model_name, 'Supply bandgap value model'
        SVRmodel_tmp = pickle.load(open(model_name, 'rb')) # Bandgap model
        params = SVRmodel_tmp['svm'].get_params()
    else:
        params =  model_name.best_estimator_['svm'].get_params()
    params.pop('epsilon')
    SVCEgNatureModel = SVC(kernel="rbf",tol= 1e-5,cache_size=10000)
    SVCEgNatureModel.set_params(**params)
    svc_pipe = Pipeline(steps=[("scaler", StandardScaler()), ("svc", SVCEgNatureModel)])
    return svc_pipe

def PrintSVC_output(y_test_nature, y_svc, y_all_nature, y_svc_all,add_extra=''):
    print(f"{add_extra}(o1) The out-of-sample accuracy_score for prediction: {accuracy_score(y_test_nature, y_svc):.3f}")
    print(f"{add_extra}(o2) The out-of-sample balanced_accuracy_score for prediction: {balanced_accuracy_score(y_test_nature, y_svc):.3f}")
    print(f"{add_extra}(a1) The all-sample accuracy_score for prediction: {accuracy_score(y_all_nature, y_svc_all):.3f}")
    print(f"{add_extra}(a2) The all-sample balanced_accuracy_score for prediction: {balanced_accuracy_score(y_all_nature, y_svc_all):.3f}")           
    print('Classification report out-of-sample:\n',classification_report(y_test_nature, y_svc))
    print('Classification report all-sample:\n',classification_report(y_all_nature, y_svc_all))
    
def PrintSVR_output(y_test_eg, y_svr, Y_full_, Y_predict_all_,data_unit_label='eV',add_extra=''):
    print(f"{add_extra}(o1) The out-of-sample r2_score for prediction: {r2_score(y_test_eg, y_svr):.3f}")
    print(f"{add_extra}(o2) The out-of-sample root_mean_squared_error for prediction: {mean_squared_error(y_test_eg, y_svr,squared=False):.3f} {data_unit_label}")
    print(f"{add_extra}(o3) The out-of-sample mean_absolute_error for prediction: {mean_absolute_error(y_test_eg, y_svr):.3f} {data_unit_label}")
    print(f"{add_extra}(o4) The out-of-sample max_error for prediction: {max_error(y_test_eg, y_svr):.3f} {data_unit_label}")
    # print('\n')
    print(f"{add_extra}(a1) The all-sample r2_score for prediction: {r2_score(Y_full_,Y_predict_all_):.3f}")
    print(f"{add_extra}(a2) The all-sample root_mean_squared_error for prediction: {mean_squared_error(Y_full_,Y_predict_all_,squared=False):.3f} {data_unit_label}")
    print(f"{add_extra}(a3) The all-sample mean_absolute_error for prediction: {mean_absolute_error(Y_full_,Y_predict_all_):.3f} {data_unit_label}")
    print(f"{add_extra}(a4) The all-sample max_error for prediction: {max_error(Y_full_,Y_predict_all_):.3f} {data_unit_label}")

def mysvrmodel_parametersearch_V0(X, y, refit = True, multiregression=False, AppendTestSET=None,
                                  regressorchain=False, SVCclassification=False,SplitInputData=True,
                                  scoringfn='r2', njobs=-1, PlotResults=False,x_test=None, y_test=None,
                                  save=False, savepath='.', figname='TruePrediction.png',
                                  SVRdependentSVC = False,SVRSVC_y_values=None,svr_data_unit='eV'):
    # Fit regression model
    # Radial Basis Function (RBF) kernel
    if SplitInputData:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=None, test_size=0.25)
        if SVRdependentSVC:
            assert SVRSVC_y_values is not None, 'must supply bandgap nature'
            y_train_nature = SVRSVC_y_values.iloc[y_train.index]
            y_test_nature = SVRSVC_y_values.iloc[y_test.index]
            y_all_nature = SVRSVC_y_values.copy()
            
        if AppendTestSET is not None: 
            X_test = pd.concat([X_test,AppendTestSET[0]], ignore_index=True)
            y_test = pd.concat([y_test,AppendTestSET[1]],ignore_index=True)
    else:
        assert (x_test is not None) and (y_test is not None),'Must supply test set.'
        X_train, X_test, y_train, y_test = X.copy(), x_test.copy(), y.copy(), y_test.copy() 
    
    # Define a Standard Scaler to normalize inputs
    scaler = StandardScaler()
    tol = 1e-5;cache_size=10000
    svm_model = SVC(kernel="rbf",tol=tol,cache_size=cache_size) if SVCclassification \
                    else SVR(kernel="rbf",tol=tol,cache_size=cache_size)
    
    if regressorchain: multiregression=False
    # Set model
    if multiregression:
        svm_model = MultiOutputRegressor(svm_model) # wrapper
    elif regressorchain:
        svm_model = RegressorChain(svm_model)
    else:
        pass
    
    # Set pipeline
    pipe = Pipeline(steps=[("scaler", scaler), ("svm", svm_model)])
    
    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    # C_range = [1e0, 1e1, 1e2, 1e3]
    # gamma_range = np.logspace(-2, 2, 5)
    C_range = [1e0, 1e1, 50, 1e2, 500, 1e3]
    gamma_range = [1.e-02, 5.e-02, 1.e-01, 5.e-01, 1.e+00, 5.e+00, 1.e+01]
    param_grid={"svm__C": C_range, 
                "svm__gamma": gamma_range}
    if multiregression:
        param_grid['svm__estimator__C'] = param_grid.pop('svm__C')
        param_grid['svm__estimator__gamma'] = param_grid.pop("svm__gamma")
    elif regressorchain:
        param_grid['svm__base_estimator__C'] = param_grid.pop('svm__C')
        param_grid['svm__base_estimator__gamma'] = param_grid.pop("svm__gamma")
    else:
        pass

    svmgrid = GridSearchCV(estimator=pipe,
                           param_grid=param_grid,
                           cv = 5,
                           scoring = scoringfn,
                           n_jobs=njobs,
                           refit=refit)
    
    t0 = time.time()
    svmgrid.fit(X_train, y_train)
    svm_fit = time.time() - t0
    print(f'Training dataset size: {X_train.shape[0]}')
    print(f'Testing dataset size: {X_test.shape[0]}')
    print("SVM complexity and bandwidth selected and model fitted in %.3f s" % svm_fit)

    if PlotResults:
        CVfigname = 'BandgapNatureGridSearchTestScore.png' if SVCclassification else 'BandgapValueGridSearchTestScore.png' 
        _ = PlotCV_results(svmgrid, list(param_grid.keys()),scoringfn,C_range, gamma_range, titletxt=None,save=save, savepath=savepath, figname=CVfigname)
    if refit or ((not refit) and (len(scoringfn)<2)): 
        print("Best parameter (CV score=%0.3f):" % svmgrid.best_score_, svmgrid.best_params_)
        if SVCclassification:
            Classes = svmgrid.best_estimator_['svm'].classes_
            print("The classes are:", Classes)
            for I in Classes:
                Number1 = np.count_nonzero(y_train==I)
                Number2 = np.count_nonzero(y_test==I)
                print(f'Samples in class-{I}: TrainSet-{Number1};TestSet-{Number2}; Total-{Number1+Number2}')
                
            print("The class weights:", svmgrid.best_estimator_['svm'].class_weight_)
        nsupports = svmgrid.best_estimator_['svm'].n_support_
        print("The grid_search cross-validation: %d fold" % svmgrid.n_splits_)
        print(f"Number of support vectors: Total={sum(nsupports)} ;  for each class=", nsupports)
        #print('* The scorer used:', inspect.getsource(svmgrid.scorer_))
    else:
        sys.exit("For multi-metric evaluation with refit False, 'GridsearCV' does not have best_score_ attribute. Can not continue.")
        

    if refit:
        print('Refitting timing for the best model on the whole training dataset:%.3f s' % svmgrid.refit_time_)
        t0 = time.time()
        y_svm = svmgrid.predict(X_test)
        # CrosValidateDataAgain(svmgrid, X_test, y_test, scoringfn,txtx='test set',cvsplitt=10,njobs=-1)
        svm_predict = time.time() - t0
        Y_predict_all = svmgrid.predict(X)
        # CrosValidateDataAgain(svmgrid, X, y, scoringfn,cvsplitt=10,njobs=-1)
        print("SVM prediction for %d (test) inputs in %.3f s" % (X_test.shape[0], svm_predict))
        if SVCclassification:
            print(f"The out-of-sample accuracy_score for prediction: {accuracy_score(y_test, y_svm):.3f}")
            print(f"The out-of-sample balanced_accuracy_score for prediction: {balanced_accuracy_score(y_test, y_svm):.3f}")
            print(f"The all-sample accuracy_score for prediction: {accuracy_score(y, Y_predict_all):.3f}")
            print(f"The all-sample balanced_accuracy_score for prediction: {balanced_accuracy_score(y, Y_predict_all):.3f}")
            if PlotResults:
                confucion_display_labels = LabelConversionFn(svmgrid.best_estimator_['svm'].classes_)
                PlotConfusionMatrix(y_test, y_svm, confucion_display_labels,save=save, savepath=savepath, figname='/NatureTestSet'+figname)
                PlotConfusionMatrix(y, Y_predict_all, confucion_display_labels,save=save, savepath=savepath, figname='/NatureFullSet'+figname)           
            print('Classification report out-of-sample:\n',classification_report(y_test, y_svm))
            print('Classification report all-sample:\n',classification_report(y, Y_predict_all))
        else:
            print(f"(o1) The out-of-sample r2_score for prediction: {r2_score(y_test, y_svm):.3f}")
            print(f"(o2) The out-of-sample root_mean_squared_error for prediction: {mean_squared_error(y_test, y_svm,squared=False):.3f} {svr_data_unit}")
            print(f"(o3) The out-of-sample mean_absolute_error for prediction: {mean_absolute_error(y_test, y_svm):.3f} {svr_data_unit}")
            print(f"(o4) The out-of-sample max_error for prediction: {max_error(y_test, y_svm):.3f} {svr_data_unit}")
            # print('\n')
            print(f"(a1) The all-sample r2_score for prediction: {r2_score(y,Y_predict_all):.3f}")
            print(f"(a2) The all-sample root_mean_squared_error for prediction: {mean_squared_error(y,Y_predict_all,squared=False):.3f} {svr_data_unit}")
            print(f"(a3) The all-sample mean_absolute_error for prediction: {mean_absolute_error(y,Y_predict_all):.3f} {svr_data_unit}")
            print(f"(a4) The all-sample max_error for prediction: {max_error(y,Y_predict_all):.3f} {svr_data_unit}")
            print(" ")
            if PlotResults:
                Plot_Prediction_Actual_results(y_test, y_svm, tsxt=None,data_unit_label=svr_data_unit,save=save, savepath=savepath, figname='TestSet'+figname)     #tsxt='Bandgap prediction on test set'
                plot_err_dist(y_test, y_svm, data_unit_label=svr_data_unit,save=save, savepath=savepath, figname='TestSetTruePredictErrorHist.png')
                Plot_Prediction_Actual_results(y,Y_predict_all, tsxt=None,data_unit_label=svr_data_unit,save=save, savepath=savepath, figname='FullSet'+figname) #tsxt='Bandgap prediction over all data'
                plot_err_dist(y,Y_predict_all, data_unit_label=svr_data_unit,save=save, savepath=savepath, figname='FullSetTruePredictErrorHist.png')
                
            if SVRdependentSVC:
                print("\n***************************************************************************")
                print('The SVC prediction using the optimized hyperparameters from SVR (SVRdependentSVC)::')
                svc_pipe = SVRdependentSVCmodel(svmgrid)
                svc_pipe.fit(X_train, y_train_nature)
                y_svc = svc_pipe.predict(X_test)
                y_svc_all = svc_pipe.predict(X)
                print(f"The out-of-sample accuracy_score for prediction: {accuracy_score(y_test_nature, y_svc):.3f}")
                print(f"The out-of-sample balanced_accuracy_score for prediction: {balanced_accuracy_score(y_test_nature, y_svc):.3f}")
                print(f"The all-sample accuracy_score for prediction: {accuracy_score(y_all_nature, y_svc_all):.3f}")
                print(f"The all-sample balanced_accuracy_score for prediction: {balanced_accuracy_score(y_all_nature, y_svc_all):.3f}")
                if PlotResults:
                    confucion_display_labels = LabelConversionFn(svc_pipe['svc'].classes_)
                    PlotConfusionMatrix(y_test_nature, y_svc, confucion_display_labels,save=save, savepath=savepath, figname='/NatureTestSet_SVRdependentSVC'+figname)
                    PlotConfusionMatrix(y_all_nature, y_svc_all, confucion_display_labels,save=save, savepath=savepath, figname='/NatureFullSet_SVRdependentSVC'+figname)           
                print('Classification report out-of-sample:\n',classification_report(y_test_nature, y_svc))
                print('Classification report all-sample:\n',classification_report(y_all_nature, y_svc_all))
        # model_best_scorer = scoringfn if isinstance(refit,bool)  else refit
        # print(f"Out-of-sample score on prediction using the 'default' scorer ({model_best_scorer}): {svrgrid.score(X_test, y_test): .3f}")
        #print("Number of support vectors:", len(svrgrid.best_estimator_['svm'].n_support_))
        return svmgrid.best_estimator_,svmgrid.best_score_
    else:
        return svmgrid.best_params_,svmgrid.best_score_

def my_custom_rmse_fix(y_true, y_pred):
    y_pred_update = np.where(y_pred<0,0,y_pred)
    return np.sqrt(np.average((y_true - y_pred_update)**2, axis=0))*(-1)

def my_custom_rmse_abs(y_true, y_pred):
    return np.sqrt(np.average((y_true - np.abs(y_pred))**2, axis=0))*(-1)

def UpdatePredictionValues(predictedvalues, which_refit_metric):
    if which_refit_metric == 'my_rmse_fix':
        print("Shifting the negative predictions to 0.")
        return np.where(predictedvalues<0,0,predictedvalues)
    elif which_refit_metric == 'my_rmse_abs':
        print("Shifting the negative predictions to positive scale (taking absolute values).")
        return  np.abs(predictedvalues)
    
def mysvrmodel_parametersearch(X, y, refit_metic_list, DoNotReset, multiregression=False, AppendTestSET=None,
                               regressorchain=False, SVCclassification=False,SplitInputData=True,
                               scoringfn='r2', njobs=-1, PlotResults=False,x_test=None, y_test=None,
                               save=False, savepath='.', figname='TruePrediction.png',DumpData=False,
                               SVRdependentSVC = False,SVRSVC_y_values=None,svr_data_unit='eV',
                               dirpathSystem_='.',BinaryConversion=False,save_model=False):
    # Fit regression model
    # Radial Basis Function (RBF) kernel
    if SplitInputData:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=None, test_size=0.25)           
        if AppendTestSET is not None: 
            X_test = pd.concat([X_test,AppendTestSET[0]], ignore_index=True)
            y_test = pd.concat([y_test,AppendTestSET[1]],ignore_index=True)
    else:
        assert (x_test is not None) and (y_test is not None),'Must supply test set.'
        X_train, X_test, y_train, y_test = X.copy(), x_test.copy(), y.copy(), y_test.copy() 
    
    X_full = pd.concat([X_train,X_test])
    Y_full = pd.concat([y_train,y_test])
    
    if SVRdependentSVC:
        if SVCclassification:
            assert SVRSVC_y_values is not None, 'must supply bandgap manitudes'  
        else:
            assert SVRSVC_y_values is not None, 'must supply bandgap nature'

        y_train_SVRSVC = SVRSVC_y_values.iloc[y_train.index]
        y_test_SVRSVC = SVRSVC_y_values.iloc[y_test.index]
        y_all_SVRSVC = SVRSVC_y_values.iloc[Y_full.index].copy()
        x_all_SVRSVC = X_full.copy()

    # Define a Standard Scaler to normalize inputs
    scaler = StandardScaler()
    tol = 1e-5;cache_size=10000
    svm_model = SVC(kernel="rbf",tol=tol,cache_size=cache_size) if SVCclassification \
                    else SVR(kernel="rbf",tol=tol,cache_size=cache_size)
    
    if not SVCclassification:
        scoringfn = {nname:nname for nname in scoringfn}
        if 'my_rmse_abs' in scoringfn:
            scoringfn['my_rmse_abs'] = make_scorer(my_custom_rmse_abs,greater_is_better=True)
        if 'my_rmse_fix' in scoringfn:
            scoringfn['my_rmse_fix'] = make_scorer(my_custom_rmse_fix,greater_is_better=True)
    
    if regressorchain: multiregression=False
    # Set model
    if multiregression:
        svm_model = MultiOutputRegressor(svm_model) # wrapper
    elif regressorchain:
        svm_model = RegressorChain(svm_model)
    else:
        pass
    
    # Set pipeline
    pipe = Pipeline(steps=[("scaler", scaler), ("svm", svm_model)])
    
    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    # C_range = [1e0, 1e1, 1e2, 1e3]
    # gamma_range = np.logspace(-2, 2, 5)
    C_range = [1e0, 1e1, 50, 1e2, 500, 1e3]
    gamma_range = [1.e-02, 5.e-02, 1.e-01, 5.e-01, 1.e+00, 5.e+00, 1.e+01]
    param_grid={"svm__C": C_range, 
                "svm__gamma": gamma_range}
    if multiregression:
        param_grid['svm__estimator__C'] = param_grid.pop('svm__C')
        param_grid['svm__estimator__gamma'] = param_grid.pop("svm__gamma")
    elif regressorchain:
        param_grid['svm__base_estimator__C'] = param_grid.pop('svm__C')
        param_grid['svm__base_estimator__gamma'] = param_grid.pop("svm__gamma")
    else:
        pass

    
    svmgrid = GridSearchCV(estimator=pipe,
                           param_grid=param_grid,
                           cv = 5,
                           scoring = scoringfn,
                           n_jobs=njobs,
                           refit=False)
    
    # t0 = time.time()
    svmgrid.fit(X_train, y_train)
    # svm_fit = time.time() - t0
           

    if PlotResults:
        CVfigname = 'BandgapNatureGridSearchTestScore.png' if SVCclassification else 'BandgapValueGridSearchTestScore.png'
        fname_tmp = dirpathSystem_+'/Figs/'
        os.makedirs(fname_tmp,exist_ok=True)
        _ = PlotCV_results(svmgrid, list(param_grid.keys()),scoringfn,C_range, gamma_range, 
                           titletxt=None,save=save, savepath=fname_tmp, figname=CVfigname)
                   
    for refit_metric in refit_metic_list:
        savepath, _, _, modelPATHS, DataFolderPath, OutPutTxtFile, svr_bandgap,\
            svc_EgNature, _, _, _ = CreateResultDirectories(dirpathSystem_,BinaryConversion, refit_metric, DoNotReset[refit_metric],DumpData)
        DoNotReset[refit_metric] = True   
        svm_model = SVC(kernel="rbf",tol=tol,cache_size=cache_size) if SVCclassification \
                        else SVR(kernel="rbf",tol=tol,cache_size=cache_size) 

        with redirect_stdout(open(OutPutTxtFile, 'a')):
            print(f"Training date: {datetime.now()}")
            print("="*100)
            print(f"Refit metrics: {refit_metric}")
            print("*"*75)
            if SVCclassification:
                print("* Model training for bandgap nature prediction [SVC(RBF)]:")
                filename = svc_EgNature+'.sav'
            else:
                print("* Model training for bandgap magnitude prediction [SVR(RBF)]:")
                filename = svr_bandgap+'.sav'
                
            print(f'Training dataset size: {X_train.shape[0]}')
            print(f'Testing dataset size: {X_test.shape[0]}')
            refit_model = Pipeline(steps=[("scaler", scaler), ("svm", svm_model)])
            best_index_ = svmgrid.cv_results_[f"rank_test_{refit_metric}"].argmin()
            best_score_ = svmgrid.cv_results_[f"mean_test_{refit_metric}"][best_index_]
            best_params_ = svmgrid.cv_results_['params'][best_index_]
            print("Best parameter (CV score=%0.6f ):" % best_score_, best_params_)
            print("The grid_search cross-validation: %d fold" % svmgrid.n_splits_)
            refit_model.set_params(**best_params_)
            refit_model.fit(X_train, y_train)
            y_svm = refit_model.predict(X_test)
            Y_predict_all = refit_model.predict(X_full)
            
            if refit_metric in ['my_rmse_fix','my_rmse_abs']:
                y_svm = UpdatePredictionValues(y_svm, refit_metric)
                Y_predict_all = UpdatePredictionValues(Y_predict_all, refit_metric)
            
            if SVCclassification:
                Classes = refit_model['svm'].classes_
                print("The classes are:", Classes)
                for I in Classes:
                    Number1 = np.count_nonzero(y_train==I)
                    Number2 = np.count_nonzero(y_test==I)
                    print(f'Samples in class-{I}: TrainSet-{Number1};TestSet-{Number2}; Total-{Number1+Number2}')
                    
                print("The class weights:", refit_model['svm'].class_weight_)
    
            nsupports = refit_model['svm'].n_support_
            print(f"Number of support vectors: Total={sum(nsupports)} ;  for each class=", nsupports)
    
            if SVCclassification:
                PrintSVC_output(y_test, y_svm, Y_full, Y_predict_all)
                if PlotResults:
                    confucion_display_labels = LabelConversionFn(refit_model['svm'].classes_)
                    PlotConfusionMatrix(y_test, y_svm, confucion_display_labels,save=save, savepath=savepath, figname='/NatureTestSet'+figname)
                    PlotConfusionMatrix(Y_full, Y_predict_all, confucion_display_labels,save=save, savepath=savepath, figname='/NatureFullSet'+figname)
                    
                if SVRdependentSVC:
                    print("*"*75)
                    print('The SVR prediction using the optimized hyperparameters from SVC (SVCdependentSVR)::')
                    SVREgModel = SVR(kernel="rbf",tol= 1e-5,cache_size=10000)
                    svr_pipe = Pipeline(steps=[("scaler", StandardScaler()), ("svm", SVREgModel)])
                    svr_pipe.set_params(**best_params_)
                    svr_pipe.fit(X_train, y_train_SVRSVC)
                    y_svrsvc = svr_pipe.predict(X_test)
                    y_svrsvc_all = svr_pipe.predict(x_all_SVRSVC)
                    
                    if refit_metric in ['my_rmse_fix','my_rmse_abs']:
                        y_svrsvc = UpdatePredictionValues(y_svrsvc, refit_metric)
                        y_svrsvc_all = UpdatePredictionValues(y_svrsvc_all, refit_metric)
                        
                    PrintSVR_output(y_test_SVRSVC, y_svrsvc, y_all_SVRSVC, y_svrsvc_all,add_extra='svrsvc_')
                    print(" ")
                    if PlotResults:
                        Plot_Prediction_Actual_results(y_test_SVRSVC, y_svrsvc, tsxt=None,data_unit_label=svr_data_unit,save=save, savepath=savepath, figname='TestSet_SVCdependentSVR'+figname)   
                        plot_err_dist(y_test_SVRSVC, y_svrsvc, data_unit_label=svr_data_unit,save=save, savepath=savepath, figname='TestSetTruePredictErrorHist_SVCdependentSVR.png')
                        Plot_Prediction_Actual_results(y_all_SVRSVC,y_svrsvc_all, tsxt=None,data_unit_label=svr_data_unit,save=save, savepath=savepath, figname='FullSet_SVCdependentSVR'+figname) 
                        plot_err_dist(y_all_SVRSVC,y_svrsvc_all, data_unit_label=svr_data_unit,save=save, savepath=savepath, figname='FullSetTruePredictErrorHist_SVCdependentSVR.png')
            else:
                PrintSVR_output(y_test, y_svm, Y_full, Y_predict_all)
                print(" ")
                if PlotResults:
                    Plot_Prediction_Actual_results(y_test, y_svm, tsxt=None,data_unit_label=svr_data_unit,save=save, savepath=savepath, figname='TestSet'+figname)     #tsxt='Bandgap prediction on test set'
                    plot_err_dist(y_test, y_svm, data_unit_label=svr_data_unit,save=save, savepath=savepath, figname='TestSetTruePredictErrorHist.png')
                    Plot_Prediction_Actual_results(Y_full,Y_predict_all, tsxt=None,data_unit_label=svr_data_unit,save=save, savepath=savepath, figname='FullSet'+figname) #tsxt='Bandgap prediction over all data'
                    plot_err_dist(Y_full,Y_predict_all, data_unit_label=svr_data_unit,save=save, savepath=savepath, figname='FullSetTruePredictErrorHist.png')
                    
                if SVRdependentSVC:
                    print("*"*75)
                    print('The SVC prediction using the optimized hyperparameters from SVR (SVRdependentSVC)::')
                    SVCEgNatureModel = SVC(kernel="rbf",tol= 1e-5,cache_size=10000)
                    svc_pipe = Pipeline(steps=[("scaler", StandardScaler()), ("svm", SVCEgNatureModel)])
                    svc_pipe.set_params(**best_params_)
                    svc_pipe.fit(X_train, y_train_SVRSVC)
                    y_svc = svc_pipe.predict(X_test)
                    y_svc_all = svc_pipe.predict(x_all_SVRSVC)
                    PrintSVC_output(y_test_SVRSVC, y_svc, y_all_SVRSVC, y_svc_all,add_extra='svrsvc_')
                    if PlotResults:
                        confucion_display_labels = LabelConversionFn(svc_pipe['svm'].classes_)
                        PlotConfusionMatrix(y_test_SVRSVC, y_svc, confucion_display_labels,save=save, savepath=savepath, figname='/NatureTestSet_SVRdependentSVC'+figname)
                        PlotConfusionMatrix(y_all_SVRSVC, y_svc_all, confucion_display_labels,save=save, savepath=savepath, figname='/NatureFullSet_SVRdependentSVC'+figname)
                
            if save_model:
                pickle.dump(refit_model, open(filename, 'wb'))
                print(" ")
                print("*"*75)
                print(f"* Model saved: \n\t{filename}")
                print("*"*75) 
                
        if DumpData:
            DumpDf = [X_test,y_test]
            if SVRdependentSVC:
                DumpDf += [y_test_SVRSVC]
            with sq.connect(DataFolderPath+'/TestSetData.db') as conn:
                pd.concat(DumpDf,join='outer',axis=1).to_sql('TestSet',conn,if_exists='replace')
                pd.DataFrame(svmgrid.cv_results_).drop(columns='params',inplace=False).to_sql('HyperParameters',conn,if_exists='replace')
            
def savemodel(BestModel, X, Y, filename, refit, multiregression=False, \
                               regressorchain=False, SVCclassification=False,\
                                   scoringfn='r2', njobs=-1, save=True):        
    if refit:
        npipe = BestModel
    else:
        # Define a Standard Scaler to normalize inputs
        scaler = StandardScaler()
        # Set model

        svm_model = SVC(kernel="rbf") if SVCclassification else SVR(kernel="rbf")

        if regressorchain: multiregression=False
        # Set model
        if multiregression:
            svm_model = MultiOutputRegressor(svm_model) # wrapper
        elif regressorchain:
            svm_model = RegressorChain(svm_model)
        else:
            pass
    
        # Set pipeline
        npipe = Pipeline(steps=[("scaler", scaler), ("svm", svm_model)])
        
        # Parameters of pipelines can be set using ‘__’ separated parameter names:
        npipe.set_params(**BestModel)
        
    # print("-----------------------------------------------")
    # # print('Model details:', npipe.get_params())
    # # print("-----------------------------------------------")
    # cvsplitt = 10
    # print(f"Evaluating {cvsplitt} cross validation over whole data set.")
    # kfold = KFold(n_splits=cvsplitt, random_state=1, shuffle=True)
    # if isinstance(scoringfn, str):   
    #     cv_results = cross_val_score(npipe, X, Y, cv=kfold, scoring=scoringfn, n_jobs=njobs)
    #     print(f"Score over all samples ({scoringfn}): {cv_results.mean():.3f}±{cv_results.std():.3f}")
    # else:
    #     cv_results = cross_validate(npipe, X, Y, cv=kfold, scoring=scoringfn, n_jobs=njobs)
    #     for I in scoringfn:
    #         print(f"Score over all samples ({I}): {cv_results['test_'+I].mean():.3f}±{cv_results['test_'+I].std():.3f}")
    
    # npipe.fit(X, Y)
    
    if save:
        pickle.dump(npipe, open(filename, 'wb'))
        print(f"* Model saved: {filename}")
    print("***************************************************************************\n")
    return

def my_ml_model_training(df, filename,DoNotReset, xfeatures=['ANTIMONY','STRAIN'], yfeatures=['BANDGAP', 'SO'],
                         refit=True, multiregression=False, regressorchain=False, REPEAT_loop=1,
                         SVCclassification=False, ydatadivfac=1,scoringfn='r2', save_model=True,
                         PlotResults=False, LearningCurve=False,SplitInputData=True,test_set=None,
                         saveFig=False, savepath='.', figname='TruePrediction.png',svr_data_unit='eV',
                         LearningCurveT3=False,SVRdependentSVC=False,LearningCurveV0=False,
                         dirpathSystem_='.',BinaryConversion=False,SaveResults=False):
    if LearningCurveV0:
        df__ = GenerateDataV2(df, xfeatures=xfeatures, yfeatures=yfeatures)
        SUBDIVISION = max(10, len(df__)//100) # Create subset with 100 or 0.1% whichever is small, elements
        if len(df__) < SUBDIVISION: 
            print("Less than 10 elements are there in set. Not recomended for subsetiing that small number of elements.")
            SUBDIVISION = 1
        print(f"The the input set will be divided into {SUBDIVISION} subset.")
        df_subset = np.array_split(df__, SUBDIVISION) 
        RANdomPICK = np.arange(len(df_subset))
        for loop in range(REPEAT_loop):
            print(f"TEST4 repetation loop: {loop}")
            print("x"*50)
            np.random.shuffle(RANdomPICK)
            TRAINSET = df_subset[RANdomPICK[0]]
            for I in range(len(RANdomPICK)):
                if I>0: TRAINSET = pd.concat([TRAINSET,df_subset[RANdomPICK[I]]],ignore_index=True)
                X, y = GenerateData(TRAINSET, xfeatures=xfeatures, yfeatures=yfeatures)
                y = y/ydatadivfac
                _ = mysvrmodel_parametersearch_V0(X, y, refit=refit, SplitInputData=SplitInputData, svr_data_unit=svr_data_unit,
                                                  multiregression=multiregression,regressorchain=regressorchain,
                                                  SVCclassification=SVCclassification,scoringfn=scoringfn,PlotResults=False)
    elif LearningCurve:
        # df__ = GenerateDataV2(df, xfeatures=xfeatures, yfeatures=yfeatures)
        X, y = GenerateData(df, xfeatures=xfeatures, yfeatures=yfeatures)
        y = y/ydatadivfac
        splits_array = [0.01,0.025,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75]
        ShuffleSplit_n_splits = 5
        print(f"Total convergence test points: {len(splits_array)}; Training sizes are:",*splits_array)
        print(f"ShuffleSplit: {ShuffleSplit_n_splits}-fold")
        save_model_ = False
        PlotResults_ = False
        save_final_data = False
        SVRSVC_y_values = None
        dirpathSystem_tmp = dirpathSystem_
        for loop_n, train_size_ in enumerate(splits_array):
            print(f'Progress loop: {loop_n}, Train setsize = {train_size_}')
            #random_state=0 preserve the test set for all loop(/training set).
            #random_state=None change the test set for each loop(/training set).
            ss = ShuffleSplit(n_splits=ShuffleSplit_n_splits, train_size=train_size_, test_size=0.25, random_state=0) 
            ii = 1
            if loop_n == (len(splits_array)-1):
                save_model_ = True
                PlotResults_ = True
                save_final_data = True
                SVRdependentSVC = True
                SVRSVC_y_values = df['BANDGAP'].copy() if SVCclassification else df['NATURE'].copy()
            for train, test in ss.split(X):
                print(f"\tTrial: {ii}")
                X_train_, x_test, y_train_, y_test = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
                if loop_n == (len(splits_array)-1): 
                    dirpathSystem_tmp = f'{dirpathSystem_}/BPD/Trial{ii}/'
                if isinstance(refit,bool) or isinstance(refit,str):
                    _ = mysvrmodel_parametersearch_V0(X_train_, y_train_, refit=refit, SplitInputData=False, x_test=x_test, y_test=y_test,
                                                      multiregression=multiregression,regressorchain=regressorchain,svr_data_unit=svr_data_unit,
                                                      SVCclassification=SVCclassification,scoringfn=scoringfn,PlotResults=False)
                else:
                    _ = mysvrmodel_parametersearch(X_train_, y_train_, scoringfn, DoNotReset, SplitInputData=False, x_test=x_test, y_test=y_test,
                                                   multiregression=multiregression,regressorchain=regressorchain,save_model=save_model_,
                                                   SVCclassification=SVCclassification,scoringfn=scoringfn,PlotResults=PlotResults_,DumpData=save_final_data,
                                                   dirpathSystem_=dirpathSystem_tmp,BinaryConversion=BinaryConversion,svr_data_unit=svr_data_unit,
                                                   SVRSVC_y_values=SVRSVC_y_values,SVRdependentSVC=SVRdependentSVC,save=True)
                ii += 1
            
                
    elif LearningCurveT3:
        if SplitInputData:
            x_test, y_test = None, None
        else:
            test_sett = test_set.copy()
            reserved_test_set = test_sett.sample(frac=0.25,axis=0,ignore_index=False)
            test_sett.drop(reserved_test_set.index, inplace=True)
            test_sett.reset_index(drop=True)
            reserved_test_set.reset_index(drop=True, inplace=True)
            x_test, y_test = GenerateData(reserved_test_set, xfeatures=xfeatures, yfeatures=yfeatures)
            
        df__ = GenerateDataV2(test_sett, xfeatures=xfeatures, yfeatures=yfeatures)
        SUBDIVISION = 20 # Constant 5% split
        # SUBDIVISION = max(10, len(df__)//100) # Create subset with 100 or 10% whichever is small, elements
        ExtraElemenst = 0
        tmp_extra_df = pd.DataFrame()
        if len(df__) < SUBDIVISION: 
            print("Less than 10 elements are there in set. Not recomended for subsetiing that small number of elements.")
            SUBDIVISION = 1  
        else:
            TotalDataSet = len(df__)
            n_split = TotalDataSet//SUBDIVISION
            ExtraElemenst = TotalDataSet%n_split
            if ExtraElemenst > 0: tmp_extra_df = df__[:ExtraElemenst]
            
        print(f"The quaternary set will be divided into {SUBDIVISION} subset.")
        df_subset = np.split(df__[ExtraElemenst:], SUBDIVISION) 
        RANdomPICK = np.arange(len(df_subset))
        
        for loop in range(REPEAT_loop):
            print(f"Trial: {loop+1}")
            np.random.shuffle(RANdomPICK)
            TRAINSET = df.copy()
            AppendTestSET = None
            for I in range(len(RANdomPICK)+1):
                print(f"\tProgress loop: {I}")
                if I>0:
                    TRAINSET = pd.concat([TRAINSET,df_subset[RANdomPICK[I-1]]],ignore_index=True)
                if SplitInputData:
                    if I == len(RANdomPICK):
                        AppendTestSET = None
                    else:
                        tmp = pd.concat([tmp_extra_df,pd.concat([df_subset[IX] for IX in RANdomPICK[I:]],ignore_index=True)],ignore_index=True)
                        AppendTestSET = GenerateData(tmp, xfeatures=xfeatures, yfeatures=yfeatures)
                    
                X, y = GenerateData(TRAINSET, xfeatures=xfeatures, yfeatures=yfeatures)
                y = y/ydatadivfac
                if isinstance(refit,bool) or isinstance(refit,str):
                    _ = mysvrmodel_parametersearch_V0(X, y, refit=refit, SplitInputData=SplitInputData, AppendTestSET=AppendTestSET,x_test=x_test, y_test=y_test,
                                                      multiregression=multiregression, regressorchain=regressorchain,svr_data_unit=svr_data_unit,
                                                      SVCclassification=SVCclassification,scoringfn=scoringfn,PlotResults=False)
                else:
                    _ = mysvrmodel_parametersearch(X, y, scoringfn, DoNotReset, SplitInputData=SplitInputData, AppendTestSET=AppendTestSET,
                                                   multiregression=multiregression, regressorchain=regressorchain,svr_data_unit=svr_data_unit,
                                                   SVCclassification=SVCclassification,scoringfn=scoringfn,PlotResults=False,
                                                   dirpathSystem_=dirpathSystem_,BinaryConversion=BinaryConversion,
                                                   x_test=x_test, y_test=y_test)
    else:    
        X, y = GenerateData(df, xfeatures=xfeatures, yfeatures=yfeatures)
        y = y/ydatadivfac
        if SplitInputData:
            x_test, y_test = None, None
        else:
            assert test_set is not None, 'Must provide test set'
            x_test, y_test = GenerateData(test_set, xfeatures=xfeatures, yfeatures=yfeatures)
        
        if SVRdependentSVC:
            SVRSVC_y_values = df['BANDGAP'].copy() if SVCclassification else df['NATURE'].copy()
        else:
            SVRSVC_y_values = None
        
        if isinstance(refit,bool) or isinstance(refit,str):
            BestParameters, _ = mysvrmodel_parametersearch_V0(X, y, refit=refit, SplitInputData=SplitInputData,\
                                                              multiregression=multiregression, x_test=x_test, y_test=y_test,\
                                                                  regressorchain=regressorchain,SVRSVC_y_values=SVRSVC_y_values,\
                                                                      SVCclassification=SVCclassification,svr_data_unit=svr_data_unit,\
                                                                          scoringfn=scoringfn,PlotResults=PlotResults,SVRdependentSVC=SVRdependentSVC,
                                                                          save=saveFig, savepath=savepath, figname=figname)
            _ = savemodel(BestParameters, X, y, filename, refit, multiregression=multiregression, \
                          regressorchain=regressorchain,\
                              SVCclassification=SVCclassification,\
                                  scoringfn=scoringfn, save=save_model)
        else:
            mysvrmodel_parametersearch(X, y, scoringfn,DoNotReset, SplitInputData=SplitInputData,svr_data_unit=svr_data_unit,
                                       multiregression=multiregression, x_test=x_test, y_test=y_test,
                                       regressorchain=regressorchain,SVRSVC_y_values=SVRSVC_y_values,
                                       SVCclassification=SVCclassification,save_model=save_model,
                                       scoringfn=scoringfn,PlotResults=PlotResults,SVRdependentSVC=SVRdependentSVC,
                                       save=saveFig, savepath=savepath, figname=figname,DumpData=SaveResults,
                                       dirpathSystem_=dirpathSystem_,BinaryConversion=BinaryConversion)            

    return 

        
def CheckScoringfn(scoringfn):
    if ('r2' in scoringfn) or ('mean' in scoringfn) or ('error' in scoringfn):
        return True
    else:
        return False
    
def Set2Default(scoringfn):    
    if isinstance(scoringfn, str):
        Set2Defaultt=CheckScoringfn(scoringfn)
    elif scoringfn is not None:
        for I in scoringfn:
            Set2Defaultt=CheckScoringfn(I)
            if Set2Defaultt: break
    else:
        Set2Defaultt=True
    return Set2Defaultt

#**************** Model training **********************************************
def SVMModelTrainingFunctions(df, xfeatures, DoNotResetfolder, yfeatures=None, scaley=1, 
                           multiregression=False, regressorchain=False, 
                           IndependentOutput=False, retrainbw=False, retraineg=False,
                           retrainnature=False, retrainlp=False,SVRdependentSVC=False,
                           svr_bandgap=None, svr_bw=None, RepeatLearningCurveTimes=1,
                           svr_bw_dm=None, svc_EgNature=None, svr_lp=None,SaveResults=False,
                           scoringfn='r2',refit=True,SplitInputData=True,test_set=None,
                           save_model=True,PlotResults=False, LearningCurve=False,LearningCurveT3=False,
                           saveFig=False, savepath='.', figname='TruePrediction.png',
                           dirpathSystem_='.',BinaryConversion=False):
    """
    This function trains the different SVM() models.

    Parameters
    ----------
    df : Pandas dataframe
        The dataframe consisting the feature values.
    xfeatures : String list
        The input feature names list. Should be from df.column names.
    yfeatures : String list, optional
        The output feature names list, when Bloch weight(+) model training is requestd.
        Along with Bloch weights the bandgap and natures can also be trained within 
        same model. 
        Should be from df.column names.  
        The default is None. yfeatures must has to be supplied whenretrainbw is True.
    scaley : Float, optional
        The scaling factor for output features (yfeatures). The default is 1.
    multiregression : Bool, optional
        Whether to use MultiOutputRegressor() model. The default is False. If 
        multiregression and regressorchain both are false then direct (multi)output model
        will be trained by default.
    regressorchain : Bool, optional
        Whether to use RegressorChain() model. The default is False. If 
        multiregression and regressorchain both are false then direct (multi)output model
        will be trained by default.
    IndependentOutput : Bool, optional
        Whether to use direct (multi)output model. The default is False.
    retrainbw : Bool, optional
        Whether to retrain the Bloch weights(+). The default is False.
    retraineg : Bool, optional
        Whether to retrain Bandgaps. The default is False.
    retrainnature : Bool, optional
        Whether to retrain Bandgap natures. The default is False.
    retrainlp : Bool, optional
        Whether to retrain lattice parameter. The default is False.
    svr_bandgap : String, optional
        The file path name for saving Bandgap trained SVR() model. 
        The default is None. The filename must has to be supplied if retraineg is True.
    svr_bw : String, optional
        The file path name for saving Bloch weights(+) trained SVR() model. 
        The default is None. The filename must has to be supplied if retrainbw is True.
    svr_bw_dm : String, optional
        The file path name for saving Bloch weights(+) trained direct multioutput SVR() model. 
        The default is None. If the filename is not supplied then the 'svr_bw' will be
        used.
    svc_EgNature : String, optional
        The file path name for saving Bandgap Nature trained SVC() model. 
        The default is None. The filename must has to be supplied if retrainnature is True.
    svr_lp : String, optional
        The file path name for saving lattice parameter trained SVR() model. 
        The default is None. The filename must has to be supplied if retrainlp is True.
    scoringfn : scikitlearn scoring functions, optional
        The scoring function for ML model. The defalut is 'r2' for refression and
        'accuracy' for classification task.
    refit : bool, str, or callable, optional
        Refit an estimator using the best found parameters on the whole dataset.
        For multiple metric evaluation, this needs to be a str denoting the scorer 
        that would be used to find the best parameters for refitting the estimator at the end.
        Default is True.
    save_model : bool, optional
        Whether to save the trained model. Default is True.
    PlotResults : bool, optional
        Whether to plot the croos validation results. Default is False.
    LearningCurve : bool, optional
        If you want to get learning curve. Default is False.

    Returns
    -------
    None.

    """
    
    # Training bandgap SVR() model
    if retraineg: 
        assert svr_bandgap is not None, 'Bandgap model training is requested but no filename is supplied for saving final model.'
        print("***************************************************************************")
        print("* Model training for bandgap prediction (SVR):")
        _ = my_ml_model_training(df, svr_bandgap+'.sav', DoNotResetfolder,xfeatures=xfeatures, test_set=test_set,REPEAT_loop=RepeatLearningCurveTimes,
                                 yfeatures='BANDGAP', scoringfn=scoringfn,refit=refit,SplitInputData=SplitInputData,
                                 save_model=save_model,PlotResults=PlotResults,LearningCurve=LearningCurve,LearningCurveT3=LearningCurveT3,
                                 saveFig=saveFig, savepath=savepath, figname=figname,SVRdependentSVC=SVRdependentSVC,
                                 dirpathSystem_=dirpathSystem_,BinaryConversion=BinaryConversion,svr_data_unit='eV',SaveResults=SaveResults)
    # Training bandgap SVR() model
    if retrainlp: 
        assert svr_lp is not None, 'Lattice parameter model training is requested but no filename is supplied for saving final model.'
        
        yfeatures_='LATTICEPARAMETER1' # <-- LATTICEPARAMETER1 is the in-plane lattice parameter. If not change this.

        print("***************************************************************************")
        print("* Model training for lattice parameter prediction (SVR):")
        print(f"The model will learn and predict the {yfeatures_}. Make sure this is what you want to learn.")
        _ = my_ml_model_training(df, svr_lp+'.sav',DoNotResetfolder, xfeatures=xfeatures, 
                                 yfeatures=yfeatures_, scoringfn=scoringfn,refit=refit,SaveResults=SaveResults,
                                 saveFig=saveFig, savepath=savepath, figname=figname,svr_data_unit='$\AA$',
                                 save_model=save_model,PlotResults=PlotResults,LearningCurve=LearningCurve)
    
    # Training Bloch weights SVR() model    
    if retrainbw: 
        assert svr_bw is not None, 'Bloch weight model training is requested but no filename is supplied for saving final model.'
        assert yfeatures is not None, 'Bloch weight model training is requested but no output features(yfeatures) are supplied.'
        print("***************************************************************************")
        print("* Model training for Bloch weight prediction (SVR):")
        if multiregression:
            print("Model type: MultiRegression")
            _ = my_ml_model_training(df, svr_bw+'_multiregression.sav', DoNotResetfolder,
                                     xfeatures=xfeatures, yfeatures=yfeatures,
                                     multiregression = multiregression, 
                                     ydatadivfac=scaley, scoringfn=scoringfn,refit=refit,
                                     save_model=save_model,PlotResults=PlotResults)
        if regressorchain:
            print("Model type: RegressorChain")
            _ = my_ml_model_training(df, svr_bw+'_regressorchain.sav', DoNotResetfolder,
                                     xfeatures=xfeatures, yfeatures=yfeatures,
                                     regressorchain=regressorchain, 
                                     ydatadivfac=scaley, scoringfn=scoringfn,refit=refit,
                                     save_model=save_model,PlotResults=PlotResults)
        # Direct Multioutput: Independent model for each numerical value to be predicted
        if ((not multiregression) and (not regressorchain)) or (IndependentOutput):
            print("Model type: Direct independent multioutput")
            if svr_bw_dm is not None:  svr_bw = svr_bw_dm
            for I in yfeatures:
                save_f = svr_bw + I + '.sav'
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("* Fitting feature %s"%I)
                _ = my_ml_model_training(df, save_f, DoNotResetfolder,xfeatures=xfeatures, 
                                         yfeatures=I, ydatadivfac=scaley,
                                         scoringfn=scoringfn,refit=refit,
                                         save_model=save_model,PlotResults=PlotResults)
                print("\n")
    
    # Training bandgap nature SVC() model
    if retrainnature:
        assert svc_EgNature is not None, 'Bandgap Nature model training is requested but no filename is supplied for saving final model.'
        yfeature_='NATURE'
                
        if Set2Default(scoringfn): scoringfn = 'accuracy'; refit = True
                
        print("***************************************************************************")
        print("* Model training for bandgap nature prediction (SVC):")
        if multiregression:
            print("Model type: MultiRegression")
            _ = my_ml_model_training(df, svc_EgNature+'_multiregression.sav', DoNotResetfolder,
                                     xfeatures=xfeatures, yfeatures=yfeature_,
                                    multiregression = multiregression, 
                                    SVCclassification=True,scoringfn=scoringfn,
                                    refit=refit,save_model=save_model,PlotResults=PlotResults)
        if regressorchain:
            print("Model type: RegressorChain")
            _ = my_ml_model_training(df, svc_EgNature+'_regressorchain.sav', DoNotResetfolder,
                                     xfeatures=xfeatures, yfeatures=yfeature_,
                                     regressorchain=regressorchain,
                                     SVCclassification=True,scoringfn=scoringfn,
                                     refit=refit,save_model=save_model,PlotResults=PlotResults)
            
        if ((not multiregression) and (not regressorchain)) or (IndependentOutput):
            print("Model type: Direct independent multioutput")
            _ = my_ml_model_training(df, svc_EgNature+'.sav', DoNotResetfolder,
                                     xfeatures=xfeatures, SplitInputData=SplitInputData,REPEAT_loop=RepeatLearningCurveTimes,
                                     SVCclassification=True, yfeatures=yfeature_,test_set=test_set,
                                     scoringfn=scoringfn,refit=refit,save_model=save_model,LearningCurveT3=LearningCurveT3,
                                     PlotResults=PlotResults,LearningCurve=LearningCurve,
                                     saveFig=saveFig, savepath=savepath, figname=figname,SVRdependentSVC=SVRdependentSVC,
                                     dirpathSystem_=dirpathSystem_,BinaryConversion=BinaryConversion,SaveResults=SaveResults)
    return

#---------------- test binary models with ternary points ----------------------
def TestModelTernaryPoints(TestPoints_X, y_test, SVMModel, SVCclassification=False):
    y_svm = SVMModel.predict(TestPoints_X)
    if SVCclassification:
        y_svm = y_svm.astype(int)
        print(f"The accuracy_score for prediction: {accuracy_score(y_test, y_svm):.3f}")
    else:
        print(f"The r2_score for prediction: {r2_score(y_test, y_svm):.3f}")
        print(f"The root_mean_squared_error for prediction: {mean_squared_error(y_test, y_svm,squared=False):.3f} eV")
    return y_svm