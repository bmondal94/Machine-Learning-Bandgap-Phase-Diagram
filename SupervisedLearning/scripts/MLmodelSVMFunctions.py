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
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, cross_validate
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.metrics import r2_score, mean_squared_error, classification_report, \
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error,max_error
import time
import inspect, pickle
import matplotlib.pyplot as plt

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
    
def PlotCV_results(svrgrid, SVRgridparameters_c, SVRgridparameters_gamma):
    #https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
    SVRgridparameters = svrgrid.cv_results_['params']
    # print(svrgrid.cv_results_)
    SVRgridparameters_mean_score =  svrgrid.cv_results_['mean_test_score']
    scores = SVRgridparameters_mean_score.reshape(len(SVRgridparameters_c), len(SVRgridparameters_gamma))
    plt.figure(figsize=(8, 6))
    #plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(
        scores,
        interpolation="nearest",
        #cmap=plt.cm.hot,
        #norm=MidpointNormalize(vmin=0.2, midpoint=0.92),
       
    )
    plt.xlabel("gamma")
    plt.ylabel("C")
    plt.colorbar( label='score')
    plt.xticks(np.arange(len(SVRgridparameters_gamma)), SVRgridparameters_gamma, rotation=45)
    plt.yticks(np.arange(len(SVRgridparameters_c)), SVRgridparameters_c)
    plt.title("Validation accuracy")
    plt.show()

    return 

def Plot_Prediction_Actual_results(X, Y, tsxt=None, save=False, savepath='.', figname='TruePrediction.png'):
    plt.figure(figsize=(8, 8))
    plt.xlabel("True values (eV)")
    plt.ylabel("Predictions (eV)")
    plt.scatter(X, Y)

    lim_min, lim_max = min(X), max(X)
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)
    plt.plot([lim_min, lim_max],[lim_min, lim_max], color='k')
    
    plt.title(tsxt)
    DIFF_ = abs(X - Y)
    # ax.plot([],[],' ',label=f"Max error = {max(DIFF_):.2f} eV \n {'MAE':>11} = {np.mean(DIFF_):.2f}±{np.std(DIFF_):.2f} eV")
    plt.plot([],[],' ',label=f"Max error = {max(DIFF_):.2f} eV \nMAE = {np.mean(DIFF_):.2f}±{np.std(DIFF_):.2f} eV")
    plt.legend(handlelength=0)
    plt.tight_layout()
    if save:
        plt.savefig(savepath+'/'+figname,bbox_inches = 'tight',dpi=300)
        plt.close()
    else:
        plt.show()
    


def GenerateData(df, xfeatures=['ANTIMONY','STRAIN'], yfeatures=['BANDGAP', 'SO']):        
    return df[xfeatures], df[yfeatures]

def mysvrmodel_parametersearch(X, y, refit = True, multiregression=False, 
                               regressorchain=False, SVCclassification=False,
                               scoringfn='r2', njobs=-1, PlotResults=False,
                               save=False, savepath='.', figname='TruePrediction.png'):
    # Fit regression model
    # Radial Basis Function (RBF) kernel
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=None, test_size=0.25)
    
    # Define a Standard Scaler to normalize inputs
    scaler = StandardScaler()
    tol = 1e-5;cache_size=10000
    svr = SVC(kernel="rbf",tol=tol,cache_size=cache_size) if SVCclassification \
        else SVR(kernel="rbf",tol=tol,cache_size=cache_size)
    
    if regressorchain: multiregression=False
    # Set model
    if multiregression:
        svr = MultiOutputRegressor(svr) # wrapper
    elif regressorchain:
        svr = RegressorChain(svr)
    else:
        pass
    
    # Set pipeline
    pipe = Pipeline(steps=[("scaler", scaler), ("svrm", svr)])
    
    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    # C_range = [1e0, 1e1, 1e2, 1e3]
    # gamma_range = np.logspace(-2, 2, 5)
    C_range = [1e0, 1e1, 50, 1e2, 500, 1e3]
    gamma_range = [1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 5.e+01, 1.e+02]
    param_grid={"svrm__C": C_range, 
                "svrm__gamma": gamma_range}
    if multiregression:
        param_grid['svrm__estimator__C'] = param_grid.pop('svrm__C')
        param_grid['svrm__estimator__gamma'] = param_grid.pop("svrm__gamma")
    elif regressorchain:
        param_grid['svrm__base_estimator__C'] = param_grid.pop('svrm__C')
        param_grid['svrm__base_estimator__gamma'] = param_grid.pop("svrm__gamma")
    else:
        pass

    svrgrid = GridSearchCV(estimator=pipe,
                           param_grid=param_grid,
                           cv = 5,
                           scoring = scoringfn,
                           n_jobs=njobs,
                           refit=refit)
    
    t0 = time.time()
    svrgrid.fit(X_train, y_train)
    svr_fit = time.time() - t0
    print("SVR complexity and bandwidth selected and model fitted in %.3f s" % svr_fit)
    print(f'Training dataset size: {X_train.shape[0]}')
   
    #print('All cv results:',svrgrid.cv_results_)
    
    if PlotResults:
        _ = PlotCV_results(svrgrid, C_range, gamma_range)
    
    print("Best parameter (CV score=%0.3f):" % svrgrid.best_score_, svrgrid.best_params_)
    if SVCclassification:
        Classes = svrgrid.best_estimator_['svrm'].classes_
        print("The classes are:", Classes)
        for I in Classes:
            Number1 = np.count_nonzero(y_train==I)
            Number2 = np.count_nonzero(y_test==I)
            print(f'Samples in class-{I}: TrainSet-{Number1};TestSet-{Number2}; Total-{Number1+Number2}')
            
        print("The class weights:", svrgrid.best_estimator_['svrm'].class_weight_)
    nsupports = svrgrid.best_estimator_['svrm'].n_support_
    print(f"Number of support vectors: Total={sum(nsupports)} ;  for each class=", nsupports)
    #print('* The scorer used:', inspect.getsource(svrgrid.scorer_))

    if refit:
        print('Refitting timing for the best model on the whole training dataset:%.3f s' % svrgrid.refit_time_)
        print("The number of cross-validation splits (folds/iterations): %d" % svrgrid.n_splits_)
        
    
        t0 = time.time()
        y_svr = svrgrid.predict(X_test)
        svr_predict = time.time() - t0
        Y_predict_all = svrgrid.predict(X)
        print("SVR prediction for %d (test) inputs in %.3f s" % (X_test.shape[0], svr_predict))
        if SVCclassification:
            print(f"The out-of-sample accuracy score for prediction: {accuracy_score(y_test, y_svr):.3f}")
            
            print(f"The accuracy score for prediction: {accuracy_score(y_test, y_svr):.3f}")
            if PlotResults:
                cm1 = confusion_matrix(y_test, y_svr, labels=svrgrid.best_estimator_['svrm'].classes_)
                disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1,
                                               display_labels=svrgrid.best_estimator_['svrm'].classes_)
                disp1.plot()
                disp1.ax_.set_title('Bandgap nature prediction\n1=direct, 0=indirect')
                cm2 = confusion_matrix(y, Y_predict_all, labels=svrgrid.best_estimator_['svrm'].classes_)
                disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2,
                                               display_labels=svrgrid.best_estimator_['svrm'].classes_)
                disp2.plot()
                disp2.ax_.set_title('Bandgap nature prediction\n1=direct, 0=indirect')
                if save:
                    disp1.figure_.savefig(savepath+'/NatureTestSet'+figname,bbox_inches = 'tight',dpi=300)
                    disp2.figure_.savefig(savepath+'/Nature'+figname,bbox_inches = 'tight',dpi=300)
                    plt.close()
                else:
                    plt.show()   
                #plt.show()
            
            print('Classification report:\n',classification_report(y_test, y_svr))
            
        else:
            
            print(f"The out-of-sample r2_score for prediction: {r2_score(y_test, y_svr):.3f}")
            print(f"The out-of-sample mean_squared_error for prediction: {mean_squared_error(y_test, y_svr):.3f} eV2")
            print(f"The out-of-sample mean_absolute_error for prediction: {mean_absolute_error(y_test, y_svr):.3f} eV")
            print(f"The out-of-sample max_error for prediction: {max_error(y_test, y_svr):.3f} eV")
            print('\n')
            print(f"The all-data r2_score for prediction: {r2_score(y,Y_predict_all):.3f}")
            print(f"The oall-data mean_squared_error for prediction: {mean_squared_error(y,Y_predict_all):.3f} eV2")
            print(f"The all-data mean_absolute_error for prediction: {mean_absolute_error(y,Y_predict_all):.3f} eV")
            print(f"The all-data max_error for prediction: {max_error(y,Y_predict_all):.3f} eV")
            if PlotResults:
                Plot_Prediction_Actual_results(y_test, y_svr, tsxt='Bandgap prediction on test set',save=save, savepath=savepath, figname='TestSet'+figname)     
                Plot_Prediction_Actual_results(y,Y_predict_all, tsxt='Bandgap prediction over all data',save=save, savepath=savepath, figname=figname)
        model_best_scorer = scoringfn if isinstance(refit,bool)  else refit
        print(f"Out-of-sample score on prediction using the 'default' scorer ({model_best_scorer}): {svrgrid.score(X_test, y_test): .3f}")
        #print("Number of support vectors:", len(svrgrid.best_estimator_['svrm'].n_support_))
        return svrgrid.best_estimator_,svrgrid.best_score_
    else:
        return svrgrid.best_params_,svrgrid.best_score_

def savemodel(BestModel, X, Y, filename, refit, multiregression=False, \
                               regressorchain=False, SVCclassification=False,\
                                   scoringfn='r2', njobs=-1, save=True):        
    if refit:
        npipe = BestModel
    else:
        # Define a Standard Scaler to normalize inputs
        scaler = StandardScaler()
        # Set model

        svr = SVC(kernel="rbf") if SVCclassification else SVR(kernel="rbf")

        if regressorchain: multiregression=False
        # Set model
        if multiregression:
            svr = MultiOutputRegressor(svr) # wrapper
        elif regressorchain:
            svr = RegressorChain(svr)
        else:
            pass
    
        # Set pipeline
        npipe = Pipeline(steps=[("scaler", scaler), ("svrm", svr)])
        
        # Parameters of pipelines can be set using ‘__’ separated parameter names:
        npipe.set_params(**BestModel)
        
    print("-----------------------------------------------")
    # print('Model details:', npipe.get_params())
    # print("-----------------------------------------------")
    print("Evaluate cross validation over whole data set.")
    kfold = KFold(n_splits=10, random_state=1, shuffle=True)
    if isinstance(scoringfn, str):   
        cv_results = cross_val_score(npipe, X, Y, cv=kfold, scoring=scoringfn, n_jobs=njobs)
        print(f"Score over all samples ({scoringfn}): {cv_results.mean():.3f}±{cv_results.std():.3f}")
    else:
        cv_results = cross_validate(npipe, X, Y, cv=kfold, scoring=scoringfn, n_jobs=njobs)
        for I in scoringfn:
            print(f"Score over all samples ({I}): {cv_results['test_'+I].mean():.3f}±{cv_results['test_'+I].std():.3f}")
    
    npipe.fit(X, Y)
    
    if save:
        pickle.dump(npipe, open(filename, 'wb'))
        print(f"* Model saved: {filename}")
    print("***************************************************************************\n")
    return

def my_ml_model_training(df, filename, xfeatures=['ANTIMONY','STRAIN'], yfeatures=['BANDGAP', 'SO'],
                         refit=True, multiregression=False, regressorchain=False, 
                         SVCclassification=False, ydatadivfac=1,scoringfn='r2', save_model=True,
                         PlotResults=False, LearningCurve=False,
                         saveFig=False, savepath='.', figname='TruePrediction.png'):
    if LearningCurve:
        TotaldataSize = len(df)
        SizeArray = np.arange(TotaldataSize//2,TotaldataSize,100,dtype=int)
        #SizeArray = np.linspace(TotaldataSize//2,TotaldataSize,2,dtype=int)
        SCOREvalue = []; BestParameters_ = []
        for I in SizeArray:
            X, y = GenerateData(df[:I], xfeatures=xfeatures, yfeatures=yfeatures)
            y = y/ydatadivfac
            BestParameters, SCORE = mysvrmodel_parametersearch(X, y, refit=True, 
                                                  multiregression=multiregression, 
                                                  regressorchain=regressorchain,
                                                  SVCclassification=SVCclassification,
                                                  scoringfn=scoringfn,PlotResults=False)
            SCOREvalue.append(SCORE)
            BestParameters_.append(BestParameters['svrm'])
        print('------------------------------------------')
        print('Best parameters:\n',BestParameters_)
        plt.figure()
        plt.plot(SizeArray,SCOREvalue,'o-')
        plt.title("Learning curve")
        plt.xlabel("Data size")
        plt.ylabel('Best Score')
        plt.show()
    else:    
        X, y = GenerateData(df, xfeatures=xfeatures, yfeatures=yfeatures)
        y = y/ydatadivfac
        BestParameters, _ = mysvrmodel_parametersearch(X, y, refit=refit, \
                                                       multiregression=multiregression, \
                                                        regressorchain=regressorchain,\
                                                         SVCclassification=SVCclassification,\
                                                         scoringfn=scoringfn,PlotResults=PlotResults,
                                                         save=saveFig, savepath=savepath, figname=figname)
        _ = savemodel(BestParameters, X, y, filename, refit, multiregression=multiregression, \
                      regressorchain=regressorchain,\
                          SVCclassification=SVCclassification,\
                              scoringfn=scoringfn, save=save_model)

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
            Set2Defaultt=CheckScoringfn(scoringfn)
            if Set2Defaultt: break
    else:
        Set2Defaultt=True
    return Set2Defaultt

#**************** Model training **********************************************
def SVMModelTrainingFunctions(df, xfeatures, yfeatures=None, scaley=1, 
                           multiregression=False, regressorchain=False, 
                           IndependentOutput=False, retrainbw=False, retraineg=False,
                           retrainnature=False, retrainlp=False,
                           svr_bandgap=None, svr_bw=None, 
                           svr_bw_dm=None, svc_EgNature=None, svr_lp=None,
                           scoringfn='r2',refit=True,
                           save_model=True,PlotResults=False, LearningCurve=False,
                           saveFig=False, savepath='.', figname='TruePrediction.png'):
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
        _ = my_ml_model_training(df, svr_bandgap+'.sav', xfeatures=xfeatures, 
                                 yfeatures='BANDGAP', scoringfn=scoringfn,refit=refit,
                                 save_model=save_model,PlotResults=PlotResults,LearningCurve=LearningCurve,
                                 saveFig=saveFig, savepath=savepath, figname=figname)
    # Training bandgap SVR() model
    if retrainlp: 
        assert svr_lp is not None, 'Lattice parameter model training is requested but no filename is supplied for saving final model.'
        print("***************************************************************************")
        print("* Model training for lattice parameter prediction (SVR):")
        _ = my_ml_model_training(df, svr_lp+'.sav', xfeatures=xfeatures, 
                                 yfeatures=yfeatures, scoringfn=scoringfn,refit=refit,
                                 save_model=save_model,PlotResults=PlotResults,LearningCurve=LearningCurve)
    
    # Training Bloch weights SVR() model    
    if retrainbw: 
        assert svr_bw is not None, 'Bloch weight model training is requested but no filename is supplied for saving final model.'
        assert yfeatures is not None, 'Bloch weight model training is requested but no output features(yfeatures) are supplied.'
        print("***************************************************************************")
        print("* Model training for Bloch weight prediction (SVR):")
        if multiregression:
            print("Model type: MultiRegression")
            _ = my_ml_model_training(df, svr_bw+'_multiregression.sav', 
                                     xfeatures=xfeatures, yfeatures=yfeatures,
                                     multiregression = multiregression, 
                                     ydatadivfac=scaley, scoringfn=scoringfn,refit=refit,
                                     save_model=save_model,PlotResults=PlotResults)
        if regressorchain:
            print("Model type: RegressorChain")
            _ = my_ml_model_training(df, svr_bw+'_regressorchain.sav', 
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
                _ = my_ml_model_training(df, save_f, xfeatures=xfeatures, 
                                         yfeatures=I, ydatadivfac=scaley,
                                         scoringfn=scoringfn,refit=refit,
                                         save_model=save_model,PlotResults=PlotResults)
                print("\n")
    
    # Training bandgap nature SVC() model
    if retrainnature:
        assert svc_EgNature is not None, 'Bandgap Nature model training is requested but no filename is supplied for saving final model.'
        yfeature='NATURE'
                
        if Set2Default(scoringfn): scoringfn = 'accuracy'; refit = True
                
        print("***************************************************************************")
        print("* Model training for bandgap nature prediction (SVC):")
        if multiregression:
            print("Model type: MultiRegression")
            _ = my_ml_model_training(df, svc_EgNature+'_multiregression.sav', 
                                     xfeatures=xfeatures, yfeatures=yfeature,
                                    multiregression = multiregression, 
                                    SVCclassification=True,scoringfn=scoringfn,
                                    refit=refit,save_model=save_model,PlotResults=PlotResults)
        if regressorchain:
            print("Model type: RegressorChain")
            _ = my_ml_model_training(df, svc_EgNature+'_regressorchain.sav', 
                                     xfeatures=xfeatures, yfeatures=yfeature,
                                     regressorchain=regressorchain,
                                     SVCclassification=True,scoringfn=scoringfn,
                                     refit=refit,save_model=save_model,PlotResults=PlotResults)
            
        if ((not multiregression) and (not regressorchain)) or (IndependentOutput):
            print("Model type: Direct independent multioutput")
            _ = my_ml_model_training(df, svc_EgNature+'.sav', 
                                     xfeatures=xfeatures, 
                                     SVCclassification=True, yfeatures=yfeature,
                                     scoringfn=scoringfn,refit=refit,save_model=save_model,
                                     PlotResults=PlotResults,LearningCurve=LearningCurve,
                                     saveFig=saveFig, savepath=savepath, figname=figname)
    return

#---------------- test binary models with ternary points ----------------------
def TestModelTernaryPoints(TestPoints_X, y_test, SVMModel, SVCclassification=False):
    y_svr = SVMModel.predict(TestPoints_X)
    if SVCclassification:
        print(f"The accuracy score for prediction: {accuracy_score(y_test, y_svr):.3f}")
    else:
        print(f"The r2_score for prediction: {r2_score(y_test, y_svr):.3f}")
        print(f"The mean_squared_error for prediction: {mean_squared_error(y_test, y_svr):.3f}")
    return y_svr