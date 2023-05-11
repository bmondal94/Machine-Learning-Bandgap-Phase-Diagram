#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 12:48:20 2021

@author: bmondal
"""

#%%-------------------------- Import modules ----------------------------------
import numpy as np
import sqlite3 as sq
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import glob, sys, os, shutil
from datetime import datetime
import scipy.interpolate as inpr

import MLmodelGeneralFunctions as mlgf
import MLmodelGeneralPlottingFunctions as mlgpf


import MLmodelSVMFunctions as mlmf

np.set_printoptions(precision=3, suppress=True)
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
oldstd = sys.stdout
plt.rc('font', size=24)  

#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
#pd.reset_option('display.max_columns')
#pd.reset_option('display.max_rows')
pd.set_option('display.expand_frame_repr', True)
#pd.convert_dtypes()

eps = (0.0001,)
print(f"Start: {datetime.today()}")
#%% ------------------------- Set default parameters --------------------------
#### Data
cwd = os.getcwd()
dirpath = f'{cwd}/../'
dbname = dirpath+'/DataBase/GaPAsSb_ML_database.xlsx'
BinaryConversion=True # Convert text to number ==> 1: Direct, 0: Indirect
UseVegardsLaw = True # <== Use vegards law for equilibrium lattice parameter calculations

#### Model training
retrain_models = True # To train the models
StartFromScratch = True # <== This will delete the whole result folder and start from scratch
SVM_Use_GridSearchCV = False # GridSearchCV vs RandomizedSearchCV. If true use GridSearchCV else RandomizedSearchCV
TOTALTRAINING = {'retraineg':True, # Train bandgap values: SVR-RBF model
                 'retrainnature':True} # Train bandgap natures: SVC-RBF model

#### Plotting parameters
Disable3dFigsGrawing = True 
createmovie = False # <== Draw movie of bandgap phase diagram from snapshots
Plot_Ternary_Figs = False #<== Plot ternary plots. Require 'ternary' package
DrawHtmls = False # <== Create htmls. Require bokeh, holoviews
if Plot_Ternary_Figs: 
    import MLmodelPlottingFunctions as mlpf
    DrawHtmls = False; createmovie = False; Disable3dFigsGrawing = True 
if DrawHtmls:
    import MLmodelWebPlottingFunctions as mlwpf
    import MLmodelWebPlottingFunctionsPart2 as mlwpfP2

#%% ------------------------- Load database -----------------------------------
df, points = mlgf.CreateDataForModel(dbname, ReturnPredictPoints=False,BinaryConversion=BinaryConversion)
df = df.sample(frac=1).reset_index(drop=True)

df_qua_id = df.index[(df['PHOSPHORUS']>eps[0]) & (df['ANTIMONY']>eps[0]) & (df['ARSENIC']>eps[0])]
df_qua = df.loc[df_qua_id].reset_index(drop=True)  ## Quaternary data
df_bintern = df.drop(index=df_qua_id, inplace=False).reset_index(drop=True) ## Binary + Ternary data

xfeatures = ['PHOSPHORUS', 'ANTIMONY', 'ARSENIC',  'STRAIN']
yfeatures = None
    
# Unstrained structures
subname = {'GaAs':(5.689,1.4662),'GaP':(5.474,2.3645),\
           'Si':(5.42103,1.1915),'GaSb':(6.13383,0.641),\
               'InP':(5.93926,1.4341)} # lattice parameter, bandgap

if Plot_Ternary_Figs:
    EqulibriumData = df[df['STRAIN'].abs() < eps[0]]
    EqmDf = mlpf.generate_scatter_data(EqulibriumData[['PHOSPHORUS', 'ANTIMONY', 'ARSENIC']]) 

#%%++++++++++++++++++++++++ SVM model +++++++++++++++++++++++++++++++++++++++++
#%%% prority: PerformAll>TESTi
### BT == train binary-ternary, test quaternary
### Q == train quaternary, test binary-ternary
### BTq == train binary-ternary + quaternary fraction, test test-set + quaternary rest frac <== Leraning curveT3
### Qbt == train quaternary + binary-ternary fraction, test test-set + binary-ternary rest frac <== Leraning curveT3
### BT_ct == train binary-ternary, test 25% of quaternary
### Q_ct == train quaternary, test 25% of binary-ternary 
### BTq_ct == train binary-ternary + quaternary fraction, test 25% reserved quaternary data <== Leraning curveT3
### Qbt_ct == train quaternary + binary-ternary fraction, test 25% reserved binary-ternary <== Leraning curveT3
### BTQ == train binary-ternary-quaternary, test test-set <== Learning curve
### BPD == Final training with binary-ternary-quaternary, test test-set
### BPD_lp == Train lattice parameter model with binary-ternary-quaternary, test test-set

TESTS = {'BT':False, 'BT_ct':False, 'BTq':False, 'Q':False, 'Q_ct':False, 'Qbt':False, 'BTQ':True,'BTq_ct':False,'Qbt_ct':False, 'BPD':False, 'BPD_lp':False}
PerformAll = False
if PerformAll: 
    TESTS = {key:True for key in TESTS.keys()}

#%%%----------------------- Model paths ---------------------------------------
dirpathSystem = dirpath+'/RESULTS/'

#%%%***************** Training model type parameters***************************
#For compatibility # Do not change next 2 lines unless you know what you are doing
multiregression = False; regressorchain = False; retrainbw = False 
SaveFigPath,svr_bandgap,svc_EgNature,svr_bw,svr_bw_dm,svr_lp  = 'TEST', 'TEST', 'TEST', 'TEST', 'TEST', 'TEST'

if retrain_models: 
    if regressorchain: multiregression = False
 
    for IIII, IIII_val in TESTS.items():
        if IIII_val:
            print("+"*100)
            print(f"Test = TEST_{IIII}")
            dirpathSystem__ = dirpathSystem + f'TEST_{IIII}'
            if all(TOTALTRAINING.values()) and StartFromScratch and os.path.exists(dirpathSystem__):
                print("Deleting all files and start from scratch.")
                shutil.rmtree(dirpathSystem__)  
                os.makedirs(dirpathSystem__,exist_ok=True)

            df_train, df_test, SplitInputData, LearningCurve, LearningCurveT3,SVRdependentSVC,train_lp = \
                mlgf.SetSplitLearningParams(IIII, df, df_bintern, df_qua,trainlp=False)  
 
            if train_lp: 
                TOTALTRAINING_backup = TOTALTRAINING.copy()
                TOTALTRAINING = {x_tmp:False for x_tmp in TOTALTRAINING}
                TOTALTRAINING['retrainlp']=True  
            else:
                TOTALTRAINING['retrainlp']=False
                
            for KEYYY,VALLL in TOTALTRAINING.items():
                TMPUPDATE = {x_tmp:False for x_tmp in TOTALTRAINING}
                if VALLL:
                    TMPUPDATE[KEYYY] = VALLL 
                    if (retrainbw or TMPUPDATE['retraineg']): 
                        # scoringfn=('r2', 'neg_root_mean_squared_error','my_rmse_abs','my_rmse_fix')
                        # refit = ('r2', 'neg_root_mean_squared_error','my_rmse_abs','my_rmse_fix')
                        scoringfn=('neg_root_mean_squared_error','my_rmse_fix') # my_rmse_fix: moves the negative bandgap values to 0.
                        refit = ('neg_root_mean_squared_error','my_rmse_fix')
                    elif TMPUPDATE['retrainlp']:
                        scoringfn = 'neg_root_mean_squared_error'
                        refit = True
                    else:
                        scoringfn = ('accuracy','balanced_accuracy')
                        refit = ('accuracy','balanced_accuracy')
                    if isinstance(refit, str) or isinstance(refit, bool): 
                        SaveFigPath, SaveMoviePath, SaveHTMLPath, modelPATHS, OutPutTxtFile, svr_bandgap,\
                            svc_EgNature, svr_bw, svr_bw_dm, svr_lp = mlgf.CreateResultDirectories(dirpathSystem__,BinaryConversion, refit)
                        sys.stdout = open(OutPutTxtFile, "w")
                        print(f"Training date: {datetime.now()}\n")
                        print("="*100)
                        print(f"Refit metrics: {refit}")
                        DoNotResetFolder = None
                    else:
                        DoNotResetFolder = {r_metix:False for r_metix in refit}
                          
                    _ = mlmf.SVMModelTrainingFunctions(df_train, xfeatures, DoNotResetFolder, yfeatures=yfeatures, scaley=100, RepeatLearningCurveTimes=5,
                                                       multiregression=multiregression, regressorchain=regressorchain, 
                                                       IndependentOutput=False, retrainbw=retrainbw, retraineg=TMPUPDATE['retraineg'],
                                                       retrainnature=TMPUPDATE['retrainnature'], retrainlp=TMPUPDATE['retrainlp'],LearningCurveT3=LearningCurveT3,
                                                       svr_bw=svr_bw, svr_lp=svr_lp,svr_bandgap=svr_bandgap,SaveResults=True,
                                                       svr_bw_dm=svr_bw_dm, svc_EgNature=svc_EgNature, save_model=1,
                                                       scoringfn=scoringfn, refit=refit,SplitInputData=SplitInputData,
                                                       PlotResults=1,LearningCurve=LearningCurve,test_set=df_test,
                                                       saveFig=True, savepath=SaveFigPath,SVRdependentSVC=SVRdependentSVC,
                                                       dirpathSystem_=dirpathSystem__,BinaryConversion=BinaryConversion,
                                                       SVM_Use_GridSearCV=SVM_Use_GridSearchCV)
                    
                    if isinstance(refit, str) or isinstance(refit, bool): sys.stdout = oldstd
                    if IIII in ['BTq','Qbt','BTQ','BTq_ct','Qbt_ct']:
                        for refit__ in refit:
                            print(f"Creating figures (refit metric = {refit__})")                             
                            OutPutTxtFile = dirpathSystem__+'/'+'MODELS'+'/'+ refit__ + '/output.txt'
                            OutdataFrame = mlgf.ReadOutputTxtFile(OutPutTxtFile)
                            if IIII == 'BTQ':
                                SearchBPDTrials = glob.glob(f"{dirpathSystem__}/BPD/*/MODELS/{refit__}/output.txt")
                                LastPointData = [OutdataFrame]
                                for OutPutTxtFile in SearchBPDTrials:
                                    LastPointData.append(mlgf.ReadOutputTxtFile(OutPutTxtFile))  
                                OutdataFrame = pd.concat(LastPointData, axis=0)
                            mlgpf.PlotPostProcessingDataSetSize(OutdataFrame,save=True,savepath=f"{dirpathSystem__}/Figs/{refit__}/")
            if train_lp: TOTALTRAINING = TOTALTRAINING_backup.copy()
    print('\n*All model training successful.')

#sys.exit()
#%% ===================== Postprocessing ======================================
#### If retrain_models=True the plotting are automatic there. Don't need to run this block.
for XYZ in ['BTQ']: # ,'BTq_ct','Qbt_ct']:
    filename = f"{dirpathSystem}/TEST_{XYZ}" 
    outnamelist = [name for name in os.listdir(filename+'/MODELS/')]
    
    for fname in outnamelist:
        SaveFigPathTmp = filename+'/Figs/'+fname
        OutdataFrame = mlgf.ReadOutputTxtFile(filename+'/MODELS/'+fname+'/output.txt')
        if fname == 'BTQ':
            SearchBPDTrials = glob.glob(f"{filename}/BPD/*/MODELS/{fname}/output.txt")
            LastPointData = [OutdataFrame]
            for OutPutTxtFile in SearchBPDTrials:
                LastPointData.append(mlgf.ReadOutputTxtFile(OutPutTxtFile))  
            OutdataFrame = pd.concat(LastPointData, axis=0)
        # _ = mlgpf.PlotPostProcessingDataSetSizeV0(OutdataFrame,save=True,savepath=SaveFigPathTmp)
        # _ = mlgpf.PlotPostProcessingDataSetSize(OutdataFrame,save=True,savepath=SaveFigPathTmp)
        _ = mlgpf.PlotPostProcessingDataSetSizeLogLog(OutdataFrame,save=True,savepath=SaveFigPathTmp)
        # _ = mlgpf.PlotPostProcessingDataSetSizeLogLog_v2(OutdataFrame,save=True,savepath=SaveFigPathTmp)

#%%%********** Load model and predictions *************************************
#%%%%---------------- Load models ---------------------------------------------
LoadSVMFinalBPD = f'{dirpathSystem}'

best_cv_score_eg, corr_fname_eg = mlgf.FindBestTrialBTQ(f'{LoadSVMFinalBPD}/TEST_BTQ/BPD/*/MODELS/my_rmse_fix')
best_cv_score_n, corr_fname_n = mlgf.FindBestTrialBTQ(f'{LoadSVMFinalBPD}/TEST_BTQ/BPD/*/MODELS/accuracy')

PostProcPath = f"{LoadSVMFinalBPD}/TEST_BTQ/BPD/PostProcessingFigs/"
SaveFigPath = f"{PostProcPath}/Figs/"
SaveMoviePath = f"{PostProcPath}/MOVIE/"
SubstrateEffect = f"{PostProcPath}/SubstrateEffect/"
SaveHTMLPath = f"{PostProcPath}/HTML/"
if not os.path.isdir(SaveFigPath): 
    os.makedirs(SaveFigPath,exist_ok=True)
    os.makedirs(SubstrateEffect,exist_ok=True)
    os.makedirs(SaveHTMLPath,exist_ok=True)
    os.makedirs(SaveMoviePath,exist_ok=True)

with open(LoadSVMFinalBPD+'/TEST_BTQ/BPD/PostProcessingFigs/beastModel.txt', 'w') as f:
    f.write(f'Best SVR-RBF model: \n\t{corr_fname_eg}\n\tscore = {best_cv_score_eg}\n\n')
    f.write(f'Best SVC-RBF model: \n\t{corr_fname_n}\n\tscore = {best_cv_score_n}')

svr_bandgap = f'{corr_fname_eg}/svrmodel_bandgap'
svc_EgNature = f'{corr_fname_n}/svcmodel_EgNature_binary'
bandgapmag_model = pickle.load(open(svr_bandgap+'.sav', 'rb')) 
BandgapNatureModel = pickle.load(open(svc_EgNature+'.sav', 'rb'))
SVRdependentSVC = False
if SVRdependentSVC: 
    params = bandgapmag_model['svm'].get_params()
    params.pop('epsilon')
    BandgapNatureModel['svm'].set_params(**params)
     
WithPaddingData = False
Xp, Yp, POINTS_, CondPOSI, Zval = mlgf.CreateDataForPredictionLoopV3(resolution=101,
                                                                    compositionscale=100,
                                                                    WithPadding=WithPaddingData,
                                                                    features=xfeatures) 
#%%%%--------------- Some special extra plots ---------------------------------
# OutdataFrame = mlgf.SpecialPlots_1(dirpathSystem + '/TEST_BTQ')
# fig_tmp, ax_tmp = mlgpf.PlotPostProcessingDataSetSize_special(OutdataFrame,save=True,savepath=SaveFigPath)

# OutdataFrame_dict = mlgf.SpecialPlots_2([f'{dirpathSystem}/TEST_BTq_ct',f'{dirpathSystem}/TEST_Qbt_ct'],'root_mean_squared_error','my_rmse_fix')
# savefig_tmp=False
# fig_tmp, ax_tmp = None, None
# ax_plot_color='k'
# for OutdataFrame in OutdataFrame_dict.values():
#     fig_tmp, ax_tmp = mlgpf.PlotPostProcessingDataSetSize_special(OutdataFrame,fig=fig_tmp,ax=ax_tmp,save=savefig_tmp,
#                                                                  ax_yminortick_multi=5,ax_plot_color=ax_plot_color,#ax_y_precision='%.1f',
#                                                                  savepath=SaveFigPath,ProjectionDict = {'set1':{'root_mean_squared_error':'RMSE (meV)'}})
#     savefig_tmp=True
#     ax_plot_color=None
#     ax_tmp.yaxis.set_major_locator(MultipleLocator(5))

# OutdataFrame_dict = mlgf.SpecialPlots_2([f'{dirpathSystem}/TEST_BTq_ct',f'{dirpathSystem}/TEST_Qbt_ct'],'accuracy_score','accuracy')
# savefig_tmp=False
# fig_tmp, ax_tmp = None, None
# ax_plot_color='k'
# for OutdataFrame in OutdataFrame_dict.values():
#     fig_tmp, ax_tmp = mlgpf.PlotPostProcessingDataSetSize_special(OutdataFrame,fig=fig_tmp,ax=ax_tmp,save=savefig_tmp,
#                                                                  ax_plot_color=ax_plot_color,ax_y_precision='%.2f',
#                                                                  savepath=SaveFigPath,ProjectionDict = {'set1':{'accuracy_score':'Accuracy score'}})
#     savefig_tmp=True
#     ax_plot_color=None
#     ax_tmp.yaxis.set_major_locator(MultipleLocator(0.02))
#%%%%.................... Create Bandgap nature contours ......................
StrainMin = -5;  StrainMax = 5; StrainPoints = 101
StrainArray = np.linspace(StrainMin, StrainMax, StrainPoints)
TotalSnapShot = len(StrainArray)
Predictions={}; cnt = {}; anticnt={}; CONTOURS = {}; CONTOURS_best_model = {}
MAX_bandgap = 0
print("Creating direct-indirect transition line contour:")
for i in range(TotalSnapShot):#TotalSnapShot
    print(f"* Snapshot: {i+1}/{TotalSnapShot}")
    POINTS = POINTS_.copy()
    POINTS['STRAIN'] = StrainArray[i]
    if Plot_Ternary_Figs: bandgapEgseparationline = BandgapNatureModel.decision_function(POINTS[xfeatures])
    # POINTS['bandgap'] = bandgapmag_model.predict(POINTS[xfeatures])
    # The -ve bandgaps are fixed here. This may not necessary as in png plots we put the colorbar from 0 to ... anyway. But this is necessary for html plots.
    POINTS['bandgap_best_model'] = mlmf.UpdatePredictionValues(bandgapmag_model.predict(POINTS[xfeatures]),'my_rmse_fix') # if my_rmse_fix
    POINTS['EgN_best_model'] = BandgapNatureModel.predict(POINTS[xfeatures])
    
    BANDGAP_dataframe = pd.DataFrame()
    BANDGAP_nature_dataframe = pd.DataFrame()
    for imp, model_path_tmp in enumerate(glob.glob(f'{LoadSVMFinalBPD}/TEST_BTQ/BPD/*/MODELS/my_rmse_fix')):
        svr_bandgap_tmp = f'{model_path_tmp}/svrmodel_bandgap'
        bandgapmag_model_tmp = pickle.load(open(svr_bandgap_tmp+'.sav', 'rb')) 
        BANDGAP_dataframe[f'bandgap_{imp}'] = mlmf.UpdatePredictionValues(bandgapmag_model_tmp.predict(POINTS[xfeatures]),'my_rmse_fix')
        
    for imp, model_path_tmp in enumerate(glob.glob(f'{LoadSVMFinalBPD}/TEST_BTQ/BPD/*/MODELS/accuracy')):
        svc_EgNature_tmp = f'{model_path_tmp}/svcmodel_EgNature_binary'
        BandgapNatureModel_tmp = pickle.load(open(svc_EgNature_tmp+'.sav', 'rb'))
        if SVRdependentSVC: 
            params = bandgapmag_model_tmp['svm'].get_params()
            params.pop('epsilon')
            BandgapNatureModel_tmp['svm'].set_params(**params)
        BANDGAP_nature_dataframe[f'EgN_{imp}'] = BandgapNatureModel_tmp.predict(POINTS[xfeatures])
             
    POINTS['bandgap'] = BANDGAP_dataframe.mean(axis=1)
    POINTS['bandgap_std'] = BANDGAP_dataframe.std(axis=1)
    POINTS['EgN'] = BANDGAP_nature_dataframe.mode(axis=1).iloc[:, 0].astype(int) # Note: if mode is 50:50 the tag is 0 (==indirect).
    BANDGAP_nature_dataframe['EgN_accuracy'] = BANDGAP_nature_dataframe.mode(axis=1).iloc[:, 0].astype(int)
    POINTS['EgN_accuracy'] = BANDGAP_nature_dataframe.apply(lambda a: np.mean([1 if val==a[-1] else 0 for val in a[:-1]]), axis=1)
    
    if Plot_Ternary_Figs:
        TXTMP = POINTS['EgN'].copy()
        TXTMP[TXTMP<0.00001] = -1
        Zval_best_model = Zval.copy()
        # Zval_best_model_2 = Zval.copy()
        Zval[CondPOSI] = np.array(TXTMP)
        Zval_best_model[CondPOSI] = bandgapEgseparationline # Using decision function
        
        # TXTMP_2 = POINTS['EgN_best_model'].copy()
        # TXTMP_2[TXTMP_2<0.00001] = -1
        # Zval_best_model_2[CondPOSI] = np.array(TXTMP_2) # Using natures
        
        if WithPaddingData:
            Zval_best_model[:,0] = Zval_best_model[0,:] = -1
            for I in range(1,len(Zval_best_model)): Zval_best_model[I,-I] = -1
        else:
            Zval_best_model = mlpf.UpdateCornersForContours(Zval_best_model)   
        contours = mlpf.GetContoursf(Xp,Yp,Zval)
        anticontours = mlpf.GetContoursf(Xp,Yp,Zval,anti_contour=True)
        Contours = mlpf.GetContours(Xp,Yp,Zval, TernaryConversion=0)
        Contours_best_model = mlpf.GetContours(Xp,Yp,Zval_best_model, TernaryConversion=0)
        # Contours_best_model_2 = mlpf.GetContours(Xp,Yp,Zval_best_model_2, TernaryConversion=0)

        cnt[StrainArray[i]] = contours
        anticnt[StrainArray[i]] = anticontours
        CONTOURS[StrainArray[i]] = Contours 
        CONTOURS_best_model[StrainArray[i]] = Contours_best_model 
        
    Predictions[StrainArray[i]] = POINTS
    MAX_bandgap_tmp = POINTS['bandgap'].max()
    # MAX_bandgap_std = POINTS['bandgap_std'].max()
    if MAX_bandgap_tmp>MAX_bandgap: 
        MAX_bandgap=MAX_bandgap_tmp
        tmp_max_value_info = POINTS.iloc[POINTS['bandgap'].argmax()]

print(f"\nThe maximum bandgap value is found at \n{tmp_max_value_info}")

## import contourpy
## contourpy.contour_generator(Xp,Yp,Zval_best_model_tmp,name='serial', corner_mask=True, line_type=contourpy.LineType.SeparateCode,quad_as_tri=True,
##                fill_type=contourpy.FillType.OuterCode).lines(0)

#%%%% ---------- Plot bandgap distribution results over trial models ----------
TMP_Eg_STD_df =  pd.concat([Predictions[i]['bandgap_std']*1000 for i in Predictions]).reset_index(drop=True)
mlgpf.PlotPredictionStdFullSpace(TMP_Eg_STD_df, nbins=200, data_unit_label=' (meV)', PlotLogScale=True,
                                x_major_locator=10,
                                save=True, savepath=SaveFigPath,figname='PredictSTDHistFullSpace_Log.png')

mlgpf.PlotPredictionStdFullSpace(TMP_Eg_STD_df, nbins=200, data_unit_label=' (meV)', PlotLogScale=False,
                                x_major_locator=10,y_minor_locator=0.25*1e4,
                                save=True, savepath=SaveFigPath,figname='PredictSTDHistFullSpace.png')

TMP_Eg_STD_df =  pd.concat([Predictions[i]['EgN_accuracy'] for i in Predictions]).reset_index(drop=True)
mlgpf.PlotPredictionStdFullSpace(TMP_Eg_STD_df, nbins=200, data_unit_label='',xlabel='Accuracy',PlotLogScale=True,
                                x_major_locator=None,
                                save=True, savepath=SaveFigPath,figname='PredictAccuracyHistFullSpace_Log.png')
mlgpf.PlotPredictionStdFullSpace(TMP_Eg_STD_df, nbins=200, data_unit_label='',xlabel='Accuracy',PlotLogScale=False,
                                x_major_locator=None,y_minor_locator=0.5*1e5,
                                save=True, savepath=SaveFigPath,figname='PredictAccuracyHistFullSpace.png')

#%%%% ------------------ Plot True test results on DFT space ------------------
POINTS = df[xfeatures].copy()
bandgapEgseparationline = BandgapNatureModel.decision_function(POINTS)
# POINTS['bandgap'] = bandgapmag_model.predict(POINTS[xfeatures])
# The -ve bandgaps are fixed here. This may not necessary as in png plots we put the colorbar from 0 to ... anyway. But this is necessary for html plots.
POINTS['bandgap_best_model'] = mlmf.UpdatePredictionValues(bandgapmag_model.predict(POINTS[xfeatures]),'my_rmse_fix') # if my_rmse_fix
POINTS['EgN_best_model'] = BandgapNatureModel.predict(POINTS[xfeatures])

BANDGAP_dataframe = pd.DataFrame()
BANDGAP_nature_dataframe = pd.DataFrame()
for imp, model_path_tmp in enumerate(glob.glob(f'{LoadSVMFinalBPD}/TEST_BTQ/BPD/*/MODELS/my_rmse_fix')):
    svr_bandgap_tmp = f'{model_path_tmp}/svrmodel_bandgap'
    bandgapmag_model_tmp = pickle.load(open(svr_bandgap_tmp+'.sav', 'rb')) 
    BANDGAP_dataframe[f'bandgap_{imp}'] = mlmf.UpdatePredictionValues(bandgapmag_model_tmp.predict(df[xfeatures]),'my_rmse_fix')
    
for imp, model_path_tmp in enumerate(glob.glob(f'{LoadSVMFinalBPD}/TEST_BTQ/BPD/*/MODELS/accuracy')):
    svc_EgNature_tmp = f'{model_path_tmp}/svcmodel_EgNature_binary'
    BandgapNatureModel_tmp = pickle.load(open(svc_EgNature_tmp+'.sav', 'rb'))
    if SVRdependentSVC: 
        params = bandgapmag_model_tmp['svm'].get_params()
        params.pop('epsilon')
        BandgapNatureModel_tmp['svm'].set_params(**params)
    BANDGAP_nature_dataframe[f'EgN_{imp}'] = BandgapNatureModel_tmp.predict(POINTS[xfeatures])
         
POINTS['bandgap'] = BANDGAP_dataframe.mean(axis=1)
POINTS['bandgap_std'] = BANDGAP_dataframe.std(axis=1)
POINTS['EgN'] = BANDGAP_nature_dataframe.mode(axis=1).iloc[:, 0].astype(int) # Note: if mode is 50:50 the tag is 0 (==indirect).
BANDGAP_nature_dataframe['EgN_accuracy'] = BANDGAP_nature_dataframe.mode(axis=1).iloc[:, 0].astype(int)
POINTS['EgN_accuracy'] = BANDGAP_nature_dataframe.apply(lambda a: np.mean([1 if val==a[-1] else 0 for val in a[:-1]]), axis=1)

mlgpf.plot_test_results(df['BANDGAP'], POINTS['bandgap'], text=None, my_color=None, 
                       save=True, savepath=SaveFigPath,figname='DFTtrue_TestAverageBandgap_DFTSpace.png',
                       marker='.', data_unit_label='eV',xlabel_text="True values",ylabel_txt="Predictions")
mlmf.plot_err_dist(df['BANDGAP'], POINTS['bandgap'], text=None, data_unit_label='eV',nbins=25,
                   save=True, savepath=SaveFigPath,figname='DFTtrue_TestAverageBandgap_distribution_DFTSpace.png')
display_labels = mlmf.LabelConversionFn([0,1])
mlmf.PlotConfusionMatrix(df['NATURE'], POINTS['EgN'], display_labels,
                         save=True, savepath=SaveFigPath,figname='DFTtrue_TestConfusionMatrix_DFTSpace.png')
# fig, ax = plt.subplots()
# ax.set_title('Diff')
# ax.hist(Predictions[-3.2]['bandgap'] - Predictions[-3.2]['bandgap_best_model'], 200, lw=1) 
#%%%%-------------- Create bandgaps for substrate strain (contours) -----------
if Plot_Ternary_Figs:
    if UseVegardsLaw:
        EqulibriumDataModel = POINTS_.copy()
        EqulibriumDataModel['STRAIN'] = 0.0
        if UseVegardsLaw:
            print("Using the Vegard's law to calculate the unstrained lattice parameters.")
            EqulibriumDataModel['LATTICEPARAMETER1'] = mlgf.CalculateLatticeParametersFromVegardsLaw_ternary(EqulibriumDataModel,
                                                                                                              ['PHOSPHORUS', 'ANTIMONY', 'ARSENIC'])
        else: 
            print("Using the SVR-RBF model to calculate the unstrained lattice parameters.")
            svr_lp += '.sav'
            assert os.path.isfile(svr_lp), 'The lattice parameter model does not exists.'
            lattice_loaded_model = pickle.load(open(svr_lp, 'rb'))
            EqulibriumDataModel['LATTICEPARAMETER1'] = lattice_loaded_model.predict(EqulibriumDataModel)
            
        EqulibriumDataModel = EqulibriumDataModel.join(mlgf.CreateSubstrateStrainData(subname,EqulibriumDataModel['LATTICEPARAMETER1']))
    else:
        EqulibriumDataModel = EqulibriumData.copy()
    
    Sub_Predictions={}; Sub_cnt = {}; Sub_anticnt={}; Sub_CONTOURS = {}; Sub_CONTOURS_best_model = {}
    for SubstrateName in subname:
        print(f"Substrate = {SubstrateName}")
        POINTS = EqulibriumDataModel[['PHOSPHORUS', 'ANTIMONY', 'ARSENIC']].copy()
        POINTS['STRAIN'] = EqulibriumDataModel[SubstrateName].copy()
        
        
        bandgapEgseparationline = BandgapNatureModel.decision_function(POINTS[xfeatures])
        # POINTS['bandgap'] = bandgapmag_model.predict(POINTS[xfeatures])
        # The -ve bandgaps are fixed here. This may not necessary as in png plots we put the colorbar from 0 to ... anyway. But this is necessary for html plots.
        POINTS['bandgap_best_model'] = mlmf.UpdatePredictionValues(bandgapmag_model.predict(POINTS[xfeatures]),'my_rmse_fix') # if my_rmse_fix
        POINTS['EgN_best_model'] = BandgapNatureModel.predict(POINTS[xfeatures])
        
        BANDGAP_dataframe = pd.DataFrame()
        BANDGAP_nature_dataframe = pd.DataFrame()
        for imp, model_path_tmp in enumerate(glob.glob(f'{LoadSVMFinalBPD}/TEST_BTQ/BPD/*/MODELS/my_rmse_fix')):
            svr_bandgap_tmp = f'{model_path_tmp}/svrmodel_bandgap'
            bandgapmag_model_tmp = pickle.load(open(svr_bandgap_tmp+'.sav', 'rb')) 
            BANDGAP_dataframe[f'bandgap_{imp}'] = mlmf.UpdatePredictionValues(bandgapmag_model_tmp.predict(POINTS[xfeatures]),'my_rmse_fix')
            
        for imp, model_path_tmp in enumerate(glob.glob(f'{LoadSVMFinalBPD}/TEST_BTQ/BPD/*/MODELS/accuracy')):
            svc_EgNature_tmp = f'{model_path_tmp}/svcmodel_EgNature_binary'
            BandgapNatureModel_tmp = pickle.load(open(svc_EgNature_tmp+'.sav', 'rb'))
            if SVRdependentSVC: 
                params = bandgapmag_model_tmp['svm'].get_params()
                params.pop('epsilon')
                BandgapNatureModel_tmp['svm'].set_params(**params)
            BANDGAP_nature_dataframe[f'EgN_{imp}'] = BandgapNatureModel_tmp.predict(POINTS[xfeatures])
                 
        POINTS['bandgap'] = BANDGAP_dataframe.mean(axis=1)
        POINTS['bandgap_std'] = BANDGAP_dataframe.std(axis=1)
        POINTS['EgN'] = BANDGAP_nature_dataframe.mode(axis=1).iloc[:, 0].astype(int) # Note: if mode is 50:50 the tag is 0 (==indirect).
        BANDGAP_nature_dataframe['EgN_accuracy'] = BANDGAP_nature_dataframe.mode(axis=1).iloc[:, 0].astype(int)
        POINTS['EgN_accuracy'] = BANDGAP_nature_dataframe.apply(lambda a: np.mean([1 if val==a[-1] else 0 for val in a[:-1]]), axis=1)
        
        TXTMP = POINTS['EgN'].copy()
        TXTMP[TXTMP<0.00001] = -1
        Zval_best_model = Zval.copy()
    
        Sub_Predictions[SubstrateName] = POINTS
        
        if UseVegardsLaw:
            Zval[CondPOSI] = np.array(TXTMP)
            Zval_best_model[CondPOSI] = bandgapEgseparationline
            if WithPaddingData:
                Zval_best_model[:,0] = Zval_best_model[0,:] = -1
                for I in range(1,len(Zval_best_model)): Zval_best_model[I,-I] = -1
            else:
                Zval_best_model = mlpf.UpdateCornersForContours(Zval_best_model)   
            contours = mlpf.GetContoursf(Xp,Yp,Zval)
            anticontours = mlpf.GetContoursf(Xp,Yp,Zval,anti_contour=True)
            Contours = mlpf.GetContours(Xp,Yp,Zval, TernaryConversion=0)
            Contours_best_model = mlpf.GetContours(Xp,Yp,Zval_best_model, TernaryConversion=0)
        else:
            contours, anticontours, Contours,Contours_best_model = None, None, None, None
        Sub_cnt[SubstrateName] = contours
        Sub_anticnt[SubstrateName] = anticontours
        Sub_CONTOURS[SubstrateName] = Contours 
        Sub_CONTOURS_best_model[SubstrateName] = Contours_best_model 

#%%+++++++++++++++++++++++ Plotting +++++++++++++++++++++++++++++++++++++++++++
#%%% --------------------- Plot 2d plot of bandgap vs strain ------------------
XX = np.linspace(-5,5,21)
BANDGAP_dataframe = pd.DataFrame()
for imp, model_path_tmp in enumerate(glob.glob(f'{LoadSVMFinalBPD}/TEST_BTQ/BPD/*/MODELS/my_rmse_fix')):
    svr_bandgap_tmp = f'{model_path_tmp}/svrmodel_bandgap'
    bandgapmag_model_tmp = pickle.load(open(svr_bandgap_tmp+'.sav', 'rb')) 
    BANDGAP_dataframe[f'bandgap_{imp}'] = bandgapmag_model_tmp.predict(pd.DataFrame([[33.33,33.33,100-66.66,x] for x in XX], columns=xfeatures))
YY = BANDGAP_dataframe.mean(axis=1)

mlgpf.PlotStrainBandgap2Dplot(XX,YY,savepath=f"{SaveFigPath}", save=1)

if Plot_Ternary_Figs:
    #%%%----------------------- Ternary axis labels--------------------------------
    axislabels = ["GaAs", "GaSb", "GaP"] # Left, right, bottom
    AxesLabelcolors = None #{'b':'r','l':'b','r':'g'}
    AxisColors=None #{'b':'g','l':'r','r':'b'}
    textt = ['GaP','GaSb','GaAs'] #l,r,t    

    #%%%************************ Plotting substate strains ************************
    #%%%%............. DFT  unstrained compositions in database ...................
    _ = mlpf.DrawSnapshot(None, RawData=EqmDf,RawDataColor=EqulibriumData['NATURE'],
                          fname=f"{SubstrateEffect}/All_Strain0_dft.png", savefig=1, scale=100,
                          axislabels=["GaAs", "GaSb", "GaP"],cbarlabel='Bandgap nature (1:Direct, 0:Indirect)', #titletext=f'Substrate: {WhichSubstrate}',
                          DrawRawData=True,cbarpos='bottom',show_colorbar=True,
                          vmin=None,vmax=None) 
    #%%%%.................. Plot substate strains .................................
    for SubstrateName in subname:
        SubstrateStrainSeries = EqulibriumData[SubstrateName]   
        vmin = SubstrateStrainSeries.min(); vmax = SubstrateStrainSeries.max()
        _ = mlpf.DrawRandomConfigurationPoints(EqmDf, fname=f"{SubstrateEffect}/{SubstrateName}_sub_dft.png", #titletext=f'Substrate = {SubstrateName}', 
                                                savefig=True, scale=100, cbarpos='bottom',
                                                axislabels = ["GaAs", "GaSb", "GaP"],axiscolors=None,
                                                axislabelcolors=None, colorbar=True,cbar_label_txt='Substrate strain (%)',
                                                fontsize = 20, colors=SubstrateStrainSeries,vmin=vmin, vmax=vmax)                                
    #%%%%................ Plot 3d substate strains ................................
    if not Disable3dFigsGrawing:
        fig3d, ax3d = None, None
        for SubstrateName in subname:
            SubstrateStrainValues = EqulibriumData[SubstrateName]
            fig3d, ax3d = mlpf.Plot3DBandgapTernary(EqulibriumData['ARSENIC'], EqulibriumData['ANTIMONY'], 
                                                    SubstrateStrainValues,SubstrateStrainValues,
                                                    titletxt=None, cbar_txt=f'Strain(%) [{SubstrateName} substrate]',
                                                    textt=['GaP','GaAs','GaSb'], scale=100, ax=ax3d, fig=fig3d) 
    #%%%%%............... Plot 2d ternary data ....................................
    if UseVegardsLaw: 
        SubstrateEffect += '/VegardsEqm'
    if not os.path.isdir(SubstrateEffect): os.makedirs(SubstrateEffect,exist_ok=True)
    #%%%
    for SubstrateName in subname:
        print(f'Creating figure for {SubstrateName}')
        if UseVegardsLaw:
            Sub_Predictions[SubstrateName] = Sub_Predictions[SubstrateName][(Sub_Predictions[SubstrateName]['STRAIN']).abs() <5]
            EqmDf_ = mlpf.generate_heatmap_data(Sub_Predictions[SubstrateName][['PHOSPHORUS', 'ANTIMONY', 'ARSENIC', 'bandgap']]) 
            EqmDf_best = mlpf.generate_heatmap_data(Sub_Predictions[SubstrateName][['PHOSPHORUS', 'ANTIMONY', 'ARSENIC', 'bandgap_best_model']])
            DrawRawData = False
            EqmDf__ = None
            
            vmin = Sub_Predictions[SubstrateName]['STRAIN'].min(); vmax = Sub_Predictions[SubstrateName]['STRAIN'].max()
            tax = mlpf.DrawSnapshot(mlpf.generate_heatmap_data(Sub_Predictions[SubstrateName][['PHOSPHORUS', 'ANTIMONY', 'ARSENIC', 'STRAIN']]),
                                  fname=f"{SubstrateEffect}/{SubstrateName}_sub_strain.png", savefig=0, scale=100,
                                  show_colorbar=True,COMContourText=['DIRECT','INDIRECT'],
                                  contours=None,UseContoursText=False, 
                                  ContoursText=None,
                                  axislabels=["GaAs", "GaSb", "GaP"],cbarlabel='Strain (%)', #titletext=f'Substrate: {WhichSubstrate}',
                                  vmin=-5, vmax=5,cbarpos='bottom',DrawRawData=False)
            if SubstrateName in ['GaAs','InP']:
                lattice_math_line = EqulibriumDataModel[EqulibriumDataModel[SubstrateName].round(2).abs() < 0.05][['PHOSPHORUS', 'ANTIMONY', 'ARSENIC']]
                if len(lattice_math_line)>0:
                    z_tmp= np.polyfit(lattice_math_line['PHOSPHORUS'], lattice_math_line['ANTIMONY'], 1)
                    f_tmp = np.poly1d(z_tmp)
                    x_new = np.linspace(0, lattice_math_line['PHOSPHORUS'].max(), 50)
                    y_new = f_tmp(x_new)
                    tax.plot(list(zip(x_new,y_new)),linestyle='--',color='k',linewidth=3)
            plt.savefig(f"{SubstrateEffect}/{SubstrateName}_sub_strain.png",bbox_inches = 'tight',dpi=300)
            plt.close()
        else:
            EqmDf_ = None
            EqmDf_best = None
            EqmDf__ = mlpf.generate_scatter_data(Sub_Predictions[SubstrateName][['PHOSPHORUS', 'ANTIMONY', 'ARSENIC']]) 
            DrawRawData = True  
            _ = mlpf.DrawSnapshot(None,RawData=EqmDf__,RawDataColor=Sub_Predictions[SubstrateName]['EgN'],
                                  fname=f"{SubstrateEffect}/{SubstrateName}_sub_predict_nature", savefig=1, scale=100,
                                  axislabels=["GaAs", "GaSb", "GaP"],cbarlabel='Bandgap nature (1:Direct, 0:Indirect)', #titletext=f'Substrate: {WhichSubstrate}',
                                  DrawRawData=True,cbarpos='bottom',show_colorbar=True,
                                  vmin=None,vmax=None)  
        vmin = Sub_Predictions[SubstrateName]['bandgap'].min(); vmax = Sub_Predictions[SubstrateName]['bandgap'].max()
        _ = mlpf.DrawSnapshot(EqmDf_,fname=f"{SubstrateEffect}/{SubstrateName}_sub_predict_bandgap.svg", savefig=1, scale=100,
                              show_colorbar=True,COMContourText=['DIRECT','INDIRECT'],cmap=plt.cm.RdYlBu_r,
                              contours=[Sub_CONTOURS[SubstrateName]],UseContoursText=False, 
                              ContoursText=[Sub_cnt[SubstrateName], Sub_anticnt[SubstrateName]],
                              axislabels=["GaAs", "GaSb", "GaP"],cbarlabel='E$_{\mathrm{g}}$ (eV)', #titletext=f'Substrate: {WhichSubstrate}',
                              vmin=0, vmax=2.5,cbarpos='bottom',RawData=EqmDf__,ApplySmoothenCountour=True,
                              RawDataColor=Sub_Predictions[SubstrateName]['bandgap'],DrawRawData=DrawRawData)  
        _ = mlpf.DrawSnapshot(EqmDf_best,fname=f"{SubstrateEffect}/{SubstrateName}_sub_predict_bandgap_best_model_boundary.svg", savefig=1, scale=100,
                              show_colorbar=True,COMContourText=['DIRECT','INDIRECT'],cmap=plt.cm.RdYlBu_r,
                              contours=[Sub_CONTOURS_best_model[SubstrateName]],UseContoursText=False, 
                              ContoursText=[Sub_cnt[SubstrateName], Sub_anticnt[SubstrateName]],
                              axislabels=["GaAs", "GaSb", "GaP"],cbarlabel='E$_{\mathrm{g}}$ (eV)', #titletext=f'Substrate: {WhichSubstrate}',
                              vmin=0, vmax=2.5,cbarpos='bottom',RawData=EqmDf__,
                              RawDataColor=Sub_Predictions[SubstrateName]['bandgap_best_model'],DrawRawData=DrawRawData,
                              )
    #%%%%................ Plot 3d substate strains ................................
    if not Disable3dFigsGrawing:
        fig3d, ax3d = None, None
        for SubstrateName in subname:
            SubstrateStrainValues = Sub_Predictions[SubstrateName]['STRAIN']
            fig3d, ax3d = mlpf.Plot3DBandgapTernary(Sub_Predictions[SubstrateName]['ARSENIC'], Sub_Predictions[SubstrateName]['ANTIMONY'], 
                                                    SubstrateStrainValues,SubstrateStrainValues,
                                                    titletxt=None, cbar_txt=f'Strain(%) [{SubstrateName} substrate]',
                                                    textt=['GaP','GaAs','GaSb'], scale=100, ax=ax3d, fig=fig3d) 
#%%%********************* Plot Bandgaps ***************************************
#%%%%....................... 3D scatter plot bandgap ..........................
if not Disable3dFigsGrawing:
    pp = pd.concat(Predictions.values()).reset_index(drop=True)
    _ = mlpf.Plot3DBandgapTernary(pp['ARSENIC'], pp['ANTIMONY'], pp['STRAIN'], pp['bandgap'],textt=textt, scale=100) 

#%%
if Plot_Ternary_Figs:
    BandGapNatureHeatmap = 0
    OnlyContours = 0
    if BandGapNatureHeatmap:
        tmp_featureseg = ["PHOSPHORUS",'ANTIMONY','ARSENIC','STRAIN','EgN']
        WhicColormap = plt.cm.get_cmap('Paired')
    else:
        tmp_featureseg = ["PHOSPHORUS",'ANTIMONY','ARSENIC','STRAIN','bandgap']
        WhicColormap = plt.cm.RdYlBu_r
    mlpf.GenerateHeatmapSnapShots(StrainArray, tmp_featureseg, Predictions, movdirname=SaveMoviePath, 
                                  generatetitle=1, savefig=1, axislabels=axislabels,
                                  axiscolors=AxisColors,axislabelcolors=AxesLabelcolors,
                                  scale=100,vmin=0, vmax=2.5, cbarlabel='E$_{\mathrm{g}}$ (eV)',  #E$_{\mathrm{g}}$
                                  cmap=WhicColormap, cbarpos='bottom',contours=CONTOURS, #_best_model
                                  BandGapNatureHeatmap=BandGapNatureHeatmap,OnlyContour=OnlyContours,
                                  UseContoursText=False, ContoursText=[cnt, anticnt],ShowColorbar=True,
                                  COMContourText=['DIRECT','INDIRECT'],RawData=df,RawDataColorColumn='NATURE',DrawRawData=0,
                                  BsplineSmoothenCountour=True)
    
    EqmDf_ = mlpf.generate_heatmap_data(Predictions[0.0][tmp_featureseg])
    _ = mlpf.DrawSnapshot(EqmDf_,fname=f"{SaveFigPath}/unstrained_GaAsPSb.png", savefig=1, scale=100,
                          show_colorbar=True,COMContourText=['DIRECT','INDIRECT'],cmap=WhicColormap,
                          contours=[CONTOURS[0.0]],UseContoursText=False, 
                          ContoursText=[cnt[0.0], anticnt[0.0]],
                          axislabels=["GaAs", "GaSb", "GaP"],cbarlabel='E$_{\mathrm{g}}$ (eV)', #titletext=f'Substrate: {WhichSubstrate}',
                          vmin=0, vmax=2.5,cbarpos='bottom',RawData=None,
                          RawDataColor=None,DrawRawData=False,ApplySmoothenCountour=True) 
    EqmDf_ = mlpf.generate_heatmap_data(Predictions[0.0][["PHOSPHORUS",'ANTIMONY','ARSENIC','STRAIN','EgN']])
    _ = mlpf.DrawSnapshot(EqmDf_,fname=f"{SaveFigPath}/unstrained_GaAsPSb_smoothing.png", savefig=1, scale=100,
                          show_colorbar=True,COMContourText=['DIRECT','INDIRECT'],cmap=colors.ListedColormap(['gray','rosybrown']),
                          contours=[CONTOURS[0.0]],UseContoursText=False, BandGapNatureHeatmap=True,
                          ContoursText=[cnt[0.0], anticnt[0.0]],
                          axislabels=["GaAs", "GaSb", "GaP"],cbarlabel='E$_{\mathrm{g}}$ (eV)', #titletext=f'Substrate: {WhichSubstrate}',
                          vmin=0, vmax=2.5,cbarpos='bottom',RawData=None,
                          RawDataColor=None,DrawRawData=False,ApplySmoothenCountour=True,ShowOriginalContourAndSmoothen=True) 
    
    os.makedirs(SaveMoviePath+'/BestModelContour/',exist_ok=True)
    tmp_featureseg[-1] += '_best_model' # Also uses bandgap from best models
    mlpf.GenerateHeatmapSnapShots(StrainArray, tmp_featureseg, Predictions, movdirname=SaveMoviePath+'/BestModelContour/', 
                                  generatetitle=1, savefig=1, axislabels=axislabels,
                                  axiscolors=AxisColors,axislabelcolors=AxesLabelcolors,
                                  scale=100,vmin=0, vmax=2.5, cbarlabel='E$_{\mathrm{g}}$ (eV)',  #E$_{\mathrm{g}}$
                                  cmap=WhicColormap, cbarpos='bottom',contours=CONTOURS_best_model,
                                  BandGapNatureHeatmap=BandGapNatureHeatmap,OnlyContour=OnlyContours,
                                  UseContoursText=False, ContoursText=[cnt, anticnt],
                                  COMContourText=['DIRECT','INDIRECT'],RawData=df,RawDataColorColumn='NATURE',DrawRawData=0)
    EqmDf_ = mlpf.generate_heatmap_data(Predictions[0.0][tmp_featureseg])
    _ = mlpf.DrawSnapshot(EqmDf_,fname=f"{SaveFigPath}/unstrained_GaAsPSb_best_model_boundary.png", savefig=1, scale=100,
                          show_colorbar=True,COMContourText=['DIRECT','INDIRECT'],cmap=WhicColormap,
                          contours=[CONTOURS_best_model[0.0]],UseContoursText=False, 
                          ContoursText=[cnt[0.0], anticnt[0.0]],
                          axislabels=["GaAs", "GaSb", "GaP"],cbarlabel='E$_{\mathrm{g}}$ (eV)', #titletext=f'Substrate: {WhichSubstrate}',
                          vmin=0, vmax=2.5,cbarpos='bottom',RawData=None,
                          RawDataColor=None,DrawRawData=False) 
    
    # mlpf.GenerateHeatmapSnapShots_multiprocessing(StrainArray, tmp_featureseg, Predictions, movdirname=SaveMoviePath, 
    #                               generatetitle=1, savefig=1, axislabels=axislabels,
    #                               axiscolors=AxisColors,axislabelcolors=AxesLabelcolors,
    #                               scale=100,vmin=0, vmax=2.5, cbarlabel='E$_{\mathrm{g}}$ (eV)',  #E$_{\mathrm{g}}$
    #                               cmap=WhicColormap, cbarpos='bottom',contours=CONTOURS,
    #                               BandGapNatureHeatmap=BandGapNatureHeatmap,OnlyContour=OnlyContours,
    #                               UseContoursText=False, ContoursText=[cnt, anticnt],
    #                               COMContourText=['DIRECT','INDIRECT'],RawData=None,RawDataColorColumn=None,DrawRawData=0)
    
    
    
    
    #%%%%% ``````````````` Draw all the contours in 3D ````````````````````````````
    if not Disable3dFigsGrawing:
        figcont, axcont = mlpf.DrawAllContour3D(CONTOURS, fname=None, #cmap=None,
                                                savefig=False, scale=100, vmin=StrainMin, vmax=StrainMax,
                                                textt=['GaSb','GaP','GaAs'], ScatterPlot=False)
        
        DrawSubstrate = 1
        if DrawSubstrate:
            SubstrateName = 'GaAs'; UseThisDataFrame = EqulibriumDataModel.copy()
            MEqulibriumData  = UseThisDataFrame.mask((UseThisDataFrame[SubstrateName]<StrainMin) \
                                                     | (UseThisDataFrame[SubstrateName]>StrainMax))
            cc = MEqulibriumData[SubstrateName] #'gray'
            _ = mlpf.Plot3DBandgapTernary(MEqulibriumData['PHOSPHORUS'], MEqulibriumData['ARSENIC'],
                                          MEqulibriumData[SubstrateName],cc,
                                          ax=axcont, fig=figcont,textt=['','',''], scale=1,
                                          ShowColorbar=False) 
    #%%%%%````````````````````` Draw all contours together ````````````````````````
    _ = mlpf.DrawAllContour(CONTOURS,fname=SaveFigPath+'/AllContours.png', titletext=None,
                            axislabels = axislabels,cbarpos='bottom',
                            savefig=1, scale=100, vmin=StrainMin, vmax=StrainMax,
                            cmap=plt.cm.get_cmap('viridis'),ApplySmoothen=True)
    _ = mlpf.DrawAllContour(CONTOURS_best_model,fname=SaveFigPath+'/AllContours_best_model.png', titletext=None,
                            axislabels = axislabels,cbarpos='bottom',
                            savefig=1, scale=100, vmin=StrainMin, vmax=StrainMax,
                            cmap=plt.cm.get_cmap('viridis'))
    _ = mlpf.DrawAllContour(CONTOURS,fname=SaveFigPath+'/AllContours.svg', titletext=None,
                            axislabels = axislabels,cbarpos='bottom',
                            savefig=1, scale=100, vmin=StrainMin, vmax=StrainMax,
                            cmap=plt.cm.get_cmap('viridis'),ApplySmoothen=True)
    _ = mlpf.DrawAllContour(CONTOURS_best_model,fname=SaveFigPath+'/AllContours_best_model.svg', titletext=None,
                            axislabels = axislabels,cbarpos='bottom',
                            savefig=1, scale=100, vmin=StrainMin, vmax=StrainMax,
                            cmap=plt.cm.get_cmap('viridis'))
#%%%------------------------ Create movie from snapshots ----------------------

if createmovie:
    images = []
    imgs = sorted(glob.glob(SaveMoviePath+"conf*.png"))
    for I in imgs[50:]: #compressive=[50::-1], tensile=[50:]
        #print(I)
        images.append(plt.imread(I))
    _ = mlpf.MakeEgStrainSnapShotMovie(images, movdirname=SaveMoviePath, savefig = 1)

#%%++++++++++++++++++++++++ Draw for html +++++++++++++++++++++++++++++++++++++
if DrawHtmls:
    webdirname = SaveHTMLPath + '/'
    # os.mkdir(webdirname)
    AxisLabels = ['GaP','GaSb','GaAs'] # Bottom, right, left
    cbarlabel='Strain(%)'
    #%%% ---------------- Draw Bandgap ( + nature contours) -----------------------
    #%%%%.............. Draw heatmaps (+contours) with slider in html .............
    fname = webdirname + 'BandgGapHeatMap.html'
    # TestPredictions = {XX[0]:XX[1] for XX in list(Predictions.items())[:3]}
    # _ = mlwpfP2.DrawBandgapHeatmapWebSlider(TestPredictions, ["PHOSPHORUS",'ANTIMONY','bandgap'], [cnt, cnt, anticnt], StrainArray, 
    #                                         ContourText=['DIRECT','INDIRECT'],
    #                                         fname=fname, titletext=None,
    #                                         savefig=1, step=10,
    #                                         cmappp="viridis",color='black', 
    #                                         line_width=4, text=AxisLabels,
    #                                         scale=100,vmin=0, vmax=2.5, cbarlabel='Bandgap value (eV)'
    #                                         )
    # TestPredictions = {XX[0]:XX[1] for XX in list(Predictions.items())[:]}
    _ = mlwpfP2.DrawBandgapHeatmapWebV2Slider(Predictions, ["PHOSPHORUS",'ANTIMONY','bandgap'], [cnt, cnt, anticnt],  
                                            ContourText=['DIRECT','INDIRECT'],
                                            fname=fname, titletext=None,
                                            savefig=1, step=10,page_title='GaAsPSb Bandgap phase diagram',
                                            cmappp="viridis",color='black', 
                                            line_width=4, text=AxisLabels,
                                            scale=100,vmin=0, vmax=2.5, cbarlabel='Bandgap value (eV)'
                                            )
    #%%% ----------------------- Draw Bandgap Nature contours -----------------
    #####.............. Draw all contours with slider in html .................
    #####........ Draw all contours with Multi Select in html..................
    fname = webdirname + 'MergeLayoutsAllContourV2.html'
    sliderlayoutv2 = mlwpf.DrawAllContourWebSliderV2([cnt, cnt, anticnt], StrainArray, CoverPage=CONTOURS,
                                                     ContourText=['DIRECT','INDIRECT'],
                                                     IntializeContourText='', 
                                                     savefig=0, scale=100, vmin=StrainMin, vmax=StrainMax,fname=fname,
                                                     step=10,text=AxisLabels,cbarlabel=cbarlabel)
    multiselectlayoutv2 = mlwpf.DrawAllContourWebMultiSelectV2([cnt, cnt, anticnt], StrainArray, CoverPage=CONTOURS,
                                                               ContourText=['DIRECT','INDIRECT'],
                                                               IntializeContourText='',
                                                               savefig=0, scale=100, vmin=StrainMin, vmax=StrainMax,fname=fname,
                                                               step=10,text=AxisLabels,cbarlabel=cbarlabel)
    
    layoutlistv2=[sliderlayoutv2,multiselectlayoutv2]
    mlwpf.MergeSliderSelect(layoutlistv2,fname=fname)
    
    #%%%%............... Draw 3D scatter plot DITs ................................
    fname = webdirname + "Contour3D.html"
    mlwpf.DrawAllContour3D(CONTOURS, StrainArray, fname=fname)

    #%%%%.. Draw substrate heatmaps (+contours) with slider in html ...........
    _ = mlwpfP2.DrawBandgapHeatmapWebV2Slider(Sub_Predictions, 
                                              ["PHOSPHORUS",'ANTIMONY','bandgap'], [Sub_cnt, Sub_cnt, Sub_anticnt], 
                                              ContourText=['DIRECT','INDIRECT'],fname=f"{webdirname}/SubstrateEffectHeatMap_Bandgap.html",
                                              titletext=None,DrawNatureContours=False,
                                              savefig=1, step=10,page_title='GaAsPSb Bandgap phase diagram substrate effect',
                                              cmappp="viridis",color='black', 
                                              line_width=4, text=AxisLabels,SelectionText='Substrate:',
                                              scale=100,vmin=0, vmax=2.5, cbarlabel='Bandgap value (eV)'
                                              )
    _ = mlwpfP2.DrawBandgapHeatmapWebV2Slider(Sub_Predictions, 
                                              ["PHOSPHORUS",'ANTIMONY','STRAIN'], None,
                                              fname=f"{webdirname}/SubstrateEffectHeatMap_Strain.html", 
                                              DrawNatureContours=False,
                                              savefig=1, step=10,page_title='GaAsPSb Bandgap phase diagram substrate effect',
                                              cmappp="viridis",color='black', TOOLTIPS_z_txt='Strain', TOOLTIPS_z_txt_unit='%',
                                              line_width=4, text=AxisLabels,SelectionText='Substrate:',
                                              scale=100,vmin=0, vmax=2.5, cbarlabel='Strain (%)'
                                              )
print(f"End: {datetime.today()}")