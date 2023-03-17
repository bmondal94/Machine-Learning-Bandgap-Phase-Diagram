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
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import glob, sys, os, shutil
from datetime import datetime

import MLmodelGeneralFunctions as mlgf
import MLmodelPlottingFunctions as mlpf
import MLmodelWebPlottingFunctions as mlwpf
import MLmodelWebPlottingFunctionsPart2 as mlwpfP2
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
#%% ------------------------- Load database -----------------------------------
#UseVegardsLaw has higher priority during plotting over UseLatticeParamterModel
#If UseLatticeParamterModel is true then it also compares the predicted lattice parameters from vegards law.

dirpath = '/home/bmondal/MachineLerning/BandGapML_project/'
BinaryConversion=True # 1: Direct, 0: Indirect
OnlyGoodData = 0
UseLatticeParamterModel = True # <-- Training is based on LATTICEPARAMETER1: Check mlmf.SVMModelTrainingFunctions(). Model learns & predicts in-plane lattice parameter only.
Disable3dFigsGrawing = True
UseVegardsLaw = True # <== Use vegards law for equilibrium lattice parameter calculations
createmovie = False # <== Draw movie of bandgap phase diagram from snapshots
DrawHtmls = True # <== Create htmls
DataCuration = False # <== Create good data

if OnlyGoodData:
    dbname = dirpath+'/DATAbase/Total_BPD_MLDataBase_GaPAsSb_GoodData.db'
else:
    dbname = dirpath+'/DATAbase/Total_BPD_MLDataBase_GaPAsSb.db'

df, points = mlgf.CreateDataForModel(dbname, ReturnPredictPoints=False,BinaryConversion=BinaryConversion)
# df.drop(index=df.index[df['STRAIN'].abs()>5.0001], inplace=True)
df = df.sample(frac=1).reset_index(drop=True)
# df = df.iloc[:1000]

df_qua_id = df.index[(df['PHOSPHORUS']>eps[0]) & (df['ANTIMONY']>eps[0]) & (df['ARSENIC']>eps[0])]
df_qua = df.loc[df_qua_id].reset_index(drop=True)  ## Quaternary data
df_bintern = df.drop(index=df_qua_id, inplace=False).reset_index(drop=True) ## Binary + Ternary data

xfeatures = ['PHOSPHORUS', 'ANTIMONY', 'ARSENIC',  'STRAIN']
yfeatures = [s for s in list(df.columns) if s.startswith('BW')]

if points is not None: predict_points = pd.DataFrame(points, columns=xfeatures)

assert len(df[(df["LATTICEPARAMETER1"] == df["LATTICEPARAMETER2"]) & (df["LATTICEPARAMETER2"] == df["LATTICEPARAMETER3"]) \
    & (df["STRAIN"] != 0.0)]) == 0,'The data base include non-biaxial strain data'
    
# Unstrained structures
subname = {'GaAs':(5.689,1.4662),'GaP':(5.475,2.3645),\
           'Si':(5.42103,1.1915),'GaSb':(6.13383,0.641),\
               'InP':(5.93926,1.4341)} # lattice parameter, bandgap
EqulibriumData = df[df['STRAIN'].abs() < eps[0]]
EqulibriumData = EqulibriumData.join(mlgf.CreateSubstrateStrainData(subname,EqulibriumData['LATTICEPARAMETER1'])) # Calculate the Strain data
EqmDf = mlpf.generate_scatter_data(EqulibriumData[['PHOSPHORUS', 'ANTIMONY', 'ARSENIC']]) 

#%%----------------- BWs-Eg nature conversion ---------------------------------
# df = mlgf.ConvertBWs2BandgapNature(df, BinaryConversion,colname='CONVERTED_NATURE')
# assert df['NATURE'].equals(df['nature']), 'The bandgap nature from BWs conversion doesnot match with actual.'

#%%++++++++++++++++++++++++ SVM model +++++++++++++++++++++++++++++++++++++++++
#%%% prority: PerformAll>TESTi
### BT == train binary-ternary, test quaternary
### Q == train quaternary, test binary-ternary 
### BTq == train binary-ternary + quaternary fraction, test test-set + quaternary rest frac <== Leraning curveT3
### Qbt == train quaternary + binary-ternary fraction, test test-set + binary-ternary rest frac <== Leraning curveT3
### BTQ == train binary-ternary-quaternary, test test-set <== Learning curve
### BPD == Final training with binary-ternary-quaternary, test test-set
### BPD_lp == Train lattice parameter model with binary-ternary-quaternary, test test-set

TESTS = {'BT':False, 'BTq':False, 'Q':False, 'Qbt':False, 'BTQ':False, 'BPD':True, 'BPD_lp':False}
PerformAll = True
if PerformAll: 
    TESTS = {key:True for key in TESTS.keys()}

#%%%----------------------- Model paths ---------------------------------------
dirpathSystem = dirpath+'/GaPAsSb/RESULTS/'
dirpathSystem += '/GOODdataSET/' if OnlyGoodData else '/ORIGINALdataSET/' 

#%%%***************** Training model type parameters***************************
retrain_models = 1
multiregression = False
regressorchain = False
retrainbw = False
StartFromScratch = True # <== This will delete the whole folder and start from scratch
TOTALTRAINING = {'retraineg':True,
                 'retrainnature':True}

#For compatibility
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
                mlgf.SetSplitLearningParams(IIII, df, df_bintern, df_qua,trainlp=UseLatticeParamterModel)  
 
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
                        scoringfn=('r2', 'neg_root_mean_squared_error','my_rmse_abs','my_rmse_fix')
                        refit = ('r2', 'neg_root_mean_squared_error','my_rmse_abs','my_rmse_fix')
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
                                                       dirpathSystem_=dirpathSystem__,BinaryConversion=BinaryConversion)
                    
                    if isinstance(refit, str) or isinstance(refit, bool): sys.stdout = oldstd
                    if IIII in ['BTq','Qbt','BTQ']:
                        for refit__ in refit:
                            print(f"Creating figures (refit metric = {refit__})")
                            OutPutTxtFile = dirpathSystem__+'/'+'MODELS'+'/'+ refit__ + '/output.txt'
                            OutdataFrame = mlgf.ReadOutputTxtFile(OutPutTxtFile)
                            mlpf.PlotPostProcessingDataSetSize(OutdataFrame,save=True,savepath=f"{dirpathSystem__}/Figs/{refit__}/")
            if train_lp: TOTALTRAINING = TOTALTRAINING_backup.copy()
    print('\n*All model training successful.')

#%% ========== Postprocessing TEST_BTq,Qbt,BTQ ================================
#### If retrain_models=True the plotting are automatic there. Don't need to run this block.
filename = dirpathSystem + '/TEST_BTQ' 
outnamelist = [name for name in os.listdir(filename+'/MODELS/')]

for fname in outnamelist:
    SaveFigPathTmp = filename+'/Figs/'+fname
    OutdataFrame = mlgf.ReadOutputTxtFile(filename+'/MODELS/'+fname+'/output.txt')
    # _ = mlpf.PlotPostProcessingDataSetSizeV0(OutdataFrame,save=True,savepath=SaveFigPathTmp)
    # _ = mlpf.PlotPostProcessingDataSetSize(OutdataFrame,save=True,savepath=SaveFigPathTmp)
    _ = mlpf.PlotPostProcessingDataSetSizeLogLog(OutdataFrame,save=True,savepath=SaveFigPathTmp)


#%%%********** Load model and predictions *************************************
#%%%%---------- Test ternary trained model on known ternary -------------------
if DataCuration:
    svr_bandgap = dirpathSystem+"/TEST_BPD/MODELS/neg_root_mean_squared_error/svrmodel_bandgap"
    svc_EgNature = dirpathSystem+"/TEST_BPD/MODELS/accuracy/svcmodel_EgNature_binary"
    SVRdependentSVC=False
    print(f"Testing binary trained model on known ternary date: {datetime.now()}\n")
    TestPoints = df[xfeatures+['BANDGAP','NATURE']].copy()
    SVMBandgapModel = pickle.load(open(svr_bandgap+'.sav', 'rb')) # Bandgap model 
    SVCEgNatureModel = pickle.load(open(svc_EgNature+'.sav', 'rb')) # Bandgap nature model
    if SVRdependentSVC: 
        params = SVMBandgapModel['svm'].get_params()
        params.pop('epsilon')
        SVCEgNatureModel['svm'].set_params(**params)   
    
    TestPoints['Predictedbandgap'] = mlmf.TestModelTernaryPoints(TestPoints[xfeatures], TestPoints['BANDGAP'], SVMBandgapModel)
    TestPoints['predNATURE'] = mlmf.TestModelTernaryPoints(TestPoints[xfeatures], TestPoints['NATURE'],
                                                           SVCEgNatureModel, SVCclassification=True)
    
    TestPoints['AbsErrorBndgap'] = (TestPoints['Predictedbandgap'] - TestPoints['BANDGAP']).abs()
    BadSamples = TestPoints[TestPoints['AbsErrorBndgap'] > 0.15].copy()
    WrongPrediction = TestPoints[TestPoints['predNATURE'] != TestPoints['NATURE']].copy()
    CommonIdx = WrongPrediction.index.intersection(BadSamples.index)
    
    
    ax = mlpf.plot_true_predict_results(TestPoints['BANDGAP'], TestPoints['Predictedbandgap'], text=None,savehist=False,savepath=SaveFigPath)
    ax.scatter(BadSamples['BANDGAP'], BadSamples['Predictedbandgap'], color='r',marker='*')
    ax.scatter(WrongPrediction['BANDGAP'], WrongPrediction['Predictedbandgap'], color='k',marker='.')
    # plt.savefig(SaveFigPath+'/TruePredictBadSamples.png',bbox_inches = 'tight',dpi=300)
    #%%%%
    BADSAMPLESATOMNUMBER = BadSamples.copy()
    BADSAMPLESATOMNUMBER[['PHOSPHORUS', 'ANTIMONY', 'ARSENIC']] = (BADSAMPLESATOMNUMBER[['PHOSPHORUS', 'ANTIMONY', 'ARSENIC']] * 2.16).astype(int)
    #%%%% Create the good samples database, deleting bad data from original database
    BadSamples_atom = mlpf.generate_scatter_data(BadSamples[['PHOSPHORUS', 'ANTIMONY', 'ARSENIC']]) 
    _ = mlpf.DrawRandomConfigurationPoints(BadSamples_atom, fname=None, titletext='Bad bandgap values', 
                                           savefig=False, scale=100,DrawGridLines=True,
                                           axislabels = ["GaSb", "GaAs", "GaP"],axiscolors=None,
                                           axislabelcolors=None, colorbar=True,vmin=BadSamples['STRAIN'].min(),
                                           vmax=BadSamples['STRAIN'].max(),
                                           fontsize = 20,colors=BadSamples['STRAIN'],
                                           cbar_label_txt='Strain (%)')
    _ = mlpf.DrawRandomConfigurationPoints(BadSamples_atom, fname=None, titletext='Bad bandgap values', 
                                           savefig=False, scale=100,DrawGridLines=True,
                                           axislabels = ["GaSb", "GaAs", "GaP"],axiscolors=None,
                                           axislabelcolors=None, colorbar=True,vmin=BadSamples['AbsErrorBndgap'].min(),
                                           vmax=BadSamples['AbsErrorBndgap'].max(),
                                           fontsize = 20,colors=BadSamples['AbsErrorBndgap'],
                                           cbar_label_txt='AbsErrorBndgap (eV)')
    
    WrongPrediction_atom = mlpf.generate_scatter_data(WrongPrediction[['PHOSPHORUS', 'ANTIMONY', 'ARSENIC']]) 
    _ = mlpf.DrawRandomConfigurationPoints(WrongPrediction_atom, fname=None, titletext='Wrong bandgap nature prediction', 
                                           savefig=False, scale=100,
                                           axislabels = ["GaSb", "GaAs", "GaP"],axiscolors=None,
                                           axislabelcolors=None, colorbar=True,vmin=WrongPrediction['STRAIN'].min(),
                                           vmax=WrongPrediction['STRAIN'].max(),
                                           fontsize = 20,colors=WrongPrediction['STRAIN'],
                                           cbar_label_txt='Strain (%)')
    _ = mlpf.DrawRandomConfigurationPoints(WrongPrediction_atom, fname=None, titletext='Wrong bandgap nature prediction', 
                                           savefig=False, scale=100,
                                           axislabels = ["GaSb", "GaAs", "GaP"],axiscolors=None,
                                           axislabelcolors=None, colorbar=False,cmap='Set1',
                                           fontsize = 20,colors=WrongPrediction['NATURE'])
    
    # _ = mlpf.Plot3DBandgapTernary(BadSamples['ARSENIC'], BadSamples['ANTIMONY'], BadSamples['STRAIN'], BadSamples['AbsErrorBndgap']\
    #                               ,textt=['GaP','GaSb','GaAs'], scale=100) 
    WrongPrediction.drop(index=CommonIdx, inplace=True)
    df.drop(index=BadSamples.index,inplace=True)
    # df.drop(index=WrongPrediction.index,inplace=True)
    
    conn = sq.connect('/home/bmondal/MachineLerning/BandGapML_project/DATAbase/Total_BPD_MLDataBase_GaPAsSb_GoodData.db')
    df.to_sql("COMPUTATIONALDATA", conn, index=False,if_exists='fail')
    conn.close()
#%%%%---------------- Load models ---------------------------------------------
LoadSVMFinalBPD = '/home/bmondal/MachineLerning/BandGapML_project/GaPAsSb/RESULTS/ORIGINALdataSET/'
svr_bandgap = LoadSVMFinalBPD+'TEST_BPD/MODELS/neg_root_mean_squared_error/svrmodel_bandgap'
svc_EgNature = LoadSVMFinalBPD+'TEST_BPD/MODELS/accuracy/svcmodel_EgNature_binary'
svr_lp = LoadSVMFinalBPD+'TEST_BPD_lp/MODELS/svrmodel_lp'
SaveMoviePath = LoadSVMFinalBPD+'TEST_BPD/MOVIE/neg_root_mean_squared_error/'
SaveFigPath = LoadSVMFinalBPD+'TEST_BPD/PostProcessingFigs/'
SubstrateEffect = SaveFigPath+'/SubstrateEffect/'
SaveHTMLPath = LoadSVMFinalBPD+'TEST_BPD/HTML/'
if not os.path.isdir(SaveFigPath): 
    os.makedirs(SaveFigPath,exist_ok=True)
    os.makedirs(SubstrateEffect,exist_ok=True)
    os.makedirs(SaveHTMLPath,exist_ok=True)
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
#%%%%.................... Create Bandgap nature contours ......................
StrainMin = -5;  StrainMax = 5; StrainPoints = 101
StrainArray = np.linspace(StrainMin, StrainMax, StrainPoints)
TotalSnapShot = len(StrainArray)
Predictions={}; cnt = {}; anticnt={}; CONTOURS = {}
MAX_bandgap = 0

for i in range(TotalSnapShot):#TotalSnapShot
    print(f"* Snapshot: {i+1}/{TotalSnapShot}")
    POINTS = POINTS_.copy()
    POINTS['STRAIN'] = StrainArray[i]
    bandgapEgseparationline = BandgapNatureModel.decision_function(POINTS[xfeatures])
    POINTS['bandgap'] = bandgapmag_model.predict(POINTS[xfeatures])
    POINTS['EgN'] = BandgapNatureModel.predict(POINTS[xfeatures])
    Zval[CondPOSI] = bandgapEgseparationline
    if WithPaddingData:
        Zval[:,0] = Zval[0,:] = -1
        for I in range(1,len(Zval)): Zval[I,-I] = -1
    else:
        Zval = mlpf.UpdateCornersForContours(Zval)   
    contours = mlpf.GetContoursf(Xp,Yp,Zval)
    anticontours = mlpf.GetContoursf(Xp,Yp,Zval,anti_contour=True)
    Contours = mlpf.GetContours(Xp,Yp,Zval, TernaryConversion=0)
    Predictions[StrainArray[i]] = POINTS
    cnt[StrainArray[i]] = contours
    anticnt[StrainArray[i]] = anticontours
    CONTOURS[StrainArray[i]] = Contours 
    MAX_bandgap_tmp = POINTS['bandgap'].max()
    if MAX_bandgap_tmp>MAX_bandgap: 
        MAX_bandgap=MAX_bandgap_tmp
        tmp_max_value_info = POINTS.iloc[POINTS['bandgap'].argmax()]

print(f"\nThe maximum bandgap value is found at \n{tmp_max_value_info}")
   
#%%%%-------------- Create bandgaps for substrate strain (contours) -----------
if UseLatticeParamterModel or UseVegardsLaw:
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

Sub_Predictions={}; Sub_cnt = {}; Sub_anticnt={}; Sub_CONTOURS = {}
for SubstrateName in subname:
    print(f"Substrate = {SubstrateName}")
    POINTS = EqulibriumDataModel[['PHOSPHORUS', 'ANTIMONY', 'ARSENIC']].copy()
    POINTS['STRAIN'] = EqulibriumDataModel[SubstrateName].copy()
    bandgapEgseparationline = BandgapNatureModel.decision_function(POINTS[xfeatures])
    POINTS['bandgap'] = bandgapmag_model.predict(POINTS[xfeatures])
    POINTS['EgN'] = BandgapNatureModel.predict(POINTS[xfeatures])
    Sub_Predictions[SubstrateName] = POINTS
    if UseLatticeParamterModel or UseVegardsLaw:
        Zval[CondPOSI] = bandgapEgseparationline
        if WithPaddingData:
            Zval[:,0] = Zval[0,:] = -1
            for I in range(1,len(Zval)): Zval[I,-I] = -1
        else:
            Zval = mlpf.UpdateCornersForContours(Zval)   
        contours = mlpf.GetContoursf(Xp,Yp,Zval)
        anticontours = mlpf.GetContoursf(Xp,Yp,Zval,anti_contour=True)
        Contours = mlpf.GetContours(Xp,Yp,Zval, TernaryConversion=0)
    else:
        contours, anticontours, Contours = None, None, None
    Sub_cnt[SubstrateName] = contours
    Sub_anticnt[SubstrateName] = anticontours
    Sub_CONTOURS[SubstrateName] = Contours 

#%%+++++++++++++++++++++++ Plotting +++++++++++++++++++++++++++++++++++++++++++
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
elif UseLatticeParamterModel:
    SubstrateEffect += '/LpModelEqm'
if not os.path.isdir(SubstrateEffect): os.makedirs(SubstrateEffect,exist_ok=True)
for SubstrateName in subname:
    print(f'Creating figure for {SubstrateName}')
    if UseLatticeParamterModel or UseVegardsLaw:
        Sub_Predictions[SubstrateName] = Sub_Predictions[SubstrateName][(Sub_Predictions[SubstrateName]['STRAIN']).abs() <5]
        EqmDf_ = mlpf.generate_heatmap_data(Sub_Predictions[SubstrateName][['PHOSPHORUS', 'ANTIMONY', 'ARSENIC', 'bandgap']])   
        DrawRawData = False
        EqmDf__ = None
        vmin = Sub_Predictions[SubstrateName]['STRAIN'].min(); vmax = Sub_Predictions[SubstrateName]['STRAIN'].max()
        _ = mlpf.DrawSnapshot(mlpf.generate_heatmap_data(Sub_Predictions[SubstrateName][['PHOSPHORUS', 'ANTIMONY', 'ARSENIC', 'STRAIN']]),
                              fname=f"{SubstrateEffect}/{SubstrateName}_sub_strain.png", savefig=1, scale=100,
                              show_colorbar=True,COMContourText=['DIRECT','INDIRECT'],
                              contours=None,UseContoursText=False, 
                              ContoursText=None,
                              axislabels=["GaAs", "GaSb", "GaP"],cbarlabel='Strain (%)', #titletext=f'Substrate: {WhichSubstrate}',
                              vmin=-5, vmax=5,cbarpos='bottom',DrawRawData=False)
    else:
        EqmDf_ = None
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
                          vmin=0, vmax=2.5,cbarpos='bottom',RawData=EqmDf__,
                          RawDataColor=Sub_Predictions[SubstrateName]['bandgap'],DrawRawData=DrawRawData)  
#%%%%................ Plot 3d substate strains ................................
if not Disable3dFigsGrawing:
    fig3d, ax3d = None, None
    for SubstrateName in subname:
        SubstrateStrainValues = Sub_Predictions[SubstrateName]['STRAIN']
        fig3d, ax3d = mlpf.Plot3DBandgapTernary(Sub_Predictions[SubstrateName]['ARSENIC'], Sub_Predictions[SubstrateName]['ANTIMONY'], 
                                                SubstrateStrainValues,SubstrateStrainValues,
                                                titletxt=None, cbar_txt=f'Strain(%) [{SubstrateName} substrate]',
                                                textt=['GaP','GaAs','GaSb'], scale=100, ax=ax3d, fig=fig3d) 
#%%%%................ Compare substate strains model and vg law................
if UseLatticeParamterModel:
    if UseVegardsLaw:
        svr_lp += '.sav'
        assert os.path.isfile(svr_lp), 'The lattice parameter model does not exists.'
        lattice_loaded_model = pickle.load(open(svr_lp, 'rb'))
        XX_lp = lattice_loaded_model.predict(EqulibriumDataModel[['PHOSPHORUS', 'ANTIMONY', 'ARSENIC', 'STRAIN']])
    else:
        XX_lp = EqulibriumDataModel['LATTICEPARAMETER1']
    LATTICEPARAMETER1_vegards = mlgf.CalculateLatticeParametersFromVegardsLaw_ternary(EqulibriumDataModel,['PHOSPHORUS', 'ANTIMONY', 'ARSENIC'])
    mlpf.plot_test_results(XX_lp,LATTICEPARAMETER1_vegards,
                           xlabel_text='Model predictions',ylabel_txt="Vegard's law",data_unit_label='$\AA$',
                           save=1, savepath=f"{SubstrateEffect}", figname="VegardsLaw_lp_Modelprediction_compare.png")

#%%%********************* Plot Bandgaps ***************************************
#%%%%....................... 3D scatter plot bandgap ..........................
if not Disable3dFigsGrawing:
    pp = pd.concat(Predictions.values()).reset_index(drop=True)
    _ = mlpf.Plot3DBandgapTernary(pp['ARSENIC'], pp['ANTIMONY'], pp['STRAIN'], pp['bandgap'],textt=textt, scale=100) 
#%%%%....................... 2D heatmap plot bandgap ..........................
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
                              cmap=WhicColormap, cbarpos='bottom',contours=CONTOURS,
                              BandGapNatureHeatmap=BandGapNatureHeatmap,OnlyContour=OnlyContours,
                              UseContoursText=False, ContoursText=[cnt, anticnt],
                              COMContourText=['DIRECT','INDIRECT'],RawData=df,RawDataColorColumn='NATURE',DrawRawData=0)

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
    fname = webdirname + 'BandgGapHeatMap_t.html'
    # TestPredictions = {XX[0]:XX[1] for XX in list(Predictions.items())[:3]}
    # _ = mlwpfP2.DrawBandgapHeatmapWebSlider(TestPredictions, ["PHOSPHORUS",'ANTIMONY','bandgap'], [cnt, cnt, anticnt], StrainArray, 
    #                                         ContourText=['DIRECT','INDIRECT'],
    #                                         fname=fname, titletext=None,
    #                                         savefig=1, step=10,
    #                                         cmappp="viridis",color='black', 
    #                                         line_width=4, text=AxisLabels,
    #                                         scale=100,vmin=0, vmax=2.5, cbarlabel='Bandgap value (eV)'
    #                                         )
    TestPredictions = {XX[0]:XX[1] for XX in list(Predictions.items())[:3]}
    _ = mlwpfP2.DrawBandgapHeatmapWebV2Slider(TestPredictions, ["PHOSPHORUS",'ANTIMONY','bandgap'], [cnt, cnt, anticnt],  
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
                                              cmappp="viridis",color='black', 
                                              line_width=4, text=AxisLabels,SelectionText='Substrate:',
                                              scale=100,vmin=0, vmax=2.5, cbarlabel='Strain (%)'
                                              )