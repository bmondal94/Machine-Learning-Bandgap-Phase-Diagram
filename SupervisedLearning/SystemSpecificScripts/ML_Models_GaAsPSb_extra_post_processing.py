#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 09:48:18 2023

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

import MLmodelGeneralFunctions as mlgf
import MLmodelGeneralPlottingFunctions as mlgpf
import MLmodelPlottingFunctions as mlpf
import MLmodelWebPlottingFunctions as mlwpf
import MLmodelWebPlottingFunctionsPart2 as mlwpfP2
import MLmodelSVMFunctions as mlmf

np.set_printoptions(precision=3, suppress=True)
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
oldstd = sys.stdout

#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
#pd.reset_option('display.max_columns')
#pd.reset_option('display.max_rows')
pd.set_option('display.expand_frame_repr', True)
#pd.convert_dtypes()

eps = (0.0001,)
#%%
#%%
params = {'figure.figsize': (8, 6),
          'legend.fontsize': 18,
          'axes.labelsize': 24,
          'axes.titlesize': 24,
          'xtick.labelsize':24,
          'ytick.labelsize': 24,
          'errorbar.capsize':2}
plt.rcParams.update(params)
plt.rc('font', size=24)  
#%% ------------------------- Load database -----------------------------------
#UseVegardsLaw has higher priority during plotting over UseLatticeParamterModel
#If UseLatticeParamterModel is true then it also compares the predicted lattice parameters from vegards law.

dirpath = '/home/bmondal/MachineLerning/BandGapML_project/GaPAsSb/'
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

#%%
#%%%----------------------- Model paths ---------------------------------------
dirpathSystem = dirpath+'/RESULTS/'
dirpathSystem += '/GOODdataSET/' if OnlyGoodData else '/ORIGINALdataSET/' 

#%% ----------------------- log-log plots -------------------------------------

filename = f"{dirpathSystem}/TEST_BTQ" 
outnamelist = ['my_rmse_fix']

for fname in outnamelist:
    SaveFigPathTmp = filename+'/BPD/PostProcessingFigs/Figs/'
    OutdataFrame = mlgf.ReadOutputTxtFile(filename+'/MODELS/'+fname+'/output.txt')
    SearchBPDTrials = glob.glob(f"{filename}/BPD/*/MODELS/{fname}/output.txt")
    LastPointData = [OutdataFrame]
    for OutPutTxtFile in SearchBPDTrials:
        LastPointData.append(mlgf.ReadOutputTxtFile(OutPutTxtFile))  
    OutdataFrame = pd.concat(LastPointData, axis=0)
_ = mlgpf.PlotPostProcessingDataSetSizeLogLog(OutdataFrame[['dataset_size','out-of-sample_root_mean_squared_error']],save=True,savepath=SaveFigPathTmp)
# #%% ------------------------- Plot average predictions ------------------------
# POINTS = POINTS_.copy()
# POINTS['STRAIN'] = StrainArray[i]
# bandgapEgseparationline = BandgapNatureModel.decision_function(POINTS[xfeatures])
# # POINTS['bandgap'] = bandgapmag_model.predict(POINTS[xfeatures])
# # The -ve bandgaps are fixed here. This may not necessary as in png plots we put the colorbar from 0 to ... anyway. But this is necessary for html plots.
# POINTS['bandgap_best_model'] = mlmf.UpdatePredictionValues(bandgapmag_model.predict(POINTS[xfeatures]),'my_rmse_fix') # if my_rmse_fix
# POINTS['EgN_best_model'] = BandgapNatureModel.predict(POINTS[xfeatures])

# BANDGAP_dataframe = pd.DataFrame()
# BANDGAP_nature_dataframe = pd.DataFrame()
# for imp, model_path_tmp in enumerate(glob.glob(f'{LoadSVMFinalBPD}/TEST_BTQ/BPD/*/MODELS/my_rmse_fix')):
#     svr_bandgap_tmp = f'{model_path_tmp}/svrmodel_bandgap'
#     bandgapmag_model_tmp = pickle.load(open(svr_bandgap_tmp+'.sav', 'rb')) 
#     BANDGAP_dataframe[f'bandgap_{imp}'] = mlmf.UpdatePredictionValues(bandgapmag_model_tmp.predict(POINTS[xfeatures]),'my_rmse_fix')
    
# for imp, model_path_tmp in enumerate(glob.glob(f'{LoadSVMFinalBPD}/TEST_BTQ/BPD/*/MODELS/accuracy')):
#     svc_EgNature_tmp = f'{model_path_tmp}/svcmodel_EgNature_binary'
#     BandgapNatureModel_tmp = pickle.load(open(svc_EgNature_tmp+'.sav', 'rb'))
#     if SVRdependentSVC: 
#         params = bandgapmag_model_tmp['svm'].get_params()
#         params.pop('epsilon')
#         BandgapNatureModel_tmp['svm'].set_params(**params)
#     BANDGAP_nature_dataframe[f'EgN_{imp}'] = BandgapNatureModel_tmp.predict(POINTS[xfeatures])
         
# POINTS['bandgap'] = BANDGAP_dataframe.mean(axis=1)
# POINTS['bandgap_std'] = BANDGAP_dataframe.std(axis=1)
# POINTS['EgN'] = BANDGAP_nature_dataframe.mode(axis=1).iloc[:, 0].astype(int) # Note: if mode is 50:50 the tag is 0 (==indirect).
# BANDGAP_nature_dataframe['EgN_accuracy'] = BANDGAP_nature_dataframe.mode(axis=1).iloc[:, 0].astype(int)
# POINTS['EgN_accuracy'] = BANDGAP_nature_dataframe.apply(lambda a: np.mean([1 if val==a[-1] else 0 for val in a[:-1]]), axis=1)

#%%%%---------------- Load models ---------------------------------------------
LoadSVMFinalBPD = '/home/bmondal/MachineLerning/BandGapML_project/GaPAsSb/RESULTS/ORIGINALdataSET/'

best_cv_score_eg, corr_fname_eg = mlgf.FindBestTrialBTQ(f'{LoadSVMFinalBPD}/TEST_BTQ/BPD/*/MODELS/my_rmse_fix')
best_cv_score_n, corr_fname_n = mlgf.FindBestTrialBTQ(f'{LoadSVMFinalBPD}/TEST_BTQ/BPD/*/MODELS/accuracy')

svr_bandgap = f'{corr_fname_eg}/svrmodel_bandgap'
svc_EgNature = f'{corr_fname_n}/svcmodel_EgNature_binary'
svr_lp = LoadSVMFinalBPD+'TEST_BPD_lp/MODELS/svrmodel_lp'

PostProcPath = f"{LoadSVMFinalBPD}/TEST_BTQ/BPD/PostProcessingFigs_2/"
SaveFigPath = f"{PostProcPath}/Figs/"
SaveMoviePath = f"{PostProcPath}/MOVIE/"
SubstrateEffect = f"{PostProcPath}/SubstrateEffect/"
SaveHTMLPath = f"{PostProcPath}/HTML/"
if not os.path.isdir(SaveFigPath): 
    os.makedirs(SaveFigPath,exist_ok=True)
    os.makedirs(SubstrateEffect,exist_ok=True)
    os.makedirs(SaveHTMLPath,exist_ok=True)
    os.makedirs(SaveMoviePath,exist_ok=True)
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

#%%%----------------------- Ternary axis labels--------------------------------
axislabels = ["GaAs", "GaSb", "GaP"] # Left, right, bottom
AxesLabelcolors = None #{'b':'r','l':'b','r':'g'}
AxisColors=None #{'b':'g','l':'r','r':'b'}
textt = ['GaP','GaSb','GaAs'] #l,r,t    

#%%%%....................... 2D heatmap plot bandgap ..........................
BandGapNatureHeatmap = 0
OnlyContours = 1
if BandGapNatureHeatmap:
    tmp_featureseg = ["PHOSPHORUS",'ANTIMONY','ARSENIC','STRAIN','EgN']
    WhicColormap = plt.cm.get_cmap('Paired')
else:
    tmp_featureseg = ["PHOSPHORUS",'ANTIMONY','ARSENIC','STRAIN','bandgap']
    # WhicColormap = plt.cm.RdYlBu_r
    WhicColormap = colors.ListedColormap(['r','b'])

dfff = df.copy()
dfff['STRAIN'] = dfff['STRAIN'].round(1) #<-- the strain values are rounded to 1 decimal point 
mlpf.GenerateHeatmapSnapShots(StrainArray[:], tmp_featureseg, Predictions, movdirname=SaveMoviePath, 
                              generatetitle=1, savefig=1, axislabels=axislabels,
                              axiscolors=AxisColors,axislabelcolors=AxesLabelcolors,
                              scale=100,vmin=None, vmax=None, cbarlabel='Bandgap nature',  #E$_{\mathrm{g}}$
                              cmap=WhicColormap, cbarpos='bottom',contours=CONTOURS,
                              BandGapNatureHeatmap=BandGapNatureHeatmap,OnlyContour=OnlyContours,
                              UseContoursText=False, ContoursText=[cnt, anticnt],
                              COMContourText=['DIRECT','INDIRECT'],RawData=dfff,RawDataColorColumn='NATURE',DrawRawData=1)
#%% --------------------------------- others ----------------------------------
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
    plt.xlim(-0.15,0.15)
    if save:
        plt.savefig(savepath+'/'+figname,bbox_inches = 'tight',dpi=300)
        plt.close()
    else:
        plt.show()
    return 

dbname = '/home/bmondal/MachineLerning/BandGapML_project/GaPAsSb/RESULTS/ORIGINALdataSET/TEST_BTQ/BPD/Trial1/Data/my_rmse_fix/TestSetData.db'
conn = sq.connect(dbname)
TEST_df = pd.read_sql_query('SELECT * FROM TestSet', conn)
conn.close()

SaveFigPathTmp = '/home/bmondal/MachineLerning/BandGapML_project/GaPAsSb/RESULTS/ORIGINALdataSET/TEST_BTQ/BPD/Trial1/Figs/my_rmse_fix'

svr_bandgap = '/home/bmondal/MachineLerning/BandGapML_project/GaPAsSb/RESULTS/ORIGINALdataSET/TEST_BTQ/BPD/Trial1/MODELS/my_rmse_fix/svrmodel_bandgap.sav'
bandgapmag_model = pickle.load(open(svr_bandgap, 'rb')) 

YY = bandgapmag_model.predict(df[xfeatures])
XX = df['BANDGAP']
plot_err_dist(XX, YY, text=None, data_unit_label='eV',save=True, savepath=SaveFigPathTmp, figname='FullSetTruePredictErrorHist_.png')

YY = bandgapmag_model.predict(TEST_df[xfeatures])
XX = TEST_df['BANDGAP']
plot_err_dist(XX, YY,  data_unit_label='eV',save=True, savepath=SaveFigPathTmp, figname='TestSetTruePredictErrorHist_.png')