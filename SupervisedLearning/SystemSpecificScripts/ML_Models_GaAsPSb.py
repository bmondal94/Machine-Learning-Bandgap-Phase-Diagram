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

#pd.set_option('display.max_columns', 16)
#%% ------------------------- Load database -----------------------------------
#https://www.tensorflow.org/tutorials/keras/regression
# load dataset
dirpath = '/home/bmondal/MachineLerning/BandGapML/'
BinaryTrainedModels = 0
OnlyGoodData = 1
if BinaryTrainedModels:
    dbname = dirpath+'/DATAbase/Binary_BPD_MLDataBase_GaPAsSb.db'
else:
    if OnlyGoodData:
        dbname = dirpath+'/DATAbase/BPD_ML_Total_DataBase_GaPAsSb_GoodData.db'
    else:
        dbname = dirpath+'/DATAbase/BPD_ML_Total_DataBase_GaPAsSb.db'
    
    
BinaryConversion=True # 1: Direct, 0: Indirect

df, points = mlgf.CreateDataForModel(dbname, ReturnPredictPoints=False,BinaryConversion=BinaryConversion)
# df = df.iloc[:-2]

xfeatures = ['PHOSPHORUS', 'ANTIMONY', 'ARSENIC',  'STRAIN']
yfeatures = [s for s in list(df.columns) if s.startswith('BW')]

if points is not None: predict_points = pd.DataFrame(points, columns=xfeatures)
#%%----------------- BWs-Eg nature conversion ---------------------------------
df = mlgf.ConvertBWs2BandgapNature(df, BinaryConversion,colname='nature')

assert df['NATURE'].equals(df['nature']), 'The bandgap nature from BWs conversion doesnot match with actual.'

#%% ++++++++++++++++++++ Substrate part +++++++++++++++++++++++++++++++++++++++
EqulibriumData = df[df['STRAIN']==-4.5]
subname = ['GaAs','GaP','Si','Test']
sublattice = np.array([[5.689],[5.475],[5.43],[5.612]]) # Substrate lattice parameters

EqmDf = mlpf.generate_scatter_data(EqulibriumData[['PHOSPHORUS', 'ANTIMONY', 'ARSENIC']]) 
_ = mlpf.DrawRandomConfigurationPoints(EqmDf, fname=None, titletext=None, 
                                       savefig=False, scale=100,
                                       axislabels = ["GaSb", "GaAs", "GaP"],axiscolors=None,
                                       axislabelcolors=None, colorbar=False,
                                       fontsize = 20,colors=EqulibriumData['NATURE'])
#%%% *********************** Calculate the Strain data ************************
substraindict = {}
for I in range(len(subname)):
        substraindict[subname[I]]=(sublattice[I] - EqulibriumData['LATTICEPARAMETER1']) \
        /EqulibriumData['LATTICEPARAMETER1'] * 100
EqulibriumData = EqulibriumData.join(pd.DataFrame(substraindict))

#%%%************************** Plotting substate strains only *****************
#%%%%............... Plot 2d ternary data .....................................
WhichSubstrate = EqulibriumData[subname[0]]   
vmin = WhichSubstrate.min()
vmax = WhichSubstrate.max()
_ = mlpf.DrawRandomConfigurationPoints(EqmDf, fname=None, titletext=None, 
                                       savefig=False, scale=100,
                                       axislabels = ["GaSb", "GaAs", "GaP"],axiscolors=None,
                                       axislabelcolors=None, colorbar=True,
                                       fontsize = 20, colors=WhichSubstrate,vmin=vmin, vmax=vmax)                                    
#%%%%................ Plot 3d ternary data ....................................
_ = mlpf.Plot3DBandgapTernary(EqulibriumData['ARSENIC'], EqulibriumData['ANTIMONY'], 
                              WhichSubstrate,WhichSubstrate,
                              textt=['GaP','GaAs','GaSb'], scale=100) 

#%%++++++++++++++++++++++++ SVR model +++++++++++++++++++++++++++++++++++++++++

#%%%-------------- Model paths for Binary trained models ----------------------
dirpathSystem = dirpath+'/'+'GaPAsSb/RESULTS/GOODdataSET'
SaveFigPath = dirpathSystem + '/' +'Figs'
SaveMoviePath = dirpathSystem + '/' +'MOVIE'
SaveHTMLPath = dirpathSystem + '/' +'HTML'
modelPATHS = dirpathSystem+'/'+'MODELS'+'/'
OutPutTxtFile = modelPATHS + 'output.txt'  
svr_bandgap = modelPATHS + 'svrmodel_bandgap'
svc_EgNature = modelPATHS + 'svrmodel_EgNature'
if BinaryConversion:
    svc_EgNature += '_binary'
svr_bw = modelPATHS + 'svrmodel_bw'
svr_bw_dm = modelPATHS + 'DirectMultioutput/'
# os.makedirs(svr_bw_dm,exist_ok=True)
svr_lp = modelPATHS + 'svrmodel_lp'

#%%%***************** Training model type parameters***************************
retrain_models = 0
multiregression = False
regressorchain = False
retrainbw = False
retraineg = 1
MultipleScoringfn=True
retrainnature = 1
retrainlp = 0
if retrain_models:
    if os.path.exists(dirpathSystem):
        shutil.rmtree(dirpathSystem)  

    os.makedirs(dirpathSystem,exist_ok=True)
    os.makedirs(SaveFigPath,exist_ok=True) 
    os.makedirs(SaveMoviePath,exist_ok=True)
    os.makedirs(modelPATHS,exist_ok=True)

    if retrainlp: yfeatures='LATTICEPARAMETER1'
    
    if regressorchain: multiregression = False
    
    if (retrainbw or retraineg or retrainlp) and MultipleScoringfn: # This will automatically adjusted to 'accuracy' for SVC.
        scoringfn='r2' #('r2', 'neg_mean_squared_error')
        refit = True # 'r2'
    else:
        scoringfn = None; refit = True
        
    
    sys.stdout = open(OutPutTxtFile, "w")
    print(f"Training date: {datetime.now()}\n")
    _ = mlmf.SVMModelTrainingFunctions(df, xfeatures, yfeatures=yfeatures, scaley=100, 
                                       multiregression=multiregression, regressorchain=regressorchain, 
                                       IndependentOutput=False, retrainbw=retrainbw, retraineg=retraineg,
                                       retrainnature=retrainnature, retrainlp=retrainlp,
                                       svr_bw=svr_bw, svr_lp=svr_lp,svr_bandgap=svr_bandgap,
                                       svr_bw_dm=svr_bw_dm, svc_EgNature=svc_EgNature, save_model=1,
                                       scoringfn=scoringfn, refit=refit,
                                       PlotResults=1,LearningCurve=0,
                                       saveFig=True, savepath=SaveFigPath)
    sys.stdout = oldstd

#%%%********** Load model and predictions *************************************
#%%%---------- Test binary trained model on known ternary ---------------------
tmpmodelPATHS = dirpath+'/MODELS/GaPAsSb/BinaryTrainedModels/'
tmpsvr_bandgap = tmpmodelPATHS + 'svrmodel_bandgap'
tmpsvc_EgNature = tmpmodelPATHS + 'svrmodel_EgNature'
if BinaryConversion:
    tmpsvc_EgNature += '_binary'

BinaryConversion=True

dbnameee = dirpath+'/DATAbase/Ternary_Random_BPD_Ga_MLDataBase.db'
df_test, _ = mlgf.CreateDataForModel(dbnameee, ReturnPredictPoints=False,
                                     BinaryConversion=BinaryConversion)
dbnameeee = dirpath+'/DATAbase/Ternary_Random_BPD_Ga_MLDataBase_p2.db'
df_test_, _ = mlgf.CreateDataForModel(dbnameeee, ReturnPredictPoints=False,
                                     BinaryConversion=BinaryConversion)

df_test =pd.concat([df_test,df_test_], ignore_index=True)

TestPoints_X = df_test[xfeatures]

print(f"Testing binary trained model on known ternary date: {datetime.now()}\n")
print("* Model testing for bandgap prediction (SVR):")
TestPoints_Y_Eg = df_test['BANDGAP']
SVMBandgapModel = pickle.load(open(tmpsvr_bandgap+'.sav', 'rb')) # Bandgap model 
Y_bandgap = mlmf.TestModelTernaryPoints(TestPoints_X, TestPoints_Y_Eg, SVMBandgapModel)
mlpf.plot_true_predict_results(TestPoints_Y_Eg, Y_bandgap, text='Bandgap (eV)')

print("* Model testing for bandgap nature prediction (SVC):")
TestPoints_Y_EgNature = df_test['NATURE']
SVCEgNatureModel = pickle.load(open(tmpsvc_EgNature+'.sav', 'rb')) 
Y_BandgapNature = mlmf.TestModelTernaryPoints(TestPoints_X, TestPoints_Y_EgNature,
                                                    SVCEgNatureModel, SVCclassification=True)
mlpf.plot_true_predict_results(TestPoints_Y_EgNature, Y_BandgapNature, text='Bandgap nature')
cm = confusion_matrix(TestPoints_Y_EgNature, Y_BandgapNature)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
#%%%---------- Test ternary trained model on known ternary --------------------
SVRdependentSVC=False
print(f"Testing binary trained model on known ternary date: {datetime.now()}\n")
TestPoints = df[xfeatures+['BANDGAP','NATURE']].copy()
SVMBandgapModel = pickle.load(open(svr_bandgap+'.sav', 'rb')) # Bandgap model 
SVCEgNatureModel = pickle.load(open(svc_EgNature+'.sav', 'rb')) # Bandgap nature model
if SVRdependentSVC: 
    params = SVMBandgapModel['svrm'].get_params()
    params.pop('epsilon')
    SVCEgNatureModel['svrm'].set_params(**params)
    SaveFigPath += '/SVRdependentSVC/'
    SaveMoviePath += '/SVRdependentSVC/'
    os.mkdir(SaveFigPath)
    os.mkdir(SaveMoviePath)
   

TestPoints['Predictedbandgap'] = mlmf.TestModelTernaryPoints(TestPoints[xfeatures], TestPoints['BANDGAP'], SVMBandgapModel)
TestPoints['predNATURE'] = mlmf.TestModelTernaryPoints(TestPoints[xfeatures], TestPoints['NATURE'],
                                              SVCEgNatureModel, SVCclassification=True)

TestPoints['AbsErrorBndgap'] = (TestPoints['Predictedbandgap'] - TestPoints['BANDGAP']).abs()
BadSamples = TestPoints[TestPoints['AbsErrorBndgap'] > 0.15].copy()
WrongPrediction = TestPoints[TestPoints['predNATURE'] != TestPoints['NATURE']].copy()
CommonIdx = WrongPrediction.index.intersection(BadSamples.index)


ax = mlpf.plot_true_predict_results(TestPoints['BANDGAP'], TestPoints['Predictedbandgap'], text='Bandgap over all samples (eV)',savehist=True,savepath=SaveFigPath)
mlpf.plot_true_predict_results(BadSamples['BANDGAP'], BadSamples['Predictedbandgap'],ax=ax, my_color='r',marker='*',ShowLegend=False)
mlpf.plot_true_predict_results(WrongPrediction['BANDGAP'], WrongPrediction['Predictedbandgap'],ax=ax, my_color='k',marker='.',
                               save=True,figname='TruePredictFinal.png',savepath=SaveFigPath,ShowLegend=False)

if SVRdependentSVC: 
    sys.stdout = open(OutPutTxtFile, "a")
    print("\n***************************************************************************")
    print('The SVC prediction using the optimized hyperparameters from SVR (SVRdependentSVC)::')
    print('Classification report:\n',classification_report(TestPoints['NATURE'], TestPoints['predNATURE']))
    print(f"The accuracy score over all samples: {accuracy_score(TestPoints['NATURE'], TestPoints['predNATURE']):.3f}")
    sys.stdout = oldstd
    cm = confusion_matrix(TestPoints['NATURE'], TestPoints['predNATURE'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    disp.ax_.set_title('Bandgap nature prediction\n1=direct, 0=indirect')
    disp.figure_.savefig(SaveFigPath+'/NatureTruePredict.png',bbox_inches = 'tight',dpi=300)
    plt.close()
    print("***************************************************************************\n")
#%%%% Create the good samples database, deleting bad data from original database
BadSamples_atom = mlpf.generate_scatter_data(BadSamples[['PHOSPHORUS', 'ANTIMONY', 'ARSENIC']]) 
_ = mlpf.DrawRandomConfigurationPoints(BadSamples_atom, fname=None, titletext=None, 
                                       savefig=False, scale=100,
                                       axislabels = ["GaSb", "GaAs", "GaP"],axiscolors=None,
                                       axislabelcolors=None, colorbar=True,
                                       fontsize = 20,colors=BadSamples['STRAIN'],
                                       cbar_label_txt='Strain (%)')

WrongPrediction_atom = mlpf.generate_scatter_data(WrongPrediction[['PHOSPHORUS', 'ANTIMONY', 'ARSENIC']]) 
_ = mlpf.DrawRandomConfigurationPoints(WrongPrediction_atom, fname=None, titletext=None, 
                                       savefig=False, scale=100,
                                       axislabels = ["GaSb", "GaAs", "GaP"],axiscolors=None,
                                       axislabelcolors=None, colorbar=True,vmin=WrongPrediction['STRAIN'].min(),
                                       vmax=WrongPrediction['STRAIN'].max(),
                                       fontsize = 20,colors=WrongPrediction['STRAIN'],
                                       cbar_label_txt='Strain (%)')
_ = mlpf.DrawRandomConfigurationPoints(WrongPrediction_atom, fname=None, titletext=None, 
                                       savefig=False, scale=100,
                                       axislabels = ["GaSb", "GaAs", "GaP"],axiscolors=None,
                                       axislabelcolors=None, colorbar=False,cmap='Set1',
                                       fontsize = 20,colors=WrongPrediction['NATURE'])

# _ = mlpf.Plot3DBandgapTernary(BadSamples['ARSENIC'], BadSamples['ANTIMONY'], BadSamples['STRAIN'], BadSamples['AbsErrorBndgap']\
#                               ,textt=['GaP','GaSb','GaAs'], scale=100) 
WrongPrediction.drop(index=CommonIdx, inplace=True)
df.drop(index=BadSamples.index,inplace=True)
# df.drop(index=WrongPrediction.index,inplace=True)

conn = sq.connect('/home/bmondal/MachineLerning/BandGapML//DATAbase/BPD_ML_Total_DataBase_GaPAsSb_GoodData.db')
df.to_sql("COMPUTATIONALDATA", conn, index=False,if_exists='fail')
conn.close()
#%%%-------------------- Predictions ------------------------------------------
#%%%%--------------- Bloch weight models --------------------------------------
if multiregression or regressorchain:            
    loaded_model = pickle.load(open(svr_bw+'.sav', 'rb'))
    AllBws = loaded_model.predict(predict_points)
else:
    for I in yfeatures:
        save_f = svr_bw_dm + I
        loaded_model = pickle.load(open(save_f+'.sav', 'rb'))
        
#%%%%-------------- Lattice parameter model -----------------------------------
lattice_loaded_model = pickle.load(open(svr_lp+'.sav', 'rb'))
EqulibriumDataModel = predict_points[predict_points['STRAIN']==0].reset_index(drop=True)
LatticeParameterData = lattice_loaded_model.predict(EqulibriumDataModel)
substraindict = {}
for I in range(len(subname)):
        substraindict[subname[I]]=(sublattice[I] - LatticeParameterData)/\
            LatticeParameterData * 100
EqulibriumDataModel = EqulibriumDataModel.join(pd.DataFrame(substraindict))

#%%%%%............... Plot 2d ternary data ....................................
movdirname = "/home/bmondal/MachineLerning/BandGapML/MOVIE/GaPAsSb/SubstrateStrainHeatmap/"
WhichSubstrate = 'GaAs'
EqmDf = mlpf.generate_heatmap_data(EqulibriumDataModel[['PHOSPHORUS', 'ARSENIC', \
                                                        'ANTIMONY',WhichSubstrate]])   
vmin = EqulibriumDataModel[WhichSubstrate].min()
vmax = EqulibriumDataModel[WhichSubstrate].max()
_ = mlpf.DrawSnapshot(EqmDf,fname=movdirname+WhichSubstrate+'_sub.png', savefig=1, scale=100,
                      axislabels=["GaSb", "GaAs", "GaP"],cbarlabel='Substrate strain (%)',
                      vmin=-5, vmax=5,titletext=f'Substrate: {WhichSubstrate}') 

#%%%%---------------- Load Bandgap models -------------------------------------
bandgapmag_model = pickle.load(open(svr_bandgap+'.sav', 'rb')) 
SVRdependentSVC = True
saveFig=False
BandgapNatureModel = pickle.load(open(svc_EgNature+'.sav', 'rb'))
if SVRdependentSVC: 
    params = bandgapmag_model['svrm'].get_params()
    params.pop('epsilon')
    BandgapNatureModel['svrm'].set_params(**params)
    SaveFigPath += '/SVRdependentSVC/'
    SaveMoviePath += '/SVRdependentSVC/'
    os.mkdir(SaveFigPath)
    os.mkdir(SaveMoviePath)
    Y_predict_all = np.asarray(BandgapNatureModel.predict(df[xfeatures]), dtype=int)
    print(f"The accuracy score for prediction: {accuracy_score(df['NATURE'],Y_predict_all):.3f}")   
    labels = np.asarray(BandgapNatureModel['svrm'].classes_, dtype=int)    
    cm2 = confusion_matrix(df['NATURE'], Y_predict_all, labels=labels)
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2,display_labels=labels)
    disp2.plot()
    disp2.ax_.set_title('Bandgap nature prediction\n1=direct, 0=indirect')
    if saveFig:
        disp2.figure_.savefig(SaveFigPath+'/NatureTruePrediction.png',bbox_inches = 'tight',dpi=300)
        plt.close()
    else:
        plt.show()   
    # print('Classification report:\n',classification_report(df[yfeatures],Y_predict_all))
    
#%%%%.................... Create Bandgap nature contours ......................
WithPaddingData = False
Xp, Yp, POINTS_, CondPOSI, Zval = mlgf.CreateDataForPredictionLoopV3(resolution=101,
                                                                    compositionscale=100,
                                                                    WithPadding=WithPaddingData,
                                                                    features=xfeatures) 
StrainMin = -5;  StrainMax = 5; StrainPoints = 101
StrainArray = np.linspace(StrainMin, StrainMax, StrainPoints)
TotalSnapShot = len(StrainArray)
Predictions={}; cnt = {}; anticnt={}; CONTOURS = {}
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

#%%+++++++++++++++++++++++++++++++ Plotting +++++++++++++++++++++++++++++++++++
axislabels = ["GaAs", "GaSb", "GaP"] # Left, right, bottom
AxesLabelcolors = None #{'b':'r','l':'b','r':'g'}
AxisColors=None #{'b':'g','l':'r','r':'b'}
textt = ['GaP','GaSb','GaAs'] #l,r,t
#%%% ---------------------------- Plot Bandgap Ternary plot -------------------
#%%%%....................... 3D scatter plot bandgap ..........................
pp = pd.concat(Predictions.values()).reset_index(drop=True)
_ = mlpf.Plot3DBandgapTernary(pp['ARSENIC'], pp['ANTIMONY'], pp['STRAIN'], pp['bandgap'],textt=textt, scale=100) 
#%%%%....................... 2D heatmap plot bandgap ..........................
BandGapNatureHeatmap = 1
OnlyContours = 0
if BandGapNatureHeatmap:
    tmp_featureseg = ["PHOSPHORUS",'ANTIMONY','ARSENIC','STRAIN','EgN']
    WhicColormap = plt.cm.get_cmap('Paired')
else:
    tmp_featureseg = ["PHOSPHORUS",'ANTIMONY','ARSENIC','STRAIN','bandgap']
    WhicColormap = plt.cm.RdYlBu_r
mlpf.GenerateHeatmapSnapShots(StrainArray[:1], tmp_featureseg, Predictions, movdirname=SaveMoviePath, 
                              generatetitle=1, savefig=1, axislabels=axislabels,
                              axiscolors=AxisColors,axislabelcolors=AxesLabelcolors,
                              scale=100,vmin=0, vmax=2.5, cbarlabel='E$_{\mathrm{g}}$ (eV)',
                              cmap=WhicColormap, cbarpos='bottom',contours=CONTOURS,
                              BandGapNatureHeatmap=BandGapNatureHeatmap,OnlyContour=OnlyContours,
                              UseContoursText=False, ContoursText=[cnt, anticnt],
                              COMContourText=['D','I'],RawData=df,RawDataColorColumn='NATURE',DrawRawData=0)

#%%%%% ``````````````` Draw all the contours in 3D ````````````````````````````
figcont, axcont = mlpf.DrawAllContour3D(CONTOURS, fname=None, #cmap=None,
                                        savefig=False, scale=100, vmin=StrainMin, vmax=StrainMax,
                                        textt=['GaSb','GaP','GaAs'], ScatterPlot=False)

DrawSubstrate = 0
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
                        savefig=0, scale=100, vmin=StrainMin, vmax=StrainMax,
                        cmap=plt.cm.get_cmap('viridis'))
#%%%------------------------ Create movie from snapshots ----------------------
createmovie = True
if createmovie:
    images = []
    imgs = sorted(glob.glob(SaveMoviePath+"conf*.png"))
    for I in imgs[50:]: #compressive=[50::-1], tensile=[50:]
        #print(I)
        images.append(plt.imread(I))
    _ = mlpf.MakeEgStrainSnapShotMovie(images, movdirname=SaveMoviePath, savefig = 1)

#%%++++++++++++++++++++++++ Draw for html +++++++++++++++++++++++++++++++++++++
DrawHtmls = 1
if DrawHtmls:
    webdirname = SaveHTMLPath + '/'
    os.mkdir(webdirname)
    AxisLabels = ['GaP','GaSb','GaAs'] # Bottom, right, left
    cbarlabel='Strain(%)'
    #%%% ---------------- Draw Bandgap ( + nature contours) -----------------------
    #%%%%.............. Draw heatmaps (+contours) with slider in html .............
    fname = webdirname + 'BandgGapHeatMap.html'
    TestPredictions = {XX[0]:XX[1] for XX in list(Predictions.items())[:3]}
    _ = mlwpfP2.DrawBandgapHeatmapWebSlider(TestPredictions, ["PHOSPHORUS",'ANTIMONY','bandgap'], [cnt, cnt, anticnt], StrainArray, 
                                            ContourText=['D','I'],
                                            fname=fname, titletext=None,
                                            savefig=1, step=10,
                                            cmappp="viridis",color='black', 
                                            line_width=4, text=AxisLabels,
                                            scale=100,vmin=0, vmax=2.5, cbarlabel='Eg'
                                            )
    #%%% ----------------------- Draw Bandgap Nature contours -----------------
    #####.............. Draw all contours with slider in html .................
    #####........ Draw all contours with Multi Select in html..................
    fname = webdirname + 'MergeLayoutsAllContourV2.html'
    sliderlayoutv2 = mlwpf.DrawAllContourWebSliderV2([cnt, cnt, anticnt], StrainArray, CoverPage=CONTOURS,
                                                     ContourText=['D','I'],
                                                     IntializeContourText='', 
                                                     savefig=0, scale=100, vmin=StrainMin, vmax=StrainMax,fname=fname,
                                                     step=10,text=AxisLabels,cbarlabel=cbarlabel)
    multiselectlayoutv2 = mlwpf.DrawAllContourWebMultiSelectV2([cnt, cnt, anticnt], StrainArray, CoverPage=CONTOURS,
                                                               ContourText=['D','I'],
                                                               IntializeContourText='',
                                                               savefig=0, scale=100, vmin=StrainMin, vmax=StrainMax,fname=fname,
                                                               step=10,text=AxisLabels,cbarlabel=cbarlabel)
    
    layoutlistv2=[sliderlayoutv2,multiselectlayoutv2]
    mlwpf.MergeSliderSelect(layoutlistv2,fname=fname)
    
    #%%%%............... Draw 3D scatter plot DITs ................................
    fname = webdirname + "Contour3D.html"
    mlwpf.DrawAllContour3D(CONTOURS, StrainArray, fname=fname)
