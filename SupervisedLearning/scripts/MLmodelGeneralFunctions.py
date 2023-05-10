#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 16:24:05 2022

@author: bmondal
"""

import numpy as np
import sqlite3 as sq
import pandas as pd
import random
import os, glob

def CreateResultDirectories(dirpathSystem_,BinaryConversion, refitm):
    SaveFigPath = dirpathSystem_ +'/Figs/' 
    SaveMoviePath = dirpathSystem_ + '/MOVIE/' 
    SaveHTMLPath = dirpathSystem_ + '/' +'HTML'
    modelPATHS = dirpathSystem_+'/'+'MODELS'+'/'
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
    os.makedirs(SaveHTMLPath,exist_ok=True)
    os.makedirs(modelPATHS,exist_ok=True)
    
    return SaveFigPath, SaveMoviePath, SaveHTMLPath, modelPATHS, OutPutTxtFile, svr_bandgap,\
        svc_EgNature, svr_bw, svr_bw_dm, svr_lp
        
def CreateSubstrateStrainData(sub_lp_eg_dict,eqm_lp):
    substraindict = {}
    for SubstrateName in sub_lp_eg_dict:
        substraindict[SubstrateName]=(sub_lp_eg_dict[SubstrateName][0] - eqm_lp) / eqm_lp *100
    return pd.DataFrame(substraindict)

def CalculateLatticeParametersFromVegardsLaw_ternary(EqulibriumDataModel,CompoundList,GroupIII='Ga',
                                             LatticeParameters_bin = {'GaAs':5.689,'GaP':5.475,'GaSb':6.13383,'InAs':6.13756,'InP':5.93926,'InSb':6.55635},
                                             GroupV_map = {'ARSENIC':'As','PHOSPHORUS':'P','ANTIMONY':'Sb'}):
    lp = EqulibriumDataModel[CompoundList[0]]*LatticeParameters_bin[GroupIII+GroupV_map[CompoundList[0]]]
    for compound in CompoundList[1:]:
        if GroupIII is None: 
            print('GroupIII can not be none.')
            GroupIII='Ga'
        bin_cmpd =GroupIII+GroupV_map[compound]
        lp += EqulibriumDataModel[compound]*LatticeParameters_bin[bin_cmpd]
    return 0.01*lp
        
def SetSplitLearningParams(IIII, df, df_bintern, df_qua, SplitInputData=False,trainlp=False, 
                           LearningCurve = False, LearningCurveT3 = False,SVRdependentSVC=False):
    trainlatticeparameter = False
    if IIII == 'BT':
        df_train = df_bintern.copy(); df_test=df_qua.copy()
    elif IIII == 'Q':
        df_train = df_qua.copy(); df_test=df_bintern.copy()
    if IIII == 'BT_ct':
        df_train = df_bintern.copy(); df_test=df_qua.sample(frac=0.25,axis=0,ignore_index=False).reset_index(drop=True)
    elif IIII == 'Q_ct':
        df_train = df_qua.copy(); df_test=df_bintern.sample(frac=0.25,axis=0,ignore_index=False).reset_index(drop=True)
    elif IIII == 'BTq':
        df_train = df_bintern.copy(); df_test=df_qua.copy()
        SplitInputData=True
        LearningCurveT3 = True
    elif IIII == 'Qbt':
        df_train = df_qua.copy(); df_test=df_bintern.copy()
        SplitInputData=True
        LearningCurveT3 = True
    elif IIII == 'BTq_ct':
        df_train = df_bintern.copy(); df_test=df_qua.copy()
        SplitInputData=False
        LearningCurveT3 = True
    elif IIII == 'Qbt_ct':
        df_train = df_qua.copy(); df_test=df_bintern.copy()
        SplitInputData=False
        LearningCurveT3 = True
    elif IIII == 'BTQ':
        df_train = df.copy(); df_test=None
        LearningCurve = True
        SVRdependentSVC=False
    elif IIII == 'BPD':
        df_train = df.copy(); df_test=None
        SplitInputData=True
        SVRdependentSVC=True
    elif IIII == 'BPD_lp':
        df_train = df.copy(); df_test=None
        SplitInputData=True
        SVRdependentSVC=False
        trainlatticeparameter=True
    else:
        print(f"warning: The condition TEST_{IIII} is implemented.")

    return df_train, df_test, SplitInputData, LearningCurve, LearningCurveT3,SVRdependentSVC,trainlatticeparameter
    
def ReturnGridResolution(gridResolution=None):
    # Number of points in composition and strain
    if gridResolution == 'ultrahigh':
        grn = [1001j, 1001j, 1001j, 101j]
    elif gridResolution == 'high':
        grn = [101j, 101j, 101j, 51j]
    elif gridResolution == 'middle':
        grn = [51j, 51j, 51j, 21j]
    else:
        grn = [11j, 11j, 11j, 11j]
    print(
        f"* The gridresolution for predictions is chossen {gridResolution}: {grn}")
    return grn


def CreateDataForModel(dbname, ReturnPredictPoints=True, adddummies=False,
                       BinaryConversion=False, gridResolution='middle',NatureMap={2: 1, 3: 0, 4: 0},
                       start=[0, 0, 0, -5], end=[100, 100, 100, 5], compositionscale=100):

    if 'xlsx' in dbname:
    	df = pd.read_excel(dbname, sheet_name='DFT_COMPUTATIONAL_DATA') 
    elif 'db' in dbname:
    	conn = sq.connect(dbname)
    	df = pd.read_sql_query('SELECT * FROM COMPUTATIONALDATA', conn)
    	conn.close()
    else:
    	print('wrong database format. Only sqlite3 .db and excel .xlsx is allowed.')
    df = df.dropna()

    if adddummies:
        df['NATUREDUMMIES'] = df['NATURE'].map(
            {1: 'D', 2: 'd', 3: 'I', 4: 'i'})
        df = pd.get_dummies(
            df, columns=['NATUREDUMMIES'], prefix='', prefix_sep='')
    elif BinaryConversion:
        df = df.replace({"NATURE": NatureMap})
        
    if ReturnPredictPoints:
        grn = ReturnGridResolution(gridResolution=gridResolution)

        xc, yc, zc, strainp = np.mgrid[start[0]:end[0]:grn[0], start[1]:end[1]:grn[1],
                                       start[2]:end[2]:grn[2], start[3]:end[3]:grn[3]]
        xyz = np.vstack((xc.flat, yc.flat, zc.flat, strainp.flat)).T
        points = xyz[np.sum(xyz[:, :3], axis=1) == compositionscale]
    else:
        points = None
    return df, points


def CreateDataForPrediction(gridResolution='middle', start=[0, 0, 0, -5],
                            end=[100, 100, 100, 5], compositionscale=100):
    if isinstance(gridResolution, list) or isinstance(gridResolution, tuple):
        grn = gridResolution
    else:
        grn = ReturnGridResolution(gridResolution=gridResolution)

    xc, yc, zc, strainp = np.mgrid[start[0]:end[0]:grn[0], start[1]:end[1]:grn[1],
                                   start[2]:end[2]:grn[2], start[3]:end[3]:grn[3]]
    xyz = np.vstack((xc.flat, yc.flat, zc.flat, strainp.flat)).T
    points = xyz[np.sum(xyz[:, :3], axis=1) == compositionscale]
    return points


def CreateDataForPredictionLoop(strain=[-5, 5], resolution=[1001, 101],
                                columns=None, compositionscale=100,
                                compositionstart=0):
    """
    This functions creates the prediction points with less memory.

    Parameters
    ----------
    strain : float list, optional
        The strain list [start,end]. The default is [-5,5].
    resolution : integer list, optional
        The resolution list [composiotion_resolution, strain_resolution]. The default is [1001,101].
    columns : string list
        The name of the columns. The default is None.
    compositionscale : float/int, optional
        The scale of composition. The default is 100.
    compositionstart: float/int, optional    
        The start scale of composition. The default is 0.
    Returns
    -------
    points : nd numpy array
        The points coordinate for predictions.

    """
    XX = np.linspace(compositionstart, compositionscale, resolution[0])
    ZZ = np.linspace(strain[0], strain[1], resolution[1])
    points = []
    for _, I in enumerate(XX):
        for J in XX:
            K = compositionscale-I-J
            if K < 0:
                break
            for L in ZZ:
                points.append([I, J, K, L])

    return pd.DataFrame(points, columns=columns)


def CreateDataForPredictionLoopV2(strain=[-5, 5], resolution=[1001, 101],
                                  columns=None, compositionscale=100,
                                  compositionstart=0, compositionend=None):
    """
    This functions creates the prediction points with less memory.

    Parameters
    ----------
    strain : float list, optional
        The strain list [start,end]. The default is [-5,5].
    resolution : integer list, optional
        The resolution list [composiotion_resolution, strain_resolution]. The default is [1001,101].
    columns : string list
        The name of the columns. The default is None.
    compositionscale : TYPE, optional
        The scale of composition. The default is 100.
    compositionstart: float/int, optional    
        The start scale of composition. The default is 0.
    compositionend: float/int, optional    
        The end scale of composition. The default is None. If None the end
        scale will be set to compositionscale.

    Returns
    -------
    points : nd numpy array
        The points coordinate for predictions.

    """
    if compositionend is None:
        compositionend = compositionscale

    XX = np.linspace(compositionstart, compositionend, resolution[0])
    ZZ = np.linspace(strain[0], strain[1], resolution[1])
    points = []
    for I in XX:
        for J in XX:
            K = compositionscale-I-J
            if K < 0:
                break
            points.append([I, J, K, 0])

    points = np.array(points)
    Ppoints = []
    for L in ZZ:
        points[:, -1] = L
        Ppoints.append(np.copy(points))

    return pd.DataFrame(np.concatenate(Ppoints, axis=0), columns=columns)


def CreateDataForPredictionLoopV3(resolution=101, compositionscale=100,
                                  compositionstart=0, compositionend=None,
                                  WithPadding=False, features=None):
    if compositionend is None:
        compositionend = compositionscale
    XX = np.linspace(compositionstart, compositionend, resolution)
    if WithPadding:
        padding = 0.1/compositionscale
        XX = np.concatenate([[compositionstart-padding],
                            XX, [compositionend+padding]])
    Xp, Yp = np.meshgrid(XX, XX)
    if WithPadding:
        for I in range(1, len(Yp)):
            Yp[I, -I] = Yp[I-1, -I]+padding
    Zval = np.empty(np.shape(Xp))
    Zval[:] = np.nan
    CondPOSI = (Xp + Yp <= compositionscale)
    Xpp, Ypp = Xp[CondPOSI], Yp[CondPOSI]
    if features is None:
        POINTS = np.stack((Xpp, Ypp, compositionscale-Xpp -
                          Ypp, np.zeros(len(Xpp))), axis=-1)
    else:   
        POINTS = pd.DataFrame({features[0]:Xpp, features[1]:Ypp, features[2]:compositionscale-Xpp-Ypp})
    return Xp, Yp, POINTS, CondPOSI, Zval


def CreateSnapEgNatureData(p):
    return pd.DataFrame(p)


def CreateRandomDataPointsTernary(strain=[-5, 5], resolution=[1001, 101],
                                  columns=None, compositionscale=100,
                                  compositionstart=0, compositionend=None,
                                  npoints=1):
    """
    This functions creates the random points in the configuration space.

    Parameters
    ----------
    strain : float list, optional
        The strain list [start,end]. The default is [-5,5].
    resolution : integer list, optional
        The resolution list [composiotion_resolution, strain_resolution]. The default is [1001,101].
    columns : string list
        The name of the columns. The default is None.
    compositionscale : float/int, optional
        The scale of composition. The default is 100.
    compositionstart: float/int, optional    
        The start scale of composition. The default is 0.
    compositionend: float/int, optional    
        The end scale of composition. The default is None. If None the end
        scale will be set to compositionscale.
    npoints: int, optional
        Number of random points to generate. Default is 1. 
    Returns
    -------
    points : nd numpy array
        The random points coordinate.

    """
    predict_points = CreateDataForPredictionLoopV2(strain=strain,
                                                   resolution=resolution,
                                                   compositionscale=compositionscale,
                                                   columns=columns,
                                                   compositionstart=compositionstart,
                                                   compositionend=compositionend)

    randomchoice = random.sample(range(0, len(predict_points)), npoints)

    pp = predict_points.iloc[randomchoice]

    return pp.reset_index(drop=True)


def CreateRandomData_AB_CDE(strain=[-5, 5],
                            columns=None, compositionscale=100, npoints=1,
                            compositionstart=0, compositionend=None):
    """
    This functions creates the prediction points with less memory.

    Parameters
    ----------
    strain : float list, optional
        The strain list [start,end]. The default is [-5,5].
    columns : string list
        The name of the columns. The default is None.
    compositionscale : TYPE, optional
        The scale of composition. The default is 100.
    compositionstart: float/int, optional    
        The start scale of composition. The default is 0.
    compositionend: float/int, optional    
        The end scale of composition. The default is None. If None the end
        scale will be set to compositionscale.
    npoints: int, optional
        Number of random points to generate. Default is 1. 

    Returns
    -------
    points : nd numpy array
        The points coordinate for predictions.

    """

    if compositionend is None:
        compositionend = compositionscale

    # https://stackoverflow.com/a/47418580
    x = np.random.rand(4, npoints)
    x[1:3] = np.sort(x[1:3], axis=0)
    # Triangle to cartesian
    # v = np.array([(0,0),(50.0, 86.60254037844386), (100, 0)])
    # As_P_Sb = np.column_stack([x[1], x[2]-x[1], 1.0-x[2]]) @ v  # Matrix multiplication
    
    x[:3] = x[:3] * (compositionend - compositionstart) + compositionstart

    # In, Ga, As, P, Sb, strain
    data = np.column_stack([x[0], 100.0-x[0], x[1], x[2]-x[1],
                           100.0-x[2], x[3]*(strain[1]-strain[0]) + strain[0]])
    return pd.DataFrame(data, columns=columns) #, As_P_Sb

def CreateRandomData_A_BC(strain=[-5, 5],
                          columns=None, compositionscale=100, npoints=1,
                          compositionstart=0, compositionend=None):
    """
    This functions creates the prediction points with less memory.

    Parameters
    ----------
    strain : float list, optional
        The strain list [start,end]. The default is [-5,5].
    columns : string list
        The name of the columns. The default is None.
    compositionscale : TYPE, optional
        The scale of composition. The default is 100.
    compositionstart: float/int, optional    
        The start scale of composition. The default is 0.
    compositionend: float/int, optional    
        The end scale of composition. The default is None. If None the end
        scale will be set to compositionscale.
    npoints: int, optional
        Number of random points to generate. Default is 1. 

    Returns
    -------
    points : nd numpy array
        The points coordinate for predictions.

    """

    if compositionend is None:
        compositionend = compositionscale

    # https://stackoverflow.com/a/47418580
    x = np.random.rand(2, npoints)  
    x[0] = x[0] * (compositionend - compositionstart) + compositionstart

    # P, strain
    data = np.column_stack([x[0], x[1]*(strain[1]-strain[0]) + strain[0]])
    return pd.DataFrame(data, columns=columns) 

def DropCommonDataPreviousDataBase(dbname, UnqueDB):
    """
    This function drops the rows that matches with the supplied database.

    Parameters
    ----------
    dbname : string
        Path of the database to compare with.
    UnqueDB : sqlite3 database
        Database to compare for.

    Returns
    -------
    sqlite3 database
        Droped unique database.

    """

    conn = sq.connect(dbname)

    df = pd.read_sql_query('SELECT * FROM ATOMNUMBER', conn)

    FindCommonIndex = []
    for index, row in UnqueDB.iterrows():
        if (df == row).all(1).any():
            FindCommonIndex.append(index)

    return df, UnqueDB.drop(FindCommonIndex)

def ReadOutputTxtFile(filename__):
    pattern_3 = 'Training dataset size'
    pattern_4 = ['out-of-sample','all-sample']
    pattern_4_map = {'out-of-sample':'(o','all-sample':'(a'}
    pattern_5 = ['r2_score','root_mean_squared_error','mean_absolute_error','max_error','accuracy_score','balanced_accuracy_score']
    training_dataset = []; tmp_dict = {}
    with open(filename__, 'r') as searchfile:
        for line in searchfile.readlines():
            if pattern_3 in line:
                training_dataset.append(tmp_dict)
                tmp_dict = {}
                tmp_dict['dataset_size'] = int(line.split()[-1]) 
            else:
                for pattern_4_tmp in pattern_4:
                    if (pattern_4_tmp in line) and (line.startswith(pattern_4_map[pattern_4_tmp])):
                        for pattern_5_tmp in pattern_5:
                            if pattern_5_tmp in line:
                                split_val = (line.split(':')[-1]).split()
                                put_val = float(split_val[0])
                                if split_val[-1] == 'eV':
                                    put_val *= 1000 #<-- eV to meV conversion 
                                tmp_dict[pattern_4_tmp+'_'+pattern_5_tmp] = put_val
        training_dataset.append(tmp_dict)
    pp = pd.DataFrame(training_dataset[1:]) 
    return pp

def FindBestTrialBTQ(filename__):
    SearchBPDTrials = glob.glob(filename__)
    cv_score = []
    fname = []
    for SearchBPDT in SearchBPDTrials:
        with open(SearchBPDT+'/output.txt', 'r') as SearchBPDTrial:
            for line in SearchBPDTrial.readlines():
                if line.startswith('Best parameter'):
                    cv_score.append(float(line.split()[3].split('=')[-1]))
                    fname.append(SearchBPDT)
                    break
    best_cv_score = np.argmax(cv_score)
    return cv_score[best_cv_score], fname[best_cv_score]

def SpecialPlots_1(filename):
    OutdataFrame_c = {}
    for fname in ['my_rmse_fix','accuracy']:
        OutdataFrame = ReadOutputTxtFile(filename+'/MODELS/'+fname+'/output.txt')
        if fname == 'BTQ':
            SearchBPDTrials = glob.glob(f"{filename}/BPD/*/MODELS/{fname}/output.txt")
            LastPointData = [OutdataFrame]
            for OutPutTxtFile in SearchBPDTrials:
                LastPointData.append(ReadOutputTxtFile(OutPutTxtFile))  
            OutdataFrame = pd.concat(LastPointData, axis=0)
        OutdataFrame_c[fname] = OutdataFrame
    TMP = pd.concat(OutdataFrame_c.values())
    return TMP

def SpecialPlots_2(filename_list,metric,metric_nickname):
    OutdataFrame_c = {}
    for filename in filename_list:
        OutdataFrame = ReadOutputTxtFile(f"{filename}/MODELS/{metric_nickname}/output.txt")
        OutdataFrame_c[filename] = OutdataFrame[['dataset_size','out-of-sample_'+metric]]
    return OutdataFrame_c
# %%----------------- BWs-Eg nature conversion ---------------------------------


def ConvertBWs2BandgapNature(df, BinaryConversion, colname='nature'):
    BWnames = [s for s in list(df.columns) if s.startswith('BW')]
    BWvbnames = [s for s in BWnames if s.startswith('BWvb')]
    BWcbnames = [s for s in BWnames if s.startswith('BWcb')]
    # BW..1: Gamma; BW..2: 0.166667; BW..3: 0.33333; BW..4: X;
    # BW..5: (0.166667 0.166667); BW..6: others; BW..7: (0.33333 0.33333); BW..8: L
    VBdata = df[BWvbnames]
    CBdata = df[BWcbnames]

    BWcb_pos = CBdata.idxmax(axis=1).str[-1:].astype(int)
    BWvb_pos = VBdata.idxmax(axis=1).str[-1:].astype(int)
    # BWcb = CBdata.max(axis=1)
    # BWvb = VBdata.max(axis=1)
    #CBM_VBM = pd.concat([BWcb_pos,BWvb_pos,BWcb,BWvb],axis=1)
    if BinaryConversion:
        # 1: Direct, 0: Indirect
        df[colname] = 0
        df.loc[BWcb_pos == BWvb_pos, colname] = 1
    else:
        # 1==Direct bandgap at the Gamma point
        # 2==Direct bandgap at some other k-point
        # 3==Indirect bandgap with VBM at the Gamma point
        # 4==Indirect bandgap with VBM at some other k-point
        df[colname] = 4
        df.loc[BWvb_pos == 0, colname] = 3
        df.loc[BWcb_pos == BWvb_pos, colname] = 2
        df.loc[((BWcb_pos == BWvb_pos) & (BWvb_pos == 0)), colname] = 1
    return df
