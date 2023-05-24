#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 14:37:24 2022

@author: bmondal
"""

import numpy as np
import sqlite3 as sq
import pandas as pd
import argparse
import os
from datetime import datetime
import HeaderTxt
#%%
def AssertNoContradiction(TrainEgNature=True,SVRdependentSVC=False,TrainNatureOnly=False,
                          FinalSVRDependentSVC=False,TrainEgOnly=False,Static_m_Models=False,
                          randdom_sampling_prediction=False,PredictProb=False):
    '''
    Priority order: TrainEgNature (and/or SVRdependentSVC) > TrainNatureOnly > FinalSVRDependentSVC > TrainEgOnly
    '''
    
    if TrainEgNature:
        TrainEgOnly = False
        TrainNatureOnly = False
        FinalSVRDependentSVC = False
        ModelDescription = 'Train both bandgap magnitude and nature models together sequentially: \n\tSVR(rbf) + SVC(rbf) model'
        Add2PathALdirectory = 'SVRSVC'
        if SVRdependentSVC:
            ModelDescription = 'Train bandgap magnitude model and dependent nature models sequentially: \n\tSVR(rbf) --> SVC(rbf) model'
            Add2PathALdirectory = 'SVRdependentSVC'
    elif TrainNatureOnly: 
        TrainEgOnly = False
        FinalSVRDependentSVC = False
        ModelDescription = 'Only bandgap nature prediction is chossen: \n\tSVC(rbf) model'
        Add2PathALdirectory = 'SVC'
    elif FinalSVRDependentSVC:
        TrainEgOnly = True
        TrainNatureOnly = False
        TrainEgNature = False
        ModelDescription = 'Train bandgap magnitude model only and finally predict\n both bandgap magnitude and nature: \n\tSVR(rbf) model, SVR(rbf) dependent SVC(rbf) model'
        Add2PathALdirectory = 'FinalSVRDependentSVC'
    else: # TrainEgOnly=True
        TrainNatureOnly = False
        TrainEgNature = False
        ModelDescription = 'Only bandgap magnitude prediction is chossen: \n\tSVR(rbf) model'
        Add2PathALdirectory = 'SVR'
        
    if randdom_sampling_prediction: Add2PathALdirectory += '_RandomSamplingPred'
    if Static_m_Models: Add2PathALdirectory += '_StaticMmodels'
    if PredictProb: Add2PathALdirectory += '_PredictProb'
    return ModelDescription,Add2PathALdirectory,TrainEgNature,SVRdependentSVC,TrainNatureOnly,FinalSVRDependentSVC,TrainEgOnly

def WelcomeMsg():
    print(f"{'='*78}\n{'Welcome to Active Learning':.^78}")
    My_MSG='''
            The code was written by Badal Mondal as a part of the PhD project. 
                More details can be found here:
            '''
    print(f"{My_MSG}")
    
def convert_conc2atomnumber(concsrray,natoms=216):
    N_atom = (concsrray[['INDIUM']]*(natoms/100)).round(0).astype(int)
    N_atom['GALLIUM'] =natoms - N_atom['INDIUM']
    N_atom[['PHOSPHORUS', 'ARSENIC']] = (concsrray[['PHOSPHORUS', 'ARSENIC']]*(natoms/100)).round(0).astype(int)
    N_atom['ANTIMONY'] = natoms-N_atom[['PHOSPHORUS', 'ARSENIC']].sum(axis=1)
    N_atom['STRAIN'] = concsrray['STRAIN']
    return N_atom

def CreateDataForModel(dbname, adddummies=False, BinaryConversion=False, ):

    conn = sq.connect(dbname)
    df = pd.read_sql_query('SELECT * FROM COMPUTATIONALDATA', conn)
    df = df.dropna()
    conn.close()
    if adddummies:
        df['NATUREDUMMIES'] = df['NATURE'].map(
            {1: 'D', 2: 'd', 3: 'I', 4: 'i'})
        df = pd.get_dummies(
            df, columns=['NATUREDUMMIES'], prefix='', prefix_sep='')
    elif BinaryConversion:
        df = df.replace({"NATURE": {1: 0, 2: 0, 3: 1, 4: 1}})

    return df

def CreateRandomData_AB_CDE(strain=[-5, 5],
                            columns=None, compositionscale=100, npoints=1,
                            compositionstart=0, compositionend=None, Ternary=False):
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
    Ternary: bool, optional
        If ternary material. Default is False.

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
    
    if Ternary:
        x[0] = 0
        x[1] = 0

    # In, Ga, P, As, Sb, strain
    data = np.column_stack([x[0], 100.0-x[0], x[2]-x[1], x[1], 
                           100.0-x[2], x[3]*(strain[1]-strain[0]) + strain[0]])
    return pd.DataFrame(data, columns=columns) #, As_P_Sb

def CreateVaspFolders(dirpath,df,natoms, setf=0,setcounter=0,nfilesperfolder=None):
    if nfilesperfolder == 0 or nfilesperfolder is None: nfilesperfolder = len(df)
    
    for I in df[['INDIUM','GALLIUM','PHOSPHORUS', 'ARSENIC', 'ANTIMONY','STRAIN']].values:       
        path = dirpath + f'/SET{setcounter:03d}/In{I[0]}Ga{I[1]}P{I[2]}As{I[3]}Sb{I[4]}'
        os.makedirs(path + '/S{I[5]:.6f}/conf01')
        os.makedirs(path + '/EQM/conf01')
        if not os.path.exists(path): # Minimizes unstrained geometry optimization
            os.makedirs(path + '/RNDSTR')
            filename1 = path + '/RNDSTR/rndstr.in'
            filename2 = path + '/RNDSTR/sqscell.out'
            if abs(I[0] - 0) < 1e-5:
                ttt = f"In={I[0]/natoms}"
                if abs(I[1] - 0) < 1e-5:
                    ttt += f", Ga={I[1]/natoms}"
            elif abs(I[1] - 0) < 1e-5:
                ttt = f"Ga={I[1]/natoms}"
            
            if abs(I[2] - 0) < 1e-5:
                ttt2 = f"P={I[2]/natoms}"
                if abs(I[3] - 0) < 1e-5:
                    ttt2 += f", As={I[3]/natoms}"
                    if abs(I[4] - 0) < 1e-5:
                        ttt2 += f", Sb={I[4]/natoms}"  
                elif abs(I[4] - 0) < 1e-5:
                    ttt2 += f", Sb={I[4]/natoms}"
            elif abs(I[3] - 0) < 1e-5:
                ttt2 = f"As={I[3]/natoms}"
                if abs(I[4] - 0) < 1e-5:
                    ttt2 += f", Sb={I[4]/natoms}"
            elif abs(I[4] - 0) < 1e-5:
                ttt2 = f", Sb={I[4]/natoms}"
                
            f1text = f"""1.   1.   1.   90   90   90
0.0   0.5   0.5
0.5   0.0   0.5
0.5   0.5   0.0
0.0   0.0   0.0  {ttt}
0.25  0.25  0.25  {ttt2}
            """
            f2text = """1
0  3  3
3  0  3
3  3  0 
            """
            with open(filename1, 'w') as temp_file:
                temp_file.write(f1text)
            with open(filename2, 'w') as temp_file:
                temp_file.write(f2text)
        setf += 1
        if setf % nfilesperfolder == 0:
            setcounter += 1

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

# def DumpModelAccuracyData(LoopIndex,model_accuracy,conn,TrainEgOnly=True,\
#                           TrainNatureOnly=False,TrainEgNature=False,SVRdependentSVC=False):
#     if TrainEgOnly or TrainNatureOnly or (TrainEgNature and SVRdependentSVC):
#         model_accuracy['LoopIndex'] = LoopIndex
#         pd.DataFrame([model_accuracy]).to_sql("ModelAccuracy", conn, index=False,if_exists='append') 
#     else:
#         model_accuracy[0]['LoopIndex'] = LoopIndex
#         pd.DataFrame([model_accuracy[0]]).to_sql("ModelAccuracy", conn, index=False,if_exists='append')
#         model_accuracy[1]['LoopIndex'] = LoopIndex
#         pd.DataFrame([model_accuracy[1]]).to_sql("NatureModelAccuracy", conn, index=False,if_exists='append')
#     return
def DumpModelAccuracyData(LoopIndex,SetSize,model_accuracy,conn,TrainEgOnly=True,\
                          TrainNatureOnly=False,TrainEgNature=False,SVRdependentSVC=False):
    if TrainEgOnly or TrainNatureOnly or (TrainEgNature and SVRdependentSVC):
        model_accuracy['LoopIndex'] = LoopIndex
        model_accuracy['TrainingSamleCount'] = SetSize
        pd.DataFrame(model_accuracy).to_sql("ModelAccuracy", conn, index=True,if_exists='append') 
    else:
        model_accuracy[0]['LoopIndex'] = LoopIndex
        model_accuracy[0]['TrainingSamleCount'] = SetSize
        pd.DataFrame(model_accuracy[0]).to_sql("ModelAccuracy", conn, index=True,if_exists='append')
        model_accuracy[1]['LoopIndex'] = LoopIndex
        model_accuracy[1]['TrainingSamleCount'] = SetSize
        pd.DataFrame(model_accuracy[1]).to_sql("NatureModelAccuracy", conn, index=True,if_exists='append')
    return

def GenerateBadSamples(PredictOvernSamples,BadSamplesCondition,conn,XFEATURES,
                       ntry_max,LoopIndex,nature_accuracy_cutoff,natoms=216,
                       LoopNonConvergenceCondition5=False,TrainingReachedAccuracy=False,
                       SVRdependentSVC=False,SVRSVCmodel=False,SVRModel=False,
                       TakeFracBadSamples=0.05):
    if LoopNonConvergenceCondition5: # Loop convergence condition 5
        TestPickRandomConfiguration = PredictOvernSamples[(BadSamplesCondition).values].reset_index(drop=True)
        GetFrac = min(TakeFracBadSamples, 100/len(TestPickRandomConfiguration))
        print(f"Note: Only {GetFrac*100:.2f}% of the bad samples will be added to the feed-back batch.")
        ntry = 0
        dff = pd.read_sql_query('SELECT * FROM TotalBatchs', conn)
        while True:
            TestPickRandomConfiguration_ = TestPickRandomConfiguration.sample(frac=GetFrac, ignore_index=True) 
            PickRandomConfiguration = TestPickRandomConfiguration_[XFEATURES]
            ##### Convert concentration to atom numbers
            PickRandomConfigurationAtom = convert_conc2atomnumber(PickRandomConfiguration,natoms=natoms)
            ##### Drop and collect the duplicates
            PickRandomConfigurationAtom__ = pd.concat([dff, PickRandomConfigurationAtom])
            DuplicatedIndices = PickRandomConfigurationAtom__[PickRandomConfigurationAtom__.duplicated()].index
            Duplicates = PickRandomConfigurationAtom.iloc[DuplicatedIndices].drop_duplicates(keep=False).reset_index(drop=True)

            if Duplicates.empty:
                break # Break while loop
            else:
                PickRandomConfigurationAtom = PickRandomConfigurationAtom.drop(DuplicatedIndices).reset_index(drop=True)
                if not PickRandomConfigurationAtom.empty:
                    Duplicates.to_sql(f"DuplicateBATCH_{LoopIndex}", conn, index=False,if_exists='fail')
                    if ntry > 0:
                        print(f'\tTry-{ntry}: picked bad samples = {len(PickRandomConfiguration)}, duplication = {len(Duplicates)}')
                    break # Break while loop
                else:
                    if ntry == 0:
                        print(f'No new samples. Picked bad samples = {len(PickRandomConfiguration)}, duplication = {len(Duplicates)}.')
                        print('Trying sample again.')                                  
                    else:
                        print(f'\tTry-{ntry}: picked bad samples = {len(PickRandomConfiguration)}, duplication = {len(Duplicates)}')
                    
                    if ntry == ntry_max: # Loop convergence condition 6
                        TrainingReachedAccuracy = True 
                        if SVRSVCmodel and not SVRdependentSVC:
                            print('Maximum # of try to find bad samples is reached. Terminating try.')
                        else:
                            print('Maximum # of try to find bad samples is reached. Terminating try.')
                            print('\n** Model is reached to limit. Given the cut-offs current AL model cannot perform better.')
                            print('** Terminating AL loop.')
                            print('Hint: Try to increase the "model_accuracy_error_cutoff" and/or "std_cutoff".')
                        # print(ntry, TestPickRandomConfiguration_)
                        break # Break while loop
                    
            ntry += 1
        TestPickRandomConfiguration = TestPickRandomConfiguration_.drop(DuplicatedIndices).reset_index(drop=True)
        # print(PickRandomConfigurationAtom)
        return TrainingReachedAccuracy, (PickRandomConfigurationAtom,PickRandomConfigurationAtom,pd.DataFrame()), TestPickRandomConfiguration
    else:
        if SVRSVCmodel:
            if SVRdependentSVC:
                print(f'\n* RMSE (over samples) of bandgap magnitude prediction (mean over models) reached cutoff={nature_accuracy_cutoff} eV')
                print('* Training complete. As the model is SVRdependentSVC, training depends only on SVR. Exiting AL loop.') 
            else:
                print(f' - RMSE (over samples) of bandgap magnitude prediction (mean over models) reached cutoff={nature_accuracy_cutoff} eV')

        else:
            extra_txt = 'magnitude' if SVRModel else 'nature'
            unit_x = 'eV' if SVRModel else ''
            print(f'\n* Average samples reached to cutoff bandgap {extra_txt} prediction of {nature_accuracy_cutoff} {unit_x} over models.')
            print('* Training complete. Exiting AL loop.')  
        return True, (pd.DataFrame(),), None

def GenerateBadSamples_SVRSVC(PredictOvernSamples,BadSamplesCondition,conn,XFEATURES,
                              PickRandomConfigurationAtom_svr,TestPickRandomConfiguration_svr,ntry_max,LoopIndex,natoms=216,
                              TrainingReachedAccuracy=False,TakeFracBadSamples=0.05):
    TestPickRandomConfiguration = PredictOvernSamples[(BadSamplesCondition).values].reset_index(drop=True)
    GetFrac = min(TakeFracBadSamples, 100/len(TestPickRandomConfiguration))
    print(f"Note: Only {GetFrac*100:.2f}% of the bad samples will be added to the feed-back batch.")
    ntry = 0
    dff_ = pd.read_sql_query('SELECT * FROM TotalBatchs', conn)
    ##### Drop duplicates from SVR chossen bad samples
    dff = pd.concat([dff_, PickRandomConfigurationAtom_svr])
    dff.drop_duplicates(inplace=True,ignore_index=True)
    
    while True:
        TestPickRandomConfiguration_ = TestPickRandomConfiguration.sample(frac=GetFrac, ignore_index=True) 
        PickRandomConfiguration = TestPickRandomConfiguration_[XFEATURES]
        ##### Convert concentration to atom numbers
        PickRandomConfigurationAtom = convert_conc2atomnumber(PickRandomConfiguration,natoms=natoms)
        ##### Drop and collect the duplicates
        PickRandomConfigurationAtom__nature = pd.concat([dff_, PickRandomConfigurationAtom])
        DuplicatedIndicesnature = PickRandomConfigurationAtom__nature[PickRandomConfigurationAtom__nature.duplicated(keep='first')].index
        PickRandomConfigurationAtom__nature = PickRandomConfigurationAtom.drop(DuplicatedIndicesnature).reset_index(drop=True)  
        # print(dff_,PickRandomConfigurationAtom,PickRandomConfigurationAtom__nature)
        
        PickRandomConfigurationAtom__ = pd.concat([dff, PickRandomConfigurationAtom])
        DuplicatedIndices = PickRandomConfigurationAtom__[PickRandomConfigurationAtom__.duplicated()].index
        Duplicates = PickRandomConfigurationAtom.iloc[DuplicatedIndices].drop_duplicates(keep=False).reset_index(drop=True)

        if Duplicates.empty:
            break # Break while loop
        else:
            PickRandomConfigurationAtom = PickRandomConfigurationAtom.drop(DuplicatedIndices).reset_index(drop=True)
            if not PickRandomConfigurationAtom.empty:
                Duplicates.to_sql(f"DuplicateBATCH_{LoopIndex}", conn, index=False,if_exists='append')
                if ntry > 0:
                    print(f'\tTry-{ntry}: picked bad samples = {len(PickRandomConfiguration)}, duplication = {len(Duplicates)}')
                break # Break while loop
            else:
                if ntry == 0:
                    print(f'No new samples. Picked bad samples = {len(PickRandomConfiguration)}, duplication = {len(Duplicates)}.')
                    print('Trying sample again.')                                  
                else:
                    print(f'\tTry-{ntry}: picked bad samples = {len(PickRandomConfiguration)}, duplication = {len(Duplicates)}')
                
                if ntry == ntry_max: 
                    TrainingReachedAccuracy = True
                    print('Maximum # of try to find bad samples is reached. Terminating try.')
                    break # Break while loop
                
        ntry += 1
    TestPickRandomConfiguration = TestPickRandomConfiguration_.drop(DuplicatedIndices).reset_index(drop=True)
    return TrainingReachedAccuracy, \
        (pd.concat([PickRandomConfigurationAtom_svr,PickRandomConfigurationAtom]).reset_index(drop=True),\
         PickRandomConfigurationAtom_svr,PickRandomConfigurationAtom__nature),\
        pd.concat([TestPickRandomConfiguration_svr,TestPickRandomConfiguration]).reset_index(drop=True)
#%%----------------------------------------------------------------------------
def BPD_header(mytxt):
    """
    This function is printing out the supplied text in terminal as a big text.

    Parameters
    ----------
    mytxt : String
        The text that you want to print in terminal.

    Returns
    -------
    None.

    """
    split_list = []; result_list = []
    for I in mytxt:
        split_list.append(HeaderTxt.standard_dic[I].split("\n"))
    
    for i in range(len(split_list[0])):
        temp=""
        for j, item in enumerate(split_list):
            if j>0 and (i==1 or i==len(split_list[0])-1):
                temp += ""
            temp += item[i]
        result_list.append(temp)       
    
    result = ('\n').join(result_list)
    print(result)
    return

def star(func):
    """
    Decorator for Header function, HeaderDecorator(). 
    """
    def inner(*args, **kwargs):
        print('\n'+"*" * 151)
        func(*args, **kwargs)
        print("*" * 151+'\n')
    return inner

def percent(func):
    '''
    Decorator for Header function, HeaderDecorator().
    '''
    def inner(*args, **kwargs):
        print("%" * 151)
        func(*args, **kwargs)
        print("%" * 151)
    return inner

@star
@percent
def HeaderDecorator():
    """
    The header function to print the decorated text on top of BPDmpi/serial.py output.
    """
    now = datetime.now()
    BPD_header('BANDGAP     PHASE    DIAGRAM')
    print(f"{now.strftime('%Y-%m-%d  %H:%M:%S'):^151}\n")
    print(f"{'Welcome to Active Learning':^151}")
    print(f"{'The code was written by Badal Mondal as a part of the PhD project.':^151}")
    print(f"{'More details can be found here:':^151}")
    return 

def ParserOptions():
    """
    This finction defines the parsers.

    Returns
    -------
    Parser arguments

    """
    # HeaderDecorator()
    # sys.stdout.flush()
    parser = argparse.ArgumentParser(prog='ActiveLearning.py', description='This script creates the database for active learning bandgap phase diagram.', epilog='Have fun!')
    parser.add_argument('-d', metavar='DIRECTORYNAME', default=".", help='AL database path (default: current directory). e.g. /home/mondal/VASP/test/')
    parser.add_argument('-l', type=int, default=100, help='Maximum number of active learning loop (default: 100).')
    parser.add_argument('-m', type=int, default=5, help='Total number of independent models (default=5).')
    parser.add_argument('-Static', action='store_true', default=False, help='If Number of independent models static? Else best models will be picked dynamically based on scoring function criteria.(default: False).')
    parser.add_argument('-N', type=int, default=10000, help='Maximum number of samples allowed in AL training (default: 10000).')
    parser.add_argument('-PredictProb', action='store_true', default=False, help='If True => nature_accuracy_cutoff is based on predict probablity definition. (default: False).')
    parser.add_argument('-t', type=int, default=10, help='Maximum number of try to find bad samples (default: 10). Loop convergence condition 6. Model saturates in performence but can not reach to the required cut-off accuracies.')
    parser.add_argument('-k', type=int, default=5, help='Compare last k AL loops to check saturation in model performance (default=5).')
    parser.add_argument('-NAC', type=float, default=0.95, help='Loop convergence condition 5.1 (default: 0.01). The condition is: mean_sample(mean_model(1(y_predict=mode_model(y_predict)))) < NAC')
    parser.add_argument('-s', type=float, default=0.01, help='Loop convergence condition 5.2 (default: 0.01). The condition is: mean_samples(STD_model(y_prediction)) < s')
    parser.add_argument('-S', type=float, default=0.95, help='Loop convergence condition 3.1 (default: 0.01). The condition is: mean_model(accuracy_score_out_sample) < AS')
    parser.add_argument('-S', type=float, default=0.01, help='Loop convergence condition 3.2 (default: 0.01). The condition is: mean_model(mean_absolute_error_out_sample) < S')
    parser.add_argument('-e', type=float, default=0.001, help='Loop convergence condition 4. Model saturates in MAE if abs(model_accuracy_mean - model_accuracy_mean_last_k) < e')
    parser.add_argument('-P', type=float, default=10, help='Percentage of full prediction space to initialize the AL (default=10).')
    parser.add_argument('-p', type=float, default=10, help='Percentage of full prediction space to create the out-of-samples set (default=10).')
    parser.add_argument('-R', action='store_true', default=False, help='The prediction space is randomly sampled or not? = True or False (default: False).')
    parser.add_argument('-r', type=float, default=10, help='In case of random prediction space sampling, the percentage to choose over the full prediction space (default=10).')
    parser.add_argument('-TrainEgNature', action='store_true', default=False, help='Train both bandgap magnitude and nature? = True or False (default: False and SVRdependentSVC=False).')
    parser.add_argument('-SVRdependentSVC', action='store_true', default=False, help='Train bandgap magnitude but choose bad samples from SVR and SVRdependentSVC? = True or False. This is applicable only when TrainEgNature=True. (default: False).')
    parser.add_argument('-TrainNatureOnly', action='store_true', default=False, help='Train bandgap nature only? = True or False (default: False).')
    parser.add_argument('-FinalSVRDependentSVC', action='store_true', default=False, help='Train bandgap magnitude only? And use final hyperpapameters for SVR and SVC prediction = True or False (default: False).')
    parser.add_argument('-TrainEgOnly', action='store_false', default=True, help='Train bandgap magnitude only? = True or False (default: True).')
    parser.add_argument('-DrawPlots', action='store_True', default=False, help='Draw the figures? = True or False (default: False).')
    parser.add_argument('-SaveFigs', action='store_True', default=False, help='Save the figures? = True or False (default: False).')
    parser.add_argument('-ShowBatch', action='store_True', default=False, help='Create the movies for batch samples? = True or False (default: False).')
    parser.add_argument('-savemovie', action='store_True', default=False, help='Save the movies? = True or False (default: False).')
    # parser.add_argument('-NKP', type=int, default=1, help='Total number of KPOINTS. (default: 1)')
    # parser.add_argument('-SF', nargs='+', type=int, default=[6,6,6], help='Supercell dimensions (must be int) in a, b and c lattice vector directions respectively. (default: [6,6,6])')   
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    
    return parser.parse_args()

def SaveFinalModelBestParameters(AL_dbname,best_model_parameters,TrainEgOnly=True,\
                          TrainNatureOnly=False,TrainEgNature=False,SVRdependentSVC=False):
    conn = sq.connect(AL_dbname)
    if TrainEgOnly or TrainNatureOnly or (TrainEgNature and SVRdependentSVC):
        pd.DataFrame(best_model_parameters).to_sql("BestModelsParameters", conn, index=True,index_label='HyperParameters',if_exists='replace') 
    else:
        pd.DataFrame(best_model_parameters[0]).to_sql("BestModelsParameters", conn, index=True,index_label='HyperParameters',if_exists='replace')
        pd.DataFrame(best_model_parameters[1]).to_sql("NatureBestModelsParameters", conn, index=True,index_label='HyperParameters',if_exists='replace')
    conn.close()
    return

def SetExtraParameters(TrainEgNature=True,SVRdependentSVC=False,TrainNatureOnly=False,
                       FinalSVRDependentSVC=False,TrainEgOnly=False):
    yfeatures,PlotFEATURES,cbarlabel_text,cbarlimit_,EgNature = \
        None, None, None, (None,None), False
    if TrainEgOnly:
        ##### Training part        
        yfeatures='BANDGAP' # Labels in database for specific system
        # scoringfn = 'r2'  # max_error, neg_mean_absolute_error, neg_mean_squared_error, neg_root_mean_squared_error, r2, neg_mean_absolute_percentage_error,
        ##### Plotting part
        PlotFEATURES = ['PHOSPHORUS','STRAIN','BANDGAP']
        cbarlabel_text = 'E$_{\mathrm{g}}$ (eV)'
        cbarlimit_ = (0,2.5)
        EgNature = False
    elif TrainNatureOnly:
        ##### Training part
        yfeatures='NATURE' # Labels in database for specific system
        ##### Plotting part
        PlotFEATURES = ['PHOSPHORUS','STRAIN','NATURE']
        cbarlabel_text = None
        cbarlimit_ = (None, None)
        EgNature = True
    elif TrainEgNature:
        ##### Training part
        yfeatures=None
        ##### Plotting part
        PlotFEATURES = ['PHOSPHORUS','STRAIN','BANDGAP'] 
        cbarlabel_text = 'E$_{\mathrm{g}}$ (eV)'
        cbarlimit_ = (0, 2.5)
        EgNature = False
    return yfeatures,PlotFEATURES,cbarlabel_text,cbarlimit_,EgNature

def UpdateMeanModelAccuarcyList4ConvergenceSaturation(LoopIndex,Check_model_accuracy4last_k,
                                                      model_accuracy_mean,model_err_last_k,model_accuracy_last_k,
                                                      TrainEgOnly=True,TrainNatureOnly=False,TrainEgNature=False,SVRdependentSVC=False):
    PosIndex = (LoopIndex)%(Check_model_accuracy4last_k)
    if TrainEgOnly or (TrainEgNature and SVRdependentSVC):
        model_err_last_k[PosIndex] = model_accuracy_mean
    elif TrainNatureOnly:
        model_accuracy_last_k[PosIndex] = model_accuracy_mean
    else:
        model_err_last_k[PosIndex] = model_accuracy_mean[0]
        model_accuracy_last_k[PosIndex] = model_accuracy_mean[1]
    return model_err_last_k, model_accuracy_last_k



