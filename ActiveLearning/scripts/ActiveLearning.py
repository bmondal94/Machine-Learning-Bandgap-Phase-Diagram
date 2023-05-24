#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 10:19:21 2022

@author: bmondal
"""

#%%-------------------------- Import modules ----------------------------------
import numpy as np
import sqlite3 as sq
import pandas as pd
import os, shutil, sys
from contextlib import redirect_stdout
# import glob
# import pickle
# import seaborn as sns
# import matplotlib.pyplot as plt

# from datetime import datetime
# import multiprocessing
# import matplotlib.cm as cm

from sklearn.metrics import r2_score, mean_squared_error, classification_report, \
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error,max_error
import ActiveLearningSVMfunctions as alsf
import ActiveLearningGeneralFunctions as algf
import ActiveLearningPlotFunctions as alpf
import MLmodelGeneralFunctions as mlgf

# import MLmodelNNFunctions as mlnnf
# import MLmodelPlottingFunctions as mlpf
# import MLmodelWebPlottingFunctions as mlwpf
# import MLmodelWebPlottingFunctionsPart2 as mlwpfP2
# import MLmodelSVMFunctions as mlmf

#%%--------------------------- main program -----------------------------------
algf.HeaderDecorator()
if __name__ == '__main__':
    #%%% ---------- Set default AL model settings -----------------------------
    '''
    ActiveLearningDirectory: string
        The directory where AL database, figures and VASP folders will be created.
    MaxLoopCount: int
        Maximum number of AL loop. Loop convergence condition 1.
    m_models: int
        Number of independent models. 
    Static_m_Models: bool
        If Number of independent models static? Else best models will be picked dynamically
        based on scoring function criteria.
    TotalMaxSampleAllowed: int
        Maximum total number of samples allowed in AL for training. Loop convergence condition 2.
    BadSamplesPredictProb: bool
        If True => nature_accuracy_cutoff := mean_sample(max(mean_model(y_predict_direct_prob), mean_model(y_predict_indirect_prob)) each sample)
    nature_accuracy_cutoff: float [:= mean_sample(mean_model(1(y_predict=mode_model(y_predict))))]
        The accuracy score cutoff for bandgap nature prediction. Loop convergence condition 5.1.
    std_cutoff: float [Unit is eV := mean_sample(STD_model(y_prediction))]
        Loop convergence condition 5.2. The condition is: mean_samples(STD_model(y_prediction)) < std_cutoff
    model_accuracy_cutoff: float [Unit is eV  := mean_model(accuracy_score_out_sample) ]
        Loop convergence condition 3.1. The condition is: mean_model(accuracy_score_out_sample) < model_accuracy_cutoff
    model_error_cutoff: float [Unit is eV  := mean_model(mean_absolute_error_out_sample) ]
        Loop convergence condition 3.2. The condition is: mean_model(mean_absolute_error_out_sample) < model_error_cutoff
    ntry_max: int    
        Maximum number of try to find bad samples.
        Loop convergence condition 6. Model saturates in performence but can not reach to the required cut-off accuracies.
    Check_model_accuracy4last_k: int
        Compare last k AL loops to check saturation in model performance
    model_saturates_epsilon: float
        Loop convergence condition 4. Model saturates in MAE if abs(model_accuracy_mean - model_accuracy_mean_last_k) < model_saturates_epsilon 
    N_InitialPoints_Percent: float  
        Percentage of full prediction space to initialize the AL.
    n_Points_test_percent: float
        Percentage of full prediction space to create the out-of-samples set.
    randdom_sampling_prediction: bool
        The prediction space is randomly sampled or not?
    random_sample_prediction_percent: float
        In case of random prediction space sampling, the percentage to choose over the full prediction space.
    FracBadSamplesFeedBack: float
        The fraction of bad total bad samples will be added to the feed back batch.
        Take only small fraction of total wrong samples. Don't need the all wrong samples to learn from.
        If no. of samples from FracBadSamplesFeedBack is really high, pick maximum of 100 samples only.
    '''
    print('Intializing parameters for active learning ...')
    ##### ======================== Parsing ====================================
    # args = algf.ParserOptions()
    # ActiveLearningDirectory = args.d 
    # MaxLoopCount = args.l ; m_models = args.m ; Static_m_Models = args.Static
    # TotalMaxSampleAllowed = args.N ; BadSamplesPredictProb = args.PredictProb 
    # nature_accuracy_cutoff = args.NAC ; model_accuracy_cutoff = args.AS ;
    # std_cutoff = args.s ; model_error_cutoff = args.S ; model_saturates_epsilon = args.e
    # ntry_max = args.t ; Check_model_accuracy4last_k = args.k ; randdom_sampling_prediction = args.R
    # InitialPoints_Percent = args.P ; Test_percent = args.p ; random_sample_prediction_percent = args.r
    #####======================================================================
    ActiveLearningDirectory = f'{os.path.expanduser("~")}/MachineLerning/BandGapML_project/GaAsPSb/ActiveLearning'
    MaxLoopCount = 1000
    m_models = 10 
    Static_m_Models = True # Default is dynamic_m_models
    TotalMaxSampleAllowed = 10000 
    BadSamplesPredictProb = False # If True => nature_accuracy_cutoff := mean_sample(max(mean_model(y_predict_direct_prob), mean_model(y_predict_indirect_prob)) each sample)
    nature_accuracy_cutoff = 0.95 # := mean_sample(mean_model(1(y_predict=mode_model(y_predict)))) # pretending y_model is the prediction and y_mode_model is the true value 
    std_cutoff = 0.001 # eV  := mean_sample(STD_model(y_prediction))
    model_accuracy_cutoff = 0.95 # := mean_model(accuracy_score_out_sample)
    model_error_cutoff = 0.001 # eV  := mean_model(RMSE_out_sample) 
    ntry_max = 10
    Check_model_accuracy4last_k = 4
    model_saturates_epsilon = 2E-6 # Delta RMSE (eV)
    InitialPoints_Percent = 1
    Test_percent = 25
    randdom_sampling_prediction = False
    random_sample_prediction_percent = 30 # Recommendation: if random_sample_prediction_percent = y% then increase FracBadSamplesFeedBack to y frac times
    FracBadSamplesFeedBack = 0.1
    #####======================================================================
    #### Priority: DrawPlots > SaveFigs and/or ShowBatch > savemovie
    DrawPlots=True # If to draw the plots
    SaveFigs=True  # If to save the plots
    ShowBatch = True # If to create the movies for batch samples
    savemovie = True # If to save the movie   
    DrawDITLines = True # Draw the DFT DIT lines
    LessThanQuaternarySystems = False # Iff. the system is less than quaternary the raw data can be plotted
    #####======================================================================
    ##### Priority order: TrainEgNature (and/or SVRdependentSVC) > TrainNatureOnly > FinalSVRDependentSVC > TrainEgOnly
    TrainEgNature = 1
    SVRdependentSVC = 1
    TrainNatureOnly = 0
    FinalSVRDependentSVC = 1
    TrainEgOnly = 1
    
    ModelDescription,Add2PathALdirectory,TrainEgNature,SVRdependentSVC,TrainNatureOnly,FinalSVRDependentSVC,TrainEgOnly=\
        algf.AssertNoContradiction(TrainEgNature=TrainEgNature,SVRdependentSVC=SVRdependentSVC,TrainNatureOnly=TrainNatureOnly,
                                   FinalSVRDependentSVC=FinalSVRDependentSVC,TrainEgOnly=TrainEgOnly,PredictProb=BadSamplesPredictProb,
                                   randdom_sampling_prediction=randdom_sampling_prediction,Static_m_Models=Static_m_Models)
    ActiveLearningDirectory = ActiveLearningDirectory + '/' + Add2PathALdirectory
    #####======================================================================
    XFEATURES = ['INDIUM','GALLIUM','PHOSPHORUS', 'ARSENIC', 'ANTIMONY', 'STRAIN'] # Features in generalized complete database
    if LessThanQuaternarySystems:
        xfeatures = ['PHOSPHORUS','ARSENIC','STRAIN'] # Features in database for specific system   
    else:
        xfeatures = ['PHOSPHORUS','ARSENIC','ANTIMONY','STRAIN'] 
        # xfeatures = ['INDIUM','GALLIUM','PHOSPHORUS', 'ARSENIC', 'ANTIMONY', 'STRAIN']
    #####======================================================================
    model_accuracy_last_k = [np.nan, 1] + [np.nan]*(Check_model_accuracy4last_k-2) 
    model_err_last_k = [np.nan, 1] + [np.nan]*(Check_model_accuracy4last_k-2)
    TrainingReachedAccuracy = False
    AL_dbname = ActiveLearningDirectory + '/' + 'ActiveLearning.db'
    FiguresPath = ActiveLearningDirectory + '/' + 'Figs'
    if randdom_sampling_prediction:
        n_points_random_sample_prediction = random_sample_prediction_percent/100 # For known test fraction
        # n_points_random_sample_prediction = int(TheoreticalPredictionPoints * random_sample_prediction_percent/100) # Random n points for prediction   
    if os.path.exists(AL_dbname):
        os.remove(AL_dbname)  

    if os.path.exists(FiguresPath):
        shutil.rmtree(FiguresPath)

    os.makedirs(FiguresPath+'/BANDGAP',exist_ok=True) 
    os.makedirs(FiguresPath+'/NATURE',exist_ok=True)
    os.makedirs(FiguresPath+'/VASP',exist_ok=True)
    OutPutTxtFile = ActiveLearningDirectory + '/output.txt'

    ##### ============= Parameters for training and plotting ==================
    yfeatures,PlotFEATURES,cbarlabel_text,cbarlimit_,EgNature = \
        algf.SetExtraParameters(TrainEgNature=TrainEgNature,SVRdependentSVC=SVRdependentSVC,TrainNatureOnly=TrainNatureOnly,\
                                FinalSVRDependentSVC=FinalSVRDependentSVC,TrainEgOnly=TrainEgOnly)
    xaxis_label = 'P(%)'
    yaxis_label = 'Strain(%)'
    with redirect_stdout(open(OutPutTxtFile, 'w')):
        algf.HeaderDecorator()
        print(f'{"="*78}\nModel description::\n {ModelDescription}\n{"="*78}\n')
    #%%%% ------------------------- Load database ----------------------------- 
    dirpath = f'{os.path.expanduser("~")}/MachineLerning/BandGapML_project/ActiveLearningDataBases'
    DIT_DFT_FILE = f'{dirpath}/GaAsP_DFT_DITs'
    dbname = f'{dirpath}/Total_BPD_MLDataBase_GaPAsSb.db'
    # dbname = f'{dirpath}/QuaBin_BPD_MLDataBase_InPAsSb.db'
        
    BinaryConversion=True # 0: Direct, 1: Indirect
    df_full = algf.CreateDataForModel(dbname,BinaryConversion=BinaryConversion)
        
    #%%%% ----------------------- Get particular system -----------------------
    StrainLimit = (-5, 5)
    CompLimit = (0, 100)
    ###### ====================== GaAsP system ================================
    # df_full.drop(index=df_full.index[df_full['ANTIMONY'] > 0], inplace=True)
    df = df_full.copy() #[(df_full['STRAIN'] > StrainLimit[0]) & (df_full['STRAIN'] < StrainLimit[1])]
    df = df.sample(frac=1).reset_index(drop=True)
    
    #%%%% --------------------- Plot the raw data -----------------------------
    if LessThanQuaternarySystems:
        alpf.PlotRawDataTernary(df, PlotFEATURES, title=None,xLimit=CompLimit, yLimit=StrainLimit,nature=EgNature,
                                xlabel=xaxis_label, ylabel=yaxis_label, cbarlabel=cbarlabel_text, vmin=cbarlimit_[0],vmax=cbarlimit_[1],
                                save=SaveFigs, savepath=FiguresPath,figname='/AllDFTset.png')    # title='All DFT samples'
    
    #%%%% ------------------------ Create prediction points -------------------
    #%%%% ================== Predict over full space ==========================
    # composition_ = np.linspace(CompLimit[0],CompLimit[1],101)
    # strain_ = np.linspace(StrainLimit[0],StrainLimit[-1],21)
    # x_predict = np.meshgrid(composition_,strain_)
    # X_predict = pd.DataFrame(np.array(x_predict).T.reshape(-1, 2),columns=CollectColumns[0:2])
    # y_prediction_model = pd.DataFrame()
    #%%%% --------------- Predict over preDFT points --------------------------
    y_prediction_modell = df.copy()
    TheoreticalPredictionPoints = len(y_prediction_modell)
    ##### ======= Chose random N_InitialPoints points to start the AL loop ====
    # TheoreticalPredictionPoints = 500000
    N_InitialPoints =  int(TheoreticalPredictionPoints * InitialPoints_Percent/100)
    ##### ====== Reserve completely independent test set :=out-of-samples =====
    n_Points_test = int(TheoreticalPredictionPoints * Test_percent/100)
    
    with redirect_stdout(open(OutPutTxtFile, 'a')):
        print(f'* Total # of DFT samples = {len(df)}')
        print(f'* # of samples reserved for out-of-sample testing during AL = {n_Points_test}\n')
        
    PreservedTestSamples = df.sample(n=n_Points_test)
    DropIndecies = PreservedTestSamples.index
    PreservedTestSamples= PreservedTestSamples.reset_index(drop=True)
    df_reduced = df.copy().drop(DropIndecies).reset_index(drop=True)
    #%%%% --------------------- Plot the raw data part-2 ----------------------
    if LessThanQuaternarySystems:
        ##### ================= Plot reserved raw data ============================
        alpf.PlotRawDataTernary(PreservedTestSamples, PlotFEATURES, title=None,xLimit=CompLimit, yLimit=StrainLimit,nature=EgNature,
                                xlabel=xaxis_label, ylabel=yaxis_label, cbarlabel=cbarlabel_text, vmin=cbarlimit_[0],vmax=cbarlimit_[1],
                                save=SaveFigs, savepath=FiguresPath,figname='/TestSet.png') # title='Reserved test sample set'
        ##### ================== Plot reduced raw data ============================
        alpf.PlotRawDataTernary(df_reduced, PlotFEATURES, title=None,xLimit=CompLimit, yLimit=StrainLimit, nature=EgNature,
                                 xlabel=xaxis_label, ylabel=yaxis_label, cbarlabel=cbarlabel_text, vmin=cbarlimit_[0],vmax=cbarlimit_[1],
                                 save=SaveFigs, savepath=FiguresPath,figname='/ReducedDFTset.png') # title='Reduced DFT sample set'
            
    #%%%% ---------------------- Active learning loop -------------------------
    '''
    Active learning::
    # Loop convergence condition 1: maximum allowed learning loop is MaxLoopCount
    # Loop convergence condition 2: maximum total number of sample after each feed-back loop allowed is TotalMaxSampleAllowed
    # Loop convergence condition 3.1: mean_model(accuracy_score) > model_accuracy_cutoff
    # Loop convergence condition 3.2: mean_model(REMSE_out_sample) < model_error_cutoff
    # Loop convergence condition 4: Model saturates in MAE if abs(model_accuracy_mean - model_accuracy_mean_last_k) < model_saturates_epsilon
    # Loop convergence condition 5.1: mean_sample(mean_model(1(y_prediction=mode_model(y_prediction)))) > nature_accuracy_cutoff
    # Loop convergence condition 5.2: mean_sample(STD_model(y_prediction)) < std_cutoff  
    # Loop convergence condition 6: model saturates in performence but can not reach to the required cut-off accuracies.
    Bad samples::
    # Bandgap nature bad samples: accuracy:=mean_model(1(y_prediction=mode_model(y_prediction))); Samples with accuracy < mean_sample(accuracy)
    # Bandgap magnitude bad samples: Samples with STD_model(y_prediction) > mean_sample(STD_model(y_prediction))
    '''
    print('Actively learning ...')
    with redirect_stdout(open(OutPutTxtFile, 'a')):
    # for _ in [1]:
        print(f'{"="*78}\nActive learning::\n')
        ##### =====================================================================
        if not randdom_sampling_prediction:
            PredictOvernSamples = df_reduced.copy()
            X_predict = PredictOvernSamples[xfeatures].copy()
        ##### =====================================================================
        for LoopIndex in range(MaxLoopCount): # Loop convergence condition 1 
            conn = sq.connect(AL_dbname)       
            if LoopIndex > 0:
                ML_df = pd.read_sql_query('SELECT * FROM ALLDATA', conn)
                if len(ML_df) > TotalMaxSampleAllowed: # Loop convergence condition 2
                    TrainingReachedAccuracy = True 
                    print(f"\n** Training set: {len(ML_df)} samples, Test set: {n_Points_test} samples")
                    print(f'** Model is reached to total allowed training samples limit (={TotalMaxSampleAllowed}). Too many samples are needed for model. Not good.')
                    print('** Terminating AL loop. Training incomplete. The results may be wrong.')
                else:
                    if randdom_sampling_prediction:
                        PredictOvernSamples = df_reduced.sample(frac=n_points_random_sample_prediction).reset_index(drop=True)
                        X_predict = PredictOvernSamples[xfeatures].copy()
                    ##### ==================== ML training ========================
                    print(f"\nActive learning loop: {LoopIndex}, Training set: {len(ML_df)} samples, Test set: {n_Points_test} samples")
                    
                    model_accuracy, model_accuracy_mean, best_model_parameters,TrainingReachedAccuracy,\
                        PickRandomConfigurationAtom, TestPickRandomConfiguration = \
                           alsf.ALmodels(model_accuracy_last_k, model_err_last_k,
                                         conn, ML_df, PreservedTestSamples, PredictOvernSamples, X_predict,
                                         model_accuracy_cutoff,model_error_cutoff, 
                                         model_saturates_epsilon, nature_accuracy_cutoff, std_cutoff,
                                         Check_model_accuracy4last_k,ntry_max,
                                         XFEATURES,xfeatures,yfeatures,LoopIndex,TakeFracBadSamples=FracBadSamplesFeedBack,
                                         random_state=None, m_models=m_models,Static_m_Models=Static_m_Models,
                                         SVRModel=TrainEgOnly,SVCmodel=TrainNatureOnly,PredictProb=BadSamplesPredictProb,
                                         SVRSVCmodel=TrainEgNature,SVRdependentSVC=SVRdependentSVC,
                                         SVR_scoringfn=['neg_root_mean_squared_error','r2'])
                    ##### ======== Save model accuracy data in database ===========
                    # if not TrainingReachedAccuracy:
                    algf.DumpModelAccuracyData(LoopIndex,len(ML_df),model_accuracy,conn,TrainEgOnly=TrainEgOnly,\
                                               TrainNatureOnly=TrainNatureOnly,TrainEgNature=TrainEgNature,SVRdependentSVC=SVRdependentSVC)
                    ##### ======== Dump model accuracy data in list ===============
                    ##### ======== for convergence saturation check later ========= 
                    model_err_last_k, model_accuracy_last_k = \
                        algf.UpdateMeanModelAccuarcyList4ConvergenceSaturation(LoopIndex,Check_model_accuracy4last_k,
                                                                               model_accuracy_mean,model_err_last_k,model_accuracy_last_k,
                                                                               TrainEgOnly=TrainEgOnly,TrainNatureOnly=TrainNatureOnly,
                                                                               TrainEgNature=TrainEgNature,SVRdependentSVC=SVRdependentSVC)        
            else:
                # PickRandomConfiguration = algf.CreateRandomData_AB_CDE(strain=StrainLimit,
                #                           columns=XFEATURES, compositionscale=CompLimit[1], npoints=N_InitialPoints,
                #                           compositionstart=CompLimit[0], Ternary=True)
                print(f'Initialization loop: {LoopIndex}') # Let's create SQS structures of N systems
                TestPickRandomConfiguration = df_reduced.sample(n=N_InitialPoints) #,random_state=20)
                PickRandomConfiguration = TestPickRandomConfiguration[XFEATURES]
                PickRandomConfiguration2Atom = algf.convert_conc2atomnumber(PickRandomConfiguration,natoms=216)
                ##### ========== Convert concentration to atom numbers ============
                PickRandomConfigurationAtom = (PickRandomConfiguration2Atom, PickRandomConfiguration2Atom, PickRandomConfiguration2Atom)
    
            ##### ============= Break AL loop if training is finished =============
            if TrainingReachedAccuracy:
                print(f'{"="*78}')
                conn.close()
                break  # Break for loop
            else:
                ##### =============== Save to database ============================
                if not PickRandomConfigurationAtom[0].empty:
                    PickRandomConfigurationAtom[0].to_sql(f"BATCH_{LoopIndex}", conn, index=False,if_exists='fail')
                    PickRandomConfigurationAtom[0].to_sql("TotalBatchs", conn, index=False,if_exists='append')
                    TestPickRandomConfiguration.to_sql("ALLDATA", conn, index=False,if_exists='append')
                if TrainEgNature and not PickRandomConfigurationAtom[1].empty:
                        PickRandomConfigurationAtom[1].to_sql(f"MAGNITUDEBATCH_{LoopIndex}", conn, index=False,if_exists='fail')
                if TrainEgNature and not PickRandomConfigurationAtom[2].empty:
                        PickRandomConfigurationAtom[2].to_sql(f"NATUREBATCH_{LoopIndex}", conn, index=False,if_exists='fail')
                        
                conn.close()
                ##### ============== Create folders for DFT =======================
                # algf.CreateVaspFolders(ActiveLearningDirectory + '/VASP',PickRandomConfigurationAtom,216)
                
            if LoopIndex == MaxLoopCount :
                print('\n** Model is reached to maximum AL loop count limit. The learning is prematuarly terminated. The results may be wrong. Please check.')
                print(f'{"="*78}\n')
    
        ##### ========= Collect the hyper-parameters for last best m models =======
        algf.SaveFinalModelBestParameters(AL_dbname,best_model_parameters,
                                          TrainEgOnly=TrainEgOnly,TrainNatureOnly=TrainNatureOnly,
                                          TrainEgNature=TrainEgNature,SVRdependentSVC=SVRdependentSVC)
    # print('Active learning complete.')
    #%% -------------------- Plotting --------------------------------------------
    if DrawPlots:
        #%%% ----------------------------------------------------------------------
        print('Plotting ...')
        with redirect_stdout(open(OutPutTxtFile, 'a')):
            ##### ======= Plot the Bandgap magnitudes AL over whole space =========
            if TrainEgOnly or FinalSVRDependentSVC:
                print(f'{"-"*78}\n{"="*78}\nFinal results over whole space::')
                X_predict_, _ = alsf.GenerateBandgapFigsData(AL_dbname,df,xfeatures,'BANDGAP',FixPredictionValues='my_rmse_fix')  
                if FinalSVRDependentSVC:
                    X_predict, labels = alsf.GenerateBandgapFigsData(AL_dbname,df,xfeatures,'NATURE')
                    print('\t- Bandgap nature predictions were based on final hyper-parameters from only SVR training.')   
                print(f"{'='*78}\n{'-'*78}\n{'='*78}\nPlotting figures:: bandgap magnitude")
                alpf.PlotALbandgapMagnitudeFeatures(X_predict_,AL_dbname,PlotFEATURES,xLimit=CompLimit,yLimit=StrainLimit,
                                                    xlabel=xaxis_label,ylabel=yaxis_label,cbarlimit_=cbarlimit_,
                                                    SaveFigs=SaveFigs,savepath=FiguresPath+'/BANDGAP',
                                                    LessThanQuaternarySystems=LessThanQuaternarySystems)
                if LessThanQuaternarySystems:
                    print("Creating movies for batch samples:: bandgap magnitude\n\t...")
                    alpf.CreateMovieBatchSamples(AL_dbname,PlotFEATURES[:2],natoms=216,
                                                 ShowBatch=ShowBatch,ShowbatchInteractiveMovie=True,
                                                 cumulative=True,NoColor=True,xlabel=xaxis_label,ylabel=yaxis_label,
                                                 xLimit=CompLimit, yLimit=StrainLimit,savemovie=savemovie,save_movie_path=FiguresPath+'/BANDGAP') 
                
                if FinalSVRDependentSVC:
                    alsf.PrintConfusionMatrix(X_predict,labels,save=SaveFigs,savepath=FiguresPath+'/NATURE')
                    alpf.PlotALbandgapNatureFeaturesP1(X_predict,PlotFEATURES,xLimit=CompLimit,yLimit=StrainLimit,
                                                       xlabel=xaxis_label,ylabel=yaxis_label,
                                                       SaveFigs=SaveFigs,savepath=FiguresPath+'/NATURE',
                                                       LessThanQuaternarySystems=LessThanQuaternarySystems)        
                print(f"\nDone\n{'='*78}\n")
                
            ##### =========== Plot the Bandgap nature AL over whole space =============
            if TrainNatureOnly:
                print(f'{"-"*78}\n{"="*78}\nFinal results over whole space::')
                X_predict, labels = alsf.GenerateBandgapFigsData(AL_dbname,df,xfeatures,'NATURE')   
                print(f"{'='*78}\n{'-'*78}\n{'='*78}\nPlotting figures:: bandgap nature")   
                alsf.PrintConfusionMatrix(X_predict,labels,save=SaveFigs,savepath=FiguresPath+'/NATURE')
                alpf.PlotALbandgapNatureFeatures(X_predict,AL_dbname,PlotFEATURES,xLimit=CompLimit,yLimit=StrainLimit,
                                                 xlabel=xaxis_label,ylabel=yaxis_label,
                                                 SaveFigs=SaveFigs,savepath=FiguresPath+'/NATURE',
                                                 PlotDFT_DIT=DrawDITLines,DIT_DFT_FILE=DIT_DFT_FILE,
                                                 LessThanQuaternarySystems=LessThanQuaternarySystems)
                if LessThanQuaternarySystems:
                    print("Creating movies for batch samples:: bandgap nature\n\t...")
                    alpf.CreateMovieBatchSamples(AL_dbname,PlotFEATURES[:2],natoms=216,
                                                 ShowBatch=ShowBatch,ShowbatchInteractiveMovie=True,
                                                 cumulative=True,NoColor=True,xlabel=xaxis_label,ylabel=yaxis_label,
                                                 xLimit=CompLimit, yLimit=StrainLimit,savemovie=savemovie,save_movie_path=FiguresPath+'/NATURE')
                print(f"\nDone\n{'='*78}\n")
            
            ##### ======= Plot the Bandgap mag and nature AL over whole space =========
            if TrainEgNature:   
                print(f'{"-"*78}\n{"="*78}\nFinal results over whole space::')
                if SVRdependentSVC:
                    X_predict_, _ = alsf.GenerateBandgapFigsData(AL_dbname,df,xfeatures,'BANDGAP',FixPredictionValues='my_rmse_fix')  
                    X_predict, labels = alsf.GenerateBandgapFigsData(AL_dbname,df,xfeatures,'NATURE')
                    print('\t- Bandgap nature predictions based on hyper-parameters from SVRdependentSVC training.')
                    print(f"{'='*78}\n{'-'*78}\n{'='*78}\nPlotting figures:: bandgap magnitude")
                    alpf.PlotALbandgapMagnitudeFeatures(X_predict_,AL_dbname,PlotFEATURES,xLimit=CompLimit,yLimit=StrainLimit,
                                                        xlabel=xaxis_label,ylabel=yaxis_label,cbarlimit_=cbarlimit_,TargetTablePattern='MAGNITUDEBATCH',
                                                        SaveFigs=SaveFigs,savepath=FiguresPath+'/BANDGAP',
                                                        LessThanQuaternarySystems=LessThanQuaternarySystems)
                    print("Plotting figures:: bandgap nature")
                    alsf.PrintConfusionMatrix(X_predict,labels,save=SaveFigs,savepath=FiguresPath+'/NATURE')
                    alpf.PlotALbandgapNatureFeaturesP1(X_predict,PlotFEATURES,xLimit=CompLimit,yLimit=StrainLimit,
                                                       xlabel=xaxis_label,ylabel=yaxis_label,
                                                       SaveFigs=SaveFigs,savepath=FiguresPath+'/NATURE',
                                                       PlotDFT_DIT=DrawDITLines,DIT_DFT_FILE=DIT_DFT_FILE,
                                                       LessThanQuaternarySystems=LessThanQuaternarySystems)
                    alpf.PlotBatchSampleCount(AL_dbname,TargetTablePattern='NATUREBATCH',save=SaveFigs,savepath=FiguresPath+'/NATURE')
                    print("Plotting figures:: general\n\t...")
                    alpf.PlotBatchSampleCount(AL_dbname,TargetTablePattern='BATCH',save=SaveFigs,savepath=FiguresPath)
                    if LessThanQuaternarySystems:
                        alpf.PlotBatchDataTernary(AL_dbname,PlotFEATURES,xLimit=CompLimit,yLimit=StrainLimit,
                                                  xaxis_label=xaxis_label,yaxis_label=yaxis_label,
                                                  save=SaveFigs,savepath=FiguresPath)
                else:
                    X_predict, labels = alsf.GenerateBandgapBothFigsData(AL_dbname,df,xfeatures,FixPredictionValues='my_rmse_fix')
                    print(f"{'='*78}\n{'-'*78}\n{'='*78}\nPlotting figures::")
                    alsf.PrintConfusionMatrix(X_predict,labels,save=SaveFigs,savepath=FiguresPath+'/NATURE')
                    alpf.PlotALbandgapMagNatureFeatures(X_predict,AL_dbname,PlotFEATURES,xLimit=CompLimit,yLimit=StrainLimit,
                                                        xlabel=xaxis_label,ylabel=yaxis_label,cbarlimit_=cbarlimit_,
                                                        SaveFigs=SaveFigs,savepath1=FiguresPath+'/BANDGAP',
                                                        savepath2=FiguresPath+'/NATURE',savepath3=FiguresPath,
                                                        PlotDFT_DIT=DrawDITLines,DIT_DFT_FILE=DIT_DFT_FILE,
                                                        LessThanQuaternarySystems=LessThanQuaternarySystems)
                if LessThanQuaternarySystems:
                    print("Creating movies for batch samples:: bandgap magnitude\n\t...")
                    alpf.CreateMovieBatchSamples(AL_dbname,PlotFEATURES[:2],TargetTablePattern='MAGNITUDEBATCH',natoms=216,
                                                 ShowBatch=ShowBatch,ShowbatchInteractiveMovie=True,
                                                 cumulative=True,NoColor=True,xlabel=xaxis_label,ylabel=yaxis_label,
                                                 xLimit=CompLimit, yLimit=StrainLimit,savemovie=savemovie,save_movie_path=FiguresPath+'/BANDGAP') 
                    print("Creating movies for batch samples:: bandgap nature\n\t...")
                    alpf.CreateMovieBatchSamples(AL_dbname,PlotFEATURES[:2],TargetTablePattern='NATUREBATCH',natoms=216,
                                                 ShowBatch=ShowBatch,ShowbatchInteractiveMovie=True,
                                                 cumulative=True,NoColor=True,xlabel=xaxis_label,ylabel=yaxis_label,
                                                 xLimit=CompLimit, yLimit=StrainLimit,savemovie=savemovie,save_movie_path=FiguresPath+'/NATURE') 
                    if ShowBatch:
                        outer_ani = alpf.ShowbatchInteractiveBoth(AL_dbname, PlotFEATURES[:2], natoms=216, 
                                                                  cumulative=1, NoColor=True,
                                                                  save_movie=savemovie, save_movie_path=FiguresPath+'/BatchSamples.mp4', 
                                                                  xLimit=CompLimit, yLimit=StrainLimit, xlabel=xaxis_label,ylabel=yaxis_label)
                print(f"\nDone\n{'='*78}\n")
    print(f"Output folder: {ActiveLearningDirectory}")
    print("All tasks complete.")
