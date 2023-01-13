#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 18:00:00 2022

@author: bmondal
"""

import numpy as np
import glob
import sys
sys.path.append("/home/bmondal/MachineLerning/BandGapML/scripts/DBscripts")
sys.path.append("/home/bmondal/MachineLerning/BandGapML/scripts/Modelscripts")
import SQLdatabaseFunctions as dbf
import sqlite3 as sq
import math
from datetime import datetime
import MLmodelSVMFunctions as mlmf
import pandas as pd
import pickle
import matplotlib.pyplot as plt

#%%
allfile1 = glob.glob(
    "/home/bmondal/MachineLerning/BandGapML/GaPAsSb/ConvergedS0POSCAR/SESSION2/SET*/SUBSET*/P*/S0/conf01/POSCAR")
allfile2 = glob.glob(
    "/home/bmondal/MachineLerning/BandGapML/GaPAsSb/ConvergedS0POSCAR/SESSION1/SET*/P*/S0/conf01/POSCAR")
allfile3 = glob.glob(
    "/home/bmondal/MachineLerning/BandGapML/InPAsSb/ConvergedS0POSCAR/SESSION*/SET*/SUBSET*/P*/S0/conf01/POSCAR")

allfile = allfile1 + allfile2 + allfile3

dbname = '/home/bmondal/MachineLerning/BandGapML/MODELS/InGaPAsSb/Predict_preopt_lp.db'
conn = sq.connect(dbname)
Tabletext = '''CREATE TABLE IF NOT EXISTS LatticeParameterDATA
                (GALLIUM  REAL default 0,
                 INDIUM  REAL default 0,
                 PHOSPHORUS   REAL default 0,
                 ARSENIC  REAL default 0,
                 ANTIMONY  REAL default 0,
                 LATTICEPARAMETER1  REAL,
                 LATTICEPARAMETER2  REAL)'''

with conn:
    conn.execute(Tabletext)
conn.close()

item_list = []
sdim = np.array([6,6,6])
ref_lp = {'GaAs': 5.689, 'GaP': 5.475, 'GaSb': 6.134, 'InP': 5.939, 'InAs': 6.138, 'InSb': 6.556}

for I in allfile:
    tmp_dict = {'Ga':0, 'In': 0, 'P': 0, 'As': 0, 'Sb': 0}
    lattice_parameters = dbf.getLP(I,sdim,ClusterType='zincblende') 
    assert math.isclose(lattice_parameters[0], lattice_parameters[1], rel_tol=1e-04), "lps don't match"
    assert math.isclose(lattice_parameters[1], lattice_parameters[2], rel_tol=1e-04), "lps don't match"
    
    atoms = np.genfromtxt(I, max_rows=2, skip_header=5,dtype=str)
    ATOMs = atoms[0]
    Con_atoms = np.array(atoms[1], dtype=float) / 216 * 100
    
    for i, j in enumerate(ATOMs):
        tmp_dict[j] = Con_atoms[i]
        
    lp_vegards = tmp_dict['Ga'] * (tmp_dict['P'] * ref_lp['GaP'] + tmp_dict['As'] * ref_lp['GaAs'] + tmp_dict['Sb'] * ref_lp['GaSb']) + \
                 tmp_dict['In'] * (tmp_dict['P'] * ref_lp['InP'] + tmp_dict['As'] * ref_lp['InAs'] + tmp_dict['Sb'] * ref_lp['InSb'])
                 
    lp_vegards /= 1E4    
        
    item_list.append(tuple(list(tmp_dict.values())+[lattice_parameters[0]]+[lp_vegards]))
        

conn = sq.connect(dbname)
with conn:
    conn.executemany('''INSERT INTO LatticeParameterDATA(GALLIUM, INDIUM, PHOSPHORUS, ARSENIC, ANTIMONY,LATTICEPARAMETER1,LATTICEPARAMETER2) VALUES (?, ?, ?, ?, ?, ?, ?)''', item_list )

conn.close()
#%%
conn = sq.connect(dbname)
df = pd.read_sql_query('SELECT * FROM LatticeParameterDATA', conn)
df.plot.scatter('LATTICEPARAMETER1','LATTICEPARAMETER2')
df_diff = df['LATTICEPARAMETER1'].sub(df['LATTICEPARAMETER2'], axis = 0)

#%%++++++++++++++++++++++++ SVR model +++++++++++++++++++++++++++++++++++++++++

#%%%-------------- Model paths for Binary trained models -----------------------
modelPATHS = '/home/bmondal/MachineLerning/BandGapML/MODELS/InGaPAsSb/'
svr_lp = modelPATHS + 'svrmodel_lp'
#%%%***************** Training model type parameters***************************
yfeatures='LATTICEPARAMETER1'
scoringfn='r2' #('r2', 'neg_mean_squared_error')
refit = True # 'r2'


xfeatures = ['GALLIUM', 'INDIUM', 'PHOSPHORUS', 'ARSENIC', 'ANTIMONY']

print(f"Training date: {datetime.now()}\n")
_ = mlmf.SVMModelTrainingFunctions(df, xfeatures, yfeatures=yfeatures, 
                                retrainlp=True,
                                svr_lp=svr_lp, save=0,
                                scoringfn=scoringfn, refit=refit,
                                PlotResults=1,LearningCurve=0)

#%%%********** Load model and predictions *************************************

dbname = '/home/bmondal/MachineLerning/BandGapML/DATAbase/RandomConfigInGaPAsSb.db'
conn = sq.connect(dbname)
dff = pd.read_sql_query('SELECT * FROM ATOMNUMBER', conn)
dff_unique = dff.drop_duplicates(subset=['INDIUM','GALLIUM','PHOSPHORUS', 'ARSENIC', 'ANTIMONY'])
lattice_loaded_model = pickle.load(open(svr_lp+'.sav', 'rb'))
LatticeParameterData_ = lattice_loaded_model.predict(dff_unique)

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
Y_bandgap = mlmf.TestBinaryModelTernaryPoints(TestPoints_X, TestPoints_Y_Eg, SVMBandgapModel)
mlpf.plot_true_predict_results(TestPoints_Y_Eg, Y_bandgap, 'Bandgap')

print("* Model testing for bandgap nature prediction (SVC):")
TestPoints_Y_EgNature = df_test['NATURE']
SVCEgNatureModel = pickle.load(open(tmpsvc_EgNature+'.sav', 'rb')) 
Y_BandgapNature = mlmf.TestBinaryModelTernaryPoints(TestPoints_X, TestPoints_Y_EgNature,
                                                    SVCEgNatureModel, SVCclassification=True)
mlpf.plot_true_predict_results(TestPoints_Y_EgNature, Y_BandgapNature, 'Bandgap nature')
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix
cm = confusion_matrix(TestPoints_Y_EgNature, Y_BandgapNature)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
#%%%-------------------- Predictions ------------------------------------------
#%%%% ------------------ Create predict points --------------------------------
predict_points =  mlgf.CreateDataForPredictionLoopV2(strain=[-5,5], 
                                                     resolution=[101,101],
                                                     compositionscale=100,
                                                     columns=xfeatures)

#%%%%-------------- Lattice parameter model -----------------------------------
lattice_loaded_model = pickle.load(open(svr_lp+'.sav', 'rb'))
EqulibriumDataModel = predict_points[predict_points['STRAIN']==0].reset_index(drop=True)
LatticeParameterData = lattice_loaded_model.predict(EqulibriumDataModel)
substraindict = {}
for I in range(len(subname)):
        substraindict[subname[I]]=(sublattice[I] - LatticeParameterData)/\
            LatticeParameterData * 100
EqulibriumDataModel = EqulibriumDataModel.join(pd.DataFrame(substraindict))