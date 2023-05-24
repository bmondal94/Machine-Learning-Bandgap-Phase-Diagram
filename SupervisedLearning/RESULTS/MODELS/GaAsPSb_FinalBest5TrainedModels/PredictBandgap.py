#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 21:32:52 2023
This script predict the bandgap of GaAsPSb with the trained machine learning model.
@author: bmondal
"""
import numpy as np
import pickle
import pandas as pd
import glob, os, sys

#%%
df = input("Write the composition and strain values in this order: PHOSPHORUS, ARSENIC,  STRAIN. \n(e.g. 50,45,-5 for 5% compressive biaxially strained Ga100P50As45Sb5)\n")
# df='50,50,1'
dff = np.array(df.split(','), dtype=float)
if len(dff)<3:
    print("Error: Must supply 3 comma separated values.")
    sys.exit()
if sum(dff[:2])>100:
    print('Error: Total composition can not greater than 100.')
    sys.exit()
dff = np.insert(dff,1,100-sum(dff[:2]))
xfeatures = ['PHOSPHORUS', 'ANTIMONY', 'ARSENIC',  'STRAIN']
dfff = pd.DataFrame.from_dict({xfeatures[i]: [dff[i]] for i,_ in enumerate(dff)})
LoadSVMFinalBPD = os.getcwd()

BANDGAP_dataframe = pd.DataFrame()
BANDGAP_nature_dataframe = pd.DataFrame()

F_MODEL_LIST = glob.glob(f'{LoadSVMFinalBPD}/*/MODELS')
for imp, model_path_tmp in enumerate(F_MODEL_LIST):
    svr_bandgap_tmp = f'{model_path_tmp}/svrmodel_bandgap'
    bandgapmag_model_tmp = pickle.load(open(svr_bandgap_tmp+'.sav', 'rb')) 
    bandgap = bandgapmag_model_tmp.predict(dfff)
    BANDGAP_dataframe[f'bandgap_{imp}'] = 0 if bandgap[0]<0 else bandgap
    
for imp, model_path_tmp in enumerate(F_MODEL_LIST):
    svc_EgNature_tmp = f'{model_path_tmp}/svcmodel_EgNature_binary'
    BandgapNatureModel_tmp = pickle.load(open(svc_EgNature_tmp+'.sav', 'rb'))
    BANDGAP_nature_dataframe[f'EgN_{imp}'] = BandgapNatureModel_tmp.predict(dfff)

EG_val = BANDGAP_dataframe.mean(axis=1)
EG_val_std  = BANDGAP_dataframe.std(axis=1)
EgN = BANDGAP_nature_dataframe.mode(axis=1).iloc[:, 0].astype(int) # Note: if mode is 50:50 the tag is 0 (==indirect)
mapping_nature = {1: 'Direct', 0: 'Indirect'}
print(f"Compound = Ga100P{dff[0]:.2f}As{dff[2]:.2f}Sb{dff[1]:.2f}")
print(f"Strain = {dff[3]:.2f} %\nBandgap value = {EG_val[0]:.3f} +/- {EG_val_std[0]:.3f} eV\nBandgap nature = {mapping_nature[EgN[0]]}")

