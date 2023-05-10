#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 11:35:55 2023

@author: bmondal
"""

import numpy as np
import sqlite3 as sq
import pandas as pd

dirpath = '/home/bmondal/MachineLerning/BandGapML_project/GaPAsSb/'
dbname = dirpath+'/DATAbase/Total_BPD_MLDataBase_GaPAsSb.db'
## load data
conn = sq.connect(dbname)
df = pd.read_sql_query('SELECT * FROM COMPUTATIONALDATA', conn)
df = df.dropna()
df['NATURE'] = df['NATURE'].map({1: 'direct', 2: 'direct', 3: 'indirect', 4: 'indirect'})
conn.close()

## Round
df[['PHOSPHORUS','ARSENIC','ANTIMONY','BANDGAP']] = df[['PHOSPHORUS','ARSENIC','ANTIMONY','BANDGAP']].round(3)
df['STRAIN'] = df['STRAIN'].round(2)

## Randomize
df = df.sample(frac=1).reset_index(drop=True)[['PHOSPHORUS','ARSENIC','ANTIMONY','STRAIN','BANDGAP','STRAIN','BANDGAP','NATURE']]

## Create column tag desciptions
dict1 = pd.DataFrame({"PHOSPHORUS": 'Phosphorus(%) in GaPAsSb sample', "ARSENIC":'Arsenic(%) in GaPAsSb sample','ANTIMONY': 'Antimony(%) in GaPAsSb sample',
         'STRAIN': 'Biaxial strain(%)', 'BANDGAP': 'Bandgap values(eV)', 'NATURE': 'Bandgap nature'}, index=['Description'])

## Save as excel sheet
with pd.ExcelWriter(f"{dirpath}/GaPAsSb_ML_database.xlsx") as writer:
    df.to_excel(writer,sheet_name='DFT_COMPUTATIONAL_DATA',index=False)
    dict1.T.to_excel(writer,sheet_name='Column_name_descriptions',index_label='Column_tag')