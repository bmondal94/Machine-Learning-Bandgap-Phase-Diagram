#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 12:49:42 2022

@author: bmondal
"""

import sqlite3 as sq
from shutil import copyfile

#%%
DIRpath = '/home/bmondal/MachineLerning/BandGapML/DATAbase/'
dbname = DIRpath + 'BPD_ML_Total_DataBase.db'
dbnamelist = [DIRpath+'Binary_BPD_MLDataBase.db', 
              DIRpath+'Ternary_Random_BPD_MLDataBase.db',
              DIRpath+'Ternary_Random_BPD_MLDataBase_p2.db']
copyfile(dbnamelist[0], dbname)

for I in dbnamelist[1:]:  
    print(I)
    conn = sq.connect(dbname)
    conn.execute('ATTACH DATABASE "{}" AS db2;'.format(I))
    conn.execute('INSERT INTO main.COMPUTATIONALDATA SELECT * FROM db2.COMPUTATIONALDATA;')
    #conn.execute('DETACH DATABASE "{}";'.format(I))
    conn.commit()    

    conn.close()

