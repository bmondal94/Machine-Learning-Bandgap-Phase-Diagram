#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 13:26:49 2022

@author: bmondal
"""

import numpy as np      
import glob
import SQLdatabaseFunctions as sqlf
import sqlite3 as sq
from shutil import copyfile


dirname = '/home/bmondal/MyFolder/VASP/InSb/DATA/'
filepath1 = dirname + 'InSbBiaxialMLDatabase.dat'

data = np.genfromtxt(filepath1)

HOMOKP = data[:,11].astype(int)
LUMOKP = data[:,12].astype(int)

DATA = []
BWvb = [0]*8
BWvb[0] = 100
for i in range(len(data)):    
    BWcb = [0]*8
    if HOMOKP[i]==LUMOKP[i]:
        Nature = 1
        
        if LUMOKP[i]==1:
            BWcb[0] = 100
        else:
            BWcb[3] = 100
    else:
        Nature = 3
        BWcb[3] = 100
        
    dlist = tuple((*data[i,:11], *BWvb, *BWcb, data[i,-1], Nature))
    
    DATA.append(dlist)
    
BW_len = 8
dbname = dirname+'/BPD_MLDataBase.db'
conn = sqlf.OPENdatabase(dbname, BW_len)
_ = sqlf.DataInsertion(DATA, conn)


#%%
def CreateDataBaseFn(dbnamelist, dbname):
    print(dbnamelist[0])
    copyfile(dbnamelist[0], dbname)
    
    for I in dbnamelist[1:]:  
        print(I)
        conn = sq.connect(dbname)
        conn.execute('ATTACH DATABASE "{}" AS db2;'.format(I))
        conn.execute('INSERT INTO main.COMPUTATIONALDATA SELECT * FROM db2.COMPUTATIONALDATA;')
        # conn.execute('DETACH DATABASE "{}";'.format(I))
        conn.commit()    
    
        conn.close()
        
dirname = '/home/bmondal/MyFolder/VASP/'
DIRpath = '/home/bmondal/MachineLerning/BandGapML_project/InPAsSb/DATAbase/'
dbnameB = DIRpath + 'Binary_BPD_MLDataBase_InPAs.db'
# llist = ['GaAs','GaP','GaSb']
llist = ['InAs','InP']
dbnamelist = [f'{dirname}/{I}/DATA/BPD_MLDataBase.db' for I in llist] 
CreateDataBaseFn(dbnamelist, dbnameB)  

#%%
dbnamelist = ['/home/bmondal/MachineLerning/BandGapML_project/InPAsSb/DATAbase/Binary_BPD_MLDataBase_InPAs.db',
              '/home/bmondal/MachineLerning/BandGapML_project/InPAsSb/DATAbase/Quaternary_BPD_MLDataBase_InPAsSb.db'] 
CreateDataBaseFn(dbnamelist, '/home/bmondal/MachineLerning/BandGapML_project/InPAsSb/DATAbase/QuaBin_BPD_MLDataBase_InPAsSb.db')  


