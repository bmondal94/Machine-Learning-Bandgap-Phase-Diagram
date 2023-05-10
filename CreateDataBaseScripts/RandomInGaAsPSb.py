#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 10:36:01 2022

@author: bmondal
"""
from collections import defaultdict
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import MLmodelGeneralFunctions as mlgf
import MLmodelPlottingFunctions as mlpf
import sqlite3 as sq


xfeatures = ['INDIUM','GALLIUM','PHOSPHORUS', 'ARSENIC', 'ANTIMONY', 'STRAIN']

pp = mlgf.CreateRandomData_AB_CDE(strain=[-5, 5],
                                  columns=xfeatures, compositionscale=100,
                                  compositionstart=0,npoints=10000)
#%%
v = pd.DataFrame([(0,0),(0.5, np.sqrt(3)/2), (1, 0)], index=['PHOSPHORUS', 'ARSENIC', 'ANTIMONY'])
dp = pp[['PHOSPHORUS', 'ARSENIC', 'ANTIMONY']] @ v
dp.columns = ['PHOSPHORUS', 'ARSENIC']
dp['STRAIN'] = pp['STRAIN']
dp.plot.scatter(x='PHOSPHORUS',y='ARSENIC',c='STRAIN',colormap='viridis')
dp.hist('STRAIN', bins=20)

#%%
#predict_points = predict_points.drop(predict_points[predict_points.PHOSPHORUS == 0].index)

# ********************** Conversion concentration to atom numbers *************
'''
216: Total number of cationic/anaionic sites in supercell
Natom = 216/100*x for x% concentration
'''

N_atom = (pp[['INDIUM']]*2.16).round(0).astype(int)
N_atom['GALLIUM'] = 216 - N_atom['INDIUM']
N_atom[['PHOSPHORUS', 'ARSENIC']] = (pp[['PHOSPHORUS', 'ARSENIC']]*2.16).round(0).astype(int)
N_atom['ANTIMONY'] = 216-N_atom[['PHOSPHORUS', 'ARSENIC']].sum(axis=1)
N_atom['STRAIN'] = pp['STRAIN']

N_atom.plot.scatter(x='PHOSPHORUS',y='ARSENIC',c='STRAIN',colormap='viridis')
N_atom.hist('STRAIN', bins=20)
N_atom.hist('INDIUM', bins=216)
# # Drop the data if that composition already exists in previously created data base.
# # Comment out next 4 line if no previous data base exists or don't want to compare.
# print(f'Total number of composition points (before dropping): {len(N_atom)}')
# dbname = '/home/bmondal/MachineLerning/BandGapML/DATAbase/RandomConfigInGaPAsSb.db'
# _, N_atom = mlgf.DropCommonDataPreviousDataBase(dbname, N_atom)
# print(f'Total number of composition points (after dropping): {len(N_atom)}')

# %% -------------------------- Plot in ternary plot ---------------------------
# ********** Conversion back to concentration **********************************
# N_conc = N_atom[['PHOSPHORUS', 'ARSENIC', 'ANTIMONY']]/216*100
# N_conc['STRAIN'] = N_atom['STRAIN']

# vmax = N_conc['STRAIN'].max()
# vmin = N_conc['STRAIN'].min()

# #N_conc['testconc'] = 100
# #N_conc['totalconc'] = N_conc[['PHOSPHORUS', 'ARSENIC', 'ANTIMONY']].sum(axis=1)
# # pd.testing.assert_frame_equal(N_conc[['totalconc']], N_conc[['testconc']], check_dtype=False)
# # N_conc[['totalconc']].astype(int).equals(N_conc[['testconc']])

# dd = mlpf.generate_scatter_data(N_conc) 
# _ = mlpf.DrawRandomConfigurationPoints(dd, fname=None, titletext=None, 
#                                        savefig=False, scale=100,
#                                        axislabels = ["GaSb", "GaAs", "GaP"],axiscolors=None,
#                                        axislabelcolors=None,
#                                        vmin=vmin, vmax=vmax,cmap=plt.cm.get_cmap('viridis'),
#                                        fontsize = 20,
#                                        colors=N_conc['STRAIN'])

# %% -------------------- SAVE to Data Base -----------------------------------
dirpath = '/home/bmondal/MachineLerning/BandGapML/'
dbname = dirpath+'/DATAbase/RandomConfigInGaPAsSb.db'
conn = sq.connect(dbname)
N_atom.to_sql("ATOMNUMBER", conn, if_exists='replace', index=False)
conn.close()

# %%----------------------------------------------------------------------------
# ******* Create the random structures for only Unique concentrations *********
print('Duplicated atom numbers:')
print(N_atom[N_atom.duplicated(['INDIUM','GALLIUM','PHOSPHORUS', 'ARSENIC', 'ANTIMONY'],keep=False)])
print('--------------')
N_atom_unque = N_atom.drop_duplicates(subset=['INDIUM','GALLIUM','PHOSPHORUS', 'ARSENIC', 'ANTIMONY'])
dirpath = '/home/bmondal/MachineLerning/BandGapML/InGaPAsSb/Allrndstr/SESSION1/'
setf = 0
setcounter = 0
# Number of files per SET folder. This helps to 
nfilesperfolder = 40  

for I in N_atom_unque[['INDIUM','GALLIUM','PHOSPHORUS', 'ARSENIC', 'ANTIMONY']].values:
    path = dirpath + f'/SET{setcounter:03d}/In{I[0]}Ga{I[1]}P{I[2]}As{I[3]}Sb{I[4]}'
    if not os.path.exists(path):
        os.makedirs(path)
        filename1 = path + '/rndstr.in'
        filename2 = path + '/sqscell.out'
        f1text = f"""1.   1.   1.   90   90   90
0.0   0.5   0.5
0.5   0.0   0.5
0.5   0.5   0.0
0.0   0.0   0.0  In={I[0]/216}, Ga={I[1]/216}
0.25  0.25  0.25  P={I[2]/216}, As={I[3]/216}, Sb={I[4]/216}
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


# %% Check if the compositions from generated random structures are consistence
# This is after all the SQS cells are generated
# %%% Generate tmp db with the collected atom numbers from final SQS POSCARs
allfile = glob.glob(
    "/home/bmondal/MachineLerning/BandGapML/InGaPAsSb/AllNonConvergedPoscar/SESSION1/SET*/In*/EQM/conf01/POSCAR")
testd = defaultdict(list)
for I in allfile:
    Natom = np.genfromtxt(I, skip_header=5, max_rows=2, dtype=str)
    NAT = Natom[0]
    NN = Natom[1].astype(int)
    tmp_dict = {"In": 0, "Ga": 0, "P": 0, "As": 0, 'Sb': 0}
    
    for ind, x in enumerate(NAT):
        tmp_dict[x] = NN[ind]    
    for key, val in tmp_dict.items():
        testd[key].append(val)

testdf = pd.DataFrame.from_dict(testd)
testdf = testdf.rename(
    columns={"In": 'INDIUM', "Ga": 'GALLIUM', "P": 'PHOSPHORUS', "As": 'ARSENIC', 'Sb': 'ANTIMONY'})
testdf = testdf[['INDIUM','GALLIUM','PHOSPHORUS', 'ARSENIC', 'ANTIMONY']]
testdf_final = testdf.sort_values(
    by=['INDIUM','GALLIUM','PHOSPHORUS', 'ARSENIC', 'ANTIMONY']).reset_index(drop=True)
# %%% Compare the final SQS POSCAR atom numbers to the existed database atom numbers
dbname = '/home/bmondal/MachineLerning/BandGapML/DATAbase/RandomConfigInGaPAsSb.db'
conn = sq.connect(dbname)

# conn.execute("""
#              DELETE FROM ATOMNUMBER
#              WHERE ANTIMONY = 0;
#              """)
# conn.commit()
# conn.close()

df = pd.read_sql_query(
    'SELECT INDIUM, GALLIUM,PHOSPHORUS, ARSENIC, ANTIMONY FROM ATOMNUMBER', conn)

df_unique = df.drop_duplicates(subset=['INDIUM','GALLIUM','PHOSPHORUS', 'ARSENIC', 'ANTIMONY'])
df_final = df_unique.sort_values(
    by=['INDIUM','GALLIUM','PHOSPHORUS', 'ARSENIC', 'ANTIMONY']).reset_index(drop=True)

pd.testing.assert_frame_equal(testdf_final, df_final, check_dtype=False)

# %% ========= Creating folders with different strains ========================

dbname = '/home/bmondal/MachineLerning/BandGapML/DATAbase/RandomConfigInGaPAsSb.db'
conn = sq.connect(dbname)
df = pd.read_sql_query('SELECT * FROM ATOMNUMBER', conn)

dirpath = '/home/bmondal/MachineLerning/BandGapML/InGaPAsSb/AllRandomFolders'
for I in df.values:
    path = dirpath + \
        f'/In{int(I[0])}Ga{int(I[1])}P{int(I[2])}As{int(I[3])}Sb{int(I[4])}/S{I[5]:.6f}/conf01'
    # print(path)

    if os.path.exists(path):
        print(f'warning: {path} exists.')
    else:
        os.makedirs(path)

allfile1 = glob.glob(
    "/home/bmondal/MachineLerning/BandGapML/InGaPAsSb/AllNonConvergedPoscar/SESSION1/SET*/In*")
allfile2 = glob.glob(
    "/home/bmondal/MachineLerning/BandGapML/InGaPAsSb/AllRandomFolders/In*")
assert len(allfile1) == len(
    allfile2), 'In* folder numbers does not match founds'
