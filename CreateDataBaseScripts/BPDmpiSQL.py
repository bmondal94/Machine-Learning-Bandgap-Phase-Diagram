#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 15:43:48 2021

@author: bmondal
"""

#%%-------------------- Importing modules -------------------------------------
import numpy as np      
import glob
from mpi4py import MPI
import SQLdatabaseFunctions as sqlf

#%%----------------- main() ---------------------------------------------------
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    status = MPI.Status()
    
    if rank == 0:
        try:
            args = sqlf.ParserOptions()
            ## +++++++++ Define the parser variables ++++++++++++++++++++++++++
            fuldir = args.d
            NKPOINTS = args.NKP
            NELECT = args.N
            NionNumber = args.NN
            ISPIN = args.ispin
            NCOL = args.ncol
            CompareMean = args.CompareMean
            SuprecellDimension = np.asarray(args.SF)
            
            ## +++++++++++ Define other necessary variables +++++++++++++++++++
            files = glob.glob(fuldir + '/*/**/WAVECAR_spinor1.f2b', recursive=True)
            BW_Group, FileSpecificKP = sqlf.GroupBws(files[0], SuprecellDimension, CompareMean=CompareMean)
            print('KPOINT coordinates:\n',FileSpecificKP)
            print('Symmetry equivalent k-points grouping:\n', BW_Group, '\n')
            
            if ISPIN == 2 : NCOL = True
            if NKPOINTS > 1:
                print("This scripts is only valid for single Gamma point calculation.")
                print("Multiple kpoints is not implemented yet.")
                exit(1)       

            CB_index = NELECT  if NCOL else NELECT//2
            
            pos_list = [CB_index, SuprecellDimension, NionNumber, BW_Group]
            #pos_list = [NCOL, SuprecellDimension, NionNumber]
                
        except:
            files = None

    else:
        files = None
        pos_list = None

    files = comm.bcast(files, root=0)
    if files is None:
        exit(0)

    pos_list = comm.bcast(pos_list, root=0)
    
    #%% +++++++++ Collect the data ++++++++++++++++++++++++++++++++++++++++++++  
    data = {}
    n = len(files)
    ineachNode = n//size
    num_larger_procs = n - size*ineachNode
    if rank<num_larger_procs:
        ineachNode += 1
        ini = rank*ineachNode
    else:
        ini = rank*ineachNode+num_larger_procs

    start_time = MPI.Wtime()
    for file in files[ini:ini+ineachNode]:
        print(f"File: {file} in Process:{rank}")
        fullfilepath = file.split("/")

        Strain = float(sqlf.CollectDigit(fullfilepath[-3]))   
               
        CB_index, SuprecellDimension, NionNumber, BW_Group = pos_list[0], pos_list[1], pos_list[2], pos_list[3]
        NUNFOLDED = SuprecellDimension[0]*SuprecellDimension[1]*SuprecellDimension[2]
        
        Bandgap, BW_DATA_vb, BW_DATA_cb, EgNature = sqlf.CollectBWs(file, NUNFOLDED, CB_index, BW_Group)


        poscar_filepath = '/'.join(fullfilepath[:-1])+'/POSCAR'
        llp = sqlf.getLP(poscar_filepath, SuprecellDimension)

        composition = sqlf.FindComposition(poscar_filepath, NionNumber)

        
        data[file] =  tuple((*composition, Strain, *llp, *BW_DATA_vb, *BW_DATA_cb, Bandgap, EgNature))
        
        print(f"\t* Processing successful in Process:{rank}")
            
    finish_time = MPI.Wtime() 
    
    #%% ++++++++++++++ Final data gather and storing ++++++++++++++++++++++++++
    if rank == 0:
        DATA = list(data.values())
        for i in range(1, size, 1): 
            rdata = comm.recv(source=i, tag=11, status=status)
            if rdata is not None:
                DATA = DATA + list(rdata.values())
            
        print("\n***************** Data Collection finished *******************\n")
        print("Program:"+' '*21 +'BPD_mpi.py')
        Total_time = finish_time - start_time
        Time_per_file = Total_time/n 
        print(f"\nDirectory path: {' '*13}{fuldir}\nTotal number of files: {' '*6}{n}\nThe number of mpi process: {' '*2}{size}")
        print(f"Total time: {' '*17}{Total_time:.3f} s\n")
        print("\n******************** Creating Database ***********************\n")
        BW_len = len(BW_DATA_vb)
        dbname = fuldir+'/BPD_MLDataBase.db'
        conn = sqlf.OPENdatabase(dbname, BW_len)
        _ = sqlf.DataInsertion(DATA, conn)
        print(f'+ Database creation successfull: {dbname}')
        print("\n*************** Congratulation: All done *********************\n")

    else:
        request = comm.send(data, dest=0, tag=11)
    
#------------------------------------------------------------------------------             
