#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 14:24:31 2021

@author: bmondal
"""
import sys
import re
import json
import pandas as pd
import numpy as np
import sqlite3 as sq
from collections import defaultdict
from itertools import combinations
import HeaderTxt
from math import isclose
import argparse
from datetime import datetime

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
    print(f"{' '*64} Date: {now.strftime('%Y-%m-%d  %H:%M:%S')}\n")
    return 


def ParserOptions():
    """
    This finction defines the parsers.

    Returns
    -------
    Parser arguments

    """
    HeaderDecorator()
    sys.stdout.flush()
    parser = argparse.ArgumentParser(prog='BPDmpiSQL.py', description='This script creates the database for Bandgap phase diagram from VASP WAVECAR and POSCAR file.', epilog='Have fun!')
    parser.add_argument('-d', metavar='DIRECTORYNAME', default=".", help='The file path where the tree of strain folders are (default: current directory). e.g. /home/mondal/VASP/test/')
    parser.add_argument('-N', type=int, help='Total number of electrons.')
    parser.add_argument('-NN', type=int, help='Total number of ion w.r.t which concentration will be calculated.')
    parser.add_argument('-NKP', type=int, default=1, help='Total number of KPOINTS. (default: 1)')
    parser.add_argument('-SF', nargs='+', type=int, default=[6,6,6], help='Supercell dimensions (must be int) in a, b and c lattice vector directions respectively. (default: [6,6,6])')
    parser.add_argument('-ispin', type=int, default=1, help='If ISPIN=1 or 2  (default: 1)')
    parser.add_argument('-ncol', action='store_true', default=False, help='Noncolliner calculation = True or False (default: False).')
    parser.add_argument('-CompareMean', action='store_true', default=False, help='Average over equivalent k-points = True or False (default: False).')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    
    return parser.parse_args()
#%%----------------------------------------------------------------------------
def OPENdatabase(dbname, NUNFOLDED):
    conn = sq.connect(dbname)
    #conn = sq.connect(':memory:')
    
    Tabletextstart = "CREATE TABLE IF NOT EXISTS COMPUTATIONALDATA" 
    Tabletext1 = "(GALLIUM  REAL," \
                "INDIUM  REAL," \
                "NITROGEN   REAL,"\
                "PHOSPHORUS   REAL,"\
                "ARSENIC  REAL,"\
                "ANTIMONY  REAL,"\
                "BISMUTH  REAL,"
                
    Tabletext2 = "STRAIN   REAL,"
    
    Tabletext3 = "LATTICEPARAMETER1  REAL,"\
                "LATTICEPARAMETER2  REAL,"\
                "LATTICEPARAMETER3  REAL,"
                
    Tabletextend = "BANDGAP  REAL,"\
                 "NATURE   INTEGER"\
                 ")"

    TabletextBWvb = ','.join([' BWvb'+str(i)+'  REAL' for i in range(NUNFOLDED) ])           
    TabletextBWcb = ','.join([' BWcb'+str(i)+'  REAL' for i in range(NUNFOLDED) ])             
    Tabletext =  Tabletextstart + Tabletext1 + Tabletext2 + Tabletext3 + TabletextBWvb \
        + ',' + TabletextBWcb + ',' + Tabletextend

    conn.execute(Tabletext)

    return conn

def DataInsertion(mydata, conn):        
    tabletext = 'INSERT INTO COMPUTATIONALDATA VALUES ('+','.join(['?']*len(mydata[0]))+')'
    conn.executemany(tabletext,mydata) 
    conn.commit()
    conn.close()
    return

def insert_data(mydata, conn, C):
    with conn:
        C.execute("INSERT INTO COMPUTATIONALDATA\
                    VALUES (:ga, :in, :n, :p, :as, :sb, :bi, :st, :lp1, :lp2, :lp3, :eg, :nt)",\
                        {'ga':mydata.gallium,'in':mydata.indium, 'n':mydata.nitrogen,\
                         'p':mydata.phosphorus,'as':mydata.arsenic,'sb':mydata.antimony,'bi':mydata.bismuth,\
                             'st':mydata.strain,'lp1':mydata.lp1,'lp2':mydata.lp2,'lp3':mydata.lp3,\
                                 'eg':mydata.bandgap,'nt':mydata.nature})
                
class Compound(object):
    '''Creates Compound class'''
    elements = ['Ga','In','N','P','As','Sb','Bi']
    elementsSet = set(elements)
    
    def __init__(self, compoundname, strain=0,lp=None,bandgap=0,nature=None):
        cstring = self.ElementsComposition(compoundname)
        self.gallium = cstring[0]
        self.indium = cstring[1]
        self.nitrogen = cstring[2]
        self.phosphorus = cstring[3]
        self.arsenic = cstring[4]
        self.antimony = cstring[5]
        self.bismuth = cstring[6]
        self.strain = strain
        self.lp1 = lp[0]
        self.lp2 = lp[1]
        self.lp3 = lp[2]
        self.bandgap = bandgap
        self.nature = nature
        
    def ElementsComposition(self, compoundname):
        composition = [0]*len(self.elements)
        elefullist = re.findall(r'[A-Za-z]+|[0-9.]+', compoundname)
        ele = elefullist[::2]
        comp = elefullist[1::2]
        assert set(ele).issubset(self.elementsSet), f'The compound {compoundname} contain element which is not implemented in the database.'
        for i, e in enumerate(ele):
            epos = self.elements.index(e)
            composition[epos] += float(comp[i])  
        return composition

#%%----------------------------------------------------------------------------
def SymmetryEquivKpoints_zincblende_Gfolding(N, dmprecesion):
    """
    This function calculates the symmetry equivalent k-points for zincblende
    structures with only Gamma point band folding.

    Parameters
    ----------
    N : Integer array
        The array containing 3 supercell dimensions.
    dmprecesion : Integer
        The decimal precision that will be used for rounding float.

    Returns
    -------
    Float nd-list
        The k-points coordinates in reciprocal space.

    """
    NN = np.array(N)/2

    kp = [np.arange(1,N[0],1),np.arange(1,N[1],1),np.arange(1,N[2],1)]
    KP = [set(np.around(abs(np.where(I>NN[i], I-N[i], I))/N[i], decimals=dmprecesion)) for i, I in enumerate(kp)]
    KP2intersection = [ i[0] & i[1] for i in combinations(KP,2) ]
    KP3intersection = KP[0] & KP[1] & KP[2]   
    
    KPP = defaultdict(list)
    KPP[frozenset([0.0])] = [[0,0,0]]
    for i, I in enumerate(KP):
        for J in I:
            kptemp = [0,0,0]
            kptemp[i]=J
            KPP[frozenset([J])].append(kptemp) 
            
    for I in KP3intersection:
        KPP[frozenset([I])].append([I]*3)
            
    for i, I in enumerate(KP2intersection):
        x = list(I)
        kptemp = np.vstack((x,x,x)).T
        kptemp[:,2-i]= 0
        for J in kptemp:
            KPP[frozenset(J)].append(J)
    
    return list(KPP.values())   

def FindSymmetryEquivalentKPpos(FileSpecificKP, FoldFactor, CompareMean=False, dmprecesion=6, structure='zincblende'):
    """
    This function find the symmetry equivalent positions in the current file k-point
    list.

    Parameters
    ----------
    FileSpecificKP : Float array/list
        The k-point coordinate list for the current file.
    FoldFactor : Integer list/array
        The supercell dimensions (X, Y, Z).
    CompareMean : Bool
        Whether the symmetry averaged or maximum BW (among symmetry equivalent 
        k-points) will be calculated.
    dmprecesion : Integer, optional
        The decimal precision for rounding. The default is 6.
    structure : String, optional
        The crystal structure. The default is 'zincblende'.

    Returns
    -------
    Integer nd-list
        The list containg the position index of the symmetry equivalent k-points.

    """
    if 'zincblende' in structure:
        KPlist = SymmetryEquivKpoints_zincblende_Gfolding(FoldFactor, dmprecesion=dmprecesion)
    else:
        sys.exit("Error: Only zincblende struture is implemented so far.")
      
    FileKpPos = defaultdict(list)
    for i, I in enumerate(FileSpecificKP):
        notsymmetric = True
        for j, J in enumerate(KPlist):
             for JJ in J:
                 if np.allclose(abs(I),JJ): 
                     FileKpPos[j].append(i)
                     notsymmetric = False
                     
        if notsymmetric: 
            FileKpPos[i].append(i) if CompareMean  else FileKpPos['other'].append(i)
        
    return list(FileKpPos.values())
    
def GroupBws(fname, FoldFactor, CompareMean=False):
    FullData = np.genfromtxt(fname)
    NUNFOLDED = FoldFactor[0]*FoldFactor[1]*FoldFactor[2]
    FileSpecificKP = FullData[:NUNFOLDED,:3]
    BW_posgroup_array = FindSymmetryEquivalentKPpos(FileSpecificKP, FoldFactor, CompareMean=CompareMean)
    return BW_posgroup_array, FileSpecificKP

def CollectBWs(fname, NUNFOLDED, CB_index, BW_Group):
        FullData = np.genfromtxt(fname)
        FullData = np.split(FullData, len(FullData)//NUNFOLDED)[CB_index-1:CB_index+1]
        Bandgap = FullData[1][0,-2] - FullData[0][0,-2]
        
        FileSpecificKP = FullData[0][:,:3]
        BW_DATA_vb = FullData[0][:,-1] * 100 # BW Percentage conversion
        BW_DATA_cb = FullData[1][:,-1] * 100
        VBdata = np.array([[round(max(BW_DATA_vb[ff]), 4), ff[np.argmax(BW_DATA_vb[ff])]] for ff in BW_Group],dtype=object)
        CBdata = np.array([[round(max(BW_DATA_cb[ff]), 4), ff[np.argmax(BW_DATA_cb[ff])]] for ff in BW_Group],dtype=object)

        BW1_pos = CBdata[np.argmax(CBdata[:,0]), 1]
        BW2_pos = VBdata[np.argmax(VBdata[:,0]), 1]
        CheckGpoint = np.allclose(FileSpecificKP[BW2_pos], [0, 0, 0])
        """
        1==Direct bandgap at the Gamma point
        2==Direct bandgap at some other k-point
        3==Indirect bandgap with VBM at the Gamma point
        4==Indirect bandgap with VBM at some other k-point
        """
        if BW1_pos==BW2_pos:
            EgNature = 1 if CheckGpoint else 2
        else:
            EgNature = 3 if CheckGpoint else 4 

        return Bandgap, VBdata[:,0], CBdata[:,0], EgNature
    
def FindComposition(poscar_filepath, NionNumber):
    elements = np.genfromtxt(poscar_filepath, max_rows=2, skip_header=5, dtype=str)
    ee = elements[0]
    targetelements = ['Ga','In','N','P','As','Sb','Bi']
    composition = [0]*len(targetelements)
    
    if set(ee).issubset(targetelements):
        Number = elements[1].astype(float)
        conc = Number/NionNumber * 100.
        if len(ee) != len(set(ee)):
            edata = defaultdict(list)
            for i, I in enumerate(ee):
                edata[I].append(Number[i])
            ee = list(edata.keys())
            conc = [sum(J)/NionNumber*100. for J in edata.values()]
        for i, I in enumerate(ee): 
            composition[targetelements.index(I)] = conc[i] 
    else:
        print('POSCAR file:', poscar_filepath)
        print('* The POSCAR file contain element which is not implemented in the database.')

    return composition

"""
For streched zincblende:
x, y, z == Lattice parameter of streched zincblende structure.
xy, yz, and xz == Diagonals of streched zincblende or 2 times lattice vactors
Note1: Angle b/w x, y, z are always 90 degree in (streched)zincblende structure
        Angle b/w xy, xz and yz might be differ from 60 in streched case
Note2: 6x6 primitive supercell is equivalent to 3x3 unit cell for zincblende

(2*xy)**2 = x**2 + y**2; (2*yz)**2 = y**2 + z**2; (2*xz)**2 = x**2 + z**2

Solving above 3 equations we get;
        y**2 = xy**2 + yz**2 - xz**2
        x**2 = xy**2 + xz**2 - yz**2
        z**2 = yz**2 + xz**2 - xy**2
"""
def vec(a):
    """
    Calculates the lattice vector magnitude.

    Parameters
    ----------
    a : Float
        [x, y, and z] coordinate of the lattice vector.

    Returns
    -------
    Float
        Lattice vector magnitude.

    """
    return (np.sqrt(np.sum(a**2, axis=1)))

def CalculateLatticeParameter(vector,crys_str):
    """
    This function calculates the lattice parameter for a given crystal structure. So far only
    cubic and zincblende structure is implemented.

    Parameters
    ----------
    vector : Float
        3 lattice vector magnitudes.
    crys_str : String
        The crystal structure.

    Returns
    -------
    lp : Float
        3 Lattice parameters.

    """
    if(crys_str == 'cubic'):
        lp = vector
    elif(crys_str == 'zincblende'): 
        nv = vector**2
        lp =  np.array([np.sqrt(nv[0] + nv[2] - nv[1]),
                        np.sqrt(nv[0] + nv[1] - nv[2]),
                        np.sqrt(nv[1] + nv[2] - nv[0])])

        lp *= np.sqrt(2)
    else:
        print("Only supports cubic,zincblende. The other symmetry are not implemented yet.")
        sys.exit()
    
    return lp

def getLP(file, SupercellDimension, ClusterType='zincblende'):
    """
    This function reads the lattice vector file (e.g. POSCAR) and calculates lattice
    parameters using vec() and CalculateLatticeParameter().

    Parameters
    ----------
    file : String filename
        Filename (e.g. POSCAR).
    SupercellDimension : Integer
        Dimensions of supercell in array.
    ClusterType : String
        Common name of the crystal system. So far only cubic and zincblende structure is implemented.
        Default is 'zincblende'.

    Returns
    -------
    lp : Float
        3 Lattice parameters.

    """
    fac = np.genfromtxt(file, max_rows=1, skip_header=1)
    lvec = np.genfromtxt(file, max_rows=3, skip_header=2) * fac
    plvec = np.divide(vec(lvec), SupercellDimension)
    lp = CalculateLatticeParameter(plvec, ClusterType)
    return lp

def CollectDigit(mystring):
    """
    This function returns the signed number in a string.

    Parameters
    ----------
    mystring : String
        The string from which you want to extract the numbers.

    Returns
    -------
    String
        All non-overlapping matches of pattern in string, as a list of strings or tuples.

    """
    return re.findall('-?\+?\d+\.?\d*', mystring)[0]
#%%----------------------------------------------------------------------------
def DI2Number(mystring):
    if mystring.upper() == 'D':
        return 0
    elif mystring.upper() == 'I':
        return 1
    else:
        return 2
    
