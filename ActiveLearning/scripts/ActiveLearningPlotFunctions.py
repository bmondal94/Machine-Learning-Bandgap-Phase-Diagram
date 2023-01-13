#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 17:11:17 2022

@author: bmondal
"""

import numpy as np
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
import ternary
# from matplotlib import animation
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import Normalize
import sqlite3 as sq
import pandas as pd

params = {'figure.figsize': (8, 6),
          'legend.fontsize': 18,
          'axes.labelsize': 24,
          'axes.titlesize': 24,
          'xtick.labelsize':24,
          'ytick.labelsize': 24,
          'errorbar.capsize':2}
plt.rcParams.update(params)
#%% ===========================================================================
def BatchPlotGrabFrame(AL_dbname, columns, TargetTablePattern='BATCH',
                       natoms=216, cumulative=True, NoColor=False,SVRSVC=True,
                       save_movie_path=None, xLimit=None, yLimit=None, xlabel=None, ylabel=None):        
    if not save_movie_path:
        print('Please provide the save_movie_path.')
        return
    conn = sq.connect(AL_dbname)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master  WHERE type='table';")
    tables = cursor.fetchall()
    BATCH_tables = [table[0] for table in tables if 'BATCH' in table[0]]
    
    colors = ['k']*len(BATCH_tables) if NoColor else cm.rainbow(np.linspace(0, 1, len(BATCH_tables)))
    writer = FFMpegWriter(fps=2)
    fig, ax = plt.subplots()
    x, y = pd.DataFrame(),pd.DataFrame()
    l, = ax.plot([], [], marker='o', ls='')
    if NoColor: l.set_color('k')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xLimit)
    plt.ylim(yLimit)
    
    with writer.saving(fig, save_movie_path, 100):
        for i, table_name in enumerate(BATCH_tables):
            if table_name.startswith(TargetTablePattern):
                print(f"Adding {table_name}")
                batch_samples = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                batch_samples[columns[0]] = batch_samples[columns[0]] / natoms * 100 # Convert atom number to concentration
                plt.title(table_name)
                x = batch_samples[columns[0]]
                y = batch_samples[columns[1]]
                if cumulative:
                    ax.plot(x,y, marker='o', ls='', color=colors[i])
                else:
                    l.set_data(x, y)
                writer.grab_frame()
    conn.close()
    return

def PlotBatchData(df, coulmns, title=None, xLimit=None, yLimit=None, xlabel=None, ylabel=None,
                  save=False,savepath='.',figname='TotalTrainigSamples.png'):
    xlabel_text = coulmns[0] if xlabel is None else xlabel
    ylabel_text = coulmns[1] if ylabel is None else ylabel
    ax = df.plot(coulmns[0],coulmns[1],kind='scatter',title=title,xlim=xLimit,ylim=yLimit,xlabel=xlabel_text,ylabel=ylabel_text)
    if save:
        ax.figure.savefig(savepath+'/'+figname,bbox_inches = 'tight',dpi=300)
        plt.close()
    # fig, ax = plt.subplots()
    # plt.title(title)
    # ax.scatter(df[coulmns[0]],df[coulmns[1]])
    # ax.set_xlabel(xlabel_text)
    # ax.set_ylabel(ylabel_text)
    # ax.set_xlim(xLimit)
    # ax.set_ylim(yLimit) 
    # plt.tight_layout()
    # if save:
    #     plt.savefig(savepath+figname,bbox_inches = 'tight',dpi=300)
    # else:
    #     plt.show()
    return ax

def PlotBatchSampleNumber(XX, YY, title=None, base=None, xLimit=(-0.1,None), 
                          yLimit=(0,None), xlabel=None, ylabel=None,
                          save=False,savepath='.',figname='BatchSampleCount.png'):
    fig, ax = plt.subplots()
    plt.title(title)
    ax.plot(XX,YY, marker='o',color='k')
    ax2 = ax.twinx()
    ax2.plot(XX,np.cumsum(YY), marker='o',color='r')
    ax2.tick_params(colors='r', which='both') 
    if base is not None: ax.xaxis.set_major_locator(ticker.MultipleLocator(base=base))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xLimit)
    ax.set_ylim(yLimit) 
    plt.tight_layout()
    if save:
        plt.savefig(savepath+'/'+figname,bbox_inches = 'tight',dpi=300)
        plt.close()
    else:
        plt.show()
    return fig, ax

def PlotCheckData_ternary(df, coulmns, z, title=None, xLimit=None, yLimit=None, 
                          xlabel=None, ylabel=None, cbarlabel=None, vmin=None,vmax=None):
    xlabel_text = coulmns[0] if xlabel is None else xlabel
    ylabel_text = coulmns[1] if ylabel is None else ylabel
    fig, ax = plt.subplots()
    ax.set_title(title)
    im = ax.scatter(df[coulmns[0]],df[coulmns[1]],c=z,cmap=plt.cm.RdYlBu_r,vmin=vmin,vmax=vmax)
    cbar = fig.colorbar(im,format='%.1f', ax=[ax], pad=0.03)
    cbar.ax.set_ylabel(cbarlabel)
    ax.set_xlabel(xlabel_text)
    ax.set_ylabel(ylabel_text)
    ax.set_xlim(xLimit)
    ax.set_ylim(yLimit) 
    # plt.tight_layout()
    plt.show()
    return fig, ax

def PlotModelAccuary(df, coulmns, title=None, base=None, xLimit=None, yLimit=None, 
                     xlabel=None, ylabel=None, save=False, savepath='.',figname='ModelAccuracy.png'):
    fig, ax = plt.subplots()
    plt.title(title)
    ax.errorbar(df.index,df[coulmns[0]], yerr=df[coulmns[1]]) #, marker='o',color='k')
    if base is not None: ax.xaxis.set_major_locator(ticker.MultipleLocator(base=base))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xLimit)
    ax.set_ylim(yLimit) 
    plt.tight_layout()
    if save:
        plt.savefig(savepath+'/'+figname,bbox_inches = 'tight',dpi=300)
        plt.close()
    else:
        plt.show()
    return fig, ax

def PlotRawDataTernary(df, coulmns, title=None, xLimit=None, yLimit=None, nature=False,
                        xlabel=None, ylabel=None, cbarlabel=None, vmin=None,vmax=None,
                        save=False, savepath='.',figname='FIG.png'):
    assert len(coulmns) == 3, 'Should be only 3 columns for X, Y, Color values'
    xlabel_text = coulmns[0] if xlabel is None else xlabel
    ylabel_text = coulmns[1] if ylabel is None else ylabel
    zlabel_text = coulmns[2] if cbarlabel is None else cbarlabel
    fig, ax = plt.subplots()
    ax.set_title(title)
    cmapp='Set1' if nature else plt.cm.RdYlBu_r
    im = ax.scatter(df[coulmns[0]],df[coulmns[1]],c=df[coulmns[2]],cmap=cmapp,vmin=vmin,vmax=vmax)
    if not nature:
        cbar = fig.colorbar(im,format='%.1f', ax=[ax], pad=0.03)
        cbar.ax.set_ylabel(zlabel_text)
    ax.set_xlabel(xlabel_text)
    ax.set_ylabel(ylabel_text)
    ax.set_xlim(xLimit)
    ax.set_ylim(yLimit) 
    # plt.tight_layout()
    if save:
        plt.savefig(savepath+'/'+figname,bbox_inches = 'tight',dpi=300)
        plt.close()
    else:
        plt.show()
    return fig, ax

def plot_test_results(XX, YY, text=None, save=False, savepath='.',figname='TruePrediction.png'):
    plt.figure()
    ax = plt.subplot(aspect='equal')
    ax.scatter(XX, YY)
    ax.set_title(text)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predictions')
    p1 = max(XX) #max(max(YY), max(XX))
    p2 = min(XX) #min(min(YY), min(XX))
    ax.plot([p1, p2], [p1, p2], 'b-')
    DIFF_ = abs(XX - YY)
    # ax.plot([],[],' ',label=f"Max error = {max(DIFF_):.2f} eV \n {'MAE':>11} = {np.mean(DIFF_):.2f}±{np.std(DIFF_):.2f} eV")
    ax.plot([],[],' ',label=f"Max error = {max(DIFF_):.2f} eV \nMAE = {np.mean(DIFF_):.2f}±{np.std(DIFF_):.2f} eV")
    ax.legend(handlelength=0)
    plt.tight_layout()
    if save:
        plt.savefig(savepath+'/'+figname,bbox_inches = 'tight',dpi=300)
        plt.close()
    else:
        plt.show()
    return ax

##### =========================================================================
def ShowbatchInteractive(AL_dbname, columns, natoms=216, TargetTablePattern='BATCH',
                         save_movie=False, cumulative=True, NoColor=False,
                         save_movie_path=None, xLimit=None, yLimit=None, xlabel=None, ylabel=None):
    conn = sq.connect(AL_dbname)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master  WHERE type='table';")
    tables = cursor.fetchall()
    BATCH_tables = [table[0] for table in tables if table[0].startswith(TargetTablePattern)]
    conn.close() 

    colors = ['k']*len(BATCH_tables) if NoColor else cm.rainbow(np.linspace(0, 1, len(BATCH_tables)))
    fig, ax = plt.subplots()
    x, y = pd.DataFrame(),pd.DataFrame()
    sc = ax.scatter([],[],marker='o')
    if NoColor: sc.set_color('k')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xLimit)
    plt.ylim(yLimit)

    def animate(i, cumulative=False):
        global x, y
        if i == 0 and cumulative:       
            plt.cla()
            plt.xlim(xLimit)
            plt.ylim(yLimit)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

        table_name = BATCH_tables[i]
        conn = sq.connect(AL_dbname)
        batch_samples = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        batch_samples[columns[0]] = batch_samples[columns[0]] / natoms * 100 # Convert atom number to concentration
        conn.close() 
        x = batch_samples[columns[0]]
        y = batch_samples[columns[1]]
        if cumulative:
            ax.scatter(x,y,color=colors[i])
        else:
            sc.set_offsets(np.c_[x,y])
        
        ax.set_title(table_name)
        
    ani = FuncAnimation(fig, animate, frames=len(BATCH_tables), fargs=(cumulative,), interval=800, repeat_delay=100, repeat=True) 
    if save_movie:
        if save_movie_path:
            ani.save(save_movie_path)
            # plt.close()
        else:
            print('Please provide the save_movie_path.')
    else:
        plt.show()
    return ani

def ShowbatchInteractiveBoth(AL_dbname, columns, natoms=216, 
                             save_movie=False, cumulative=True, NoColor=False,
                             save_movie_path=None, xLimit=None, yLimit=None, xlabel=None, ylabel=None):
    
    conn = sq.connect(AL_dbname)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master  WHERE type='table';")
    tables = cursor.fetchall()
    BATCH_tables1 = [table[0] for table in tables if table[0].startswith('MAGNITUDEBATCH')]
    BATCH_tables2 = [table[0] for table in tables if table[0].startswith('NATUREBATCH')]
    conn.close() 
    
    BATCH_tables1_indices = [int(x.split('_')[1]) for x in BATCH_tables1]
    BATCH_tables2_indices = [int(x.split('_')[1]) for x in BATCH_tables2]
    
    FrameStart = min(min(BATCH_tables1_indices), min(BATCH_tables2_indices))
    FrameEnd = max(max(BATCH_tables1_indices), max(BATCH_tables2_indices))
    FRAMES = np.arange(FrameStart,FrameEnd+1, dtype=int)
    # print(FrameStart,FrameEnd,FRAMES)

    colors1 = ['k']*(FrameEnd+1) if NoColor else cm.rainbow(np.linspace(0, 1, (FrameEnd+1)))
    colors2 = ['r']*(FrameEnd+1) if NoColor else cm.rainbow(np.linspace(0, 1, (FrameEnd+1)))
    fig, ax = plt.subplots()
    sc = ax.scatter([],[],marker='o')
    if NoColor: sc.set_color('k')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xLimit)
    plt.ylim(yLimit)
    TXXT1 = fig.text(0.35, 0.9,'', ha="center", va="bottom", size=plt.rcParams['axes.titlesize']*0.5)
    fig.text(0.515, 0.9,'+', ha="center", va="bottom")
    TXXT2 = fig.text(0.65, 0.9,'', ha="center", va="bottom", size=plt.rcParams['axes.titlesize']*0.5)

    def animate(i, FrameStart, cumulative=False):
        if i == FrameStart and cumulative:       
            plt.cla()
            plt.xlim(xLimit)
            plt.ylim(yLimit)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

        if f'MAGNITUDEBATCH_{i}' in BATCH_tables1:
            table_name1 = f'MAGNITUDEBATCH_{i}'
            conn = sq.connect(AL_dbname)
            batch_samples1 = pd.read_sql_query(f"SELECT * FROM {table_name1}", conn)
            batch_samples1[columns[0]] = batch_samples1[columns[0]] / natoms * 100 # Convert atom number to concentration
            conn.close() 
            x1 = batch_samples1[columns[0]]
            y1 = batch_samples1[columns[1]]
            TXXT1.set_text(table_name1); TXXT1.set_color(colors1[i])
        else:
            x1=pd.DataFrame(); y1=pd.DataFrame()
            if not cumulative: TXXT1.set_text('MAGNITUDEBATCH_')
        
        if f'NATUREBATCH_{i}' in BATCH_tables2:
            table_name2 = f'NATUREBATCH_{i}'
            conn = sq.connect(AL_dbname)
            batch_samples2 = pd.read_sql_query(f"SELECT * FROM {table_name2}", conn)
            batch_samples2[columns[0]] = batch_samples2[columns[0]] / natoms * 100 # Convert atom number to concentration
            conn.close() 
            x2 = batch_samples2[columns[0]]
            y2 = batch_samples2[columns[1]]
            TXXT2.set_text(table_name2); TXXT2.set_color(colors2[i])
        else:
            x2=pd.DataFrame(); y2=pd.DataFrame()
            if not cumulative: TXXT2.set_text('NATUREBATCH_')

        if cumulative:
            ax.scatter(x1,y1,color=colors1[i],label='Magnitude')
            ax.scatter(x2,y2,color=colors2[i],label='Nature')
        else:
            sc.set_offsets(np.c_[pd.concat([x1,x2]),pd.concat([y1,y2])])
            sc.set_color(['k']*len(x1) + ['r']*len(x2)) 
            
        # ax.set_title(table_name1 + '+' + table_name2)
        
    ani = FuncAnimation(fig, animate, frames=FRAMES, fargs=(FrameStart, cumulative), interval=800, repeat_delay=100, repeat=True) 
    plt.show()
    if save_movie:
        if save_movie_path:
            ani.save(save_movie_path)
            # plt.close()
        else:
            print('Please provide the save_movie_path.')
    else:
        plt.show()
    return ani
##### =========================================================================
def CreateMovieBatchSamples(AL_dbname,PlotFEATURES,TargetTablePattern='BATCH',natoms=216,
                            ShowBatch=False,ShowbatchInteractiveMovie=True,
                            cumulative=True,NoColor=False,xlabel=None,ylabel=None,
                            xLimit=None, yLimit=None,savemovie=False,save_movie_path='.',MovieName='BatchSamples.mp4'):
    
    savemoviedir = save_movie_path + '/' + MovieName
    if ShowBatch:
        if ShowbatchInteractiveMovie:
            outer_ani = ShowbatchInteractive(AL_dbname, PlotFEATURES, TargetTablePattern=TargetTablePattern,natoms=natoms, 
                                             cumulative=cumulative, NoColor=NoColor,
                                             save_movie=savemovie, save_movie_path=savemoviedir, 
                                             xLimit=xLimit, yLimit=yLimit, xlabel=xlabel, ylabel=ylabel)   
        else:
            BatchPlotGrabFrame(AL_dbname, PlotFEATURES, natoms=natoms, TargetTablePattern=TargetTablePattern,
                               cumulative=cumulative, NoColor=NoColor,save_movie_path=savemoviedir, 
                               xLimit=xLimit, yLimit=yLimit, xlabel=xlabel, ylabel=ylabel)
#%%----------------------------------------------------------------------------
##### =========================================================================
def PlotBatchDataTernary(AL_dbname,PlotFEATURES,xLimit=None,yLimit=None,xaxis_label=None,yaxis_label=None,
                         save=False,savepath='.',figname='TotalTrainigSamples.png'):
    conn = sq.connect(AL_dbname)
    ML_All_df = pd.read_sql_query('SELECT * FROM TotalBatchs', conn)
    ML_All_df['CONC'] = ML_All_df[PlotFEATURES[0]] / 216 * 100 # Convert atom number to concentration
    conn.close()
    PlotBatchData(ML_All_df,['CONC',PlotFEATURES[1]],title='AL final db samples', xLimit=xLimit, yLimit=yLimit,
                  xlabel=xaxis_label, ylabel=yaxis_label,save=save,savepath=savepath,figname=figname)

def PlotBatchSampleCount(AL_dbname,TargetTablePattern='BATCH',save=False,savepath='.',figname='BatchSampleCount.png'):
    conn = sq.connect(AL_dbname)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master  WHERE type='table';")
    tables = cursor.fetchall()
    BATCH_tables = [table[0] for table in tables if 'BATCH' in table[0]]
    
    N_samples = []; Loopindex = []
    for table_name in BATCH_tables:
        if table_name.startswith(TargetTablePattern):
            # print(table_name)
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            N_samples.append(cursor.fetchone()[0])
            Loopindex.append(int(table_name.split('_')[1]))
    conn.close()        
    Loopindex_sort_id = np.argsort(Loopindex)
    PlotBatchSampleNumber(np.array(Loopindex,dtype=int)[Loopindex_sort_id], np.array(N_samples)[Loopindex_sort_id], \
                          xlabel='AL loop index', ylabel='Feed-back batch size',title='Training set', base=1,
                          save=save,savepath=savepath,figname=figname)  
        
def PlotModelAccuaryTernary(AL_dbname,tablename='ModelAccuracy',ylabel=None, save=False,savepath='.',figname='ModelAccuracy.png'):
    conn = sq.connect(AL_dbname)
    Model_Accuracy_Convergence = pd.read_sql_query(f"SELECT * FROM {tablename}", conn, index_col='LoopIndex')
    conn.close()
    Model_Accuracy_Convergence['MEAN'] = Model_Accuracy_Convergence.mean(axis=1)
    Model_Accuracy_Convergence['STD'] = Model_Accuracy_Convergence.std(axis=1)
    PlotModelAccuary(Model_Accuracy_Convergence, ['MEAN', 'STD'], title='AL model convergency',
                     ylabel=ylabel, xlabel='AL loop index', base=1,
                     save=save,savepath=savepath,figname=figname)
##### =========================================================================        
def PlotALbandgapMagnitudeFeatures(X_predict,AL_dbname,PlotFEATURES,xLimit=None,yLimit=None,cbarlimit_=(None,None),
                                   xlabel=None,ylabel=None,SaveFigs=False,savepath='.',TargetTablePattern='BATCH',
                                   ModelAccuaryTable='ModelAccuracy'):
    '''
    # PredictMeanBandgap = mean_model(bandgap_predict) # for each sample
    # PredictSTDBandgap = std_model(bandgap_predict) # for each sample
    # BandgapError = PredictMeanBandgap - DFTbandgap # for each sample
    '''
    print("\tAL prediction Eg = mean_model(bandgap_predict); for each sample")
    print("\tSTDmodel(Eg predict) = std_model(bandgap_predict); for each sample")
    print("\tΔEg = [mean_model(bandgap_predict) - DFT bandgap]; for each sample")
    PlotRawDataTernary(X_predict, PlotFEATURES[:2] + ['PredictMeanBandgap'], title='AL prediction',xLimit=xLimit, yLimit=yLimit,
                       xlabel=xlabel, ylabel=ylabel, cbarlabel='E$_{\mathrm{g}}$ (eV)', vmin=cbarlimit_[0],vmax=cbarlimit_[1],
                       save=SaveFigs, savepath=savepath,figname='EgPrediction.png')
    
    PlotRawDataTernary(X_predict, PlotFEATURES[:2] + ['PredictSTDBandgap'], xLimit=xLimit, yLimit=yLimit, 
                       title='AL prediction', xlabel=xlabel, ylabel=ylabel, 
                       cbarlabel='STD$_\mathrm{model}$(E$_{\mathrm{g}}^{\mathrm{predict}}$)', 
                       save=SaveFigs, savepath=savepath,figname='EgPredictionSTD.png')
    
    PlotRawDataTernary(X_predict, PlotFEATURES[:2] + ['BandgapError'], xLimit=xLimit, yLimit=yLimit, 
                       title='$\Delta$E$_{\mathrm{g}}$ = Prediction - DFT', xlabel=xlabel, ylabel=ylabel, 
                       cbarlabel='$\Delta$E$_{\mathrm{g}}$ (eV)', 
                       save=SaveFigs, savepath=savepath,figname='EgPredictionError.png') #vmin=cbarlimit_[0],vmax=cbarlimit_[1],
    
    
    
    ##### ================= Plot True and predicted value =====================
    plot_test_results(X_predict['TrueBandgap'], X_predict['PredictMeanBandgap'], text='Bandgap (eV)',
                      save=SaveFigs, savepath=savepath)
    
    ##### ==================== Plot STD over models ===========================
    
    
    ##### =========== Plot model accuracy over active learning loop ===========
    '''
    # MAE out_sample = mean_model(MAE_over_all_out_samples)
    '''
    print('\tMAE out_sample = mean_model(MAE_over_all_out_samples)')
    PlotModelAccuaryTernary(AL_dbname,tablename=ModelAccuaryTable,ylabel='MAE out_sample (eV)',save=SaveFigs,savepath=savepath)

    ##### ============ Plot the all samples used in learning ==================
    if not TargetTablePattern == 'MAGNITUDEBATCH':
        PlotBatchDataTernary(AL_dbname,PlotFEATURES,xLimit=xLimit, yLimit=yLimit,
                             xaxis_label=xlabel, yaxis_label=ylabel,
                             save=SaveFigs,savepath=savepath)
  
    ##### ====================== Plot batche sample count =====================
    PlotBatchSampleCount(AL_dbname,TargetTablePattern=TargetTablePattern,save=SaveFigs,savepath=savepath)
    
def PlotALbandgapNatureFeaturesP1(X_predict,PlotFEATURES,xLimit=None,yLimit=None,
                                  xlabel=None,ylabel=None,SaveFigs=False,savepath='.'):
    '''
    # TrueBandgapNature = DFTbandgap nature
    # PredictBandgapNature = mode_model(bandgapnature_prediction) # for each sample 
    # NatureAccuracyTag = accuracy_score(nature_prediction_mode_models,nature_prediction_model) # for each sample    
    # NatureAccuracyWRTtrue = accuracy_score(DFTnature,nature_prediction_model) # for each sample
    '''
    print('\tAL prediction nature = mode_model(bandgapnature_prediction_per_model); for each sample')
    print('\tAccuracy = accuracy_score(DFTnature,nature_prediction_models); for each sample')
    print('\tAccuracy = accuracy_score(nature_prediction_mode_models,nature_prediction_model) # for each sample')
    PlotRawDataTernary(X_predict, PlotFEATURES[:2] + ['TrueBandgapNature'], title='DFT nature',xLimit=xLimit, yLimit=yLimit,
                       xlabel=xlabel,ylabel=ylabel,nature=True,save=SaveFigs, savepath=savepath,figname='DFTNature.png') 
    PlotRawDataTernary(X_predict, PlotFEATURES[:2] + ['PredictBandgapNature'], title='AL prediction nature',xLimit=xLimit, yLimit=yLimit,
                       xlabel=xlabel,ylabel=ylabel,nature=True,save=SaveFigs, savepath=savepath,figname='ALpredictionNature.png')
    PlotRawDataTernary(X_predict, PlotFEATURES[:2] + ['NatureAccuracyTag'], title='AL prediction nature',xLimit=xLimit, yLimit=yLimit,
                       xlabel=xlabel,ylabel=ylabel,cbarlabel='Accuracy',vmin=0,vmax=1,
                       save=SaveFigs, savepath=savepath,figname='ALpredictionNatureAccuracy.png')
    PlotRawDataTernary(X_predict, PlotFEATURES[:2] + ['NatureAccuracyWRTtrue'], title='Compare prediction nature',xLimit=xLimit, yLimit=yLimit,
                       xlabel=xlabel,ylabel=ylabel,cbarlabel='Accuracy w.r.t true value',vmin=0,vmax=1,
                       save=SaveFigs, savepath=savepath,figname='PredictionNatureAccuracyWRTtrue.png')

def PlotALbandgapNatureFeatures(X_predict,AL_dbname,PlotFEATURES,xLimit=None,yLimit=None,
                                xlabel=None,ylabel=None,SaveFigs=False,savepath='.',
                                TargetTablePattern='BATCH',ModelAccuaryTable='ModelAccuracy'):
    PlotALbandgapNatureFeaturesP1(X_predict,PlotFEATURES,xLimit=xLimit,yLimit=yLimit,xlabel=xlabel,ylabel=ylabel,SaveFigs=SaveFigs,savepath=savepath)
    ##### =========== Plot model accuracy over active learning loop ===========
    '''
    # Accuracy out_sample = mean_model(accuracy_score_over_all_out_sample)
    '''
    print('\tAccuracy out_sample = mean_model(accuracy_score_over_all_out_sample)')
    PlotModelAccuaryTernary(AL_dbname,tablename=ModelAccuaryTable,ylabel='Accuracy out_sample',save=SaveFigs,savepath=savepath)
    
    ##### ============ Plot the all samples used in learning ==================
    if not TargetTablePattern == 'NATUREBATCH':
        PlotBatchDataTernary(AL_dbname,PlotFEATURES,xLimit=xLimit, yLimit=yLimit,
                             xaxis_label=xlabel, yaxis_label=ylabel,
                             save=SaveFigs,savepath=savepath)
  
    ##### ====================== Plot batche sample count =====================
    PlotBatchSampleCount(AL_dbname,TargetTablePattern=TargetTablePattern,save=SaveFigs,savepath=savepath)
    
def PlotALbandgapMagNatureFeatures(X_predict,AL_dbname,PlotFEATURES,xLimit=None,yLimit=None,cbarlimit_=(None,None),
                                   xlabel=None,ylabel=None,SaveFigs=False,savepath1='.',savepath2='.',savepath3='.'):
    PlotALbandgapMagnitudeFeatures(X_predict,AL_dbname,PlotFEATURES,xLimit=xLimit, yLimit=yLimit,TargetTablePattern='MAGNITUDEBATCH',
                                   xlabel=xlabel, ylabel=ylabel,cbarlimit_=cbarlimit_,SaveFigs=SaveFigs, savepath=savepath1)
    PlotALbandgapNatureFeatures(X_predict,AL_dbname,PlotFEATURES,xLimit=xLimit, yLimit=yLimit,
                                xlabel=xlabel, ylabel=ylabel,ModelAccuaryTable='NatureModelAccuracy',
                                SaveFigs=SaveFigs,savepath=savepath2,TargetTablePattern='NATUREBATCH')   
    
    ##### ************* Plot the all samples used in learning *****************
    PlotBatchDataTernary(AL_dbname,PlotFEATURES,xLimit=xLimit, yLimit=yLimit,
                         xaxis_label=xlabel, yaxis_label=ylabel,
                         save=SaveFigs,savepath=savepath3)
  
    ##### **************** Plot batche sample count ***************************
    PlotBatchSampleCount(AL_dbname,save=SaveFigs,savepath=savepath3)
##### =========================================================================