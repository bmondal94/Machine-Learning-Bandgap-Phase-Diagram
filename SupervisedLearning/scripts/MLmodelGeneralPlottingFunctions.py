#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 12:48:20 2021

@author: bmondal
"""
import numpy as np
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator,LogLocator,ScalarFormatter,LogFormatter,LogFormatterMathtext,LogFormatterExponent,LogFormatterSciNotation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.colors import Normalize
import scipy.interpolate as inpr

# import tensorflow_docs as tfdocs
# import tensorflow_docs.modeling
# import tensorflow_docs.plots

# from scipy.signal import savgol_filter

#%%
# params = {'legend.fontsize': 18,
#          'axes.labelsize': 24,
#          'axes.titlesize': 24,
#          'xtick.labelsize':24,
#          'ytick.labelsize': 24,
#          'errorbar.capsize':2}
# plt.rcParams.update(params)
plt.rc('font', size=24)  
#%% ---------------------------------------------------------------------------
def plot_loss(history):
    plt.figure()
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    #plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    
def plot_metrices(history, metrix_name, tn=1, nrows=1, ncols=1, index=1):
    plt.subplot(nrows, ncols, index)
    plt.title(metrix_name)
    plt.plot(history.history[metrix_name], label='train')
    plt.plot(history.history['val_'+metrix_name], label='test')
    #plt.ylim([0, 10])
    if index > (tn-ncols):
        plt.xlabel('Epoch')
    if index%ncols == 1:
        plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    
def PlotMatrices(metrices, history, nrows=2):   
    TotalPlotN = len(metrices)     
    ncols = (TotalPlotN//nrows) + (TotalPlotN%nrows)
    fig = plt.figure(figsize=(12,7),constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=4 / 72, hspace=0.05, wspace=0.05)
    for i, I in enumerate(metrices):
        plot_metrices(history, I, tn=TotalPlotN, nrows=nrows, ncols=ncols, index=(i+1))
    return 

def PlotStrainBandgap2Dplot(XX,YY,save=False, savepath='.', figname='EaxmpleBandgapStrain2dMap.png'):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_xlabel('Strain (%)')
    ax.set_ylabel('E$_{\mathrm{g}}$ (eV)')
    ax.plot(XX, YY, 'o-',c='k')
    ax.axvline(x=0,ls='--',color='k')
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis='both',which='major',length=10,width=2)
    ax.tick_params(axis='both',which='minor',length=6,width=2)
    plt.tight_layout()
    if save:
        plt.savefig(savepath+'/'+figname,bbox_inches = 'tight',dpi=300)
        plt.close()
    else:
        plt.show()
    return fig, ax
    
def plot_test_results(XX, YY, text=None, my_color=None, tn=1, nrows=1, ncols=1, ShowLegend=True,
                      index=1, save=False, savepath='.', figname='TruePrediction.png',marker=None,
                      data_unit_label='eV',xlabel_text="True values",ylabel_txt="Predictions"):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(nrows, ncols, index, aspect='equal')
    plt.scatter(XX, YY, c=my_color,marker=marker)
    plt.title(text)
    if index > (tn-ncols):
        plt.xlabel(f"{xlabel_text} ({data_unit_label})")
    if index%ncols == 1 or ncols==1:
        plt.ylabel(f"{ylabel_txt} ({data_unit_label})")
    p1 = max(max(YY), max(XX)) #max(XX)
    p2 = min(min(YY), min(XX)) #min(XX)
    plt.plot([p1, p2], [p1, p2], 'k-')
    plt.xlim(p2, p1)
    plt.ylim(p2,p1)
    # plt.ylim(min(min(YY), min(XX)), max(max(YY),max(XX)))
    # DIFF_ = np.sqrt(np.sum((XX - YY)**2)/len(XX))
    # ax.plot([],[],' ',label=f"Max error = {max(DIFF_):.2f} eV \n {'MAE':>11} = {np.mean(DIFF_):.2f}±{np.std(DIFF_):.2f} eV")
    # plt.plot([],[],' ',label=f"RMSE = {DIFF_:.2f} eV")
    # if ShowLegend: plt.legend(handlelength=0)
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.4))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.4))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.gca().tick_params(axis='both',which='major',length=10,width=2)
    plt.gca().tick_params(axis='both',which='minor',length=6,width=2)
    plt.tight_layout()
    if save:
        plt.savefig(savepath+'/'+figname,bbox_inches = 'tight',dpi=300)
        plt.close()
    else:
        plt.show()
    return ax

def plot_test_results_separateExp_Qua(XX, YY, ax=None, lm1=None, lm2=None,
                                      fig=None,text=None, my_color=None, ShowLegend=True,LABEL=None,
                                      save=False, savepath='.', figname='TruePrediction.png',marker=None,
                                      data_unit_label='eV',xlabel_text="True values",ylabel_txt="Predictions"):
    if fig is None:
        fig = plt.figure(figsize=(8, 8))
    if ax is None:
        ax = fig.add_subplot(1, 1, 1, aspect='equal')
        ax.set_title(text)
        ax.set_xlabel(f"{xlabel_text} ({data_unit_label})")
        ax.set_ylabel(f"{ylabel_txt} ({data_unit_label})")
        
    ax.plot(XX, YY, c=my_color,lw=2, label=LABEL) #,marker=marker
    p1 = max(max(YY), max(XX)) #max(XX)
    p2 = min(min(YY), min(XX)) #min(XX)
    if (lm1 is None) or (p1>lm1):lm1 = p1
    if (lm2 is None) or (p2<lm2):lm2 = p2
    ax.set_xlim(lm2, lm1)
    ax.set_ylim(lm2, lm1)
    ax.plot([lm1, lm2], [lm1, lm2], 'k-')
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.tick_params(axis='both',which='major',length=10,width=2)
    ax.tick_params(axis='both',which='minor',length=6,width=2)
    plt.tight_layout()
    if ShowLegend: ax.legend()
    if save:
        # if ShowLegend: plt.legend()
        plt.savefig(savepath+'/'+figname,bbox_inches = 'tight',dpi=300)
        plt.close()
    else:
        plt.show()
    return fig, ax, lm1, lm2
    
def PlotTruePredictionCorrelation(Y_test, Y_pred, ttext, nrows=3):
    nproperties = Y_test.shape[1]
    ncols = (nproperties//nrows) + (nproperties%nrows)
    plt.figure(figsize=(12,7),constrained_layout=True)
    for I in range(nproperties):    
        plot_test_results(Y_test[:,I], Y_pred[:,I], ttext[I],\
                          tn=nproperties, nrows=nrows, ncols=ncols, index=(I+1))
    
def plot_err_dist(XX, YY, text, tn=1, nrows=1, ncols=1, index=1,data_unit_label='eV',save=False, savepath='.', figname='TruePredictErrorHist.png'):
    # Check error distribution
    plt.subplot(nrows, ncols, index)
    plt.title(text)
    error = XX - YY
    plt.hist(error, bins=25)
    plt.gca().tick_params(axis='both',which='major',length=10,width=2)
    plt.gca().tick_params(axis='both',which='minor',length=6,width=2)
    if index > (tn-ncols):
        plt.xlabel(f'Prediction error ({data_unit_label})')
    if index%ncols == 1:
        plt.ylabel('Count (arb.)')
    if save:
        plt.savefig(savepath+'/'+figname,bbox_inches = 'tight',dpi=300)
        plt.close()
    else:
        plt.show()
    return 

def PlotPredictionStdFullSpace(X, text=None, nbins=10, data_unit_label=' (eV)',save=False, savepath='.', figname='PredictAccuracyHistFullSpace.png',
                               xlabel='Prediction STD',PlotLogScale=True,y_minor_locator=1e5,x_major_locator=10):
    fig, ax = plt.subplots()
    ax.hist(X, bins=nbins, lw=1) #, ec="yellow", fc="green", alpha=0.5)
    ax.set_xlabel(f'{xlabel}{data_unit_label}')
    ax.set_ylabel('Count (arb.)')
    ax.set_title(None)
    ax.tick_params(axis='both',which='major',length=10,width=2)
    ax.tick_params(axis='both',which='minor',length=6,width=2)
    if PlotLogScale: 
        ax.set_yscale('log')
    else:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3,2))
        ax.yaxis.set_major_formatter(formatter)
        # ax.yaxis.set_minor_locator(MultipleLocator(0.25*1e4))
        ax.yaxis.set_minor_locator(MultipleLocator(y_minor_locator))

    if x_major_locator is not None: ax.xaxis.set_major_locator(MultipleLocator(x_major_locator))
    
    if save:
        plt.savefig(savepath+'/'+figname,bbox_inches = 'tight',dpi=300)
        plt.close()
    else:
        plt.show()
    return 
    
def PlotTruePredictionErrorDistribution(Y_test, Y_pred, ttext, nrows=3):
    nproperties = Y_test.shape[1]
    ncols = (nproperties//nrows) + (nproperties%nrows)
    plt.figure(figsize=(12,7),constrained_layout=True)
    for I in range(nproperties):    
        plot_err_dist(Y_test[:,I], Y_pred[:,I], ttext[I],\
                          tn=nproperties, nrows=nrows, ncols=ncols, index=(I+1))


#%%
def plot_true_predict_results(XX, YY, ax=None, my_color=None, save=False, CreateHist=False, savehist=False, savepath='.', marker=None, ShowLegend=True,
                              text=None, tn=1, nrows=1, ncols=1, index=1,figname='TruePrediction.png', fignameHist='TruePredictErrorHist.png'):
    if ax is None:
        if CreateHist:
            plt.figure()
            plot_err_dist(XX, YY, text=text, tn=tn, nrows=nrows, ncols=ncols, index=index,save=savehist, savepath=savepath,figname=fignameHist)
        # plt.figure()
        ax = plot_test_results(XX, YY, text=text, my_color=my_color, tn=tn, nrows=nrows, ncols=ncols,marker=marker, 
                               index=index,save=save, savepath=savepath,figname=figname,ShowLegend=ShowLegend)
        return ax
    

def PlotPostProcessingDataSetSizeV0(pp,save=False, savepath='.'):      
    ProjectionDict = {'root_mean_squared_error':'RMSE (meV)','r2_score':'$\mathrm{R}^2$','mean_absolute_error':'MAE (meV)',\
                      'max_error':'Max error (meV)','accuracy_score':'Accuracy','balanced_accuracy_score': 'Balanced accuracy'}
    AllColumns = list(pp.columns)[1:]
    
    pp_mean = pp.groupby('dataset_size', as_index=False).mean(numeric_only=True) 
    pp_std = pp.groupby('dataset_size', as_index=False).std(numeric_only=True) 
    # print("\nCreating figures:")
    for II in AllColumns:
        print(f"\t{II}")
        WhichMetric = ProjectionDict.get(II.split('_',1)[-1],None)
        fig, ax = plt.subplots()
        ax.set_xlabel('Training set size')   
        ax.set_ylabel(WhichMetric)    
        ax.errorbar(pp_mean['dataset_size'],pp_mean[II],yerr=pp_std[II],linestyle='-',color='k')  
        plt.tight_layout()
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if save:
            plt.savefig(savepath+'/'+II+'.png',bbox_inches = 'tight',dpi=300)
            plt.close()
        else:
            plt.show()
    return ax

def PlotPostProcessingDataSetSize(pp,save=False, savepath='.',ProjectionDict=None,figformat='png'): 
    if ProjectionDict is None:
        ProjectionDict = {'set1':{'root_mean_squared_error':'RMSE (meV)','r2_score':'$\mathrm{R}^2$'},'set2':{'mean_absolute_error':'MAE (meV)'},\
                          'set3':{'max_error':'Max error (meV)'},'set4':{'accuracy_score':'Accuracy score','balanced_accuracy_score': 'Balanced accuracy score'},\
                              'set5':{'accuracy_score':'Accuracy score'},'set6':{'r2_score':'$\mathrm{R}^2$'},'set7':{'balanced_accuracy_score': 'Balanced accuracy score'},
                              'set8':{'root_mean_squared_error':'RMSE (meV)'}}
        # ProjectionDict = {'set1':{'root_mean_squared_error':'RMSE (meV)'},'set2':{'mean_absolute_error':'MAE (meV)'},\
        #                   'set3':{'max_error':'Max error (meV)'},\
        #                       'set5':{'accuracy_score':'Accuracy score'},'set6':{'r2_score':'$\mathrm{R}^2$'},'set7':{'balanced_accuracy_score': 'Balanced accuracy score'},
        #                       }
    elif isinstance(ProjectionDict,dict):
        pass
    else:
        print("ProjectionDict must be a dictionary.")
        
    AllColumns = list(pp.columns)[1:]
    pp_mean = pp.groupby('dataset_size', as_index=False).mean(numeric_only=True) 
    pp_std = pp.groupby('dataset_size', as_index=False).std(numeric_only=True) 
    # print("\nCreating figures:")
    for LL in ['out-of-sample','all-sample']:
        for whichset, I in ProjectionDict.items():
            WhichMetric = list(I.keys())
            II = f'{LL}_{WhichMetric[0]}'
            if II in AllColumns:
                fig, ax = plt.subplots()
                ax.set_xlabel('Training set size')
                ax.set_ylabel(ProjectionDict[whichset][WhichMetric[0]])    
                ax.errorbar(pp_mean['dataset_size'],pp_mean[II],yerr=pp_std[II],marker='o',linestyle='-',color='k')
                if WhichMetric[0] in ['root_mean_squared_error','mean_absolute_error','max_error']:
                    PrecissionStr = '%.1f' 
                    ax.yaxis.set_minor_locator(MultipleLocator(5))
                else:
                    PrecissionStr = '%.2f'
                    if WhichMetric[0] == 'r2_score': 
                        PrecissionStr = '%.3f'
                    else:
                        ax.yaxis.set_major_locator(MultipleLocator(0.02))
                        ax.yaxis.set_minor_locator(MultipleLocator(0.01))
                    
                    ax.yaxis.set_major_formatter(FormatStrFormatter(PrecissionStr))
                if len(WhichMetric) == 2:
                    II = f'{LL}_{WhichMetric[1]}'
                    if II in AllColumns:
                        ax2 = ax.twinx()
                        ax2.set_ylabel(ProjectionDict[whichset][WhichMetric[1]])  
                        ax2.errorbar(pp_mean['dataset_size'],pp_mean[II],yerr=pp_std[II],marker='o',linestyle='-',color='tab:blue')  
                        ax2.tick_params(axis='y', labelcolor='tab:blue')
                        if WhichMetric[1] in ['root_mean_squared_error','mean_absolute_error','max_error']:
                            PrecissionStr = '%.1f' 
                        else:
                            PrecissionStr = '%.2f'
                            if WhichMetric[1] == 'r2_score': 
                                PrecissionStr = '%.3f'
                            else:
                                ax2.yaxis.set_major_locator(MultipleLocator(0.02))
                                ax2.yaxis.set_minor_locator(MultipleLocator(0.01))
                        ax2.yaxis.set_major_formatter(FormatStrFormatter(PrecissionStr))
                        ax2.tick_params(axis='both',which='major',length=10,width=2)
                        ax2.tick_params(axis='both',which='minor',length=6,width=2)
                ax.xaxis.set_minor_locator(MultipleLocator(100))
                ax.tick_params(axis='both',which='major',length=10,width=2)
                ax.tick_params(axis='both',which='minor',length=6,width=2)
                if save:
                    SaveFileName = f"{savepath}/{LL}_{'-'.join(WhichMetric)}.{figformat}"
                    plt.savefig(SaveFileName,bbox_inches = 'tight',dpi=300)
                    plt.close()
                else:
                    plt.show()
    return 

def PlotPostProcessingDataSetSize_special(pp,save=False, ax=None, fig=None, savepath='.',ProjectionDict=None, 
                                          ax_y_precision=None,ax_yminortick_multi=5,ax_plot_color='k',figformat='png'): 
    if ProjectionDict is None:
        ProjectionDict = {'set1':{'root_mean_squared_error':'RMSE (meV)','accuracy_score':'Accuracy score'}}
    elif isinstance(ProjectionDict,dict):
        pass
    else:
        print("ProjectionDict must be a dictionary.")
        
    AllColumns = list(pp.columns)[1:]
    pp_mean = pp.groupby('dataset_size', as_index=False).mean(numeric_only=True) 
    pp_std = pp.groupby('dataset_size', as_index=False).std(numeric_only=True) 
    # print("\nCreating figures:")
    for LL in ['out-of-sample']:
        for whichset, I in ProjectionDict.items():
            WhichMetric = list(I.keys())
            II = f'{LL}_{WhichMetric[0]}'
            if II in AllColumns:
                if ax is None:
                    fig, ax = plt.subplots()
                ax.set_xlabel('Training set size')
                ax.set_ylabel(ProjectionDict[whichset][WhichMetric[0]])    
                ax.errorbar(pp_mean['dataset_size'],pp_mean[II],yerr=pp_std[II],marker='o',linestyle='-',color=ax_plot_color)
                # PrecissionStr = '%.1f' if WhichMetric[0] in ['root_mean_squared_error','mean_absolute_error','max_error'] else '%.3f'
                if ax_y_precision is not None and isinstance(ax_y_precision,str):
                    ax.yaxis.set_major_formatter(FormatStrFormatter(ax_y_precision))
                if len(WhichMetric) == 2:
                    II = f'{LL}_{WhichMetric[1]}'
                    if II in AllColumns:
                        ax2 = ax.twinx()
                        ax2.set_ylabel(ProjectionDict[whichset][WhichMetric[1]])  
                        ax2.errorbar(pp_mean['dataset_size'],pp_mean[II],yerr=pp_std[II],marker='o',linestyle='-',color='tab:blue')  
                        ax2.tick_params(axis='y', labelcolor='tab:blue')
                        # ax2.tick_params(axis='both', length=2,width=3)
                        PrecissionStr = '%.1f' if WhichMetric[1] in ['root_mean_squared_error','mean_absolute_error','max_error'] else '%.2f'
                        ax2.yaxis.set_major_formatter(FormatStrFormatter(PrecissionStr))
                        ax2.yaxis.set_minor_locator(MultipleLocator(0.02))
                        ax2.yaxis.set_major_locator(MultipleLocator(0.04))
                        ax2.tick_params(axis='both',which='major',length=10,width=2)
                        ax2.tick_params(axis='both',which='minor',length=6,width=2)
                ax.xaxis.set_minor_locator(MultipleLocator(100))
                ax.yaxis.set_minor_locator(MultipleLocator(ax_yminortick_multi))
                # ax.yaxis.set_major_locator(MultipleLocator(10))
                ax.tick_params(axis='both',which='major',length=10,width=2)
                ax.tick_params(axis='both',which='minor',length=6,width=2)
                if save:
                    SaveFileName = f"{savepath}/{LL}_{'-'.join(WhichMetric)}.{figformat}"
                    plt.savefig(SaveFileName,bbox_inches = 'tight',dpi=300)
                    plt.close()
                else:
                    pass
                    #plt.show()
    return fig, ax

class OOMFormatter(ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format

def PlotPostProcessingDataSetSizeLogLog(pp,save=False, savepath='.',figformat='png'):      
    ProjectionDict = {'set1':{'root_mean_squared_error':'RMSE (meV)'},'set2':{'mean_absolute_error':'MAE (meV)'},\
                          'set3':{'accuracy_score':'Accuracy-score'}}
    # ProjectionDict = {'set1':{'root_mean_squared_error':'RMSE (meV)'}}
    AllColumns = list(pp.columns)[1:]
    pp_mean = pp.groupby('dataset_size', as_index=False).mean(numeric_only=True) 
    pp_std = pp.groupby('dataset_size', as_index=False).std(numeric_only=True) 
    # print("\nCreating figures:")
    for LL in ['out-of-sample','all-sample']:
        for whichset, I in ProjectionDict.items():
            WhichMetric = list(I.keys())
            II = f'{LL}_{WhichMetric[0]}'
            if II in AllColumns:
                fig, ax = plt.subplots()
                ax.set_xlabel('Training set size')
                ax.set_ylabel(ProjectionDict[whichset][WhichMetric[0]])  
                ax.set_xscale('log', nonpositive='clip')
                ax.set_yscale('log', nonpositive='clip')
                ax.errorbar(pp_mean['dataset_size'],pp_mean[II],yerr=pp_std[II],marker='o',linestyle='-',color='k')
                ax.tick_params(axis='both',which='major',length=10,width=2)
                ax.tick_params(axis='both',which='minor',length=6,width=2)
                # y_major = LogLocator(base = 10.0, numticks = 100)
                # ax.yaxis.set_minor_locator(y_major)
                # ax.yaxis.set_minor_formatter(LogFormatterSciNotation(minor_thresholds=(1,10)))
                ax.yaxis.set_minor_formatter(FormatStrFormatter('%.d'))

                # ax.yaxis.set_minor_formatter(LogFormatter(minor_thresholds=(1,10)))
                if len(WhichMetric) == 2:
                    II = f'{LL}_{WhichMetric[1]}'
                    if II in AllColumns:
                        ax2 = ax.twinx()
                        ax2.set_ylabel(ProjectionDict[whichset][WhichMetric[1]])  
                        ax2.set_xscale('log', nonpositive='clip')
                        ax2.set_yscale('log', nonpositive='clip')
                        ax2.errorbar(pp_mean['dataset_size'],pp_mean[II],yerr=pp_std[II],marker='o',linestyle='-',color='tab:blue')  
                        ax2.tick_params(axis='y', which='both', labelcolor='tab:blue')
                        ax2.tick_params(axis='both',which='major',length=10,width=2)
                        ax2.tick_params(axis='both',which='minor',length=6,width=2)
                if save:
                    SaveFileName = f"{savepath}/{LL}_{'-'.join(WhichMetric)}_LogLog.{figformat}"
                    print(f'\t{SaveFileName}')
                    plt.savefig(SaveFileName,bbox_inches = 'tight',dpi=300)
                    plt.close()
                else:
                    plt.show()
    return 

def PlotPostProcessingDataSetSizeLogLog_v2(pp,save=False, savepath='.',figformat='png'):      
    ProjectionDict = {'set1':{'root_mean_squared_error':'log$_{10}$(RMSE) (meV)'}}
    pp_mean = pp.groupby('dataset_size', as_index=False).mean(numeric_only=True) 
    pp_std = pp.groupby('dataset_size', as_index=False).std(numeric_only=True) 
    AllColumns = list(pp.columns)[1:]
    # print("\nCreating figures:")
    for LL in ['out-of-sample','all-sample']:
        for whichset, I in ProjectionDict.items():
            WhichMetric = list(I.keys())
            II = f'{LL}_{WhichMetric[0]}'
            if II in AllColumns:
                fig, ax = plt.subplots()
                ax.set_xlabel('log$_{10}$(Training set size)')
                ax.set_ylabel(ProjectionDict[whichset][WhichMetric[0]])  
                ax.errorbar(np.log10(pp_mean['dataset_size']),np.log10(pp_mean[II]),yerr=0,linestyle='-',color='k')
                ax.tick_params(axis='both',which='major',length=10,width=2)
                ax.tick_params(axis='both',which='minor',length=6,width=2)
                ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            if save:
                SaveFileName = f"{savepath}/{LL}_{'-'.join(WhichMetric)}_LogLog_v2.{figformat}"
                print(f'\t{SaveFileName}')
                plt.savefig(SaveFileName,bbox_inches = 'tight',dpi=300)
                plt.close()
            else:
                plt.show()
    return 
