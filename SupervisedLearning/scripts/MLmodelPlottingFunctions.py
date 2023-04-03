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
import ternary
from matplotlib import animation
from matplotlib.colors import Normalize

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

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
    error = YY - XX
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
    
def PlotTruePredictionErrorDistribution(Y_test, Y_pred, ttext, nrows=3):
    nproperties = Y_test.shape[1]
    ncols = (nproperties//nrows) + (nproperties%nrows)
    plt.figure(figsize=(12,7),constrained_layout=True)
    for I in range(nproperties):    
        plot_err_dist(Y_test[:,I], Y_pred[:,I], ttext[I],\
                          tn=nproperties, nrows=nrows, ncols=ncols, index=(I+1))

#%% ---------------------------------------------------------------------------
def DataConversion(X,Y):
    """
    This function converts the cartesian x,y coordinate to ternary coordinate.

    Parameters
    ----------
    X : numpy float array
        The x coordinate array.
    Y : numpy float array
        The y coordinate array.

    Returns
    -------
    x : numpy float array
        The x coordinate array after ternary conversion.
    y : numpy float array
        The y coordinate array after ternary conversion.

    """
    x = X + Y*0.5
    y = 0.8660254037844386 * Y
    return x, y

def HideLabels(ax):
    """
    This function haides the matplotlib axis ticks.

    Parameters
    ----------
    ax : matplotlib axis
        The axis object on which the operations will be applied.

    Returns
    -------
    ax : matplotlib axis
         Matplotlib axis after getting rid of panes, spines and ticks. The
         z-axis spines and ticks are kept.

    """
    # Get rid of the panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    # Get rid of the spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    #ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    
    # Get rid of the ticks
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    #ax.set_zticks([])
    return ax

def AxisLabels(ax, textt, scale, fontsize=24):
    """
    This function add the axis labels in ternary plot.

    Parameters
    ----------
    ax : matplotib axis
        The axis object on which the labels will be added.
    textt : string list
        The list of 3 label strings.
    scale : float/int
        The scale of the ternary plot.
    fontsize : matplotlib text fontsize object, optional
        The fontsize of the labels. The default is 24.

    Returns
    -------
    ax : matplotlib axis

    """
    textpos = np.array([[-0.2,0],[1,0],[0.5,0.866]])*scale
    for i, I in enumerate(textpos):
        ax.text(I[0],I[1],0, textt[i], fontsize=fontsize)
    return ax

def Plot3DBandgapTernary(X,Y,Z,c,textt=['a','b','c'], scale=1, zaxis_label=None,
                         label_fontsize=16,cbarlabel_fontsize=16,cbar_txt = 'Strain(%)',
                         ax=None, fig=None, ShowColorbar=True,titletxt=None):   
    """
    This function plots the bandgap scatter plot in 3D.

    Parameters
    ----------
    X : Float numpy array
        The x coordinate array.
    Y : Float numpy array
        The y coordinate array.
    Z : Float numpy array
        The z coordinate array.
    c : Float numpy array
        The color array. The magnitude of bandgap array.
    textt : String list, optional
        The list of axis labels. The default is ['a','b','c']. The order is 
        [left corner, right corner, top corner]
    scale : float/int, optional
        The composition scale of the ternary plot. The default is 1.
    label_fontsize : matplotlib text fontsize, optional
        The fontsize of the labels. The default is 16.
    cbarlabel_fontsize : matplotlib text fontsize, optional
        The fontsize of the colorbar. The default is 16.
    ax : matplotlib 3d axis object, optional
        Already created matplotlib 3d axis object. The default is None. 
    fig : matplotlib figure object, optional
        Already created matplotlib 3d figure object. The default is None.
    ShowColorbar : matplotlib colorbar object, optional
        If the colorbar will be shown. The default is True
    Returns
    -------
    None.

    """
    if ax is None:
        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlabel(zaxis_label)
        ax = HideLabels(ax)
        #ax.set_axis_off()
        ax.set_title(titletxt)
        ax = AxisLabels(ax, textt, scale=1, fontsize=label_fontsize)
    
    x,y = DataConversion(X,Y)
    x = x / scale
    y = y / scale 
    surf = ax.scatter(x,y,Z,c=c)
    if fig is None: ShowColorbar=False
    if ShowColorbar:
        cbar = fig.colorbar(surf, shrink=0.5) #, aspect=5)
        cbar.ax.set_ylabel(cbar_txt, fontsize = cbarlabel_fontsize)
    return fig, ax
    

def generate_heatmap_data(pp):
    """
    This function generates the ternary heatmap cpmpatible data format.

    Parameters
    ----------
    pp : pandas dataframe
        The dataframe with first 3 column as x,y,z coordinate and last column
        for color bar data (e.g. bandgap magnitude).

    Returns
    -------
    d : python dictionary
        Dictionary containg the data in ternary heatmap format.

    """
    d = {x[:3]:x[-1] for x in pp.itertuples(index=False)}
    return d

def generate_scatter_data(pp):
    d = [x[:3] for x in pp.itertuples(index=False)]
    return d

def color_map_color(value, cmap=plt.cm.get_cmap('viridis'), vmin=0, vmax=1):
    norm = Normalize(vmin=vmin, vmax=vmax)
    return cmap(norm(value))

def ConversionCartesian2Ternary(x,y, SQRT3o2 = 0.8660254037844386):
    XX = x + y*0.5
    YY = SQRT3o2 * y
    return XX, YY

def DrawBoundary(p, scale, z=0, color='gray', line_width=4):
    ax = np.array([0,scale,0,0])
    ay = np.array([0,0,scale,0])    
    XX, YY = ConversionCartesian2Ternary(ax,ay)
    p.plot(XX,YY, zs=z, color=color, lw=line_width)
    return  p


def DrawAllContour3D(cnt, textt=['a','b','c'], fname=None, titletext=None,
                 savefig=False, scale=1, vmin=-1, vmax=1,
                 cmap=plt.cm.get_cmap('viridis'),label_fontsize=16,
                 cbarlabel_fontsize=16, ScatterPlot=False):
    
    DATA = []
    for strain,contours in cnt.items():
        for contour in contours.allsegs:
            for seg in contour: 
                dd = np.insert(seg[:, 0:2],2, strain,axis=-1)
                DATA.append(dd)
                
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=90, azim=-90)
    ax.set_box_aspect(aspect = (1,0.866,1))
    ax.set_xlim(0, scale)
    ax.set_ylim(0, scale*0.8660254037844386)
    ax.set_zlim(vmin, vmax)
    ax.set_zlabel('Strain(%)')
    ax.set_axis_off()
    ax = AxisLabels(ax, textt, scale=scale, fontsize=label_fontsize)
    
    for s,_ in cnt.items():
        ax = DrawBoundary(ax, scale, s)

    if ScatterPlot:
        DATA = np.concatenate(DATA)
        x,y = DataConversion(DATA[:,0],DATA[:,1])
        if cmap is not None:
            surf = ax.scatter(x,y,DATA[:,2],c=DATA[:,2], cmap=cmap)
            cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
            cbar.ax.set_ylabel('Strain(%)', fontsize = cbarlabel_fontsize)
        else:
            surf = ax.scatter(x,y,DATA[:,2])
    else:
        norm=plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, shrink=0.5, aspect=10, ax=ax)
        cbar.ax.set_ylabel('Strain(%)', fontsize = cbarlabel_fontsize)
        for data in DATA:
            x,y = DataConversion(data[:,0],data[:,1])
            ax.plot(x,y,data[:,2],c=cmap(norm(data[0,2])))
    return fig, ax

def DrawAllContour(cnt,fname=None, titletext=None,
                   axislabels = ["A", "B", "C"],
                   savefig=False, scale=1, vmin=-1, vmax=1,
                   cmap=plt.cm.get_cmap('viridis'),
                   fontsize = 20, cbarpos = 'right'):
    axes_colors = {'b':'k','l':'k','r':'k'}
    #figure, tax = ternary.figure(scale=scale)
    figure, ax = plt.subplots()
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)
    if (cbarpos == 'right') or (cbarpos == 'left'):
        figure.set_size_inches(10, 8) 
    else:
        figure.set_size_inches(10, 10)
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    tax.boundary(linewidth=2.0, axes_colors={'b':'k','l':'k','r':'k'})
    
    if titletext is not None:
        tax.set_title(titletext, fontsize=24, loc='left')
    tax.left_axis_label(axislabels[0], offset=0.12, color=axes_colors['l'], fontsize=fontsize)
    tax.right_axis_label(axislabels[1], offset=0.13, color=axes_colors['r'], fontsize=fontsize)
    tax.bottom_axis_label(axislabels[2], offset=0.02, color=axes_colors['b'], fontsize=fontsize)
    
    tax.ticks(axis='blr', linewidth=2, clockwise=False, multiple=10,
              axes_colors=axes_colors, offset=0.02, tick_formats="%d",
              fontsize=20)
    
    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()
    for strain,contours in cnt.items():
        for contour in contours.allsegs:
            for seg in contour:
                tax.plot(seg[:, 0:2], color=cmap(norm(strain)),antialiased=True)
        
    tax.get_axes().set_aspect(1)
    ax.text(25,35,'DIRECT',fontsize=fontsize,rotation=56)
    ax.text(65,3,'INDIRECT',fontsize=fontsize,rotation=45)
    
    cb = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),ax=ax,
                        format='%.1f',shrink=0.9,anchor=(0.5,0.8), pad=0.01,location=cbarpos)
    cb.set_label(label='Strain(%)',size=fontsize)
    cb.ax.tick_params(labelsize=20,length=8,width=2)
    tax._redraw_labels()
    #plt.tight_layout()
    if savefig:
        plt.savefig(fname,bbox_inches = 'tight',dpi=300)
        plt.close(figure)
    else:
        tax.show()
        
    return figure

def center_of_mass(X):
    """
    This function calulates the cenroid of contour polygon.

    Parameters
    ----------
    X : 2d array
        The point coordinates of contours. 1st column is for x coordinate and
        2nd column for y coordinate.

    Returns
    -------
    numpy array
        x and y coordinate array (after ternary data conversion).

    """
    # https://en.wikipedia.org/wiki/Centroid#Centroid_of_a_polygon
    # calculate center of mass of a closed polygon
    x = X[:,0]
    y = X[:,1]
    g = (x[:-1]*y[1:] - x[1:]*y[:-1])
    #A = 0.5*g.sum() ; fact = 1./(6*A)
    #COM = 1./(6*A)*np.array([cx,cy])
    fact = 1./(3*g.sum())
    cx = (((x[:-1] + x[1:])*g).sum())*fact
    cy = (((y[:-1] + y[1:])*g).sum())*fact
    return DataConversion(cx, cy)

def DrawIndividualContour_and_Text(ternaryax, ax, contours, COMContourText, fontsize=20):
    for contour in contours.allsegs:
        for seg in contour:
            ternaryax.plot(seg[:, 0:2], color='k')
            COMtext = center_of_mass(seg[:, 0:2])
            ax.text(COMtext[0],COMtext[1],COMContourText,
                    ha="center", va="center",
                    fontsize=fontsize,color='k')
    return ternaryax, ax

def DrawIndividualContour(ternaryax, contours):
    for contour in contours.allsegs:
        for seg in contour:
            ternaryax.plot(seg[:, 0:2], color='k')
    return ternaryax
            
def DrawIndividualContourText(ax, contours, COMContourText, fontsize=20):
    for contour in contours.allsegs:
        for seg in contour:
            COMtext = center_of_mass(seg[:, 0:2])
            ax.text(COMtext[0],COMtext[1],COMContourText,
                    ha="center", va="center",
                    fontsize=fontsize,color='k')
    return ax

def DrawSnapshot(dd, fname=None, titletext=None, BandGapNatureHeatmap=False, contours=None,
                 UseContoursText=False, ContoursText=None, savefig=False, 
                 scale=1, titletext_loc='right',show_colorbar=True,
                 axislabels = ["A", "B", "C"],axiscolors=None,
                 axislabelcolors=None,COMContourText=['test'],
                 vmin=-1, vmax=1,cmap=plt.cm.get_cmap('viridis'),
                 fontsize = 20, cbarlabel=None,cbarpos='right',
                 RawData=None, RawDataColor=None, DrawRawData=False):
    """
    This function draw the bandgap(nature) heatmap and contours.

    Parameters
    ----------
    dd : dictionary
        Ternary plot data.
    fname : string/path, optional
        The filename or filepath to save the figure. The default is None.
    titletext : string, optional
        The title of the figure. The default is None.
    BandGapNatureHeatmap : Bool, optional
        If to plot bandgap nature heatmap. The default is False.
    contours : matplotlib contour object list, optional
        The countour object for contour plot. The default is None.
    UseContoursText : bool, optional
        If this is true then the contour texts will be annotated using the supplied
        contours list. If false the ContoursText list will be used for contour
        texts. The default is False. 
    ContoursText : matplotlib contour object list, optional
        The countour object for contour plot texts. The texts will be annotated
        at the centroid of countour polygons. The default is None.
    savefig : Bool, optional
        Save the figure. The default is False.
    scale : float/int, optional
        Scale of the ternary plot. The default is 1.
    axislabels : list, optional
        The list of axis labels. The default is ["A", "B", "C"]. Format [left, right, bottom].
    axiscolors : dictionary, optional
        The color of the axes, e.g {'b':'g','l':'r','r':'b'}. The default is None.
    axislabelcolors : dictionary, optional
        The color of the axes labels, e.g. {'b':'g','l':'r','r':'b'}. The default is None.
    COMContourText : string list, optional
        The text in the centroid of (anti) contour ploygons. The default is ['test'].
        Length of COMContourText should be >= len(contours) or len(ContoursText).
    vmin : float, optional
        The minima of color bar. The default is -1.
    vmax : float, optional
        Maxima of color bar. The default is 1.
    cmap : matplotlib color map, optional
        Color map for heat map. The default is plt.cm.get_cmap('viridis').
    fontsize : matplotlib fontsize, optional
        Fontsize for axislabels, COMContourText and colorbar label. The default is 20.
    cbarlabel : string, optional
        The latel text of the colorbar. The default is None.
    cbarpos : string, optional
        The position of the colorbar. The default is 'right.'
    RawData : dataframe, optional
        The data for DrawRawData. Default is None.
    RawDataColor : dataframe, optional
        The color data for DrawRawData. Default is None.    
    DrawRawData : bool, optional
        Whether to draw the raw data provided by 'RawData' dataframe. Default is 
        False. If RawData is None, DrawRawData will be set to False. The last 
        column of 'RawData' will be used for coloring. 
    Returns
    -------
    None.

    """
    # Dictionary of axes colors for bottom (b), left (l), right (r).
    if axislabelcolors is None: axislabelcolors={'b':'k','l':'k','r':'k'}
    
    figure, ax = plt.subplots()
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)
    #figure, tax = ternary.figure(scale=scale)
    if (cbarpos == 'right') or (cbarpos == 'left'):
        figure.set_size_inches(10, 8) 
    else:
        figure.set_size_inches(10, 10)
    
    tax.boundary(linewidth=2.0, axes_colors=None) 
    
    if titletext is not None:
        tax.set_title(titletext, fontsize=24, loc=titletext_loc)
        #ax.text(100,55,titletext, fontsize=24,  rotation=270)
    tax.left_axis_label(axislabels[0], offset=0.12, color=axislabelcolors['l'], fontsize=fontsize)
    tax.right_axis_label(axislabels[1], offset=0.13, color=axislabelcolors['r'], fontsize=fontsize)
    tax.bottom_axis_label(axislabels[2], offset=0.02, color=axislabelcolors['b'], fontsize=fontsize)
    
    # tax.gridlines(multiple=10, linewidth=2,
    #               horizontal_kwargs={'color': axislabelcolors['r']},
    #               left_kwargs={'color': axislabelcolors['b']},
    #               right_kwargs={'color': axislabelcolors['l']},
    #               alpha=0.7)
    
    # # Set and format axes ticks.
    # ticks = list(np.linspace(0,100,11))
    # tax.ticks(ticks=ticks, axis='blr', linewidth=1, clockwise=False,
    #           axes_colors={'b':'r','l':'b','r':'g'}, offset=0.02, tick_formats="%0.1f")
    
    tax.ticks(axis='blr', linewidth=2, clockwise=False, multiple=10,
              axes_colors=axislabelcolors, offset=0.02, tick_formats="%d",
              fontsize=20)
    
    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()
    if BandGapNatureHeatmap:
        _, cb = tax.heatmap(dd, scale=scale, style="h", cmap=cmap, colorbar=False,)
                           #vmin=0, vmax=1,)
        # tax.scatter([(1,1,1)], marker='s', color=cmap(0), label="Indirect")
        # tax.scatter([(1,1,1)], marker='s', color=cmap(11), label="Direct")
        # tax.legend(loc=1, fontsize=18)
        if show_colorbar:
            cb.set_label(label=cbarlabel, size=fontsize)
            cb.ax.tick_params(labelsize=20,length=8,width=2)
    elif dd is not None:
        _,cb = tax.heatmap(dd, scale=scale, style="h", cmap=cmap, 
                    vmin=vmin, vmax=vmax, 
                    cb_kwargs={'format':'%.1f','shrink':0.9,'anchor':(0.6,0.8),
                               'location':cbarpos,
                               'pad':0.01}, 
                    )
        if show_colorbar:
            cb.set_label(label=cbarlabel, size=fontsize)
            cb.ax.tick_params(labelsize=20,length=8,width=2)
        
    if contours is not None:
        if UseContoursText: 
            for i, contourss in enumerate(contours):
                tax, ax = DrawIndividualContour_and_Text(tax, ax, contourss, 
                                                          COMContourText[i], 
                                                          fontsize=fontsize)
        else:
            for i, contourss in enumerate(contours):
                tax = DrawIndividualContour(tax, contourss)
            if all(ContoursText):
                for i, contourss in enumerate(ContoursText):
                    ax = DrawIndividualContourText(ax, contourss, 
                                                    COMContourText[i], fontsize=fontsize)
                        
    tax.get_axes().set_aspect(1)
    tax._redraw_labels()
    
    if DrawRawData and len(RawData)>0:  
        COlors = RawDataColor if (RawDataColor is not None) else 'k' 
        _,cb = tax.scatter(RawData, marker='s', colormap=cmap, cmap=cmap, vmax=vmax,
                           vmin=vmin, c=COlors,colorbar=show_colorbar, 
                           cb_kwargs={'format':'%.1f','shrink':0.9,'anchor':(0.6,0.8),
                                                  'location':cbarpos,
                                                  'pad':0.01})
        if show_colorbar:
            cb.set_label(label = cbarlabel, size=fontsize)
            cb.ax.tick_params(labelsize=20,length=8,width=2)
            cb.ax.set_xticks([0.25,0.75],['indirect','direct'])
    #plt.tight_layout()
    if savefig:
        plt.savefig(fname,bbox_inches = 'tight',dpi=300)
        plt.close(figure)
    else:
        tax.show()
    return tax

def GenerateHeatmapSnapShots(StrainArray, features, pp, movdirname='', 
                             tt=None, ZcolumnName='STRAIN',
                             generatetitle=True, BandGapNatureHeatmap=False,contours=None,
                             UseContoursText=False, ContoursText=None,
                             axislabels = ["A", "B", "C"],
                             axiscolors=None,axislabelcolors=None,COMContourText=['test'],
                             savefig=False, scale=100,cmap=plt.cm.get_cmap('viridis'),
                             vmin=-1, vmax=1, fontsize=20, cbarlabel=None,
                             cbarpos='right',OnlyContour=False,
                             RawData=None, RawDataColorColumn=None, DrawRawData=False):
    """
    This function plots the ternary bandgap(nature) heatmap and contours.

    Parameters
    ----------
    StrainArray : Array
        The array for the z-coordinate of snapshots.
    features : list
        The list of features. 3 ternary cordinates column name, 1 z-values column
        name, 1 color values column name.
    pp : pandas dataframe
        Dataframe containig the data.
    movdirname : File path, optional
        The file path where the snapshots will be stored. The default is ''.
    tt : string, optional
        Title of the figures. The default is None.
    ZcolumnName: string, optional
        The column name that will be used to sort the data from pp according to 
        StrainArray.
    generatetitle : Bool, optional
        If the title will be generated. The format is f"Strain = {StrainSnap:.1f}".
        The default is True.
    BandGapNatureHeatmap : Bool, optional
        If to plot the bandgap nature heatmap. The default is False.        
    contours : matplotlib contour object list, optional
        The countour object for contour plot. The default is None.
    UseContoursText : bool, optional
        If this is true then the contour texts will be annotated using the supplied
        contours list. If false the ContoursText list will be used for contour
        texts. The default is False. 
    ContoursText : matplotlib contour object list, optional
        The countour object for contour plot texts. The texts will be annotated
        at the centroid of countour polygons. The default is None.
    axislabels : list, optional
        The list of axis labels. The default is ["A", "B", "C"]. Format [left, right, bottom].
    axiscolors : dictionary, optional
        The color of the axes, e.g {'b':'g','l':'r','r':'b'}. The default is None.
    axislabelcolors : dictionary, optional
        The color of the axes labels, e.g. {'b':'g','l':'r','r':'b'}. The default is None.
    COMContourText : string list, optional
        The text in the centroid of contour ploygons. The default is ['test'].
        Length of COMContourText should be >= len(contours) or len(ContoursText).
    savefig : Bool, optional
        Save the figure. The default is False.
    scale : float/int, optional
        Scale of the ternary plot. The default is 100.
    cmap : matplotlib color map, optional
        Color map for heat map. The default is plt.cm.get_cmap('viridis').
    vmin : float, optional
        The minima of color bar. The default is -1.
    vmax : float, optional
        Maxima of color bar. The default is 1.      
    fontsize : matplotlib fontsize, optional
        Fontsize for axislabels, COMContourText and colorbar label. The default is 20.   
    cbarlabel : string, optional
        The latel text of the colorbar. The default is None.
    cbarpos : string, optional
        The position of the colorbar. The default is 'right.'
    OnlyContour : bool, optional
        Whether to create only the contour lines. Default is False. If True this will set BandGapNatureHeatmap to False.
    RawData : dataframe, optional
        The data for DrawRawData. Default is None.
    RawDataColorColumn : string, optional
        The column label of RawData that will be used for coloring. Default is None.    
    DrawRawData : bool, optional
        Whether to draw the raw data provided by 'RawData' dataframe. Default is 
        False. If RawData is None, DrawRawData will be set to False. The last 
        column of 'RawData' will be used for coloring. 
    Returns
    -------
    None.

    """
    TotalSnapShot = len(StrainArray)
    RawDataTmp_color = None; RawDataTmp = None
    if OnlyContour: BandGapNatureHeatmap = False
    if RawData is None: DrawRawData=False
    
    if contours is None: contours={i:None for i in StrainArray}
    if ContoursText is None: ContoursText=[{i:None for i in StrainArray}]
    
    HowManyCNTtext = len(ContoursText)
    
    for i, StrainSnap in enumerate(StrainArray):
        print(f"* Snapshot: {i+1}/{TotalSnapShot}")
        ppp = pp[StrainSnap][features]
        dd = None if OnlyContour else generate_heatmap_data(ppp)
        if DrawRawData: 
            RawData_ = RawData[RawData['STRAIN']==StrainSnap]
            RawDataTmp = generate_scatter_data(RawData_[features[:3]])
            if RawDataColorColumn is not None: RawDataTmp_color = RawData_[RawDataColorColumn] 
        if generatetitle:
            tt = f"Strain = {StrainSnap:.1f} %"
        CNTtext = [ContoursText[JJ][StrainSnap] for JJ in range(HowManyCNTtext)]
        _ = DrawSnapshot(dd, fname=movdirname+f'conf{i:03d}.png', titletext=tt, \
                         BandGapNatureHeatmap=BandGapNatureHeatmap, savefig=savefig, scale=scale,
                         axislabels=axislabels,axiscolors=axiscolors,
                         axislabelcolors=axislabelcolors,COMContourText=COMContourText,
                         cmap=cmap, vmin=vmin, vmax=vmax,contours=[contours[StrainSnap]],
                         UseContoursText=UseContoursText, ContoursText=CNTtext,
                         fontsize=fontsize, cbarlabel=cbarlabel,cbarpos=cbarpos,
                         RawData=RawDataTmp,RawDataColor=RawDataTmp_color,DrawRawData=DrawRawData)
            
# def cube(i,pp,StrainArray,features,OnlyContour,DrawRawData,RawData,RawDataColorColumn,generatetitle,ContoursText,
#          HowManyCNTtext,movdirname,BandGapNatureHeatmap,savefig,scale,axislabels,axiscolors,axislabelcolors,COMContourText,
#          cmap,vmin,vmax,contours,UseContoursText,fontsize,cbarlabel,cbarpos,RawDataTmp_color,RawDataTmp):
#     StrainSnap = StrainArray[i]
#     print(f"* Snapshot: {i+1}/{len(StrainArray)}")
#     ppp = pp[StrainSnap][features]
#     dd = None if OnlyContour else generate_heatmap_data(ppp)
#     if DrawRawData: 
#         RawData_ = RawData[RawData['STRAIN']==StrainSnap]
#         RawDataTmp = generate_scatter_data(RawData_[features[:3]])
#         if RawDataColorColumn is not None: RawDataTmp_color = RawData_[RawDataColorColumn] 
#     if generatetitle:
#         tt = f"Strain = {StrainSnap:.1f} %"
#     CNTtext = [ContoursText[JJ][StrainSnap] for JJ in range(HowManyCNTtext)]
#     _ = DrawSnapshot(dd, fname=movdirname+f'conf{i:03d}.png', titletext=tt, \
#                      BandGapNatureHeatmap=BandGapNatureHeatmap, savefig=savefig, scale=scale,
#                      axislabels=axislabels,axiscolors=axiscolors,
#                      axislabelcolors=axislabelcolors,COMContourText=COMContourText,
#                      cmap=cmap, vmin=vmin, vmax=vmax,contours=[contours[StrainSnap]],
#                      UseContoursText=UseContoursText, ContoursText=CNTtext,
#                      fontsize=fontsize, cbarlabel=cbarlabel,cbarpos=cbarpos,
#                      RawData=RawDataTmp,RawDataColor=RawDataTmp_color,DrawRawData=DrawRawData)
# from multiprocessing import Pool    
# def GenerateHeatmapSnapShots_multiprocessing(StrainArray, features, pp, movdirname='', 
#                              tt=None, ZcolumnName='STRAIN',
#                              generatetitle=True, BandGapNatureHeatmap=False,contours=None,
#                              UseContoursText=False, ContoursText=None,
#                              axislabels = ["A", "B", "C"],
#                              axiscolors=None,axislabelcolors=None,COMContourText=['test'],
#                              savefig=False, scale=100,cmap=plt.cm.get_cmap('viridis'),
#                              vmin=-1, vmax=1, fontsize=20, cbarlabel=None,
#                              cbarpos='right',OnlyContour=False,
#                              RawData=None, RawDataColorColumn=None, DrawRawData=False):
#     TotalSnapShot = len(StrainArray)
#     RawDataTmp_color = None; RawDataTmp = None
#     if OnlyContour: BandGapNatureHeatmap = False
#     if RawData is None: DrawRawData=False
    
#     if contours is None: contours={i:None for i in StrainArray}
#     if ContoursText is None: ContoursText=[{i:None for i in StrainArray}]
    
#     HowManyCNTtext = len(ContoursText)
#     args = [(i,pp,StrainArray,features,OnlyContour,DrawRawData,RawData,RawDataColorColumn,generatetitle,ContoursText,
#              HowManyCNTtext,movdirname,BandGapNatureHeatmap,savefig,scale,axislabels,axiscolors,axislabelcolors,COMContourText,
#              cmap,vmin,vmax,contours,UseContoursText,fontsize,cbarlabel,cbarpos,RawDataTmp_color,RawDataTmp) for i in range(TotalSnapShot)]
    
    
#     with Pool(processes=4) as pool:
#         pool.starmap(cube, args)
    
  
def GenerateHeatmapSnapShotsV2(ppp, strain, movdirname=None, tt=None, 
                               generatetitle=True,contours=None,
                               UseContoursText=False, ContoursText=None,
                               BandGapNatureHeatmap=False,
                               axislabels = ["A", "B", "C"],
                               axiscolors=None,axislabelcolors=None, 
                               COMContourText=['test'], fontsize=20, 
                               cmap=plt.cm.get_cmap('viridis'), vmin=-1, vmax=1,
                               savefig=False, scale=100, cbarlabel=None,
                               cbarpos='right',
                               RawData=None, RawDataColor=None, DrawRawData=False):
    """
    This function plots the ternary bandgap(nature) heatmap and contours.

    Parameters
    ----------
    ppp : pandas dataframe
        Dataframe containig the data.
    strain : float
        The z-coordinate of snapshots (e.g. strain point).
    movdirname : File path, optional
        The file path where the snapshots will be stored. The default is ''.
    tt : string, optional
        Title of the figures. The default is None.
    generatetitle : Bool, optional
        If the title will be generated. The format is f"Strain = {StrainSnap:.1f}".
        The default is True.
    contours : matplotlib contour object list, optional
        The countour object for contour plot. The default is None. 
    UseContoursText : bool, optional
        If this is true then the contour texts will be annotated using the supplied
        contours list. If false the ContoursText list will be used for contour
        texts. The default is False. 
    ContoursText : matplotlib contour object list, optional
        The countour object for contour plot texts. The texts will be annotated
        at the centroid of countour polygons. The default is None.      
    BandGapNatureHeatmap : Bool, optional
        If to plot the bandgap nature heatmap. The default is False.        
    axislabels : list, optional
        The list of axis labels. The default is ["A", "B", "C"]. Format [left, right, bottom].
    axiscolors : dictionary, optional
        The color of the axes, e.g {'b':'g','l':'r','r':'b'}. The default is None.
    axislabelcolors : dictionary, optional
        The color of the axes labels, e.g. {'b':'g','l':'r','r':'b'}. The default is None.
    COMContourText : string list, optional
        The text in the centroid of contour ploygons. The default is ['test'].
        Length of COMContourText should be >= len(contours) or len(ContoursText).
    fontsize : matplotlib fontsize, optional
        Fontsize for axislabels, COMContourText and colorbar label. The default is 20. 
    cmap : matplotlib color map, optional
        Color map for heat map. The default is plt.cm.get_cmap('viridis').
    vmin : float, optional
        The minima of color bar. The default is -1.
    vmax : float, optional
        Maxima of color bar. The default is 1.   
    savefig : Bool, optional
        Save the figure. The default is False.
    scale : float/int, optional
        Scale of the ternary plot. The default is 100.
    cbarlabel : string, optional
        The latel text of the colorbar. The default is None.
    cbarpos : string, optional
        The position of the colorbar. The default is 'right.'
    RawData : dataframe, optional
        The data for DrawRawData. Default is None.
    RawDataColor : dataframe, optional
        The color data for DrawRawData. Default is None.    
    DrawRawData : bool, optional
        Whether to draw the raw data provided by 'RawData' dataframe. Default is 
        False. If RawData is None, DrawRawData will be set to False. The last 
        column of 'RawData' will be used for coloring. 
    Returns
    -------
    None.

    """

    dd=generate_heatmap_data(ppp) if BandGapNatureHeatmap else None
    if (RawData is None) or (not RawData): DrawRawData=False
        
    if generatetitle:
        tt = f"Strain = {strain:.1f}"
    _ = DrawSnapshot(dd, fname=movdirname, titletext=tt, \
                     axislabels=axislabels,axiscolors=axiscolors,
                     axislabelcolors=axislabelcolors,COMContourText=COMContourText,
                     savefig=savefig, scale=scale, BandGapNatureHeatmap=BandGapNatureHeatmap,
                     contours=contours,
                     UseContoursText=UseContoursText, ContoursText=ContoursText,
                     cmap=cmap,vmin=vmin, vmax=vmax, fontsize=fontsize,
                     cbarlabel=cbarlabel,cbarpos=cbarpos,
                     RawData=RawData, RawDataColor=RawDataColor, DrawRawData=DrawRawData)


def MakeEgStrainSnapShotMovie(images, movdirname=None, savefig = 0, repeatunit=1):
    fig = plt.figure(constrained_layout=True)
    ax = plt.subplot(1,1,1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.set_constrained_layout_pads(w_pad=0.2, h_pad=0.2, hspace=0.2, wspace=0.15)
    
    im=ax.imshow(images[0])
    
    def init():
        im.set_data(images[0])
        return [im]
    
    def update_img(n, repeatunit):
        tmp1 = (n%(repeatunit))/repeatunit
        tmp2 = n//repeatunit
        im.set_data((images[tmp2]*(1-tmp1))+(images[tmp2+1]*tmp1))
        return [im]
    
    def updateALL(frameNum, args):
        repeatunit = args
        update_img(frameNum, repeatunit)
        
    anim = animation.FuncAnimation(fig, updateALL, init_func=init, fargs=(repeatunit,),\
                                   frames=(len(images)-1)*repeatunit, repeat=0, interval=100, blit=False)
    
    if savefig:
        print('*** Saving movie')
        filname = movdirname+'/movie.mp4'
        #anim.save(filname, fps=20, savefig_kwargs={'bbox_inches': 'tight'}, metadata={'copyright': 'bm'}, dpi=300)
        anim.save(filname, fps=20, metadata={'copyright': 'bm'}, dpi=300) 
        
    return 

def GetCorners(X):
    # The number 1000 is arbitrary but large enough. 
    return np.sign(X)*1000

def UpdateCornersForContours(Zval):
    # Completes the polygon. This number shoud be same as the 2nd number in GetContours() levels.
    Zval[0,0] = GetCorners(Zval[0,0])
    Zval[0,-1] = GetCorners(Zval[0,-1])
    Zval[-1,0] = GetCorners(Zval[-1,0])
    return Zval

def GetContoursf(x,y,h,anti_contour=False):
    levels = [np.nanmin(h),0] if anti_contour else [0,np.nanmax(h)] 
    cnt = plt.contourf(x,y,h,levels)
    #plt.clf()
    plt.close()
    return cnt

def GetContours(x,y,h, TernaryConversion=False):
    if TernaryConversion:
        x, y = DataConversion(x, y)
    cnt = plt.contour(x,y,h,[0])
    #plt.clabel(cnt, inline=1, fontsize=10)
    plt.close()
    return cnt

def DrawRandomConfigurationPoints(dd, fname=None, titletext=None, ax=None,
                                  savefig=False, scale=1, OnlyContour=False,
                                  axislabels = ["A", "B", "C"],axiscolors=None,
                                  axislabelcolors=None,DrawGridLines=False,titletext_loc='right',
                                  vmin=-1, vmax=1,cmap=plt.cm.get_cmap('viridis'),
                                  fontsize = 20,colorbar=True,cbarpos='right',
                                  colors='k',cbar_label_txt='Strain (%)',marker='s',
                                  ):
    # Dictionary of axes colors for bottom (b), left (l), right (r).
    if axislabelcolors is None: axislabelcolors={'b':'k','l':'k','r':'k'}
            
    
    # figure, ax = plt.subplots()
    # tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)
    # #figure, tax = ternary.figure(scale=scale)
    # if (cbarpos == 'right') or (cbarpos == 'left'):
    #     figure.set_size_inches(10, 8) 
    # else:
    #     figure.set_size_inches(10, 10)
    
    # tax.boundary(linewidth=2.0, axes_colors=None) 
    
    # if titletext is not None:
    #     tax.set_title(titletext, fontsize=24, loc='right')
    #     #ax.text(100,55,titletext, fontsize=24,  rotation=270)
        
        
        
    if ax is None: 
        figure, ax = plt.subplots()
        if (cbarpos == 'right') or (cbarpos == 'left'):
            figure.set_size_inches(10, 8) 
        else:
            figure.set_size_inches(10, 10)
        
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)
    # #figure, tax = ternary.figure(scale=scale)
    # figure.set_size_inches(10, 8)
    
    tax.boundary(linewidth=2.0, axes_colors=None) 
    
    if titletext is not None:
        tax.set_title(titletext, fontsize=24, pad=20)      
    tax.left_axis_label(axislabels[0], offset=0.12, color=axislabelcolors['l'], fontsize=fontsize)
    tax.right_axis_label(axislabels[1], offset=0.13, color=axislabelcolors['r'], fontsize=fontsize)
    tax.bottom_axis_label(axislabels[2], offset=0.02, color=axislabelcolors['b'], fontsize=fontsize)
        
    tax.ticks(axis='blr', linewidth=2, clockwise=False, multiple=10,
              axes_colors=axislabelcolors, offset=0.015, tick_formats="%d",
              fontsize=20)
    if DrawGridLines:
        tax.gridlines(multiple=10, linewidth=2,
                      horizontal_kwargs={'color': axislabelcolors['b']},
                      left_kwargs={'color': axislabelcolors['l']},
                      right_kwargs={'color': axislabelcolors['r']},
                      alpha=0.7)
    
    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()
    
    sc,cb = tax.scatter(dd, marker=marker, colormap=cmap, colorbar=colorbar, vmax=vmax,
                       vmin=vmin, c=colors, cmap=cmap,
                       cb_kwargs={'format':'%.1f','shrink':0.9,'anchor':(0.6,0.8),
                                  'location':cbarpos,
                                  'pad':0.01})
    if colorbar:
        cb.set_label(label = cbar_label_txt, size=fontsize)
        cb.ax.tick_params(labelsize=20,length=8,width=2)
    
    tax.get_axes().set_aspect(1)
    tax._redraw_labels()
    plt.tight_layout()
    if savefig:
        plt.savefig(fname,bbox_inches = 'tight',dpi=300)
        plt.close(figure)
    else:
        tax.show()
    return figure, ax 

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
    
def Plot3Ddecision_function(Xp, Yp, Zval, scale=100, 
                            TernaryConversion=True, hplane=False):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    cmap = cm.get_cmap('PiYG', 2)
    # surf = ax.plot_surface(Xp, Yp, Zval, cmap=cmap,
    #                        linewidth=0, antialiased=False)
    
    X = Xp.flatten()
    Y = Yp.flatten()
    Z = Zval.flatten()
    if TernaryConversion:
        X, Y = DataConversion(X, Y) 

    surf = ax.scatter(X, Y, Z, marker='o', c=Z, cmap = cmap, vmin=-1, vmax=1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    if hplane:
        xx, yy = np.meshgrid([0,scale], [0,scale])
        zz = yy*0
        ax.plot_surface(xx, yy, zz, color='gray')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim(0,scale)
    plt.show()
    
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

def PlotPostProcessingDataSetSize(pp,save=False, savepath='.',ProjectionDict=None): 
    if ProjectionDict is None:
        ProjectionDict = {'set1':{'root_mean_squared_error':'RMSE (meV)','r2_score':'$\mathrm{R}^2$'},'set2':{'mean_absolute_error':'MAE (meV)'},\
                          'set3':{'max_error':'Max error (meV)'},'set4':{'accuracy_score':'Accuracy score','balanced_accuracy_score': 'Balanced accuracy score'},\
                              'set5':{'accuracy_score':'Accuracy score'},'set6':{'r2_score':'$\mathrm{R}^2$'},'set7':{'balanced_accuracy_score': 'Balanced accuracy score'},
                              'set8':{'root_mean_squared_error':'RMSE (meV)'}}
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
                ax.errorbar(pp_mean['dataset_size'],pp_mean[II],yerr=pp_std[II],linestyle='-',color='k')
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
                        ax2.errorbar(pp_mean['dataset_size'],pp_mean[II],yerr=pp_std[II],linestyle='-',color='tab:blue')  
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
                    SaveFileName = f"{savepath}/{LL}_{'-'.join(WhichMetric)}.png"
                    plt.savefig(SaveFileName,bbox_inches = 'tight',dpi=300)
                    plt.close()
                else:
                    plt.show()
    return 

def PlotPostProcessingDataSetSize_special(pp,save=False, ax=None, fig=None, savepath='.',ProjectionDict=None, ax_y_precision=None,ax_yminortick_multi=5,ax_plot_color='k'): 
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
                ax.errorbar(pp_mean['dataset_size'],pp_mean[II],yerr=pp_std[II],linestyle='-',color=ax_plot_color)
                # PrecissionStr = '%.1f' if WhichMetric[0] in ['root_mean_squared_error','mean_absolute_error','max_error'] else '%.3f'
                if ax_y_precision is not None and isinstance(ax_y_precision,str):
                    ax.yaxis.set_major_formatter(FormatStrFormatter(ax_y_precision))
                if len(WhichMetric) == 2:
                    II = f'{LL}_{WhichMetric[1]}'
                    if II in AllColumns:
                        ax2 = ax.twinx()
                        ax2.set_ylabel(ProjectionDict[whichset][WhichMetric[1]])  
                        ax2.errorbar(pp_mean['dataset_size'],pp_mean[II],yerr=pp_std[II],linestyle='-',color='tab:blue')  
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
                    SaveFileName = f"{savepath}/{LL}_{'-'.join(WhichMetric)}.png"
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

def PlotPostProcessingDataSetSizeLogLog(pp,save=False, savepath='.'):      
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
                ax.errorbar(pp_mean['dataset_size'],pp_mean[II],yerr=pp_std[II],linestyle='-',color='k')
                ax.tick_params(axis='both',which='major',length=10,width=2)
                ax.tick_params(axis='both',which='minor',length=6,width=2)
                # y_major = LogLocator(base = 10.0, numticks = 100)
                # ax.yaxis.set_minor_locator(y_major)
                ax.yaxis.set_minor_formatter(LogFormatterSciNotation(minor_thresholds=(1,10)))
                # ax.yaxis.set_minor_formatter(LogFormatter(minor_thresholds=(1,10)))
                if len(WhichMetric) == 2:
                    II = f'{LL}_{WhichMetric[1]}'
                    if II in AllColumns:
                        ax2 = ax.twinx()
                        ax2.set_ylabel(ProjectionDict[whichset][WhichMetric[1]])  
                        ax2.set_xscale('log', nonpositive='clip')
                        ax2.set_yscale('log', nonpositive='clip')
                        ax2.errorbar(pp_mean['dataset_size'],pp_mean[II],yerr=pp_std[II],linestyle='-',color='tab:blue')  
                        ax2.tick_params(axis='y', which='both', labelcolor='tab:blue')
                        ax2.tick_params(axis='both',which='major',length=10,width=2)
                        ax2.tick_params(axis='both',which='minor',length=6,width=2)
                if save:
                    SaveFileName = f"{savepath}/{LL}_{'-'.join(WhichMetric)}_LogLog.png"
                    print(f'\t{SaveFileName}')
                    plt.savefig(SaveFileName,bbox_inches = 'tight',dpi=300)
                    plt.close()
                else:
                    plt.show()
    return 

def PlotPostProcessingDataSetSizeLogLog_v2(pp,save=False, savepath='.'):      
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
                SaveFileName = f"{savepath}/{LL}_{'-'.join(WhichMetric)}_LogLog_v2.png"
                print(f'\t{SaveFileName}')
                plt.savefig(SaveFileName,bbox_inches = 'tight',dpi=300)
                plt.close()
            else:
                plt.show()
    return 