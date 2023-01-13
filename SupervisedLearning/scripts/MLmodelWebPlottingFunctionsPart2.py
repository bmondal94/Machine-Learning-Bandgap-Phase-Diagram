#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:18:06 2022

@author: bmondal
"""

from bokeh.util.hex import hexbin, cartesian_to_axial
import numpy as np
import holoviews as hv
from holoviews import opts
from holoviews import dim
#from bokeh.io import show, curdoc, output_notebook
from bokeh.layouts import layout
hv.extension('bokeh')
renderer = hv.renderer('bokeh')
import MLmodelWebPlottingFunctions as mlwpf
from bokeh.models import ColumnDataSource
from bokeh.models import LinearColorMapper, ColorBar, HoverTool
from bokeh.plotting import figure, output_file, show, curdoc,save
from bokeh.models.ranges import DataRange1d
from bokeh.models import Plot, CustomJSHover, MultiSelect, Select, CustomJS, Slider

#%%-------------- Using holoview and bokeh models -----------------------------
def ConversionCartesian2Ternary(x,y, SQRT3o2 = 0.8660254037844386):
    XX = x + y*0.5
    YY = SQRT3o2 * y
    return XX, YY

def DrawBoundaryV2(p, scale, color, line_width, offset_up, offset_bottom,offset_left,offset_right):
    ax = np.array([offset_left,scale+offset_right,offset_left,offset_left])
    ay = np.array([offset_bottom,offset_bottom,scale+offset_up,offset_bottom])    
    XX, YY = ConversionCartesian2Ternary(ax,ay)
    p.line(XX,YY, color=color, line_width=line_width)
    return  p

def GetContoursV2(contours):
    X, Y =  [], []
    for contour in contours.allsegs:
        for seg in contour:
            a = seg[:, 0:2][:,0]
            b = seg[:, 0:2][:,1]
            x, y = ConversionCartesian2Ternary(a,b)
            X.append(list(x))
            Y.append(list(y))
    return X, Y#, Z

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
    return ConversionCartesian2Ternary(cx, cy)

def GetContoursV2Text(contours, textt = ''):
    textx, texty, textlist = [], [], []
    tmptextx, tmptexty, tmptext = [], [], []
    for contour in contours.allsegs:
        for seg in contour:                
            COMtextx, COMtexty = center_of_mass(seg[:, 0:2])
            tmptextx.append(COMtextx)
            tmptexty.append(COMtexty)
            tmptext.append(textt)

        textx.append(tmptextx)
        texty.append(tmptexty)
        textlist.append(tmptext)

    return textx, texty, textlist

def DrawBandgapHeatmapWeb(df, strain, fname="patch.html", titletext=None,
                          savefig=False, scale=1, vmin=-1, vmax=1,
                          cmappp="Viridis256",color='black', 
                          line_width=4, text=['a','b','c'],
                          offset_bottom=-0.14, offset_left=-0.1,offset_right=0.1,
                          fontsize='3em', cbarlabel_fontsize='3em',cbarlabel='',
                          label_colors=['black','black','black'],
                          tick_colors=['black','black','black'],
                          tick_length=3, TickLabel_fontsize='3em',step=0.1,
                          SQRT3o2 = 0.8660254037844386):

    p = mlwpf.CreateSkeliton(titletext,savefig, scale, 
                          color, line_width, text,
                          offset_bottom, offset_left,offset_right,
                          fontsize, label_colors,tick_colors,
                          tick_length, TickLabel_fontsize,step,
                          SQRT3o2)
    
    
    
    color_bar = mlwpf.CreateColorBar(vmin, vmax,cmappp,cbarlabel_fontsize,cbarlabel)
    p.add_layout(color_bar, 'right')
    
    X, Y = ConversionCartesian2Ternary(df['PHOSPHORUS'], df['ARSENIC'])
    source = ColumnDataSource(data=dict(x=X, y=Y, z=df['bandgap']))

    cmap = LinearColorMapper(palette=cmappp, 
                              low = df['bandgap'].min(), 
                              high = df['bandgap'].max())

    

    hexsize=10 #p.width/scale
    halfhexsize=5
    p.hex('x','y',source=source,size=halfhexsize*2,line_color=None,angle=1.57,
          fill_color={"field":"z", "transform":cmap})
    
    # bins = hexbin(X, Y, 1)
    # p.hex_tile(q="q", r="r", size=1, line_color=None, source=bins,
    #            fill_color={"field":"counts", "transform":cmap})
    p = DrawBoundaryV2(p, scale, color, halfhexsize, 0.01*scale,-.007*scale,-0.007*scale,0.01*scale)
    #p = DrawBoundaryV2(p, scale, color, line_width, offset_bottom,offset_left,offset_right)
    #p = DrawBoundary(p, scale, color, hexsize)
    if savefig:
        output_file(fname, title="Bandgap phase diagram")
        save(p)
        return None
    else:
        return p
    
def DrawBandgapHeatmapWebV2(df, strain, fname="patch.html", titletext=None,
                            savefig=False, scale=1, vmin=-1, vmax=1,
                            cmappp="viridis",color='black', 
                            colorbar_cmap='Viridis256',
                            line_width=4, text=['a','b','c'],
                            offset_bottom=-0.14, offset_left=-0.15,offset_right=0.15,
                            fontsize='3em', cbarlabel_fontsize='3em',cbarlabel='',
                            label_colors=['black','black','black'],
                            tick_colors=['black','black','black'],
                            tick_length=3, TickLabel_fontsize='3em',step=0.1,
                            SQRT3o2 = 0.8660254037844386):
    
    

    TOOLTIPS, FORMATTER = mlwpf.DefHoverTools(text=text)
    TOOLTIPS += [("Bandgap", "@values eV")]
    tool = HoverTool(tooltips = TOOLTIPS,formatters=FORMATTER)

    X, Y = ConversionCartesian2Ternary(df['PHOSPHORUS'], df['ANTIMONY']) 
    
    hex_with_values = hv.HexTiles((X, Y, df['bandgap']), vdims=['values'])
    hex_with_values.opts(width=800, cmap=cmappp,
                         clabel="Eg",#padding=(0.1,(0.15,0.1)),
                         aggregator=np.mean, colorbar=False,
                         tools=[tool],
                         )
    p = hv.render(hex_with_values)

    p.x_range=DataRange1d(start=offset_left*scale,end=scale+offset_right*scale)
    p.y_range=DataRange1d(start=offset_bottom*scale,end=scale*SQRT3o2+(-offset_bottom)*scale)
    p.update(sizing_mode="scale_height", aspect_ratio =1.3,
               height_policy ='fit',
               )
    mlwpf.ClearMatplotlibAxis(p)
    p = mlwpf.DefinePositionsBoundayLabels(p, scale, color=color, 
                                      line_width=line_width, text=text,
                                      offset_bottom=offset_bottom,
                                      offset_left=offset_left,offset_right=offset_right,
                                      fontsize=fontsize,label_colors=label_colors,
                                      tick_colors=tick_colors,tick_length=tick_length,
                                      TickLabel_fontsize=TickLabel_fontsize,
                                      step=step)

    color_bar = mlwpf.CreateColorBar(df['bandgap'].min(), df['bandgap'].max(),
                               colorbar_cmap,cbarlabel_fontsize,cbarlabel)
    p.add_layout(color_bar, 'right')
    if savefig:
        output_file(fname,title="Bandgap phase diagram")
        save(p)
        return None
    else:
        return p

#%% ---------------- Define borders, axislabels with holoview -----------------
def AxisBorders(scale,line_width):
# Borders
    ax = np.array([0,scale,0,0])
    ay = np.array([0,0,scale,0])    
    XX, YY = ConversionCartesian2Ternary(ax,ay)
    borders = hv.Curve((XX,YY)) 
    borders.opts(xaxis=None, yaxis=None,line_color='black',line_width=line_width)
    return borders

def AxisLabelsHV(text,scale,offset_left,offset_right,offset_bottom,label_colors,
                 angle=[0,-60,60],fontsize='3em',):
    # Axislabels
    sclaehalf = scale * 0.5
    ax = np.array([sclaehalf,sclaehalf+(offset_right-0.1)*scale,offset_left*scale])
    ay = np.array([offset_bottom*scale,sclaehalf+(0.1*scale),sclaehalf])
    x, y = ConversionCartesian2Ternary(ax,ay)  
    # labels = hv.Labels((x, y, text))
    # labels.opts(text_color='black', text_font_size=fontsize, angle=0)


    labelslist = [hv.Text(x[i], y[i], text[i]).opts(text_color=label_colors[i], \
                                      text_font_size=fontsize,
                                      angle=angle[i]) for i in range(3)]
        
    return labelslist

def AxisTicks(scale,step,tick_colors, label_colors, fontsize='3em', line_width=4):
    # Axis ticks and ticks labels
    num = scale//step
    dtype = int if scale%step==0 else float
    tpos = np.linspace(0,scale,num=(num+1),dtype=dtype)
    textt = tpos.astype(str)
    zarray = np.zeros(len(tpos))

    bofset = scale * (-0.03)
    lofset = scale * 0.03
    tofset = scale * 0.03
    llofset = scale*0.02
    rofset = scale*0.035
    # Bottom ticks
    x = tpos; y = zarray-(scale * 0.07)
    BottomTicks= [hv.Curve(([xi,xi-2], [0,bofset])).opts(line_color=tick_colors[0], 
                  line_width=line_width) for xi in x]
    Blabels = hv.Labels((x-lofset, y, textt))
    Blabels.opts(text_color=label_colors[0], text_font_size=fontsize)

    # Left ticks
    
    XX, YY = ConversionCartesian2Ternary(zarray,tpos)
    LeftTicks= [hv.Curve(([xi,xi-llofset], [yi,yi+tofset])).opts(line_color=tick_colors[0]
                ,line_width=4) for yi, xi in zip(YY,XX)]
    Llabels = hv.Labels((XX-(0.04*scale), YY+(0.045*scale), np.flip(textt)))
    Llabels.opts(text_color=label_colors[1], text_font_size=fontsize)

    # Right ticks
    XX, YY = ConversionCartesian2Ternary(tpos,np.flip(tpos))
    RightTicks= [hv.Curve(([xi,xi+rofset], [yi,yi])).opts(line_color=tick_colors[0],line_width=4) 
                  for yi, xi in zip(YY,XX)]
    Rlabels = hv.Labels((XX+(0.07*scale), YY, np.flip(textt)))
    Rlabels.opts(text_color=label_colors[2], text_font_size=fontsize)
    
    AllTicks = BottomTicks+LeftTicks+RightTicks
    
    return AllTicks+[Rlabels]+[Blabels]+[Llabels]

def HexTiles_Opts(cmap,cbarlabel,cbarlabel_fontsize,tool,scale,
                  fontsize='2em',aggregator=np.mean,):
    xpadding = 0.1*scale
    HexTilesOpts = opts.HexTiles(frame_height=800, frame_width=800, cmap=cmap,
                                 #clabel="Eg",#padding=(0.1,(0.15,0.1)),
                                 aggregator=aggregator, colorbar=True,
                                 colorbar_opts = dict(title=cbarlabel,
                                                      title_text_font_size=cbarlabel_fontsize,
                                                      major_label_text_font_size=cbarlabel_fontsize),
                                 tools=[tool],
                                 xaxis=None, yaxis=None,
                                 xlim=(-xpadding,scale+xpadding),
                                 #responsive=True,
                                 fontsize=fontsize,)
    return HexTilesOpts
#------------------------------------------------------------------------------  
 
def DrawBandgapHeatmapWebSlider(pp,XYZcloumns,countourset,strain, fname="patchh.html", titletext=None,
                                ContourText=None,
                                savefig=False, scale=1, vmin=-1, vmax=1,
                                cmappp="viridis",color='black', 
                                line_width=4, text=['a','b','c'],
                                offset_bottom=-0.14, offset_left=-0.15,offset_right=0.15,
                                fontsize='3em', cbarlabel_fontsize='3em',cbarlabel='',
                                label_colors=['black','black','black'],
                                tick_colors=['black','black','black'],
                                tick_length=3, TickLabel_fontsize='3em',step=0.1,
                                SQRT3o2 = 0.8660254037844386):
    
    cntNumber = len(countourset) # Total number of extra contour
    if cntNumber>1:
        if ContourText is None: 
            ContourText = ['']*cntNumber
        else:
            assert cntNumber-1 <= len(ContourText), 'Total number of extra contours supplied is more than number of contour texts.'

    TOOLTIPS, FORMATTER = mlwpf.DefHoverTools(text=text)
    TOOLTIPS += [("Bandgap", "@values eV")]
    tool = HoverTool(tooltips = TOOLTIPS,formatters=FORMATTER)
    hmapdict = {}
    for ii, df in pp.items():
        print(f'Strain: {ii} %')
        X, Y = ConversionCartesian2Ternary(df[XYZcloumns[0]], df[XYZcloumns[1]]) 
        HexTilesOpts = HexTiles_Opts(cmappp,cbarlabel,cbarlabel_fontsize,tool,scale)
        hex_with_values = hv.HexTiles((X, Y, df[XYZcloumns[2]]), vdims=['values'])
        hex_with_values.opts(HexTilesOpts)
        
        
        borders = AxisBorders(scale,line_width)
        labels  = AxisLabelsHV(text,scale,offset_left,offset_right,offset_bottom,label_colors,
                               fontsize=fontsize)

        AxisTicksandLabelst = AxisTicks(scale,step,tick_colors, label_colors,
                                        fontsize=fontsize,line_width=line_width)
        
        #------------------ Contour line --------------------------------------
        ContourLineX, ContourLineY = GetContoursV2(countourset[0][ii])
        ContourLineList = []
        for J in range(len(ContourLineX)):
            curvepoints = (ContourLineX[J], ContourLineY[J])
            ContourLine = hv.Curve(curvepoints).opts(line_color='black',line_width=line_width)
            ContourLineList.append(ContourLine)
        #----------------- Contour texts --------------------------------------
        if cntNumber>1:
            for IIII in range(1,cntNumber):
                XXtext = GetContoursV2Text(countourset[IIII][ii], textt=ContourText[IIII-1])
                llbels = hv.Labels(XXtext).opts(text_color='black', text_font_size=fontsize)
                ContourLineList.append(llbels)
        #----------------------------------------------------------------------
        
        overlay = hv.Overlay([hex_with_values] + 
                              [borders] + labels + AxisTicksandLabelst +
                              ContourLineList
                              ) 
        hmapdict[ii]= overlay 
        
    hmap = hv.HoloMap(hmapdict, kdims='Strain')

    hv.output(widget_location='bottom')
    if savefig:
        print('**********Saving***********')
        hv.save(hmap,fname,title="Bandgap phase diagram")
        print(f'Save at {fname}')
        return None
    else:
        return hmap  
     
def DrawBandgapHeatmapWebV2Slider(pp, strain, fname="patch.html", titletext=None,
                          savefig=False, scale=1, vmin=-1, vmax=1,
                          cmappp="viridis",color='black', 
                          colorbar_cmap='Viridis256',
                          line_width=4, text=['a','b','c'],
                          offset_bottom=-0.14, offset_left=-0.15,offset_right=0.15,
                          fontsize='3em', cbarlabel_fontsize='3em',cbarlabel='',
                          label_colors=['black','black','black'],
                          tick_colors=['black','black','black'],
                          tick_length=3, TickLabel_fontsize='3em',step=0.1,
                          SQRT3o2 = 0.8660254037844386):
    
    

    TOOLTIPS, FORMATTER = mlwpf.DefHoverTools(text=text)
    TOOLTIPS += [("Bandgap", "@values eV")]
    tool = HoverTool(tooltips = TOOLTIPS,formatters=FORMATTER)
    hmapdict = {}

    for I in strain:
        df = pp[pp['STRAIN']==I]

        X, Y = ConversionCartesian2Ternary(df['PHOSPHORUS'], df['ANTIMONY']) 
    
        hex_with_values = hv.HexTiles((X, Y, df['bandgap']), vdims=['values'])
        hex_with_values.opts(width=800, cmap=cmappp,
                             clabel="Eg",#padding=(0.1,(0.15,0.1)),
                             aggregator=np.mean, colorbar=False,
                             tools=[tool],
                             )
        hmapdict[I]=hex_with_values
        
    hmap_keys = list(hmapdict.keys())
    
    def MAPPING(SnapShot):
        key = hmap_keys[SnapShot]
        return hmapdict[key]
    
    stream = hv.streams.Stream.define('SnapShots', SnapShot=0)()
    
    dmap = hv.DynamicMap(MAPPING, streams=[stream])   
    
    slider_code = """
                    var i = cb_obj.value;
                    stream[0].update(SnapShot=i)
                    """

    slider_callback = CustomJS(args = dict(stream=[stream]), 
                                code = slider_code)
    
    def modify_doc(doc):
        # Create HoloViews plot and attach the document
        # hvplot = renderer.get_plot(dmap, doc)
        
        p = renderer.get_plot(dmap) #, doc

        #p = hv.render(hex_with_values)
        
            # p.x_range=DataRange1d(start=offset_left*scale,end=scale+offset_right*scale)
            # p.y_range=DataRange1d(start=offset_bottom*scale,end=scale*SQRT3o2+(-offset_bottom)*scale)
            # p.update(sizing_mode="scale_height", aspect_ratio =1.3,
            #             height_policy ='fit',
            #             )
            # ClearMatplotlibAxis(p)
            # p = DefinePositionsBoundayLabels(p, scale, color=color, 
            #                                   line_width=line_width, text=text,
            #                                   offset_bottom=offset_bottom,
            #                                   offset_left=offset_left,offset_right=offset_right,
            #                                   fontsize=fontsize,label_colors=label_colors,
            #                                   tick_colors=tick_colors,tick_length=tick_length,
            #                                   TickLabel_fontsize=TickLabel_fontsize,
            #                                   step=step)
        
        # color_bar = CreateColorBar(df['bandgap'].min(), df['bandgap'].max(),
        #                            colorbar_cmap,cbarlabel_fontsize,cbarlabel)
        # p.add_layout(color_bar, 'right')

        start, end = 0, len(hmap_keys) - 1
        slider = Slider(start=start, end=end, value=start, step=1, title='SnapShotss', height=30, width=180)

    
        slider.js_on_change('value', slider_callback)
    
        # Combine the holoviews plot and widgets in a layout
        plot = layout([
        [p.state],
        [slider]], sizing_mode='scale_height')
        #curdoc().add_root(plot)
        doc.add_root(plot)
        return doc
    
    
    #show(modify_doc)
    #output_file(fname,title="Bandgap phase diagram")
    #save(plot)
    # plotlist.append(p)
    # hmap = hv.HoloMap(hmapdict, kdims='Strain')
    # print(plotlist)
    #hv.save(hmap,'plot.html',title="Bandgap phase diagram")
    #save(hmap)