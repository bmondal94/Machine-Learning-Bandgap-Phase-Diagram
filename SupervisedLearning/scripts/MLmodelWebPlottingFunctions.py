#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 16:55:10 2022

@author: bmondal
"""

import numpy as np
import matplotlib as mpl
from bokeh.layouts import column,row
from bokeh.plotting import figure, output_file, show, curdoc,save
from bokeh.models import ColumnDataSource, LabelSet, Text, NumeralTickFormatter, CustomJS, Slider
from bokeh.models import Plot, CustomJSHover, MultiSelect, Select
from bokeh.models import LinearColorMapper, ColorBar, HoverTool

import plotly.graph_objects as go

#%%------------- Using Bokeh models -------------------------------------------
def ConversionCartesian2Ternary(x,y, SQRT3o2 = 0.8660254037844386):
    XX = x + y*0.5
    YY = SQRT3o2 * y
    return XX, YY

def ClearMatplotlibAxis(p):
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.xgrid.visible = False
    p.ygrid.visible = False
    return

def DrawBoundary(p, scale, color, line_width):
    ax = np.array([0,scale,0,0])
    ay = np.array([0,0,scale,0])    
    XX, YY = ConversionCartesian2Ternary(ax,ay)
    p.line(XX,YY, color=color, line_width=line_width)
    return  p

def DrawAxisLabel(p, scale, text, fontsize, label_colors, offset_bottom=-0.2, 
                  offset_left=-0.2,offset_right=0.02,
                  ):
    sclaehalf = scale * 0.5
    ax = np.array([sclaehalf,sclaehalf+(offset_right-0.1)*scale,offset_left*scale])
    ay = np.array([offset_bottom*scale, sclaehalf+(0.1*scale),sclaehalf])
    x, y = ConversionCartesian2Ternary(ax,ay)
    # print(x, y)
    source = ColumnDataSource(data=dict(XX=x,
                                        YY=y,
                                        names=text,
                                        angle=[0,-60,60],
                                        colors=label_colors))   
    labels = LabelSet(x='XX', y='YY', text='names',
                      source=source, render_mode='canvas',
                      text_font_size=fontsize,angle_units='deg',
                      angle='angle', text_color = 'colors')
    p.add_layout(labels)
    return p

def TernaryTicks(p, scale, tick_colors, tick_length=2, TickLabel_fontsize='3em',step=0.1):
    num = scale//step
    dtype = int if scale%step==0 else float
    tpos = np.linspace(0,scale,num=(num+1),dtype=dtype)
    text = tpos.astype(str)
    zarray = np.zeros(len(tpos))
    # Bottom ticks
    p.ray(x=tpos, y=zarray, length=tick_length, 
          angle = -120,
          angle_units="deg", color=tick_colors[0], line_width=2)
    
    source = ColumnDataSource(data=dict(Xp=tpos-(0.05*scale),
                                        Yp=zarray-(0.06*scale),
                                        names=text)) 
    blabels = LabelSet(x='Xp',y='Yp',text='names',source=source,
                       text_font_size=TickLabel_fontsize,
                       render_mode='canvas', text_color = tick_colors[0])
    # Left ticks
    XX, YY = ConversionCartesian2Ternary(zarray,tpos)
    p.ray(x=XX, y=YY, length=tick_length, 
          angle = 120,
          angle_units="deg", color=tick_colors[1], line_width=2)
    source = ColumnDataSource(data=dict(Xp=XX-(0.07*scale),
                                        Yp=YY+(0.02*scale),
                                        names=np.flip(text)))
    llabels = LabelSet(x='Xp',y='Yp',text='names',source=source,
                       text_font_size=TickLabel_fontsize,
                       render_mode='canvas', text_color = tick_colors[0])
    # Right ticks
    XX, YY = ConversionCartesian2Ternary(tpos,np.flip(tpos))
    p.ray(x=XX, y=YY, length=tick_length, 
          angle = 0,
          angle_units="deg", color=tick_colors[1], line_width=2)
    source = ColumnDataSource(data=dict(Xp=XX+(0.02*scale),
                                        Yp=YY-(0.015*scale),
                                        names=np.flip(text)))
    rlabels = LabelSet(x='Xp',y='Yp',text='names',source=source,
                       text_font_size=TickLabel_fontsize,
                       render_mode='canvas', text_color = tick_colors[0])
    
    # Add the labels
    p.add_layout(blabels)
    p.add_layout(llabels)
    p.add_layout(rlabels)
    return p

def DefinePositionsBoundayLabels(p, scale, color='black', 
                                 line_width=4, text=['a','b','c'],
                                 offset_bottom=-0.02, 
                                 offset_left=-0.02,offset_right=0.02,
                                 fontsize='16px',
                                 label_colors=['black','black','black'],
                                 tick_colors=['black','black','black'],
                                 tick_length=2, TickLabel_fontsize='3em',step=0.1): 
    p = DrawBoundary(p, scale, color, line_width)
    p = DrawAxisLabel(p, scale, text,fontsize,label_colors,
                      offset_bottom=offset_bottom,
                      offset_left=offset_left,offset_right=offset_right,
                      )
    p = TernaryTicks(p, scale, tick_colors,tick_length=tick_length, 
                     TickLabel_fontsize=TickLabel_fontsize,step=step)
    return p
    
def GetContours(cnt, strain):
    norm = 255*mpl.cm.viridis(mpl.colors.Normalize()(strain))
    X, Y,  Z =  [], [],[]
    i = 0
    for _,contours in cnt.items():
        CC = norm[i].astype(int)
        i += 1
        for contour in contours.allsegs:
            for seg in contour:
                a = seg[:, 0:2][:,0]
                b = seg[:, 0:2][:,1]
                x, y = ConversionCartesian2Ternary(a,b)
                X.append(list(x))
                Y.append(list(y))
                Z.append("#%02x%02x%02x" % (CC[0], CC[1], CC[2]))
    return X, Y, Z

def GetContoursV2(cnt, strain):
    norm = 255*mpl.cm.viridis(mpl.colors.Normalize()(strain))
    X, Y, Z = [],[],[]
    i=0
    for _,contours in cnt.items():
        tmpX, tmpY, tmpZ =[], [], []
        CC = norm[i].astype(int)
        i += 1
        for contour in contours.allsegs:
            for seg in contour:
                a = seg[:, 0:2][:,0]
                b = seg[:, 0:2][:,1]
                x, y = ConversionCartesian2Ternary(a,b)
                tmpX.append(list(x))
                tmpY.append(list(y))
                tmpZ.append("#%02x%02x%02x" % (CC[0], CC[1], CC[2]))
        X.append(tmpX)
        Y.append(tmpY)
        Z.append(tmpZ)
    return X, Y, Z

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

def GetContoursText(cnt, strain, textt = ''):
    norm = 255*mpl.cm.viridis(mpl.colors.Normalize()(strain))
    textx, texty, text, tcolor = [], [], [], []
    if cnt is not None:
        i = 0
        for _,contours in cnt.items():
            CC = norm[i].astype(int)
            i += 1
            for contour in contours.allsegs:
                for seg in contour:
                    textx.append(0)
                    texty.append(0)
                    text.append(textt)
                    tcolor.append("#%02x%02x%02x" % (CC[0], CC[1], CC[2]))
    return textx, texty, text, tcolor

def GetContoursV2Text(cnt, strain, textt = '', AddLegendText=False):
    norm = 255*mpl.cm.viridis(mpl.colors.Normalize()(strain))
    textx, texty, textlist, tcolor = [], [], [],[]
    i = 0
    for _,contours in cnt.items():
        tmptextx, tmptexty, tmptext, tmptextcolor = [], [], [], []
        CC = norm[i].astype(int)
        for contour in contours.allsegs:
            for seg in contour:                
                COMtextx, COMtexty = center_of_mass(seg[:, 0:2])
                tmptextx.append(COMtextx)
                tmptexty.append(COMtexty)
                tmptext.append(textt)
                tmptextcolor.append("#%02x%02x%02x" % (CC[0], CC[1], CC[2]))

        if AddLegendText:
            tmptextx += [35]
            tmptexty += [94]
            tmptext += [f'Strain={strain[i]:.2f}%']
            tmptextcolor += ["#000000"]
        textx.append(tmptextx)
        texty.append(tmptexty)
        textlist.append(tmptext)
        tcolor.append(tmptextcolor)
        i += 1
    return textx, texty, textlist, tcolor

def DefHoverTools(text=['A','B','C']):
    # Default scale is 100
    # YY = 2/sqrt(3) * y; XX = x-y/sqrt(3) : Back conversion

    MyCustomx = CustomJSHover(code='''
                              const x = special_vars.x
                              const y = special_vars.y
                              const XX = x - y*0.5773502691896258
                              return XX.toFixed(2)
                          ''')
    MyCustomy = CustomJSHover(code='''
                              const y = special_vars.y
                              const YY = y*1.1547005383792517
                              return YY.toFixed(2)
                          ''')                      
    MyCustomz = CustomJSHover(code='''
                              const x = special_vars.x
                              const y = special_vars.y
                              const YY = y*1.1547005383792517
                              const XX = x - y*0.5773502691896258
                              const ZZ = 100-XX-YY
                              return ZZ.toFixed(2)
                          ''')
    TOOLTIPS = [
        (text[0], "$x{custom}%"),
        (text[1], "$y{custom}%"),
        (text[2], '$indices{custom}%'),
        #('Strain','$swatch:C')

    ]
    
    FORMATTER = {'$x':MyCustomx, '$y':MyCustomy, '$indices':MyCustomz}
    
    hover = HoverTool(tooltips = TOOLTIPS,formatters=FORMATTER)
    
    return TOOLTIPS, FORMATTER 

def MyToolTips(text=['A','B','C']):
    TOOLTIPS, FORMATTER = DefHoverTools(text=text)
    hover = HoverTool(tooltips = TOOLTIPS,formatters=FORMATTER)
    tool = [hover,'box_zoom','wheel_zoom', 'pan','reset']
    return tool

def CreateSkeliton(titletext,savefig, scale, 
                   color, line_width, text,
                   offset_bottom, offset_left,offset_right,
                   fontsize, label_colors,tick_colors,
                   tick_length, TickLabel_fontsize,step,
                   SQRT3o2):

    p = figure(sizing_mode="scale_height",
               aspect_ratio =1.155,height_policy ='fit',
               x_range=(offset_left*scale,scale+offset_right*scale),
               y_range=(offset_bottom*scale,scale*SQRT3o2+(-offset_bottom)*scale),
               tools = MyToolTips(text),)
    ClearMatplotlibAxis(p)   
    p = DefinePositionsBoundayLabels(p, scale, color=color, 
                                     line_width=line_width, text=text,
                                     offset_bottom=offset_bottom,
                                     offset_left=offset_left,offset_right=offset_right,
                                     fontsize=fontsize,label_colors=label_colors,
                                     tick_colors=tick_colors,tick_length=tick_length,
                                     TickLabel_fontsize=TickLabel_fontsize,
                                     step=step)
    
    return p

def CreateColorBar(vmin, vmax,cmappp,cbarlabel_fontsize,cbarlabel):
    color_mapper = LinearColorMapper(palette=cmappp, low=vmin, high=vmax)   
    color_bar = ColorBar(color_mapper=color_mapper,
                         major_label_text_font_size=cbarlabel_fontsize,
                         title=cbarlabel, title_text_font_size=cbarlabel_fontsize)

    return color_bar

def DrawAllContourWeb(cnt, strain, fname="patch.html", titletext=None,
                      savefig=False, scale=1, vmin=-1, vmax=1,
                      cmappp="Viridis256",color='black', 
                      line_width=4, text=['a','b','c'],
                      offset_bottom=-0.14, offset_left=-0.1,offset_right=0.1,
                      fontsize='3em', cbarlabel_fontsize='3em',cbarlabel='',
                      label_colors=['black','black','black'],
                      tick_colors=['black','black','black'],
                      tick_length=2, TickLabel_fontsize='3em',step=0.1,
                      SQRT3o2 = 0.8660254037844386):

    p = CreateSkeliton(titletext,savefig, scale, 
                       color, line_width, text,
                       offset_bottom, offset_left,offset_right,
                       fontsize, label_colors,tick_colors,
                       tick_length, TickLabel_fontsize,step,
                       SQRT3o2)
    
    color_bar = CreateColorBar(vmin, vmax,cmappp,cbarlabel_fontsize,cbarlabel)
    p.add_layout(color_bar, 'right')
    
    X, Y, Z = GetContours(cnt, strain)
    data = ColumnDataSource(dict(xx=X,
                                 yy=Y,
                                 C=Z))
    p.multi_line(xs='xx', ys='yy', color='C', source=data,  line_width=4)

    if savefig:
        output_file(fname, title="Bandgap phase diagram ")
        save(p)
        return None
    else:
        return p

def DrawAllContourWebSlider(cnt, strain, fname="patch2.html", titletext=None,
                      savefig=False, scale=1, vmin=-1, vmax=1,
                     cmappp="Viridis256",color='black', 
                     line_width=4, text=['a','b','c'],
                     offset_bottom=-0.14, offset_left=-0.1,offset_right=0.1,
                     fontsize='3em', cbarlabel_fontsize='3em',cbarlabel='',
                     label_colors=['black','black','black'],
                     tick_colors=['black','black','black'],
                     tick_length=2, TickLabel_fontsize='3em',step=0.1,
                     SQRT3o2 = 0.8660254037844386):

    p = CreateSkeliton(titletext,savefig, scale, 
                       color, line_width, text,
                       offset_bottom, offset_left,offset_right,
                       fontsize, label_colors,tick_colors,
                       tick_length, TickLabel_fontsize,step,
                       SQRT3o2)
    
    color_bar = CreateColorBar(vmin, vmax,cmappp,cbarlabel_fontsize,cbarlabel)
    p.add_layout(color_bar, 'right')
    
    X1, Y1, Z1,= GetContours(cnt, strain)
    X, Y, Z = GetContoursV2(cnt, strain)
    source = ColumnDataSource(data = dict(x=X1, y=Y1, C=Z1))
    full_source = ColumnDataSource(data = dict(x=X, y=Y, C=Z))

    p.multi_line(xs='x',ys='y', color='C', source=source, line_width=4)

    slider = Slider(start=-1, end=len(X), value=-1, value_throttled=-1, step=1, 
                    title="Snapshots",bar_color ='green',width_policy = 'fit',
                    sizing_mode ="stretch_width")
    slider_code = '''   
                    const i = cb_obj.value;
                    source.data['x'] = full_source.data['x'][i]
                    source.data['y'] = full_source.data['y'][i]
                    source.data['C'] = full_source.data['C'][i]
                    source.change.emit()
                    '''
                    
    slider_callback = CustomJS(args = dict(source=source, full_source=full_source), 
                                code = slider_code)
    
    slider.js_on_change('value', slider_callback)
    layout = column(p, slider)
    if savefig:
        output_file(fname, title="Bandgap phase diagram ")
        save(layout)
        return None
    else:
        return layout

def DrawAllContourWebSelect(cnt, strain, fname="patch3.html", titletext=None,
                            savefig=False, scale=1, vmin=-1, vmax=1,
                            cmappp="Viridis256",color='black', 
                            line_width=4, text=['a','b','c'],
                            offset_bottom=-0.14, offset_left=-0.1,offset_right=0.1,
                            fontsize='3em', cbarlabel_fontsize='3em',cbarlabel='',
                            label_colors=['black','black','black'],
                            tick_colors=['black','black','black'],
                            tick_length=2, TickLabel_fontsize='3em',step=0.1,
                            SQRT3o2 = 0.8660254037844386):
    
    p = CreateSkeliton(titletext,savefig, scale, 
                       color, line_width, text,
                       offset_bottom, offset_left,offset_right,
                       fontsize, label_colors,tick_colors,
                       tick_length, TickLabel_fontsize,step,
                       SQRT3o2)
    
    color_bar = CreateColorBar(vmin, vmax,cmappp,cbarlabel_fontsize,cbarlabel)
    p.add_layout(color_bar, 'right')
    
    X1, Y1, Z1 = GetContours(cnt, strain)
    X, Y, Z = GetContoursV2(cnt, strain)
    source = ColumnDataSource(data = dict(x=X1, y=Y1, C=Z1))
    full_source = ColumnDataSource(data = dict(x=X, y=Y, C=Z))

    p.multi_line(xs='x',ys='y', color='C', source=source, line_width=4)
    
    RANGE = np.arange(0,len(strain),dtype=int).astype(str)
    STRAIN = [f'{s:.2f}' for s in strain]
    OPTIONS = [(r, s) for r,s in zip(RANGE,STRAIN)]
    select = Select(title="Strain:",value='', options=OPTIONS)
    
    
    select_code = '''   
                    const i = cb_obj.value;
                    source.data['x'] = full_source.data['x'][i];
                    source.data['y'] = full_source.data['y'][i];
                    source.data['C'] = full_source.data['C'][i];
                    source.change.emit();
                    '''
                    
    select_callback = CustomJS(args = dict(source=source, full_source=full_source), 
                                code = select_code)
    
    select.js_on_change('value', select_callback)
    layout = column(p, select)
    if savefig:
        output_file(fname, title="Bandgap phase diagram ")
        save(layout)
        #show(layout)
        return None
    else:
        return layout
    
def DrawAllContourWebMultiSelect(cnt, strain, fname="patch5.html", titletext=None,
                                 savefig=False, scale=1, vmin=-1, vmax=1,
                                 cmappp="Viridis256",color='black', 
                                 line_width=4, text=['a','b','c'],
                                 offset_bottom=-0.14, offset_left=-0.1,offset_right=0.1,
                                 fontsize='3em', cbarlabel_fontsize='3em',cbarlabel='',
                                 label_colors=['black','black','black'],
                                 tick_colors=['black','black','black'],
                                 tick_length=2, TickLabel_fontsize='3em',step=0.1,
                                 SQRT3o2 = 0.8660254037844386):
    
    p = CreateSkeliton(titletext,savefig, scale, 
                       color, line_width, text,
                       offset_bottom, offset_left,offset_right,
                       fontsize, label_colors,tick_colors,
                       tick_length, TickLabel_fontsize,step,
                       SQRT3o2)
    
    color_bar = CreateColorBar(vmin, vmax,cmappp,cbarlabel_fontsize,cbarlabel)
    p.add_layout(color_bar, 'right')
    
    X1, Y1, Z1 = GetContours(cnt, strain)
    X, Y, Z = GetContoursV2(cnt, strain)
    source = ColumnDataSource(data = dict(x=X1, y=Y1, z=Z1))
    full_source = ColumnDataSource(data = dict(x=X, y=Y, z=Z))

    p.multi_line(xs='x',ys='y', color='z', source=source, line_width=4)

    RANGE = np.arange(0,len(strain),dtype=int).astype(str)
    STRAIN = [f'{s:.2f}' for s in strain]
    OPTIONS = [(r, s) for r,s in zip(RANGE,STRAIN)]
    select = MultiSelect(title="Strain:",value=[], options=OPTIONS)
    
    
    select_code = '''   
                    const inds = cb_obj.value;
                    const d1 = full_source.data;
                    const d2 = source.data;
                    d2['x'] = []
                    d2['y'] = []
                    d2['z'] = []
                    for (let i = 0; i < inds.length; i++) {
                            const x_dd = d1['x'][inds[i]]
                            const y_dd = d1['y'][inds[i]]
                            const z_dd = d1['z'][inds[i]]
                            for (let j = 0; j < x_dd.length; j++ ){
                                d2['x'].push(x_dd[j])
                                d2['y'].push(y_dd[j])
                                d2['z'].push(z_dd[j])
                            }
                    }
                    source.change.emit();
                    '''
                    
    select_callback = CustomJS(args = dict(full_source=full_source, source=source), 
                                code = select_code)
    
    select.js_on_change('value', select_callback)
    layout = column(p, select)
    if savefig:
        output_file(fname, title="Bandgap phase diagram ")
        save(layout)
        #show(layout)
        return None
    else:
        return layout
    
def UpdateDictionaryWithNewCNTdata(sourcedata, nkey, XX):
    sourcedata['textx'+nkey] = XX[0]
    sourcedata['texty'+nkey] = XX[1]
    sourcedata['textlist'+nkey] = XX[2]
    sourcedata['textcolor'+nkey] = XX[3]
    return sourcedata
    
def DrawAllContourWebSliderV2(cnt, strain, CoverPage=True,ContourText=None,
                              IntializeContourText='',
                              fname="patch2.html", titletext=None,
                              savefig=False, scale=1, vmin=-1, vmax=1,
                              cmappp="Viridis256",color='black', 
                              line_width=4, text=['a','b','c'],
                              offset_bottom=-0.14, offset_left=-0.1,offset_right=0.1,
                              fontsize='3em', cbarlabel_fontsize='3em',cbarlabel='',
                              label_colors=['black','black','black'],
                              tick_colors=['black','black','black'],
                              tick_length=2, TickLabel_fontsize='3em',step=0.1,
                              SQRT3o2 = 0.8660254037844386):

    p = CreateSkeliton(titletext,savefig, scale, 
                       color, line_width, text,
                       offset_bottom, offset_left,offset_right,
                       fontsize, label_colors,tick_colors,
                       tick_length, TickLabel_fontsize,step,
                       SQRT3o2)
    
    color_bar = CreateColorBar(vmin, vmax,cmappp,cbarlabel_fontsize,cbarlabel)
    p.add_layout(color_bar, 'right')
    
    if isinstance(CoverPage, bool):
        if CoverPage:
            IntializeContour = cnt[0]
        else:
            IntializeContour = None
    else:
        IntializeContour = CoverPage
        
    X_ = GetContours(IntializeContour, strain)
    X = GetContoursV2(cnt[0], strain)
    
    sourcedata = dict(x=X_[0], y=X_[1], z=X_[2])
    fullsourcedata = dict(x=X[0], y=X[1], z=X[2])
    
    cntNumber = len(cnt) # Total number of extra contour
    if cntNumber>1:
        if ContourText is None: 
            ContourText = ['']*cntNumber
        else:
            assert cntNumber-1 <= len(ContourText), 'Total number of extra contours supplied is more than number of contour texts.'
        XX_ = GetContoursText(IntializeContour, strain, textt=IntializeContourText)
        for I in range(1,cntNumber):
            nkey=str(I)
            AddLegendText=True if I==1 else False
            XX = GetContoursV2Text(cnt[I], strain, textt=ContourText[I-1], AddLegendText=AddLegendText)
            fullsourcedata = UpdateDictionaryWithNewCNTdata(fullsourcedata, nkey, XX)
            sourcedata = UpdateDictionaryWithNewCNTdata(sourcedata, nkey, XX_)
        
    source = ColumnDataSource(data = sourcedata)
    full_source = ColumnDataSource(data = fullsourcedata)
    p.multi_line(xs='x',ys='y', color='z', source=source, line_width=4)
    
    slider = Slider(start=-1, end=len(strain), value=-1, value_throttled=-1, step=1, 
                    title='Slider::Snapshots',bar_color ='green',width_policy = 'fit',
                    sizing_mode ="stretch_width",background='gray')

    slider_code = '''   
                    const i = cb_obj.value;
                    source.data['x'] = full_source.data['x'][i]
                    source.data['y'] = full_source.data['y'][i]
                    source.data['z'] = full_source.data['z'][i]
                    '''
       
    if cntNumber>1:
        for I in range(1,cntNumber):
            nkey=str(I)
            slider_code_part = f'''
                    source.data['textx{I}'] = full_source.data['textx{I}'][i]
                    source.data['texty{I}'] = full_source.data['texty{I}'][i]
                    source.data['textlist{I}'] = full_source.data['textlist{I}'][i]
                    source.data['textcolor{I}'] = full_source.data['textcolor{I}'][i]
                    '''
            slider_code += slider_code_part
            COMtexts = LabelSet(x='textx'+nkey,y='texty'+nkey, text='textlist'+nkey, 
                                text_color='textcolor'+nkey,source=source, 
                                text_font_size=fontsize)
            p.add_layout(COMtexts)
               
    slider_code += '''
                    source.change.emit();
                   '''
                        
    slider_callback = CustomJS(args = dict(source=source, full_source=full_source), 
                                code = slider_code)
    
    slider.js_on_change('value', slider_callback)
    layout = column(p, slider)
    if savefig:
        output_file(fname, title="Bandgap phase diagram ")
        save(layout)
        return None
    else:
        return layout

def DrawAllContourWebMultiSelectV2(cnt, strain, CoverPage=True,ContourText=None,
                                   IntializeContourText='',
                                   fname="patch5.html", titletext=None,
                                   savefig=False, scale=1, vmin=-1, vmax=1,
                                   cmappp="Viridis256",color='black', 
                                   line_width=4, text=['a','b','c'],
                                   offset_bottom=-0.14, offset_left=-0.1,offset_right=0.1,
                                   fontsize='3em', cbarlabel_fontsize='3em',cbarlabel='',
                                   label_colors=['black','black','black'],
                                   tick_colors=['black','black','black'],
                                   tick_length=2, TickLabel_fontsize='3em',step=0.1,
                                   SQRT3o2 = 0.8660254037844386):
    
    p = CreateSkeliton(titletext,savefig, scale, 
                         color, line_width, text,
                         offset_bottom, offset_left,offset_right,
                         fontsize, label_colors,tick_colors,
                         tick_length, TickLabel_fontsize,step,
                         SQRT3o2)
    
    color_bar = CreateColorBar(vmin, vmax,cmappp,cbarlabel_fontsize,cbarlabel)
    p.add_layout(color_bar, 'right')
    
    if isinstance(CoverPage, bool):
        if CoverPage:
            IntializeContour = cnt[0]
        else:
            IntializeContour = None
    else:
        IntializeContour = CoverPage
        
    X_ = GetContours(IntializeContour, strain)
    X = GetContoursV2(cnt[0], strain)
    
    sourcedata = dict(x=X_[0], y=X_[1], z=X_[2])
    fullsourcedata = dict(x=X[0], y=X[1], z=X[2])
    
    cntNumber = len(cnt) # Total number of extra contour
    if cntNumber>1:
        if ContourText is None: 
            ContourText = ['']*cntNumber
        else:
            assert cntNumber-1 <= len(ContourText), 'Total number of extra contours supplied is more than number of contour texts.'
        XX_ = GetContoursText(IntializeContour, strain, textt=IntializeContourText)
        for I in range(1,cntNumber):
            nkey=str(I)
            XX = GetContoursV2Text(cnt[I], strain, textt=ContourText[I-1])
            fullsourcedata = UpdateDictionaryWithNewCNTdata(fullsourcedata, nkey, XX)
            sourcedata = UpdateDictionaryWithNewCNTdata(sourcedata, nkey, XX_)
        
    source = ColumnDataSource(sourcedata)
    full_source = ColumnDataSource(fullsourcedata)
    p.multi_line(xs='x',ys='y', color='z', source=source, line_width=4)

    RANGE = np.arange(0,len(strain),dtype=int).astype(str)
    STRAIN = [f'{s:.2f}' for s in strain]
    OPTIONS = [(r, s) for r,s in zip(RANGE,STRAIN)]
    select = MultiSelect(title="Select strain(s) from below. Use shift key for multiple-select.",value=[], options=OPTIONS)
    
    select_code = '''   
                    const inds = cb_obj.value;
                    const d1 = full_source.data;
                    const d2 = source.data;
                    d2['x'] = []
                    d2['y'] = []
                    d2['z'] = []
                    for (let i = 0; i < inds.length; i++) {
                            const x_dd = d1['x'][inds[i]]
                            const y_dd = d1['y'][inds[i]]
                            const z_dd = d1['z'][inds[i]]
                            for (let j = 0; j < x_dd.length; j++ ){
                                d2['x'].push(x_dd[j])
                                d2['y'].push(y_dd[j])
                                d2['z'].push(z_dd[j])
                            }
                    }
                    '''
    
    if cntNumber>1:
        for I in range(1,cntNumber):
            nkey=str(I)
            select_code_part = f'''
                    d2['textx{I}'] = []
                    d2['texty{I}'] = []
                    d2['textlist{I}'] = []
                    d2['textcolor{I}'] = []
                    for (let i = 0; i < inds.length; i++) {{
                            const x_dd = d1['textx{I}'][inds[i]]
                            const y_dd = d1['texty{I}'][inds[i]]
                            const z_dd = d1['textlist{I}'][inds[i]]
                            const c_dd = d1['textcolor{I}'][inds[i]]
                            for (let j = 0; j < x_dd.length; j++ ){{
                                    d2['textx{I}'].push(x_dd[j])
                                    d2['texty{I}'].push(y_dd[j])
                                    d2['textlist{I}'].push(z_dd[j])
                                    d2['textcolor{I}'].push(c_dd[j])
                            }}
                    }}
                    '''
            select_code += select_code_part
            COMtexts = LabelSet(x='textx'+nkey,y='texty'+nkey, text='textlist'+nkey, 
                                text_color='textcolor'+nkey, source=source, 
                                text_font_size=fontsize, )
            p.add_layout(COMtexts)
            
    select_code += '''
                    source.change.emit();
                    '''
                
    select_callback = CustomJS(args = dict(full_source=full_source, source=source), 
                                code = select_code)
    
    select.js_on_change('value', select_callback)
    layout = column(p, select)
    if savefig:
        output_file(fname)
        save(layout)
        #show(layout)
        return None
    else:
        return layout
    
def MergeSliderSelect(layoutlist,fname='patch4.html' ):
    curdoc().clear()
    LAYOUT = row(children=layoutlist, sizing_mode="scale_height")
    curdoc().add_root(LAYOUT)
    output_file(fname, title="Bandgap phase diagram ")
    save(LAYOUT)
    
#%%------------------- Using plotly models ------------------------------------
def GetContoursPlotly(cnt):
    X, Y,  Z =  [], [],[]
    for strain,contours in cnt.items():
        for contour in contours.allsegs:
            for seg in contour:
                a = seg[:, 0:2][:,0]
                b = seg[:, 0:2][:,1]
                x, y = ConversionCartesian2Ternary(a,b)
                X+=list(x)
                Y+=list(y)

                Z+=[strain]*len(x)
    return X, Y, Z

def DrawBoundaryPlotly(scale):
    ax = np.array([0,scale,0,0])
    ay = np.array([0,0,scale,0])    
    XX, YY = ConversionCartesian2Ternary(ax,ay)
    return  XX, YY 

def DrawAllContour3D(cnt, strain, fname="file.html", 
                     savefig=False, scale=100,
                     PlotScatter=True):
    fig = go.Figure()
    if PlotScatter:
        X, Y, Z = GetContoursPlotly(cnt)
        fig.add_trace(go.Scatter3d(
                    x=X,
                    y=Y,
                    z=Z,
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=Z,                # set color to an array/list of desired values
                        colorscale='Viridis',   # choose a colorscale
                        opacity=0.8,
                        colorbar=dict(title="Strain(%)"),
                        )
                    ))
    else:
        for strain,contours in cnt.items():
            for contour in contours.allsegs:
                for seg in contour:
                    a = seg[:, 0:2][:,0]
                    b = seg[:, 0:2][:,1]
                    x, y = ConversionCartesian2Ternary(a,b)
                    fig.add_trace(go.Scatter3d(x=x,y=y,z=[strain]*len(x),
                                               mode='lines',
                                               line=dict(width=5,color=strain,                
                                                         colorscale='Viridis',   
                                                         #colorbar=dict(title="Strain(%)"),
                                                         ),
                                               )   
                                  )
        
    XX, YY= DrawBoundaryPlotly(scale)
    for zz in strain:
        fig.add_trace(go.Scatter3d(x=XX, y=YY, z=np.array([zz]*len(XX)),
                        mode='lines',
                        line=dict(color='black',width=2)
                        ))

    # tight layout
    fig.update_layout(scene = dict(
                    xaxis = dict(nticks=10, range=[0,scale],visible=False),
                    yaxis = dict(nticks=10, range=[0,0.866*scale],visible=False),
                    zaxis = dict(visible=False),),
                    showlegend=False,
                    #width=700,
                    #margin=dict(l=0, r=0, b=0, t=0),
                    )
    #fig.show()
    fig.write_html(fname)

    return 

#%%----------------------------------------------------------------------------
# All the (bandgap) heatmap plots are available in MLmodelWebPlottingFunctionsPart2.py 
