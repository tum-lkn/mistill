from array import array
import json
import glob
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as bcknd
from dataclasses import dataclass
import sys
import random
#######################################################
# run this script with command line input as follows:
# python3 plot_info.py up
# or
# python3 plot_info.py down
# or
# python3 plot_info.py react
# to plot infos for the links if tor0-eth5 is either up or down
# the plot settings are set into the following dataclasses
# react plots the reaction time of the nn to a link failure
# further settings start in line 147
#######################################################
LABELSIZE = 12
COLORS = ['#80b1d3','#fb8072','#fdb462','#bebada','#8dd3c7']
WIDTH_FACTOR = 1
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
})

@dataclass
class plot_object: # used for x and y axis
    label: str # label of this axis with unit (if necessary)
    useticks: bool # use ticks on this axis? yes or no
    ticks: int # values for the axis ticks
    tick_labels: str # labels for the axis ticks
    flip_axis: bool # sort axis values from high to low instead of low to high
@dataclass
class plot_settings: # used for mean, median and percentiles
    show: bool # show this property? yes or no
    marker: str # type of marker
    color: str # color of marker
    edgecolor: str # edgecolor of marker
    size: int # size of marker
@dataclass
class plot_info: # used as input of the plotting functions
    savename: str # filename of the resulting pdf
    x: plot_object # plot_object for the x axis data
    y: plot_object # plot_object for the y axis data
    plot_pos: int # positions of the violin plots on the x axis
    ylim_tolerance: int # tolerance around the most extreme plottet data
    lbrt: array # left, bottom, right, top: margins for the plot in the pdf file
    mea: plot_settings # plot_settings for the mean
    med: plot_settings # plot_settings for the median
    per: plot_settings # plot_settings for the percentiles
    per_tolerance: int # percentile tolerance

def read_jsons(read_path,operstate):
    # read link utilization and bytes from json into list
    # operstate is either up or down and only the specified data is extracted
    # so, for up, the data of all four links is collected if link5 is up
    # for down, the data of all four links is collected if link5 is down
    list_of_dicts = []
    for name in glob.glob(read_path):
        with open(name, 'r') as jfile:values = json.load(jfile)
        for x in values['Data']:
            if x['tor0-eth5']["operstate"] == operstate:
                list_of_dicts.append({'link5_tx': x['tor0-eth5']["tx_bytes"], 'link6_tx': x['tor0-eth6']["tx_bytes"], 'link7_tx': x['tor0-eth7']["tx_bytes"], 'link8_tx': x['tor0-eth8']["tx_bytes"]})
    return list_of_dicts

def read_jsons_reaction_patrick(read_path):
    # read link utilization and bytes from json into list
    # operstate is either up or down and only the specified data is extracted
    # so, for up, the data of all four links is collected if link5 is up
    # for down, the data of all four links is collected if link5 is down
    # SPECIAL CASE: for the reaction time
    with open(read_path, 'r') as fh:
        data = json.load(fh)
    eth5 = [d['tor0-eth5'] for d in data['Data']]
    eth6 = [d['tor0-eth6'] for d in data['Data']]
    df5 = pd.DataFrame.from_dict(eth5)
    df6=pd.DataFrame.from_dict(eth6)
    df5.loc[:, 'operstate'] = df5.loc[:, 'operstate'].apply(lambda x: float(x == 'up')).values
    df6.loc[:, 'operstate'] = df6.loc[:, 'operstate'].apply(lambda x: float(x == 'up')).values
    diff5 = df5.diff().dropna()
    diff6 = df6.diff().dropna()
    diff5.iloc[:, 0] = df5.iloc[1:, 0].values
    diff6.iloc[:, 0] = df6.iloc[1:, 0].values
    ts5_switch_off = diff5.loc[diff5.operstate == -1, 'ts']
    delays = []
    for i in range(ts5_switch_off.shape[0]):
        a = diff6.loc[diff6.ts.values > ts5_switch_off.iloc[i], :].iloc[:100]
        idx = np.argmax(a.tx_bytes.values > 140)
        dl = a.iloc[idx, 0] - ts5_switch_off.iloc[i]
        delays.append({'reaction time': ((a.iloc[idx, 0] - ts5_switch_off.iloc[i]) * 1000)})
    return delays

def __my_violin_plot(data, info):
    # function to plot a number of violin plots next to each other
    # the data is plotted to given percentiles -> the outliers are removed on both sides
    # mean and median are plotted as well
    # 
    # data contains the data, that is to be plotted
    # info contains additional information to individualize the plot

    # width as measured in inkscape
    width = 3.487 * WIDTH_FACTOR # in inches
    height = width / 1.618
    pdf = bcknd.PdfPages(info.savename)
    mpl.rc('xtick', labelsize=LABELSIZE)
    mpl.rc('ytick', labelsize=LABELSIZE)
    mpl.rc('axes', labelsize=LABELSIZE)
    mpl.rc('legend', fontsize=LABELSIZE)
    # mpl.rcParams['font.family'] =  'serif'

    fig, axes = plt.subplots()
    ylim_min = []
    ylim_max = []
    if(info.per.show == True):
        for n,dat in enumerate(data):
            smin = np.percentile(dat,info.per_tolerance)
            smax = np.percentile(dat,100-info.per_tolerance)
            print("ylim_min: " + str(smin) + " ylim_max: " + str(smax))
            ylim_min.append(smin)
            ylim_max.append(smax)

    print(ylim_min)
    print(ylim_max)
    data_clean = []
    for i,dat in enumerate(data):
        if ylim_min[i] == ylim_max[i]:
            d2 = dat[np.logical_and(dat > (ylim_min[i] - 1), (dat < ylim_max[i] + 1))]
            d2[-1] = 1
            data_clean.append(d2)
            print("ylim min = ylim max; data_clean appended")
        else:
            data_clean.append(dat[np.logical_and(dat > (ylim_min[i]-1), dat < (ylim_max[i]+1))])
            print("data cleaned and appended to data_clean")
    try:
        parts = axes.violinplot(dataset=data_clean,positions=info.plot_pos,widths=0.8,showextrema=False)
    except ValueError: # sometimes this error occured, so it is now catched and the uncleaned data is plotted instead
        print("ValueError detected: printing uncleaned data instead")
        parts = axes.violinplot(dataset=data,positions=info.plot_pos,widths=0.8,showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor(COLORS[0])
        pc.set_edgecolor(COLORS[0])
        pc.set_alpha(0.3)
        pc.set_linewidth(0.5)
    for n,dat in enumerate(data_clean):
        axes.scatter([info.plot_pos[n] for i in range(100)], dat[np.random.randint(0, dat.size, 100)], marker='o', c=COLORS[0], s=5, alpha=0.3)
    if(info.mea.show == True):
        for n,dat in enumerate(data_clean):
            sm = np.mean(dat)
            # plot the mean into each single violin
            axes.scatter(info.plot_pos[n], sm, marker=info.mea.marker,edgecolors=info.mea.edgecolor, color=info.mea.color,s=info.mea.size,linewidths=0.5,alpha=1)
    if(info.med.show == True):
        for n,dat in enumerate(data_clean):
            sme = np.median(dat)
            # plot the median into each single violin
            axes.scatter(info.plot_pos[n], sme, marker=info.med.marker, color=info.med.color,s=info.med.size,linewidths=0.5,alpha=1)

    if(info.x.flip_axis == True):
        axes.invert_xaxis()
    if(info.y.flip_axis == True):
        axes.invert_yaxis()
    plt.xlabel(info.x.label)
    if(info.x.useticks == True):
        plt.xticks(ticks=info.x.ticks,labels=info.x.tick_labels)
    plt.ylabel(info.y.label)
    if(info.y.useticks == True):
        plt.yticks(ticks=info.y.ticks,labels=info.y.tick_labels)
    # adjust the distance of the plot the borders of the file
    plt.subplots_adjust(left=info.lbrt[0],bottom=info.lbrt[1],right=info.lbrt[2],top=info.lbrt[3])
    if(info.per.show == True):
        # automatically limits the plot to the lowest and highest plotted point plus a set tolerance
        # plt.ylim([-50, 1620])
        # plt.ylim([np.min(ylim_min) - info.ylim_tolerance, np.max(ylim_max) + info.ylim_tolerance])
        plt.ylim([27,61])
        print("ylim_min: " + str(np.min(ylim_min)) + " ylim_tolerance: " + str(info.ylim_tolerance) + " ylim_max: " + str(np.max(ylim_max)))
    fig.set_size_inches(width, height)
    pdf.savefig(fig)
    pdf.close()

def __my_violin_plot2(data, info):
    # function to plot a number of violin plots next to each other
    # the data is plotted to given percentiles -> the outliers are removed on both sides
    # mean and median are plotted as well
    # 
    # data contains the data, that is to be plotted
    # info contains additional information to individualize the plot

    # width as measured in inkscape
    width = 3.487 * WIDTH_FACTOR # in inches
    height = width / 1.618
    pdf = bcknd.PdfPages(info.savename)
    mpl.rc('xtick', labelsize=LABELSIZE)
    mpl.rc('ytick', labelsize=LABELSIZE)
    mpl.rc('axes', labelsize=LABELSIZE)
    mpl.rc('legend', fontsize=LABELSIZE)
    # mpl.rcParams['font.family'] =  'serif'

    fig, axes = plt.subplots()
    ylim_min = []
    ylim_max = []
    if(info.per.show == True):
        for n,dat in enumerate(data):
            smin = np.percentile(dat,info.per_tolerance)
            smax = np.percentile(dat,100-info.per_tolerance)
            print("ylim_min: " + str(smin) + " ylim_max: " + str(smax))
            ylim_min.append(smin)
            ylim_max.append(smax)
    parts = axes.violinplot(dataset=data,positions=info.plot_pos,widths=0.8,showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor(COLORS[0])
        pc.set_edgecolor(COLORS[0])
        pc.set_alpha(0.3)
        pc.set_linewidth(0.5)
    for n,dat in enumerate(data):
        axes.scatter([info.plot_pos[n] for i in range(100)], dat[np.random.randint(0, dat.size, 100)], marker='o', c=COLORS[0], s=5, alpha=0.3)
    if(info.mea.show == True):
        for n,dat in enumerate(data):
            sm = np.mean(dat)
            # plot the mean into each single violin
            axes.scatter(info.plot_pos[n], sm, marker=info.mea.marker,edgecolors=info.mea.edgecolor, color=info.mea.color,s=info.mea.size,linewidths=0.5,alpha=1)
    if(info.med.show == True):
        for n,dat in enumerate(data):
            sme = np.median(dat)
            # plot the median into each single violin
            axes.scatter(info.plot_pos[n], sme, marker=info.med.marker, color=info.med.color,s=info.med.size,linewidths=0.5,alpha=1)

    if(info.x.flip_axis == True):
        axes.invert_xaxis()
    if(info.y.flip_axis == True):
        axes.invert_yaxis()
    plt.xlabel(info.x.label)
    if(info.x.useticks == True):
        plt.xticks(ticks=info.x.ticks,labels=info.x.tick_labels)
    plt.ylabel(info.y.label)
    if(info.y.useticks == True):
        plt.yticks(ticks=info.y.ticks,labels=info.y.tick_labels)
    # adjust the distance of the plot the borders of the file
    plt.subplots_adjust(left=info.lbrt[0],bottom=info.lbrt[1],right=info.lbrt[2],top=info.lbrt[3])
    if(info.per.show == True):
        # automatically limits the plot to the lowest and highest plotted point plus a set tolerance
        # plt.ylim([-50, 1620])
        plt.ylim([np.min(ylim_min) - info.ylim_tolerance, np.max(ylim_max) + info.ylim_tolerance])
        print("ylim_min: " + str(np.min(ylim_min)) + " ylim_tolerance: " + str(info.ylim_tolerance) + " ylim_max: " + str(np.max(ylim_max)))
    fig.set_size_inches(width, height)
    pdf.savefig(fig)
    pdf.close()

load = "/home/zeidler/dc-mininet-emulation/stat_info/"
mea = plot_settings(True,"o","w","k",30) # mean
med = plot_settings(True,"_","k","k",100) # median
per = plot_settings(True,"_","b","k",200) # percentiles
loadname = load + "test_link_ud_1000pps_cpu.json"
mode = sys.argv[1]
if mode == "react":
    data = []
    list_j = read_jsons_reaction_patrick(loadname)
    df = pd.DataFrame(list_j)
    data.append(df["reaction time"].values)
    
    # loadname = load + "test_link_ud_1000pps_gpu.json"
    # list_j = read_jsons_reaction_patrick(loadname)
    # df = pd.DataFrame(list_j)
    # data.append(df["reaction time"].values)
    plot_pos = [1]
    x_ticks = [1]
    x_tick_labels = ["Reaction"]
    # plot_pos = [1,2]
    # x_ticks = [1,2]
    # x_tick_labels = ["CPU","GPU"]
    x = plot_object('',True,x_ticks, x_tick_labels,False)
    y_ticks = [0,5,10,15,20,25,30,40,50,60,70,80]
    y_tick_labels = ["0","5","10","15","20","25","30","40","50","60","70","80"]
    # y_ticks = [0,50,100,150,200,250,300]
    # y_tick_labels = ["0","50","100","150","200","250","300"]
    y = plot_object('Time [ms]  ',True,y_ticks, y_tick_labels,False)
    savename = load + "plots/reaction_time_1000pps.pdf"
    # info = plot_info(savename,x,y,plot_pos,3,[0.185,0.27,0.99,0.99],mea,med,per,1) # patrick
    info = plot_info(savename,x,y,plot_pos,3,[0.15,0.15,0.99,0.99],mea,med,per,1)
    __my_violin_plot(data,info)
    # ---------------------------------------------------------------------------------------------
else:
    list_j = read_jsons(loadname, mode)
    df = pd.DataFrame(list_j)
    df_d = df.diff() # calculate the rates from the absolute amount of bits
    df_d.info()
    data = []
    a = []
    if mode == "up": # since data was only collected every 1.5 or so ms, 
        # this regulates back to 1ms by removing/ changing the data where no or two packets arrived in one time slot..
        for x in df_d["link5_tx"].values[1:]:
            if x == 0:
                x = 149
                a.append(int(x))
            if x == 300:
                x = 151
                a.append(int(x))
            else:
                a.append(int(x))
        data.append(pd.DataFrame(a).values)
        data.append(df_d["link6_tx"].values[1:])
    
    if mode == "down":
        for x in df_d["link6_tx"].values[1:]:
            if x == 300:
                x = 151
                a.append(int(x))
            if x == 350:
                x = 201
                a.append(int(x))
            else:
                a.append(int(x))
        data.append(df_d["link5_tx"].values[1:])
        data.append(pd.DataFrame(a).values)
    # data.append(df_d["link5_tx"].values[1:]) # the first value is 0 due to the diff() operation
    # data.append(df_d["link6_tx"].values[1:])
    data.append(df_d["link7_tx"].values[1:])
    data.append(df_d["link8_tx"].values[1:])
    print(data)
    print(pd.DataFrame(a).values)
    print(df_d["link8_tx"].values[1:])

    plot_pos = [1,2,3,4]
    x_ticks = [1,2,3,4]
    x_tick_labels = ["eth5","eth6","eth7","eth8"]
    x = plot_object('Links of tor0',True,x_ticks, x_tick_labels,False)
    y_ticks = [0,50,100,150,200,250,300,350,400]
    y_tick_labels = ["0","50","100","150","200","250","300","350","400"]
    y = plot_object('Transmitted rate [kB/s]  ',True,y_ticks, y_tick_labels,False)
    # savename = load + "plots/link_ud_h16_" + mode + "_ebpf2.pdf"
    savename = load + "plots/link_ud_1000pps_" + mode + "_cpu_clean.pdf"
    info = plot_info(savename,x,y,plot_pos,10,[0.185,0.222,0.985,0.975],mea,med,per,1)
    # __my_line_plot(data,info)
    __my_violin_plot(data,info)
    # ---------------------------------------------------------------------------------------------
    
