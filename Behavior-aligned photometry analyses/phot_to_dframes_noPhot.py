# # Script to create and save sampleframe and triframe to directory


import numpy as np
import pandas as pd
import ipywidgets as widgets
import ipympl
import csv
import matplotlib.pyplot as plt
import scipy.io
from sklearn.linear_model import Lasso
from scipy.signal import lfilter,resample,decimate,correlate
from matplotlib.lines import Line2D
import process_ephys_visitData
from TriFuncs import * #custom functions for following analysis
#get_ipython().run_line_magic('matplotlib', '')

#get variables from main script
from __main__ import *

#set path to access previously stored .mat files, etc. use scipy.io.loadmat()
datepath = filepath + date + '/'
savepath = datepath
phot_dlc = input('photometry DLC? y/n')

from photProcessingFuncs import *

ephysSR = 1500 #sampling rate of ephys system for vis
Fs = 250
VizGetter = process_ephys_visitData.EphysVisitGetter(animal,date,rec_folder,ephysSR)
VizGetter.prepData()
n_seconds = VizGetter.getSeshDurationInS()
# fill in nansig with nan vector of length Fs*n_seconds
nansig = np.full(int(Fs*(n_seconds+10)),np.nan)
#need to get visits from ephys data. should be 
visits = VizGetter.get_visit_times()
visits = visits[1:] - visits[0]
visits = np.unique(visits)
print(visits.astype(int))

shiftVisByN = int(input("adjust visits by removing n indices from start?"+\
    " (input n; if 1 is first index, input 1)"))
visits = visits[shiftVisByN:]

#get sample numbers of visit indices
visits = np.divide(visits,ephysSR/Fs)
visits = visits.astype(int)


#import arduino behavioral data and their timestamps
a = open(datepath+'arduinoraw'+str(filecount)+'.txt','r')
ardtext= a.read().splitlines()
a.close

with open(datepath+'ArduinoStamps'+str(filecount)+'.csv','r') as at:
    reader = csv.reader(at)
    ardtimes = list(reader)
ardtimes = [float(val) for sublist in ardtimes for val in sublist]

photstart = ardtimes[1] #time when pulse was sent to R series. 0 index for sig data
ardtimes = np.subtract(ardtimes,photstart) #make photometry start time zero

#convert ardtimes to sample number to match photometry data. 
#Target rate = 250/sec from 10 KHz initial sample rate (same as photometry)
ardstamps = np.round(ardtimes*(Fs/1000)).astype(int)


vis4pos = visits
x,y,vel,framestamps,acc,dlc_pos = align_pos_to_visits(Fs,vis4pos,datepath,
    phot_dlc=phot_dlc,filecount=filecount,gaus_smooth=usePosSmoothing,sigma=gausSigma,cutoff=0.4)

plt.figure()
plt.plot(framestamps)
plt.title("camera frame stamps in photometry sample time")


#Make dataframe with all data organized by sample number
visits = np.unique(visits)

#This is where the problem is... not the right length.
a = np.tile(0,(len(nansig),22))
data = np.full_like(a, np.nan, dtype=np.double) #make a sample number x variable number array of nans
#fill in nans with behavioral data. columns = x,y,ach,dlight,port,rwd,roi
#give signal columns dff values
data[:,2] = nansig 
data[:,3] = 0
data[:,16] = nansig
data[:,17] = 0
data[:,11] = nansig
block = 1
tris = 0
trial = 1
data[0,15] = 0
pA = np.nan
pB = np.nan
pC = np.nan
#get sample number of current trial to align phot data to port entries
#data = list(data)
#fill in reward and port info. Ports are coded by 1:A,2:B,3:C
for i in range(len(ardtext)):
    try:
        vsamp = visits[tris]
    except:
        print('end of trials reached')
    new = False
    if 'A Harvested' in ardtext[i] or "delivered at port A" in ardtext[i]:
        data[vsamp,4] = 0
        data[vsamp,5] = 1
        data[vsamp+1,15] = 0
        trial += 1
        tris += 1
    elif 'B Harvested' in ardtext[i] or "delivered at port B" in ardtext[i]:
        data[vsamp,4] = 1
        data[vsamp,5] = 1
        data[vsamp+1,15] = 1
        trial += 1
        tris += 1
    elif 'C Harvested' in ardtext[i] or "delivered at port C" in ardtext[i]:
        data[vsamp,4] = 2
        data[vsamp,5] = 1
        data[vsamp+1,15] = 2
        trial += 1
        tris += 1
    elif 'o Reward port A' in ardtext[i]:
        data[vsamp,4] = 0
        data[vsamp,5] = 0
        data[vsamp+1,15] = 0
        trial += 1
        tris += 1
    elif 'o Reward port B' in ardtext[i]:
        data[vsamp,4] = 1
        data[vsamp,5] = 0
        data[vsamp+1,15] = 1
        trial += 1
        tris += 1
    elif 'o Reward port C' in ardtext[i]:
        data[vsamp,4] = 2
        data[vsamp,5] = 0
        data[vsamp+1,15] = 2
        trial += 1
        tris += 1
    elif 'Lick at port A' in ardtext[i] or "beam break at port A" in ardtext[i]:
        data[ardstamps[i],18] = 1
    elif 'Lick at port B' in ardtext[i] or "beam break at port B" in ardtext[i]:
        data[ardstamps[i],19] = 1
    elif 'Lick at port C' in ardtext[i] or "beam break at port C" in ardtext[i]:
        data[ardstamps[i],20] = 1
    elif "Block" in ardtext[i]:
        block = int(ardtext[i][-1]) + 1
        new = True
        pA = ardtext[i+1][-2:]
        pB = ardtext[i+2][-2:]
        pC = ardtext[i+3][-2:]
    data[ardstamps[i],6] = block #block number
    data[ardstamps[i],7] = pA
    data[ardstamps[i],8] = pB
    data[ardstamps[i],9] = pC
    if new == True:
        trial = 0 #reset trials within block
    data[ardstamps[i],14] = trial
    data[ardstamps[i],21]

data[framestamps[:len(x)],0] = x
data[framestamps[:len(x)],1] = y
data[framestamps[:len(x)],10] = framestamps[:len(x)]
data[framestamps[:len(x)],12] = vel
data[framestamps[:len(x)],13] = acc

#if only dLight
sampledata = pd.DataFrame(data,columns = ['x','y','green','red','port','rwd','block','pA','pB','pC','frame','ref',\
    'vel','acc','tri','fromP','470','565','beamA','beamB','beamC','tot_tri'])

sampledata = sampledata.drop(['red','565'],axis=1)
visinds = sampledata.loc[sampledata.port.notnull()].index.values
tritimes = np.diff(visinds)/Fs
tts = plt.figure()
plt.title('Distribution of trial times')
plt.hist(tritimes,bins=100)
plt.ylabel('# of trials')
plt.xlabel('trial time (s)')
tts.savefig(datepath+'trial_time_dist.pdf')




#sampledata.loc[(sampledata.x.diff()==0)|(sampledata.y.diff()==0),['x','y']]=np.nan
#sampledata.loc[(sampledata.x>550)|(sampledata.x<100),['x','y']]=np.nan

plt.figure()
plt.plot(sampledata.loc[(sampledata.x.notnull()),"x"].values[:visinds[-1]],
            sampledata.loc[(sampledata.x.notnull()),"y"].values[:visinds[-1]])
plt.plot(x,y,color='darkorange')

plt.figure()
plt.scatter(sampledata.loc[visinds[20]:visinds[21],'x'],sampledata.loc[visinds[20]:visinds[21],'y'])
plt.scatter(sampledata.loc[visinds[21]:visinds[22],'x'],sampledata.loc[visinds[21]:visinds[22],'y'])


#z-score and save signal as zscore
zscored = np.divide(np.subtract(sampledata.green,sampledata.green.mean()),sampledata.green.std())
sampledata['green_z_scored'] = zscored
#zscored = np.divide(np.subtract(sampledata.red,sampledata.red.mean()),sampledata.red.std())
#sampledata['red_z_scored'] = zscored

#add column indicating recording location
sampledata['fiberloc'] = np.nan

#create list of repeating session type string for addition to data frame
sampledata['session_type']=ses_type

#note which date and animal data comes from
anlist = np.full_like(data[:,15],0)
anlist = [animal for i in anlist]
sampledata['rat']=anlist
datelist = np.full_like(data[:,15],0)
datelist = [date for i in datelist]
sampledata['date']=datelist

#sampledata['short_tri']=short
sampledata['frame']=sampledata.frame.fillna(method='ffill')
sampledata['pA']=sampledata.pA.fillna(method='ffill')
sampledata['pA']=sampledata.pA.fillna(method='bfill')
sampledata['pB']=sampledata.pB.fillna(method='ffill')
sampledata['pB']=sampledata.pB.fillna(method='bfill')
sampledata['pC']=sampledata.pC.fillna(method='ffill')
sampledata['pC']=sampledata.pC.fillna(method='bfill')
sampledata['block']=sampledata.block.fillna(method='ffill')
sampledata['tri']=sampledata.tri.fillna(method='ffill')
sampledata['fromP']=sampledata.fromP.fillna(method='ffill')

ps = ['pA','pB','pC']
tridat = sampledata.loc[sampledata.port.isnull()==False]
tridat = tridat.reset_index()
ports=tridat.port.values
nextp=ports[1:]
nextprob = []
for n in range(len(nextp)):
    nextprob.append(tridat.loc[n,ps[int(nextp[int(n)])]])
npo = pd.DataFrame()
npo['nextprob'] = nextprob
npo['nextport'] = nextp
#add back to sampledata
addto = sampledata.loc[sampledata.port.isnull()==False].index[:-1]
npo = npo.set_index(addto)
npo = npo.reindex(np.arange(sampledata.index[0],sampledata.index[-1]+1),fill_value=np.nan)
sampledata['nextprob'] = npo.nextprob.fillna(method='ffill')
sampledata['nextp'] = npo.nextport.fillna(method='ffill')
tridat['nextp'] = np.concatenate([nextp,[10]])
tridat['lrchoice'] = make_lr(tridat.port.values)

xytups = [tuple(t) for t in sampledata[['x','y']].dropna().values.tolist()] #create list of point tuples

#also want these in sampledata. Should do block by block if ses_type == 'barrier'
if ses_type=='prob':
    tridat['lenAC'] = input('length of A-C?')
    tridat['lenBC'] = input('length of B-C?')
    tridat['lenAB'] = input('length of A-B?')
else:
    tridat['lenAC']=0
    tridat['lenBC']=0
    tridat['lenAB']=0
    for b in tridat.block.unique():
        tridat.loc[tridat.block==b,'lenAC'] = input('min length of A-C in block '+str(b)+'?')
        tridat.loc[tridat.block==b,'lenBC'] = input('min length of B-C in block '+str(b)+'?')
        tridat.loc[tridat.block==b,'lenAB'] = input('min length of A-B in block '+str(b)+'?')
plengths = pd.DataFrame()
plengths['lenAC'] = tridat.lenAC
plengths['lenAB'] = tridat.lenAB
plengths['lenBC'] = tridat.lenBC
addto = visinds
plengths = plengths.set_index(addto)
plengths = plengths.reindex(np.arange(sampledata.index[0],sampledata.index[-1]+1),fill_value=np.nan)

sampledata['lenAC'] = plengths['lenAC'].fillna(method='bfill')
sampledata['lenBC'] = plengths['lenBC'].fillna(method='bfill')
sampledata['lenAB'] = plengths['lenAB'].fillna(method='bfill')

#compute a simple reward rate estimate (rewards per minute)
rr = simple_rr(sampledata)
sampledata['simple_rr'] = rr.fillna(0)

addto = visinds
pchosen=[]
for i in visinds:
    if sampledata.loc[i,'port'] == 0:
        pchosen.append(sampledata.loc[i,'pA'])
    elif sampledata.loc[i,'port'] == 1:
        pchosen.append(sampledata.loc[i,'pB'])
    elif sampledata.loc[i,'port'] == 2:
        pchosen.append(sampledata.loc[i,'pC'])
pchosen = pd.DataFrame(pchosen)
pchosen = pchosen.set_index(addto)
pchosen = pchosen.reindex(np.arange(sampledata.index[0],sampledata.index[-1]+1),fill_value=np.nan)
sampledata['pchosen'] = pchosen.fillna(method='bfill')

#add column for the distance to the next (chosen) port
addto = visinds[1:]
dtop=[]
for i in addto:
    if sampledata.loc[i,'port'] == 0 and sampledata.loc[i,'fromP']==1:
        dtop.append(sampledata.loc[i,'lenAB'])
    elif sampledata.loc[i,'port'] == 1 and sampledata.loc[i,'fromP']==0:
        dtop.append(sampledata.loc[i,'lenAB'])
    elif sampledata.loc[i,'port'] == 0 and sampledata.loc[i,'fromP'] == 2:
        dtop.append(sampledata.loc[i,'lenAC'])
    elif sampledata.loc[i,'port'] == 2 and sampledata.loc[i,'fromP'] == 0:
        dtop.append(sampledata.loc[i,'lenAC'])
    elif sampledata.loc[i,'port'] == 1 and sampledata.loc[i,'fromP'] == 2:
        dtop.append(sampledata.loc[i,'lenBC'])
    elif sampledata.loc[i,'port'] == 2 and sampledata.loc[i,'fromP'] == 1:
        dtop.append(sampledata.loc[i,'lenBC'])
dtop = pd.DataFrame(dtop)
dtop = dtop.set_index(addto)
dtop = dtop.reindex(np.arange(sampledata.index[0],sampledata.index[-1]+1),fill_value=np.nan)
sampledata['dtop'] = dtop.fillna(method='bfill')

#fill beam nans with 0s
sampledata['beamA'] = sampledata.beamA.fillna(0)
sampledata['beamB'] = sampledata.beamB.fillna(0)
sampledata['beamC'] = sampledata.beamC.fillna(0)




sampledata['nom_rwd_a'] = sampledata.pA
sampledata['nom_rwd_b'] = sampledata.pB
sampledata['nom_rwd_c'] = sampledata.pC

sampledata = reduce_mem_usage(sampledata)
#save dataframes
sampledata.to_csv(savepath+'sampleframe.csv')
tridat.to_csv(savepath+'triframe.csv')

