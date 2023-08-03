
# coding: utf-8

import math
import numpy as np
from shapely.geometry import Point,MultiPoint,Polygon,MultiPolygon
import shapely.vectorized
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
#import pyproj
from shapely.ops import transform

fs = 250

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]
def closestlong(array,value):
    return(array[np.abs(np.subtract(array,value)).argmin()])

def project_onto_path(posmat,path):
    #path = path.reshape((1, 2))  # A row vector
    c_values = path.dot(posmat)  # c values for scaling u
    # scale u by values to get projection
    projected = path.T.dot(c_values)

def dtg(x1,y1,xgoal,ygoal):
    '''Returns absolute distance to goal coordinates'''
    dist = np.sqrt((xgoal - x1)**2 + (ygoal - y1)**2)
    return dist
vecdtg = np.vectorize(dtg)

def distsbw(path):
    dists = []
    for p in range(len(path)-1):
        dists.append(dtg(path[p,0],path[p,1],path[p+1,0],path[p+1,1]))
    return dists

def naninterp(B):
    A = B
    ok = ~np.isnan(A)
    xp = ok.ravel().nonzero()[0]
    fp = A[~np.isnan(A)]
    x  = np.isnan(A).ravel().nonzero()[0]
    A[np.isnan(A)] = np.interp(x, xp, fp)
    return(A)

def define_hexbins(drawnlines):
    '''Return polygon collection object for each drawn hexbin in drawnlines'''
    #add all polygon objects to multipolygon object called polybins
    polys = [Polygon(h) for h in drawnlines]
    polybins = MultiPolygon(polys)
    return polybins

def in_approach(apzone,data):
    '''Return binary vector where 1 indicates rat is inside of the drawn and defined
    approach zone (apzone). data should be photometry dataframe'''
    zone = Polygon(apzone)
    xvec = data.x.values
    yvec = data.y.values
    inzone = shapely.vectorized.contains(zone,xvec,yvec)
    return inzone.astype(int)

def inpoly(point):
    '''Return poly bin index (identifier) and centroid array (for plotting)'''
    try: polybins
    except NameError: 'polybins not defined'
    #iterate through polygon objects, checking if point is in one of them
    ind = None
    for p in range(len(polybins)):
        if point.within(polybins[p]):
            ind = p
    return ind,np.array(polybins[ind].centroid)

def get_nearest_hex(point,centroids):
    '''return hexID whose centroid is the shortest distance from point'''
    dists = []
    for c in centroids:
        dists.append(point.distance(Point(c[0])))
    if np.min(dists) < 50:
        return centroids[np.argmin(dists)][1]
    else:
        return np.nan

def point_to_poly(points,polybins,centroids):
    '''given MultiPoint object (points) and MultiPolygon object (polybins),
    return array of bin IDs whose indices correspond to point identities. Also
    returns centroid arrays for each element in binIDs.'''
    xvec = np.array(points)[:,0]
    yvec = np.array(points)[:,1]
    binID = np.tile(np.nan,len(xvec))
    hex_centers = np.tile(np.nan,(len(xvec),2))
    for h in range(len(polybins)):
        inhex = shapely.vectorized.contains(polybins[h],xvec,yvec)
        binID[np.where(inhex==True)]=h
        hex_centers[np.where(inhex==True)]=np.array(polybins[h].centroid)
    #get positions at all indices where no hex was assigned
    #gaps = np.where(np.isnan(binID))[0]
    #for i in range(len(gaps)):
    #    p = points[gaps[i]]
    #    h = get_nearest_hex(p,centroids)
    #    binID[i]=h
    return binID,hex_centers

def make_lr(ports):
    '''returns list of left-right choices for each trial. 0 is right, 1 is left'''
    lr = [2]
    for p in range(1,len(ports)):
        if ports[p-1]==2 and ports[p]==0:
            lr.append(0)
        elif ports[p-1]==1 and ports[p]==2:
            lr.append(0)
        elif ports[p-1]==0 and ports[p]==1:
            lr.append(0)
        else:
            lr.append(1)
    return lr

def get_zdFF(raw_reference,raw_signal,smooth_win,lambd,porder,itermax,remove): 
    '''
    Calculates z-score dF/F signal based on photometry calcium-idependent 
    and calcium dependent signals
    
    Input
        raw_reference - calcium-independent signal (usually 405-420 nm excitation)
        raw_signal - calcium-dependent signal (usually 465-490 nm excitation)
        smooth_win - window for signal 
        remove - the beginning of the traces with a big slope one would like to remove
        Inputs for airPLS:
        lambd - parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z
        porder - adaptive iteratively reweighted penalized least squares for baseline fitting
        itermax
    Output
        zdFF - z-score dF/F, numpy vector
    '''
    # Smooth signal
    reference = np.array(raw_reference.rolling(window=smooth_win).mean()).reshape(len(raw_reference),1)
    signal = np.array(raw_signal.rolling(window=smooth_win).mean()).reshape(len(raw_signal),1)
    
    # Remove slope using airPLS algorithm
    r_base=airPLS(raw_reference.T,lambda_=lambd,porder=porder,itermax=itermax).reshape(len(raw_reference),1)
    s_base=airPLS(raw_signal.T,lambda_=lambd,porder=porder,itermax=itermax).reshape(len(raw_signal),1)  

    # Remove baseline and the begining of test 
    reference = (reference[remove:] - r_base[remove:])
    signal = (signal[remove:] - s_base[remove:])   

    # Standardize signals    
    z_reference = (reference - np.median(reference)) / np.std(reference)
    z_signal = (signal - np.median(signal)) / np.std(signal)
    
    # Align reference signal to calcium signal using linear regression
    from sklearn.linear_model import Lasso
    lin = Lasso(alpha=0.0001,precompute=True,max_iter=1000,
                positive=True, random_state=9999, selection='random')
    lin.fit(z_reference, z_signal)
    z_reference_fitted = lin.predict(z_reference).reshape(len(z_reference),1) 

    # z dFF    
    zdFF = (z_signal - z_reference_fitted)
    
    return zdFF

from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve

def WhittakerSmooth(x,w,lambda_,differences=1):
    '''
    Penalized least squares algorithm for background fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties
    
    output
        the fitted background vector
    '''
    X=np.matrix(x)
    m=X.size
    i=np.arange(0,m)
    E=eye(m,format='csc')
    D=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
    W=diags(w,0,shape=(m,m))
    A=csc_matrix(W+(lambda_*D.T*D))
    B=csc_matrix(W*X.T)
    background=spsolve(A,B)
    return np.array(background)

def airPLS(x, lambda_=100, porder=1, itermax=15):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
    
    output
        the fitted background vector
    '''
    m=x.shape[0]
    w=np.ones(m)
    for i in range(1,itermax+1):
        z=WhittakerSmooth(x,w,lambda_, porder)
        d=x-z
        dssn=np.abs(d[d<0].sum())
        if(dssn<0.001*(abs(x)).sum() or i==itermax):
            if(i==itermax): print('WARING max iteration reached!')
            break
        w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
        w[0]=np.exp(i*(d[d<0]).max()/dssn) 
        w[-1]=w[0]
    return z


def create_hexdata(data,hexIDs,hex_centers,hexlabels):
    addto = data.loc[data.x.isnull()==False].index
    #create new data frame of hex info to easily join with full data
    hexdata = pd.DataFrame({'hexIDs':hexIDs,'hex_centerx':hex_centers[:,0],'hex_centery':hex_centers[:,1]\
                           ,'hexlabels':hexlabels}) 
    hexdata = hexdata.set_index(addto)
    hexdata = hexdata.reindex(np.arange(data.index[0],data.index[-1]+1),fill_value=np.nan)
    
    #get sequence of hexes visited in order, without consecutive repeats
    hexseq = hexdata.loc[hexdata.hexIDs.isnull()==False,'hexlabels'].astype(int)
    hexseq = list(hexseq.loc[hexseq.diff() != 0].values)
    data.loc[:,'hexlabels']=hexdata.hexlabels
    return hexdata,data,hexseq

def make_hexlist(sampledata):
    visinds = visits = sampledata.loc[sampledata.port.isnull()==False].index.values
    triseqs = [] # list of sequences of hexes visited
    trida = []
    for i in range(1,len(visits)):
        tdat = sampledata.loc[visits[i-1]:visits[i]]
        binda = tdat.groupby('hexlabels').mean().green #dLight binned by hex occupancy
        tseq = tdat.hexlabels.unique() #sequence of hexes taken between ports WITH NO REPEATS
        triseqs.append(tseq)
        trida.append(binda.loc[tseq].values)
    
    #make all rows same length, fill initial points with nans to align all
    #fill array with nans of shape len(tridat) x len(sampledata.hexlabels.unique())
    hextot = len(sampledata.hexlabels.unique())
    distsig = np.full_like(np.zeros((len(trida),hextot)),np.nan)
    pathlens = []
    for t in range(len(trida)):
        start = hextot-len(trida[t])
        distsig[t,start:] = trida[t]
        pathlens.append(len(trida[t]))
    plt.figure()
    plt.plot(np.nanmean(distsig,axis=0))
    maxdist = len(distsig[0])
    
    distsig = pd.DataFrame(distsig)
    distsig['rwd'] = tridat['rwd']
    distsig['pA'] = tridat['pA']
    distsig['pB'] = tridat['pB']
    distsig['pC'] = tridat['pC']
    distsig['port'] = tridat['port']
    distsig['nextp'] = tridat['nextp']
    distsig['block'] = tridat['block']
    distsig['plength'] = pathlens
    
    plt.figure()
    plt.plot(distsig.plength)
    plt.ylabel('path length',fontweight='bold')
    plt.xlabel('trial #',fontweight='bold')
    sampledata['plength'] = np.full_like(np.zeros(len(sampledata)),np.nan)
    mostvis = visinds[:-1] #make in line with distsig data
    sampledata.loc[mostvis,'plength']=pathlens
    sampledata.plength = sampledata.plength.fillna(method='bfill')
    return distsig

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def findpath(fr,to):
    if fr==0 and to==1: #AB
        path = 0
    elif fr==1 and to==0: #BA
        path = 1
    elif fr==0 and to==2: #AC
        path = 2
    elif fr==2 and to==0: #CA
        path = 3
    elif fr==1 and to==2: #BC
        path = 4
    elif fr==2 and to==1: #CB
        path = 5
    else:
        path = np.nan
    return path

vfpath = np.vectorize(findpath)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
def ramptest(data,visnum):
    y = np.array(data.iloc[visinds[visnum]-time[0]*250:visinds[visnum]])
    x = np.linspace(0,len(y),len(y))#column vector with x values. linspace(0,Fs*6,Fs*6)
    x = x.reshape(-1,1)
    model = LinearRegression()
    model.fit(x,y)
    r_sq = model.score(x, y)
    coef = model.coef_[0]
    inter = model.intercept_
    return r_sq,coef,inter

def linreg(data,v1,v2):
    '''Return linear regression r squared value, coefficient, and intercept for plotting.
    v1 and v2 must already be columns in data. Ensure no nans in v1 or v2'''
    y = np.array(data.loc[:,v2])
    x = np.array(data.loc[:,v1])
    x = x.reshape(-1,1)
    model = LinearRegression()
    model.fit(x,y)
    r_sq = model.score(x, y)
    coef = model.coef_[0]
    inter = model.intercept_
    return r_sq,coef,inter

def findpath(fr,to):
    '''Return number of path taken, given which port animal went to and from'''
    if fr==0 and to==1: #AB
        path = 0
    elif fr==1 and to==0: #BA
        path = 1
    elif fr==0 and to==2: #AC
        path = 2
    elif fr==2 and to==0: #CA
        path = 3
    elif fr==1 and to==2: #BC
        path = 4
    elif fr==2 and to==1: #CB
        path = 5
    else:
        path = np.nan
    return path

vfpath = np.vectorize(findpath)

def calc_overlap(sampledata):
    '''Calculate percent of overlapping hexes from one path traversal to the next.
    Add percent overlap to dataframe'''
    #get path ID for each trial
    tridat.loc[:,'path'] = np.concatenate([[np.nan],vfpath(tridat.loc[:,'fromP'].\
        values,tridat.loc[:,'port'].values)[1:]])
    path = vfpath(tridat.loc[:,'fromP'].values,tridat.loc[:,'port'].values)[1:] #ignore first port entry (not from any)
    
    # pair paths with triseqs, where paths are 0,1,2,3,4,5 for [AB,BA,AC,CA,BC,CB]
    
    # pool trials by path, order by trial
    ABseqs = np.array(triseqs)[np.where(path==0)]
    BAseqs = np.array(triseqs)[np.where(path==1)]
    ACseqs = np.array(triseqs)[np.where(path==2)]
    CAseqs = np.array(triseqs)[np.where(path==3)]
    BCseqs = np.array(triseqs)[np.where(path==4)]
    CBseqs = np.array(triseqs)[np.where(path==5)]
    
    # use np.intersect1d() to identify common hexes to both paths. divide by len(new path) to get fraction of overlap.
    # x100 to get percent overlap
    # either compare common hexes between new path and last path or new path and rolling average hexes. Start w/ first
    ABoverlap = vec_overlap(ABseqs[:-1],ABseqs[1:])
    BAoverlap = vec_overlap(BAseqs[:-1],BAseqs[1:])
    ACoverlap = vec_overlap(ACseqs[:-1],ACseqs[1:])
    CAoverlap = vec_overlap(CAseqs[:-1],CAseqs[1:])
    BCoverlap = vec_overlap(BCseqs[:-1],BCseqs[1:])
    CBoverlap = vec_overlap(CBseqs[:-1],CBseqs[1:])
    
    # add path overlap info to tridat/sampledata
    tridat.loc[:,'path_overlap'] = np.nan
    tridat.loc[tridat.path==0,'path_overlap'] = np.concatenate([[np.nan],ABoverlap])
    tridat.loc[tridat.path==1,'path_overlap'] = np.concatenate([[np.nan],BAoverlap])
    tridat.loc[tridat.path==2,'path_overlap'] = np.concatenate([[np.nan],ACoverlap])
    tridat.loc[tridat.path==3,'path_overlap'] = np.concatenate([[np.nan],CAoverlap])
    tridat.loc[tridat.path==4,'path_overlap'] = np.concatenate([[np.nan],BCoverlap])
    tridat.loc[tridat.path==5,'path_overlap'] = np.concatenate([[np.nan],CBoverlap])
    
    sampledata.loc[:,'path_overlap'] = np.nan
    sampledata.loc[:,'path'] = np.nan

    addto = sampledata.loc[sampledata.port.isnull()==False].index
    polaps = pd.DataFrame()
    polaps['path'] = tridat['path']
    polaps['path_overlap'] = tridat['path_overlap']
    polaps = polaps.set_index(addto)
    polaps = polaps.reindex(np.arange(sampledata.index[0],sampledata.index[-1]+1),fill_value=np.nan)
    sampledata['path'] = polaps['path'].fillna(method='bfill')
    #bfill to code for degree of overlap in rat's current path
    sampledata['path_overlap'] = polaps['path_overlap'].fillna(method='bfill') 

def findpath(fr,to):
    if fr==0 and to==1: #AB
        path = 0
    elif fr==1 and to==0: #BA
        path = 1
    elif fr==0 and to==2: #AC
        path = 2
    elif fr==2 and to==0: #CA
        path = 3
    elif fr==1 and to==2: #BC
        path = 4
    elif fr==2 and to==1: #CB
        path = 5
    else:
        path = np.nan
    return path

vfpath = np.vectorize(findpath)

def perc_overlap(lastp,p):
    '''Returns percent of hex identities in current path (p) that are common
    to last taken path betwen same ports. lastp should be array with hexes of
    previous trial, p should be array with hexes of current trial.'''
    return len(np.intersect1d(lastp,p))/len(p)*100

vec_overlap = np.vectorize(perc_overlap)

def simple_rr(data,ports=[0,1,2]):
    '''Returns rolling reward rate per minute for ports of interest. Also
    returns indices of visits to port of interest. Useful for calculating rr
    by trial, etc.'''
    win = 60*250
    if len(ports)==1:
        rwds = data.loc[data.port==ports[0],'rwd'].copy()
        rwds.loc[rwds==-1]=0
        rwds = rwds.reindex(np.arange(data.index[0],data.index[-1]+1),fill_value=np.nan)
        #inds = data.loc[data.port==ports[0]].index
        return rwds.rolling(window=win,min_periods=1).sum()#,inds
    elif len(ports)==2:
        #inds = data.loc[(data.port==ports[0])|(data.port==ports[1])].index
        #make column with 1s at points of reward
        rwds = data.loc[(data.port==ports[0])|(data.port==ports[1]),'rwd'].copy()
        rwds.loc[rwds==-1]=0
        rwds = rwds.reindex(np.arange(data.index[0],data.index[-1]+1),fill_value=np.nan)
        return rwds.rolling(window=win,min_periods=1).sum()#,inds
    elif len(ports)==3:
        rwds = data.rwd.copy()
        rwds.loc[rwds==-1]=0
        #inds = data.loc[(data.rwd==0)|(data.rwd==1)].index
        #create new df/series with all rows from original, 1s in rwd indices
        return rwds.rolling(window=win,min_periods=1).sum()#,inds

#calculate number of rewards in past t trials for specified ports
def tri_rr(data,ports=[0,1,2],t=5):
    '''Return trial-by-trial estimte of previous reward history over past t trials'''
    if len(ports)==1:
        rwds = data.loc[data.port==ports[0],'rwd'].copy()
        rwds.loc[rwds==-1]=0
        rwds = rwds.rolling(window=t).sum()
        rwds = rwds.reindex(np.arange(data.index[0],data.index[-1]+1),fill_value=np.nan)
        return rwds.fillna(method='ffill')
    elif len(ports)==2:
        rwds = data.loc[(data.port==ports[0])|(data.port==ports[1]),'rwd'].copy()
        rwds.loc[rwds==-1]=0
        rwds = rwds.rolling(window=t).sum()
        rwds = rwds.reindex(np.arange(data.index[0],data.index[-1]+1),fill_value=np.nan)
        return rwds.fillna(method='ffill')
    elif len(ports)==3:
        rwds = data.loc[data.port.isnull()==False,'rwd'].copy()
        rwds.loc[rwds==-1]=0
        rwds = rwds.rolling(window=t).sum()
        rwds = rwds.reindex(np.arange(data.index[0],data.index[-1]+1),fill_value=np.nan)
        return rwds.fillna(method='ffill')

#def tri_rr_frport(data,ports,frport,t=5):
#    ''''''
#     rwds = data.loc[(data.port==ports[0])&(data.fromP==frport),'rwd'].copy()
#     rwds.loc[rwds==-1]=0
#     rwds = rwds.rolling(window=t).sum()
#     rwds = rwds.reindex(np.arange(data.index[0],data.index[-1]+1),fill_value=np.nan)
#     return rwds.fillna(method='ffill')

def reg_in_time(data,ttp=4,tfp=4,trace='green',factors=['pchosen','Qc_allo','Qc_ego','speed']):
    '''Calculate regression of behavioral factors to
    dLight values during approach. can add extra factors to regress'''
    dframe = data
    tridat = dframe.loc[dframe.port.isnull()==False]
    lags = np.arange(-fs*ttp,fs*tfp)
    vinds = dframe.loc[dframe.port.isnull()==False].index
    faclen = len(factors)
    rweights = np.zeros((faclen,len(lags)))
    rserrors = np.zeros((faclen,len(lags)))
    speed = data.vel.fillna(method='ffill')
    speed = speed.fillna(method='bfill')
    for n in range(len(lags)):
        y = dframe.loc[vinds+lags[n],trace]
        y = y.reset_index(drop=True)
        X = pd.DataFrame()
        for f in factors:
            if f=='pchosen':
                X[f] = data.loc[vinds,f].values/100
            elif f=='speed':
                X[f] = speed.loc[vinds+lags[n]].values
            else:
                X[f] = data.loc[vinds,f].values
        #scale factors to normalize for interpretable comparison of beta values
        X[factors] = scale.fit_transform(X[factors].as_matrix())
        mod = sm.GLS(y, X).fit()
        rweights[:,n] = mod.params.values
        rserrors[:,n] = mod.bse.values
        facnames = X.columns
    return facnames,rweights,rserrors

#create function to take factor of interest only for port(s) of interest
def get_avail(p):
    allports = [0,1,2]
    allports.pop(p)
    return allports

def avg_factor(factor,portz,sampledata):
    '''ports is a list of ports to average the factors over (one for each trial).
    factor must be either Q_ego, Q_allo, nom_rwd, or rhist'''
    visinds = sampledata.loc[sampledata.port.isnull()==False].index
    if factor == 'Q_ego':
        df = sampledata.loc[visinds,['Q_ego_a','Q_ego_b','Q_ego_c']].values
    elif factor == 'Q_allo':
        df = sampledata.loc[visinds,['Q_allo_a','Q_allo_b','Q_allo_c']].values
    elif factor == 'nom_rwd':
        df = sampledata.loc[visinds,['nom_rwd_a','nom_rwd_b','nom_rwd_c']].values
    elif factor == 'rhist':
        df = sampledata.loc[visinds,['rhist_a','rhist_b','rhist_c']].values
    else:
        print("factor not yet defined for grouping.")
        return None
    fval = []
    for i in range(len(portz)):
        fval.append(np.mean(df[i,portz[i]]))
    return fval

def factor_by_p_type(factor,sampledata,p_type='all'):
    '''p_type can be all, avail, or chosen. returns value of factor given p_type for every trial.'''
    visinds = sampledata.loc[sampledata.port.notnull()].index
    if p_type == 'chosen':
        portz = sampledata.loc[visinds,'port'].values.astype(int)
    elif p_type == 'avail':
        portz = [[0,1,2]]
        for p in sampledata.loc[visinds[:-1],'port'].values.astype(int):
            portz.append(get_avail(p))
    elif p_type == 'all':
        portz = np.tile([0,1,2],(len(visinds),1))
    elif p_type == 'other':
        portz = [0]
        for p in range(len(visinds)-1):
            avail = get_avail(int(sampledata.loc[visinds[p],'port']))
            chos = int(sampledata.loc[visinds[p+1],'port'])
            avail.remove(chos)
            portz.append(avail)
    elif p_type == 'last':
        portz = np.concatenate([[0],sampledata.loc[visinds[:-1],'port'].values.astype(int)])
    return avg_factor(factor,portz,sampledata)

def reg_by_ptype(factor,sampledata):
    addto = sampledata.loc[sampledata.port.notnull()].index
    bytype = pd.DataFrame()
    bytype[factor+'_avail'] = factor_by_p_type(factor,sampledata,'avail')
    bytype[factor+'_chosen'] = factor_by_p_type(factor,sampledata,'chosen')
    bytype[factor+'_all'] = factor_by_p_type(factor,sampledata,'all')
    bytype[factor+'_other'] = factor_by_p_type(factor,sampledata,'other')
    bytype[factor+'_last'] = factor_by_p_type(factor,sampledata,'last')
    bytype = bytype.set_index(addto)
    bytype = bytype.reindex(np.arange(sampledata.index[0],sampledata.index[-1]+1),fill_value=np.nan)
    
    sampledata[factor+'_avail'] = bytype[factor+'_avail'].fillna(method='bfill')
    sampledata[factor+'_avail'] = sampledata[factor+'_avail'].fillna(method='bfill')
    sampledata[factor+'_last'] = bytype[factor+'_last'].fillna(method='bfill')
    sampledata[factor+'_last'] = sampledata[factor+'_last'].fillna(method='bfill')
    sampledata[factor+'_other'] = bytype[factor+'_other'].fillna(method='bfill')
    sampledata[factor+'_other'] = sampledata[factor+'_other'].fillna(method='bfill')
    sampledata[factor+'_chosen'] = bytype[factor+'_chosen'].fillna(method='bfill')
    sampledata[factor+'_chosen'] = sampledata[factor+'_chosen'].fillna(method='bfill')
    sampledata[factor+'_all'] = bytype[factor+'_all'].fillna(method='bfill')
    sampledata[factor+'_all'] = sampledata[factor+'_all'].fillna(method='bfill')
    return sampledata

#make vectorized leakyrr calculator
from scipy.stats import spearmanr     
def leaky(vec,kernel,ind):
    vec[ind:ind+len(kernel)] = vec[ind:ind+len(kernel)] + kernel
    return vec
vleaky = np.vectorize(leaky) #this might not work, loop needs to be executed in order

def tauloop(n,tt,rwdentries,cors):
    rrtau = np.zeros(endtime)
    kernel = np.exp((-np.linspace(0,5*tau)/tau))
    for e in rwdentries:
        rrtau[e:e+len(kernel)] = rrtau[e:e+len(kernel)] + kernel
    if len(rrtau) != len(tt):
        rrtau = rrtau[:len(tt)]
    cor = spearmanr([rrtau,tt],axis=1,nan_policy='omit')[0]
    cors.append(cor)
    return cors
vtauloop = np.vectorize(tauloop,otypes=[list])
    
def optimize_tau(data,rwdentries,m=1,n=200,pad=3500):
    '''call to iterate through n tau values and return tau that gives maximum negative
    correlation between rr and tt. tt is trialtimes'''
    endtime = len(data) + fs*pad
    tt = data.vel_AUC.fillna(method='bfill')#behavioral parameter for optimization, velocity AUC
    tt = tt.fillna(method='ffill')
    cors = [] #store n row vector of correlation coefficients between rrs of dif tau and tt
    taus = np.arange(m,n+1)
    for tau in taus:
        rrtau = np.zeros(endtime)
        kernel = np.exp((-np.linspace(0,5*tau,fs*tau)/tau))
        for e in rwdentries:
            rrtau[e:e+len(kernel)] = rrtau[e:e+len(kernel)] + kernel
        if len(rrtau) != len(data):
            rrtau = rrtau[:len(data)]
        cor = spearmanr([rrtau,tt],axis=1,nan_policy='omit')[0]
        cors.append(cor)
    maxcor = np.max(cors)
    opt_tau = taus[cors.index(maxcor)]
    return opt_tau,maxcor,cors
    
def est_leakint_rr(data,ports,tau=0,optimize=True):
    '''Returns leaky integrator reward rate for an input vector of reward outcomes over n timesteps.
    grabs reward outcomes from data dataframe. tau = time constant. Optimize selects whether to optimize
    tau as value that results in maximum correlation between trial speed metric and rewrate. method
    specifies whether to consider all ports, available ports, or chosen port in leaky integrator'''
    pad = 3500 #not sure why this value, or what function is
    if len(ports) == 3:
        rwdinds = data.loc[data.rwd==1].index
    elif len(ports) == 2:
        rwddat = data.loc[(data.rwd==1)]
        rwdinds = rwddat.loc[(rwddat.port==ports[0])|(rwddat.port==ports[1])].index
    elif len(ports) == 1:
        rwdinds = data.loc[(data.rwd==1)&(data.port==ports[0])].index
    if tau==0 or optimize==True:
        tau,maxcor,corr = optimize_tau(data,rwdinds,1,400,pad)
    #now calculate optimal leaky integrator for reward rate
    endtime = len(data) + fs*pad
    kernel = np.exp((-np.linspace(0,5*tau,fs*tau)/tau))
    rr = np.zeros(endtime)
    for rwdind in rwdinds:
        rr[rwdind:rwdind+len(kernel)] = rr[rwdind:rwdind+len(kernel)] + kernel
    if len(rr) != len(data): #make sure length of rr vector == number of samples
        rr = rr[:len(data)]
    if optimize==False:
        tt = data.vel_AUC.fillna(method='bfill')#behavioral parameter for optimization, velocity AUC
        tt = tt.fillna(method='ffill')
        cor = spearmanr([rr,tt],axis=1,nan_policy='omit')[0]
    return rr,corr,tau


def tauloop_tri(n,tt,rwdentries,cors):
    rrtau = np.zeros(endtime)
    kernel = np.exp((-np.linspace(0,5*tau)/tau))
    for e in rwdentries:
        rrtau[e:e+len(kernel)] = rrtau[e:e+len(kernel)] + kernel
    if len(rrtau) != len(tt):
        rrtau = rrtau[:len(tt)]
    cor = spearmanr([rrtau,tt],axis=1,nan_policy='omit')[0]
    cors.append(cor)
    return cors
vtauloop_tri = np.vectorize(tauloop,otypes=[list])
    
def optimize_tau_tri(data,rwdentries,m=1,n=50):
    '''call to iterate through n tau values and return tau that gives maximum negative
    correlation between rr and tt. tt is trialtimes'''
    tdat = data.loc[data.port!=-100]
    endtime = len(tdat)
    tt = data.visinds.diff() #behavioral parameter for optimization, velocity AUC
    cors = [] #store n row vector of correlation coefficients between rrs of dif tau and tt
    taus = np.arange(m,n+1)
    for tau in taus:
        rrtau = np.zeros(endtime)
        kernel = np.exp((-np.linspace(0,5*tau,tau)/tau))
        for e in rwdentries:
            rrtau[e:e+len(kernel)] = rrtau[e:e+len(kernel)] + kernel
        if len(rrtau) != len(tdat):
            rrtau = rrtau[:len(tdat)]
        cor = spearmanr([rrtau,tt],axis=1,nan_policy='omit')[0]
        cors.append(cor)
    maxcor = np.max(cors)
    opt_tau = taus[cors.index(maxcor)]
    return opt_tau,maxcor,cors
    
def est_leakint_rr_tri(data,ports,tau=0,optimize=True):
    '''Returns leaky integrator reward rate for an input vector of reward outcomes over n timesteps.
    grabs reward outcomes from data dataframe. tau = time constant. Optimize selects whether to optimize
    tau as value that results in maximum correlation between trial speed metric and rewrate. method
    specifies whether to consider all ports, available ports, or chosen port in leaky integrator'''
    if len(ports) == 3:
        rwdinds = data.loc[data.rwd==1].index
    elif len(ports) == 2:
        rwddat = data.loc[(data.rwd==1)]
        rwdinds = rwddat.loc[(rwddat.port==ports[0])|(rwddat.port==ports[1])].index
    elif len(ports) == 1:
        rwdinds = data.loc[(data.rwd==1)&(data.port==ports[0])].index
    if tau==0 or optimize==True:
        tau,maxcor,corr = optimize_tau_tri(data,rwdinds,1,50)
    #now calculate optimal leaky integrator for reward rate
    endtime = len(data) + fs*pad
    kernel = np.exp((-np.linspace(0,5*tau,fs*tau)/tau))
    rr = np.zeros(endtime)
    for rwdind in rwdinds:
        rr[rwdind:rwdind+len(kernel)] = rr[rwdind:rwdind+len(kernel)] + kernel
    if len(rr) != len(data): #make sure length of rr vector == number of samples
        rr = rr[:len(data)]
    if optimize==False:
        tt = data.vel_AUC.fillna(method='bfill')#behavioral parameter for optimization, velocity AUC
        tt = tt.fillna(method='ffill')
        cor = spearmanr([rr,tt],axis=1,nan_policy='omit')[0]
    return rr,corr,tau

#def transform_geom(g1, src_proj: str, dest_proj: str):
#    '''Transform shapely multipolygon object from source projection to 
#    destination projection'''
#    project = partial(
#        pyproj.transform,
#        pyproj.Proj(init=src_proj),
#        pyproj.Proj(init=dest_proj))
#
#    g2 = transform(project, g1)
#
#    return g2
from statsmodels.discrete.discrete_model import Logit
def choice_reg(factors,data):
    y = data.loc[visinds,'lrchoice']
    X = data.loc[visinds,factors]
    X = X.drop(y.loc[y==2].index,axis=0)
    y = y.drop(y.loc[y==2].index,axis=0)
    X[factors] = scale.fit_transform(X[factors].as_matrix())
    mod = Logit(y, X).fit()
    pdf = Logit(y, X).pdf(np.linspace(X[factors[0]].min(),X[factors[0]].max(),100))
    rweights = mod.params.values
    rserrors = mod.bse.values
    return rweights,rserrors,pdf


#def reg_by_ptype(factor,sampledata):
#    addto = sampledata.loc[sampledata.port.notnull()].index
#    bytype = pd.DataFrame()
#    bytype[factor+'_avail'] = factor_by_p_type(factor,sampledata,'avail')
#    bytype[factor+'_chosen'] = factor_by_p_type(factor,sampledata,'chosen')
#    bytype[factor+'_all'] = factor_by_p_type(factor,sampledata,'all')
#    bytype[factor+'_other'] = factor_by_p_type(factor,sampledata,'other')
#    bytype[factor+'_last'] = factor_by_p_type(factor,sampledata,'last')
#    bytype = bytype.set_index(addto)
#    bytype = bytype.reindex(np.arange(sampledata.index[0],sampledata.index[-1]+1),fill_value=np.nan)
#    
#    sampledata[factor+'_avail'] = bytype[factor+'_avail'].fillna(method='bfill')
#    sampledata[factor+'_avail'] = sampledata[factor+'_avail'].fillna(method='bfill')
#    sampledata[factor+'_last'] = bytype[factor+'_last'].fillna(method='bfill')
#    sampledata[factor+'_last'] = sampledata[factor+'_last'].fillna(method='bfill')
#    sampledata[factor+'_other'] = bytype[factor+'_other'].fillna(method='bfill')
#    sampledata[factor+'_other'] = sampledata[factor+'_other'].fillna(method='bfill')
#    sampledata[factor+'_chosen'] = bytype[factor+'_chosen'].fillna(method='bfill')
#    sampledata[factor+'_chosen'] = sampledata[factor+'_chosen'].fillna(method='bfill')
#    sampledata[factor+'_all'] = bytype[factor+'_all'].fillna(method='bfill')
#    sampledata[factor+'_all'] = sampledata[factor+'_all'].fillna(method='bfill')
#    return sampledata


def get_latencies(sampledata,tridat):
    '''sampledata must contain apzone column'''
    visinds = sampledata.loc[sampledata.port.isnull()==False].index
    apstarts = [] #indices of approach start (used for latency, smearing, etc.)
    latencies = [] #list of latency for each trial
    for i in range(len(visinds)):
        apstarts.append(visinds[i]+np.where(np.diff(sampledata.loc[visinds[i]:,'apzone'])==1)[0][0])
        if i==0:
            latencies.append(visinds[i]/250)
        if i<len(visinds)-1:
            latencies.append((visinds[i+1]-apstarts[i])/250)
            
    tridat['latency']=latencies
    addto = visinds
    lats = pd.DataFrame(latencies)
    lats = lats.set_index(addto)
    lats = lats.reindex(np.arange(sampledata.index[0],sampledata.index[-1]+1),fill_value=np.nan)
    sampledata['latency'] = lats.fillna(method='bfill')
    lat = plt.figure()
    plt.title('Distribution of latencies')
    plt.hist(latencies,bins=100)
    plt.ylabel('# of trials')
    plt.xlabel('latency (s)')
    lat.savefig(datepath+'latency_dist.pdf')
    return sampledata,tridat

#def detect_events(signal,thresh=3):
#    '''return timestamp and magnitude of each detected event.
#    an event is a thold crossing of (default 3) sd from mean. signal
#    should already be z-scored, just needs to cross a value of 3.'''
#    crossings = 
#    return e_times,mags
#