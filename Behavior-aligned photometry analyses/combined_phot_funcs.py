

from phot_funcs import *
from sklearn.preprocessing import StandardScaler,MinMaxScaler
scale = StandardScaler()
valscale = MinMaxScaler()
scaler = valscale.fit(np.array([-1,0,1]).reshape(-1,1))
from scipy.stats import pearsonr,spearmanr
from scipy.signal import resample
from tqdm import tqdm
from math import log as mlog
from Q_learning_funcs import hyb_q_choice

def viz_value_map(valmap,i):
    plt.clf()
    plt.title('iteration '+str(i)+' value map',fontsize='x-large',fontweight='bold')
    plt.scatter(centdf[0].sort_index().values,centdf[1].sort_index().values,c=\
            valmap,marker='H',s=600,cmap='magma',edgecolors= "black",vmin=0,vmax=1)
    plt.colorbar()
    plt.pause(.1)

def viz_value_map_pairedState(valmap,i):
    plt.clf()
    plt.title('iteration '+str(i)+' value map',fontsize='x-large',fontweight='bold')
    plot_arrowMapFromStates(np.arange(0,126),arrowvals=valmap[i,:126]*100)
    plt.colorbar()
    plt.pause(.1)

class PhotRats(Photrat):

    cols2load = ['green_z_scored',"ref",'port','rwd','x','y','nom_rwd_a','nom_rwd_b',\
              'beamA', 'beamB', 'beamC','vel','acc','tri','block',\
             'nom_rwd_c','hexlabels','rat','date','lenAC', 'lenBC',\
             'lenAB', 'dtop','fiberloc','session_type']
    plotvals = False
    use_nom_rwds = False
    bin_size = int(250/10)
    offset = int(250*.4)
    phot_directory_prefix = "/Volumes/Tim K/Photometry Data/Triangle Maze/"
    poolByTerc = True
    iter_type='maxAvail'
    state_type = 'paired'

    def __init__(self,seshdict):
        self.seshdict = seshdict

    def set_savepath(self,rat):
        self.savepath = self.directory_prefix+rat+'/'

    def add_df(self,df):
        try:
            self.df = self.df.append(df,ignore_index=True)
        except:
            self.df = df

    def load_dfs(self):
        '''iterate through IDs and dates to load all dfs and compile into one df'''
        sesh = 1
        for rat in self.seshdict:
            for date in self.seshdict[rat]:
                newdf = self.attempt_load(rat,date)
                newdf['session'] = sesh
                self.add_df(newdf)
                sesh += 1

    def attempt_load(self,rat,date):
        try:
            newdf = reduce_mem_usage(pd.read_csv(self.directory_prefix+\
                rat+'/'+date+'/'+rat+'_'+date+'_h_sampleframe.csv',\
                usecols=self.cols2load))
            return newdf
        except:
            print('could not load df for '+rat+'; '+date)

    def reduce_dtypeMem(self):
        for f in ['port','rwd','lenAC', 'lenBC','lenAB','dtop']:
            self.df.loc[self.df[f].isnull(),f]=-100
            self.df[f] = self.df[f].astype("int8")
        for f in ['beamA', 'beamB', 'beamC']:
            self.df.loc[self.df[f].isnull(),f]=0
            self.df[f] = self.df[f].astype("int8")
        self.df.loc[self.df.hexlabels.isnull(),'hexlabels']=-100
        self.df['hexlabels']=self.df.hexlabels.astype("int8")
        for f in ['tri','block']:
            self.df[f] = self.df[f].astype("uint8")
        for f in ['x','y','green_z_scored','vel','acc']:
            self.df[f] = self.df[f].astype("float16")
        self.df['session'] = self.df.session.astype("int8")
        #self.df['tot_tri'] = self.df.tot_tri.astype("int8")

    def load_and_add_df(self,ID,date):
        newdf = reduce_mem_usage(pd.read_csv(self.directory_prefix+\
            ID+'/'+date+'/'+ID+'_'+date+'_h_sampleframe.csv',usecols=self.cols2load))
        newdf['session']=self.df.session.max()+1
        self.df = self.df.append(newdf,ignore_index=True)

    def load_pooled_df(self,variant='bar'):
        if variant == 'bar':
            self.df = reduce_mem_usage(pd.read_csv(self.directory_prefix+\
            'bar_frames_july_cohort_compressed_processed_36sesh.csv'))
        else:
            self.df = reduce_mem_usage(pd.read_csv(self.directory_prefix+\
            'all_frames_july_cohort_compressed.csv'))

    def get_barIDs(self):
        self.sesh_barIDs = {s:[] for s in self.df.session.unique()}
        for sesh in self.df.session.unique():
            self.sesh_barIDs[sesh] = self.attempt_get_barIDs(sesh)

    def attempt_get_barIDs(self,sesh):
        barIds = []
        try:
            if self.df.loc[self.df.session==sesh,'session_type'].values[0]=='prob':
                tmat = self.sesh_tmats[sesh]
                barIds = np.where(np.mean(tmat,axis=1)==0)[0][1:]-1
            else:
                for b in self.df.loc[self.df.session==sesh,'block'].unique():
                    tmat = self.sesh_tmats[sesh][int(b-1)]
                    barIds.append(np.where(np.mean(tmat,axis=1)==0)[0][1:]-1)
        except:
            print('unable to find barriers for session '+str(sesh)+' block '+str(b))
        return barIds

    def get_portQvals(self,qtype='hybrid',level="session"):
        '''Using the parameters optimized in Julia, estimate values for each port,
        on each trial. Specify which type of q value from ['hybrid','mb','mf','port'].
        Specify whether parameters were drawn from the level of rat or session.'''
        self.df.loc[:,'Q_a'] = np.nan
        self.df.loc[:,'Q_b'] = np.nan
        self.df.loc[:,'Q_c'] = np.nan
        self.load_q_params(qtype)
        if level=="session":
            seshs = self.triframe.session.unique()
            for s in range(len(seshs)):
                tsesh = self.triframe.loc[self.triframe.session==seshs[s]]
                if qtype=="hybrid":
                    Q = hyb_q_choice(tsesh,self.hyb_params.loc[s].values)
                elif qtype=="mb":
                    Q = mb_q_choice(tsesh,self.mb_params.loc[s].values)
                elif qtype=="mf":
                    Q = mf_q_choice(tsesh,self.mf_params.loc[s].values)
                elif qtype == "port":
                    Q = port_q_choice(tsesh,self.q_params.loc[s].values)
                self.addQs2dfSubset(Q,seshs[s],useSesh=True)
                
        elif level=="rat":
            rats = self.triframe.rat.unique()
            for r in range(len(rats)):
                #should maybe still do this by sesh but keep same parameters
                for s in self.triframe.loc[self.triframe.rat==rats[r],"session"].unique():
                    tsesh = self.triframe.loc[self.triframe.session==s]#rat==rats[r]]
                    if qtype=="hybrid":
                        Q = hyb_q_choice(tsesh,self.hyb_params.loc[r].values)
                    elif qtype=="mb":
                        Q = mb_q_choice(tsesh,self.mb_params.loc[r].values)
                    elif qtype=="mf":
                        Q = mf_q_choice(tsesh,self.mf_params.loc[r].values)
                    elif qtype == "port":
                        Q = port_q_choice(tsesh,self.q_params.loc[r].values)
                    #self.addQs2dfSubset(Q,rats[r],useSesh=False)
                    self.addQs2dfSubset(Q,s,useSesh=True)
        self.df.loc[:,'Q_a'] = self.df.loc[:,'Q_a'].fillna(method='bfill').astype("float16")
        self.df.loc[:,'Q_b'] = self.df.loc[:,'Q_b'].fillna(method='bfill').astype("float16")
        self.df.loc[:,'Q_c'] = self.df.loc[:,'Q_c'].fillna(method='bfill').astype("float16")

    def addQs2dfSubset(self,Qs2add,levelIndex,useSesh=True):
        if useSesh:
            poolString = "session"
        else:
            poolString = "rat"
        self.df.loc[(self.df[poolString]==levelIndex)&(self.df.port!=-100),'Q_a']=Qs2add[:,0]
        self.df.loc[(self.df[poolString]==levelIndex)&(self.df.port!=-100),'Q_b']=Qs2add[:,1]
        self.df.loc[(self.df[poolString]==levelIndex)&(self.df.port!=-100),'Q_c']=Qs2add[:,2]

    def load_q_params(self,qtype='hybrid'):
        if qtype == 'hybrid':
            self.hyb_params = pd.read_csv(self.directory_prefix+"tri_hybrid_params.csv")
        elif qtype == 'mb':
            self.mb_params = pd.read_csv(self.directory_prefix+"tri_mb_params.csv")
        elif qtype == 'mf':
            self.mf_params = pd.read_csv(self.directory_prefix+"tri_mf_params.csv")
        elif qtype == 'port':
            self.q_params = pd.read_csv(self.directory_prefix+"tri_q3port_params.csv")
        
    def get_vals_byChosenEtc(self,chosen_only=False):
        for s in self.df.session.unique():
            self.dat = self.df.loc[self.df.session==s,:]
            if chosen_only:
                self.get_valOfChosenPort()
            else:
                self.get_vals_by_portType()
            if s==self.df.session[0]:
                newdf = self.dat
            else:
                newdf = newdf.append(self.dat,ignore_index=True)
        self.df = newdf

    def add_hexDistToPort(self):
        self.df['hexDistToPort'] = np.nan
        for s in self.df.session.unique():
            print(s)
            dat = self.df.loc[self.df.session==s]
            vinds = dat.loc[dat.port!=-100].index
            hexchanges = dat.loc[dat.hexlabels.diff()!=0].index
            hexDists = []
            for i in range(len(vinds)):
                if i == 0:
                    triHexChanges = hexchanges[(hexchanges>=0)&(hexchanges<vinds[i])]
                else:
                    triHexChanges = hexchanges[(hexchanges>vinds[i-1])&(hexchanges<vinds[i])]
                hexDists = np.append(hexDists,np.flip(np.arange(1,len(triHexChanges)+1)))
            self.df.loc[hexchanges[hexchanges<vinds[-1]],'hexDistToPort']=hexDists
        self.df.loc[self.visinds,'hexDistToPort']=0
        self.df.hexDistToPort.fillna(method='bfill',inplace=True)

    #def create_hexLevelDf(self):
    #    

    def get_critChoicePoints(self):
        #depreciated, better to use manual identification and entry
        self.check_seshTmatsExist()
        self.crit_cps = {s:[] for s in self.df.session.unique()}
        self.sesh_deadEnds = {s:[] for s in self.df.session.unique()}
        self.deadEndCps = {s:[] for s in self.df.session.unique()}
        self.find_cps4allSessions()
        self.add_cps2Df()

    def get_allChoicePoints(self):
        self.check_seshTmatsExist()
        self.all_cps = {s:[] for s in self.df.session.unique()}
        self.find_allCpsBySesh()
        self.add_allCps2Df()

    def get_newlyAvailHexesBySesh(self):
        self.check_seshTmatsExist()
        barseshs = self.df.loc[self.df.session_type=='barrier','session'].unique()
        self.sesh_newlyAvailHexes = {s:[] for s in barseshs}
        self.find_newlyAvailHexes()
        self.add_adjacent2newlyAvail()

    def get_newlyBlockedHexesBySesh(self):
        self.check_seshTmatsExist()
        barseshs = self.df.loc[self.df.session_type=='barrier','session'].unique()
        self.sesh_newlyBlockedHexes = {s:[] for s in barseshs}
        self.find_newlyBlockedHexes()
        self.add_adjacent2newlyBlocked()

    def check_seshTmatsExist(self):
        try:
            self.sesh_tmats
        except:
            print("transition matrices not yet loaded. Loading now...")
            self.load_tmats()

    def find_newlyAvailHexes(self):
        for s in self.sesh_newlyAvailHexes:
            if len(self.sesh_tmats[s])==1:
                continue
            for b in range(len(self.sesh_tmats[s])-1):
                testmat = self.sesh_tmats[s][b]
                testmat1 = self.sesh_tmats[s][b+1]
                self.sesh_newlyAvailHexes[s].append(np.where((\
                    np.sum(testmat,axis=0)==0)&(np.sum(testmat1,axis=0)!=0))[0])

    def find_newlyBlockedHexes(self):
        for s in self.sesh_newlyBlockedHexes:
            #if len(self.sesh_tmats[s])==1:
            #    continue
            for b in range(len(self.sesh_tmats[s])-1):
                testmat = self.sesh_tmats[s][b]
                testmat1 = self.sesh_tmats[s][b+1]
                self.sesh_newlyBlockedHexes[s].append(np.where((\
                    np.sum(testmat,axis=0)!=0)&(np.sum(testmat1,axis=0)==0))[0])

    def add_newlyAvailHexesToDf(self):
        self.df.loc[:,"newlyAvailHex"]=0
        self.df.loc[:,"newlyAvailHex"] = self.df.loc[:,"newlyAvailHex"].astype("int8")
        for s in self.sesh_newlyAvailHexes:
            #if len(self.sesh_newlyAvailHexes[s])==1:
            #    continue
            for b in range(len(self.sesh_newlyAvailHexes[s])):
                newHex = self.sesh_newlyAvailHexes[s][b]
                dat = self.df.loc[(self.df.session==s)&(self.df.block==b+2)]
                inNewHex = np.where(dat.loc[:,'hexlabels'].isin(\
                    newHex)==1)[0]+dat.index.min()
                self.df.loc[inNewHex,"newlyAvailHex"] = 1

    def add_adjacent2newlyAvail(self):
        self.df.loc[:,"adj2newlyAvail"]=0
        self.df.loc[:,"adj2newlyAvail"] = self.df.loc[:,"adj2newlyAvail"].astype("int8")
        for s in self.sesh_newlyAvailHexes:
            for b in range(len(self.sesh_newlyAvailHexes[s])):
                newHex = self.sesh_newlyAvailHexes[s][b]
                if len(newHex)<1:
                    continue
                adjHexes = []
                for h in newHex:
                    adjHexes = np.concatenate([adjHexes,np.where(self.sesh_tmats[s][b+1][h]>0)[0]])
                dat = self.df.loc[(self.df.session==s)&(self.df.block==b+2)]
                inAdjHex = np.where(dat.loc[:,'hexlabels'].isin(\
                    adjHexes)==1)[0]+dat.index.min()
                self.df.loc[inAdjHex,"adj2newlyAvail"] = 1

    def add_newlyBlockedHexesToDf(self):
        self.df.loc[:,"newlyBlockedHex_Adj"]=0
        self.df.loc[:,"newlyBlockedHex_Adj"] = self.df.loc[:,"newlyBlockedHex_Adj"].astype("int8")
        for s in self.sesh_newlyBlockedHexes:
            if len(self.sesh_newlyBlockedHexes[s])==1:
                continue
            for b in range(len(self.sesh_newlyBlockedHexes[s])):
                newHex = self.sesh_newlyBlockedHexes[s][b]
                dat = self.df.loc[(self.df.session==s)&(self.df.block==b+2)]
                inNewHex = np.where(dat.loc[:,'hexlabels'].isin(\
                    newHex)==1)[0]+dat.index.min()
                self.df.loc[inNewHex,"newlyBlockedHex_Adj"] = 1

    def add_adjacent2newlyBlocked(self):
        self.df.loc[:,"adj2newlyBlocked"]=0
        self.df.loc[:,"adj2newlyBlocked"] = self.df.loc[:,"adj2newlyBlocked"].astype("int8")
        for s in self.sesh_newlyBlockedHexes:
            for b in range(len(self.sesh_newlyBlockedHexes[s])):
                newHex = self.sesh_newlyBlockedHexes[s][b]
                if len(newHex)<1:
                    continue
                adjHexes = []
                for h in newHex:
                    adjHexes = np.concatenate([adjHexes,np.where(self.sesh_tmats[s][b][h]>0)[0]])
                dat = self.df.loc[(self.df.session==s)&(self.df.block==b+2)]
                inAdjHex = np.where(dat.loc[:,'hexlabels'].isin(\
                    adjHexes)==1)[0]+dat.index.min()
                self.df.loc[inAdjHex,"adj2newlyBlocked"] = 1

    def load_tmats(self):
        self.sesh_tmats = {s:[] for s in self.df.session.unique()}
        for sesh in self.df.session.unique():
            if sesh<=77:
                self.attempt_loadTmats(sesh)
            else:
                print('briefly switching directory_prefix to new Nas...')
                self.directory_prefix = '/Volumes/Tim/Photometry/'
                self.attempt_loadTmats(sesh)
                self.directory_prefix = self.phot_directory_prefix
                print("directory_prefix back to nas 8")

    def attempt_loadTmats(self,sesh):
        rat = self.df.loc[self.df.session==sesh,'rat'].unique()[0]
        date = str(self.df.loc[self.df.session==sesh,'date'].unique()[0])
        if int(date[0])!=1:
            date = '0'+date
        if self.df.loc[self.df.session==sesh,'session_type'].values[0]=='prob':
            self.sesh_tmats[sesh] = np.load(self.directory_prefix+rat+'/'+\
                    date+'/'+'tmat.npy')
        else:
            for b in self.df.loc[self.df.session==sesh,'block'].unique():
                try:
                    self.sesh_tmats[sesh].append(np.load(self.directory_prefix+rat+'/'+\
                            date+'/'+'tmat_block_'+str(b)+'.0.npy'))
                except:
                    print('no tmat saved for sesh '+str(sesh)+' block '+str(b))


    def get_simpleStateTransitions(self,s,b):
        if self.df.loc[self.df.session==s,'session_type'].values[0]=='prob':
            self.bars = self.sesh_barIDs[s]
        else:
            self.tmat = self.tmats[s][int(b-1)]
            self.bars = self.sesh_barIDs[s][int(b-1)]
        self.add_bars2tmatrix50()
        self.to_state = [[0,0,0,0,0,0]]+[np.argmax(a,axis=1) \
            for a in self.bar_tmatrix]

    def find_critCps4allSessions(self):
        for s in self.df.session.unique():
            self.dat = self.df.loc[self.df.session==s]
            if self.dat.session_type.unique()[0]=='prob':
                self.tmat = self.sesh_tmats[s]
                self.filter_extraneousCPs()
                self.crit_cps[s].append(self.cps)
                self.deadEndCps[s].append(self.dEndCps)
                self.sesh_deadEnds[s].append(self.deadEnds)
            else:
                for b in self.dat.block.unique():
                    try:
                        self.tmat = self.sesh_tmats[s][b]
                    except:
                        continue
                    self.filter_extraneousCPs()
                    self.crit_cps[s].append(self.cps)
                    self.deadEndCps[s].append(self.dEndCps)
                    self.sesh_deadEnds[s].append(self.deadEnds)

    def find_allCpsBySesh(self):
        for s in self.df.session.unique():
            self.dat = self.df.loc[self.df.session==s]
            if self.dat.session_type.unique()[0]=='prob':
                self.tmat = self.sesh_tmats[s]
                self.find_allCPs()
                self.all_cps[s].append(self.cps)
            else:
                for b in self.dat.block.unique():
                    try:
                        self.tmat = self.sesh_tmats[s][b-1]
                    except:
                        continue
                    self.find_allCPs()
                    self.all_cps[s].append(self.cps)

    def filter_extraneousCPs(self):
        self.find_allCPs()
        self.find_allDeadEnds()
        self.get_deadEndCPs()
        self.remove_intermedCPs()

    def find_allCPs(self):
        self.cps = np.unique(np.where((self.tmat*10).astype("int8")==3)[0])
        
    def find_allDeadEnds(self):
        self.deadEnds = np.unique(np.where(self.tmat==1)[0])
        self.deadEnds = np.delete(self.deadEnds,np.isin(self.deadEnds,[1,2,3]))

    def get_deadEndCPs(self):
        self.dEndCps = []
        for h in self.cps:
            if self.leads2deadEnd(h,[]):
                self.dEndCps.append(h)
        self.cps = np.delete(self.cps,np.isin(self.cps,self.dEndCps))

    def leads2deadEnd(self,h,visited):
        visited.append(h)
        nexthexes = np.where(self.tmat[h]>0)[0]
        toDeadEnd = []
        for hprime in nexthexes:
            if hprime in visited or hprime in self.cps or hprime in [1,2,3]:
                toDeadEnd.append(False)
            elif hprime in self.deadEnds:
                toDeadEnd.append(True)
            else:
                toDeadEnd.append(self.leads2deadEnd(hprime,visited))
        return any(toDeadEnd)

    def remove_intermedCPs(self):
        intermed = []
        for h in self.cps:
            if not self.leads2port(h,[]):
                intermed.append(h)
        self.cps = np.delete(self.cps,np.isin(self.cps,intermed))

    def leads2port(self,h,visited):
        visited.append(h)
        nexthexes = np.where(self.tmat[h]>0)[0]
        toPort = []
        for hprime in nexthexes:
            if hprime in visited or hprime in np.setdiff1d(self.cps,[4,48,49]):
                toPort.append(False)
            elif hprime in [1,2,3] and len(visited)>1:
                toPort.append(True)
            elif hprime in [1,2,3] and len(visited)==1:
                toPort.append(False)
            else:
                toPort.append(self.leads2port(hprime,visited))
        return any(toPort)

    def add_cps2Df(self):
        self.df.loc[:,'deadEndCP'] = 0
        self.df.loc[:,'critCP'] = 0
        for s in self.df.session.unique():
            dat = self.df.loc[self.df.session==s]
            if self.df.loc[self.df.session==s,'session_type'].values[0]=='prob':
                incp = dat.loc[dat.loc[:,'hexlabels'].isin(self.crit_cps[s][0])].index
                self.df.loc[incp,"critCP"] = 1
                indecp = dat.loc[dat.loc[:,'hexlabels'].isin(self.deadEndCps[s][0])].index
                self.df.loc[indecp,"deadEndCP"] = 1
            else:
                for b in range(1,len(self.crit_cps[s])+1):#dat.block.unique():
                    incp = np.where(dat.loc[dat.block==b,'hexlabels'].isin(\
                        self.crit_cps[s][int(b-1)])==1)[0]+dat.index.min()
                    self.df.loc[incp,"critCP"] = 1
                    indecp = np.where(dat.loc[dat.block==b,'hexlabels'].isin(\
                        self.deadEndCps[s][int(b-1)])==1)[0]+dat.index.min()
                    self.df.loc[indecp,"deadEndCP"] = 1

    def add_allCps2Df(self):
        self.df.loc[:,'choicePoint'] = 0
        for s in self.df.session.unique():
            dat = self.df.loc[self.df.session==s]
            if self.df.loc[self.df.session==s,'session_type'].values[0]=='prob':
                incp = dat.loc[dat.loc[:,'hexlabels'].isin(self.all_cps[s][0])].index
                self.df.loc[incp,"choicePoint"] = 1
            else:
                for b in range(1,len(self.all_cps[s])+1):#dat.block.unique():
                    incp = np.where(dat.loc[dat.block==b,'hexlabels'].isin(\
                        self.all_cps[s][int(b-1)])==1)[0]+dat.loc[dat.block==b,:].index.min()
                    self.df.loc[incp,"choicePoint"] = 1

#for b in dat.block.unique():
#    incp = dat.loc[dat.loc[dat.block==b,'hexlabels'].isin(self.crit_cps[s][b])].index
#    self.df.loc[incp,"critCP"] = 1
#    indecp = dat.loc[dat.loc[dat.block==b,'hexlabels'].isin(self.deadEndCps[s][b])].index
#    self.df.loc[indecp,"deadEndCP"] = 1

    def get_last_leave(self):
        leaveinds = []
        next_inds = []
        for s in self.df.session.unique():
            sdat = self.df.loc[self.df.session==s]
            leave,nextind,sdat = get_leave_inds(sdat)
            leaveinds = leaveinds + list(leave)
            next_inds = next_inds + list(nextind)
        #now create a column in self.df marking these samples to be kept for plotting/spatial analysis
        lastapproach = np.zeros(len(self.df))
        lastleave = np.zeros(len(self.df))
        for l,n in zip(leaveinds,next_inds):
            lastapproach[l+1:n+1] = 1
            lastleave[l] = 1
        self.df['lastleave']=lastleave
        self.df.loc[np.where(lastapproach==1)[0],'lastleave']=2
        self.df['lastleave']=self.df.lastleave.astype("int8")

    def check_df_exists(self):
        try:
            self.df
        except:
            raise Exception("Must define dataframe before executing function")

    def get_daRPE_auc(self):
        self.check_df_exists()
        self.df['DA_RPE'] = np.nan
        daRpes = []
        for i in self.visinds:
            bline = 0#self.df.loc[i-10:i,'green_z_scored'].mean()
            daRpes.append(self.df.loc[i+0*self.fs:i+1.0*self.fs,'green_z_scored'].mean()-bline)
        self.df.loc[self.visinds,'DA_RPE']=daRpes
        self.df['DA_RPE'] = self.df['DA_RPE'].astype("float16")
        self.triframe['DA_RPE']=daRpes
        fig = plt.figure()
        sns.distplot(self.triframe.loc[self.triframe.rwd==1,'DA_RPE'],label='rwd')
        sns.distplot(self.triframe.loc[self.triframe.rwd==0,'DA_RPE'],label='omission')
        plt.legend()
        fig.savefig(self.directory_prefix+'DA_rwdVsOm_AUC.pdf')

    def get_daRPE_maxMin(self):
        self.check_df_exists()
        self.df['DA_RPE'] = np.nan
        daRpes = []
        for i in self.visinds:
            bline = 0#self.df.loc[i-10:i,'green_z_scored'].mean()
            if self.df.loc[i,'rwd']==1:
                daRpes.append(self.df.loc[i+0*self.fs:i+0.5*self.fs,'green_z_scored'].max())
            else:
                daRpes.append(self.df.loc[i+0*self.fs:i+0.5*self.fs,'green_z_scored'].min())
        self.df.loc[self.visinds,'DA_RPE']=daRpes
        self.df['DA_RPE'] = self.df['DA_RPE'].astype("float16")
        self.triframe['DA_RPE']=daRpes
        fig = plt.figure()
        sns.distplot(self.triframe.loc[self.triframe.rwd==1,'DA_RPE'],label='rwd')
        sns.distplot(self.triframe.loc[self.triframe.rwd==0,'DA_RPE'],label='omission')
        plt.legend()
        fig.savefig(self.directory_prefix+'DA_rwdVsOm_maxMin.pdf')

    
    def get_qRPE(self):
        r = self.triframe.loc[:,'rwd'].values
        portQs = self.df.loc[self.visinds,'Q_chosen'].values
        self.triframe.loc[:,"q_rpe"] = r - scaler.transform(portQs.reshape(-1,1)).T[0]

    def get_avail(self,p):
        allports = [0,1,2]
        allports.pop(p)
        return allports

    def get_lr_dif_val(self,data,factor):
        '''returns list of factor values for left choice for each trial.'''
        ports = data.loc[data.port!=-100,'port'].values
        vinds = data.loc[data.port!=-100].index
        fac = [0]
        if factor=='dist':
            for p in range(1,len(ports)): #append values of going left
                if ports[p-1]==2:
                    fac.append(data.loc[vinds[p],'lenBC']-data.loc[vinds[p],'lenAC'])
                elif ports[p-1]==1:
                    fac.append(data.loc[vinds[p],'lenAB']-data.loc[vinds[p],'lenBC'])
                elif ports[p-1]==0:
                    fac.append(data.loc[vinds[p],'lenAC']-data.loc[vinds[p],'lenAB'])
        else:
            for p in range(1,len(ports)): #append values of going left
                if ports[p-1]==2:
                    fac.append(data.loc[vinds[p],factor+'_b']-data.loc[vinds[p],factor+'_a'])
                elif ports[p-1]==1:
                    fac.append(data.loc[vinds[p],factor+'_a']-data.loc[vinds[p],factor+'_c'])
                elif ports[p-1]==0:
                    fac.append(data.loc[vinds[p],factor+'_c']-data.loc[vinds[p],factor+'_b'])
        return fac

    def get_left_val(self,data,factor):
        '''returns list of factor values for left choice for each trial.'''
        ports = data.loc[data.port!=-100,'port'].values
        vinds = data.loc[data.port!=-100].index
        fac = [0]
        if factor=='dist':
            for p in range(1,len(ports)): #append values of going left
                if ports[p-1]==2:
                    fac.append(data.loc[vinds[p],'lenBC'])
                elif ports[p-1]==1:
                    fac.append(data.loc[vinds[p],'lenAB'])
                elif ports[p-1]==0:
                    fac.append(data.loc[vinds[p],'lenAC'])
        else:
            for p in range(1,len(ports)): #append values of going left
                if ports[p-1]==2:
                    fac.append(data.loc[vinds[p],factor+'_b'])
                elif ports[p-1]==1:
                    fac.append(data.loc[vinds[p],factor+'_a'])
                elif ports[p-1]==0:
                    fac.append(data.loc[vinds[p],factor+'_c'])
        return fac
    
    def tri_rr(self,data,ports=[0,1,2],t=5):
        '''Return trial-by-trial estimte of previous reward history over past t trials'''
        if len(ports)==1:
            rwds = data.loc[data.port==ports[0],'rwd'].copy()
            rwds.loc[rwds==-1]=0
            rwds = rwds.rolling(window=t).sum()
            rwds = rwds.reindex(np.arange(data.index[0],data.index[-1]+1),fill_value=-100)
            return rwds.replace(-100,method='ffill')
        elif len(ports)==2:
            rwds = data.loc[(data.port==ports[0])|(data.port==ports[1]),'rwd'].copy()
            rwds.loc[rwds==-1]=0
            rwds = rwds.rolling(window=t).sum()
            rwds = rwds.reindex(np.arange(data.index[0],data.index[-1]+1),fill_value=-100)
            return rwds.replace(-100,method='ffill')
        elif len(ports)==3:
            rwds = data.loc[data.port!=-100,'rwd'].copy()
            rwds.loc[rwds==-1]=0
            rwds = rwds.rolling(window=t).sum()
            rwds = rwds.reindex(np.arange(data.index[0],data.index[-1]+1),fill_value=-100)
            return rwds.replace(-100,method='ffill')
    
    def get_log_pchoos_v_costNben(self):
        df = self.triframe.copy()
        df.rat = df.rat.astype('category')
        df['ratcodes'] = df.rat.cat.codes
        seshs=df.session.unique()
        for s in range(len(seshs)):
            sdf = df.loc[(df.session==seshs[s])]
            #sdf['rhist_a'] = self.tri_rr(sdf,[0],t=20).fillna(0)
            #sdf['rhist_b'] = self.tri_rr(sdf,[1],t=20).fillna(0)
            #sdf['rhist_c'] = self.tri_rr(sdf,[2],t=20).fillna(0)
            rdf = pd.DataFrame({'rdif':self.get_lr_dif_val(sdf,'nom_rwd'),\
                'ldif':self.get_lr_dif_val(sdf,'dist')})
            rdf['rhist_dif'] = get_lr_dif_val(sdf,'rhist')
            rdf['choose_L'] = sdf.lrchoice.values
            rdf['session']=s
            rdf['rat'] = sdf.ratcodes.values
            rdf['tri'] = sdf.tri.values
            rdf['block'] = sdf.block.values
            if s == 0:
                self.regdf = rdf
            else:
                self.regdf = self.regdf.append(rdf,ignore_index=True)
        self.regdf.loc[regdf.choose_L==2,'choose_L']=np.nan
        
    def plot_log_pchoos_v_costNben(self):
        plt.figure()
        plt.subplot(121)
        plt.title('p(choose L) vs len dif',fontsize='x-large',fontweight='bold')
        sns.regplot(x='ldif',y='choose_L',data=regdf,logistic=True)#,scatter=False)
        plt.ylim(.1,.9)
        plt.xlabel('distance dif',fontsize='large',fontweight='bold')
        plt.ylabel('choose L',fontsize='large',fontweight='bold')
        plt.subplot(122)
        #plt.twinx(plt.gca())
        plt.title('p(choose L) vs rwd dif',fontsize='x-large',fontweight='bold')
        sns.regplot(x='rhist_dif',y='choose_L',data=regdf.loc[(regdf.rhist_dif!=-100)&(regdf.rhist_dif!=100)],\
                    logistic=True,color='orange')#,scatter=False)
        plt.xlabel('rwd hist dif',fontsize='large',fontweight='bold')
        plt.ylabel('')
        plt.ylim(.1,0.9)
        

    def avg_factor(self,portz,visinds):
        #isinds = self.dat.loc[self.dat.port!=-100].index
        df = self.dat.loc[visinds,[self.factor+'_a',self.factor\
        +'_b',self.factor+'_c']].values
        fval = []
        for i in range(len(portz)):
            fval.append(np.mean(df[i,portz[i]]))
        return fval

    def factor_by_p_type(self,p_type='all'):
        '''p_type can be all, avail, or chosen. returns value of factor given p_type for every trial.'''
        visinds = self.dat.loc[self.dat.port!=-100].index
        if p_type == 'chosen':
            portz = self.dat.loc[visinds,'port'].values.astype(int)
        elif p_type == 'avail':
            portz = [[0,1,2]]
            for p in self.dat.loc[visinds[:-1],'port'].values.astype(int):
                portz.append(self.get_avail(p))
            visinds = np.concatenate([visinds[1:],[visinds[-1]]])
        elif p_type == 'all':
            portz = np.tile([0,1,2],(len(visinds),1))
        elif p_type == 'other':
            portz = [0]
            for p in range(len(visinds)-1):
                avail = self.get_avail(int(self.dat.loc[visinds[p],'port']))
                chos = int(self.dat.loc[visinds[p+1],'port'])
                avail.remove(chos)
                portz.append(avail)
            visinds = np.concatenate([visinds[1:],[visinds[-1]]])
        return self.avg_factor(portz,visinds)

    def get_vals_by_portType(self,dattype='float16'):
        addto = self.dat.loc[self.dat.port!=-100].index
        bytype = pd.DataFrame()
        bytype[self.factor+'_avail'] = self.factor_by_p_type('avail')
        bytype[self.factor+'_chosen'] = self.factor_by_p_type('chosen')
        bytype[self.factor+'_all'] = self.factor_by_p_type('all')
        bytype[self.factor+'_other'] = self.factor_by_p_type('other')
        bytype = bytype.set_index(addto)
        bytype = bytype.reindex(np.arange(self.dat.index[0],self.dat.index[-1]+1),fill_value=np.nan)
        
        self.dat[self.factor+'_avail'] = bytype.loc[:,self.factor+'_avail'].\
        fillna(method='ffill').astype(dattype)
        self.dat[self.factor+'_other'] = bytype.loc[:,self.factor+'_other'].\
        fillna(method='ffill').astype(dattype)
        self.dat[self.factor+'_chosen'] = bytype.loc[:,self.factor+'_chosen'].\
        fillna(method='bfill').astype(dattype)
        self.dat[self.factor+'_all'] = bytype.loc[:,self.factor+'_all'].\
        fillna(method='bfill').astype(dattype)

    def get_valOfChosenPort(self,dattype='float16'):
        addto = self.dat.loc[self.dat.port!=-100].index
        bytype = pd.DataFrame()
        bytype[self.factor+'_chosen'] = self.factor_by_p_type('chosen')
        bytype = bytype.set_index(addto)
        bytype = bytype.reindex(np.arange(self.dat.index[0],self.dat.index[-1]+1),fill_value=np.nan)
        self.dat[self.factor+'_chosen'] = bytype.loc[:,self.factor+'_chosen'].\
        fillna(method='bfill').astype(dattype)

    def get_optimizedValmapDF(self):
        seshs = self.df.session.unique()
        for sesh,gamma in zip(seshs,self.opt_gams):
            valdat = self.get_valdat(sesh,gamma)
            if sesh == seshs[0]:
                self.valdata = valdat
            else:
                self.valdata = self.valdata.append(valdat,ignore_index=True)
            del valdat

    def optimize_valParams(self):
        seshs = self.df.session.unique()
        self.allcors = []
        self.allseshs = []
        self.opt_gams = []
        gammas = np.linspace(0.5,1.0,51)
        self.opt_cors = []
        for s in tqdm(range(len(seshs))):
            self.cor_for_opt = []
            for gam in gammas:
                if self.usePairedState:
                    valdat = self.get_valdat_pairedState(seshs[s],gam)
                    self.get_cors4valOpt_pairedState(valdat,seshs[s])
                else:
                    valdat = self.get_valdat(seshs[s],gam)
                    self.get_cors4valOpt(valdat,seshs[s])
            self.opt_gams.append(gammas[np.argmax(self.cor_for_opt)])
            self.opt_cors.append(np.max(self.cor_for_opt))
            if s == 0:
                self.valdata = valdat
            else:
                self.valdata = self.valdata.append(valdat,ignore_index=True)
            del valdat
        del self.valdata

    def get_optimizedValmapDF_pairedState(self):
        seshs = self.df.session.unique()
        for sesh,gamma in zip(seshs,self.opt_gams):
            valdat = self.get_valdat_pairedState(sesh,gamma)
            if sesh == seshs[0]:
                self.valdata = valdat
            else:
                self.valdata = self.valdata.append(valdat,ignore_index=True)
            del valdat

    def get_valdat(self,sesh,gam):
        for b in self.df.loc[self.df.session==sesh,'block'].unique():
            if self.df.loc[self.df.session==sesh,'session_type'].values[0]=='prob':
                self.tmat = self.sesh_tmats[sesh]
                self.bars = self.sesh_barIDs[sesh]
            else:
                try:
                    self.tmat = self.sesh_tmats[sesh][int(b-1)]
                except:
                    continue
                self.bars = self.sesh_barIDs[sesh][int(b-1)]
            self.hexdata = self.df.loc[(self.df.session==sesh)&(self.df.block==b)]
            vals = self.hex_v_iteration(self.hexdata.loc[self.hexdata.port!=-100],gam)
            if b>1:
                valmap = np.vstack([valmap,vals])
            else:
                valmap = vals
        valdat = pd.DataFrame(valmap)
        valdat['trial']= np.arange(1,len(valdat)+1)
        valdat['block']= self.df.loc[(self.df.port!=-100)&(self.df.session==sesh),'block']\
        .values[:len(valdat)]
        valdat['port']= self.df.loc[(self.df.port!=-100)&(self.df.session==sesh),'port'].\
        values[:len(valdat)]
        valdat['fromP']= np.concatenate([[3],valdat.port.values[:-1]])
        valdat['session'] = np.tile(sesh,(len(valdat)))
        return valdat

    def get_valdat_pairedState(self,sesh,gam):
        for b in self.df.loc[self.df.session==sesh,'block'].unique():
            if self.df.loc[self.df.session==sesh,'session_type'].values[0]=='prob':
                self.tmat = self.sesh_tmats[sesh]
                self.bars = self.sesh_barIDs[sesh]
            else:
                try:
                    self.tmat = self.sesh_tmats[sesh][int(b-1)]
                except:
                    continue
                self.bars = self.sesh_barIDs[sesh][int(b-1)]
            self.hexdata = self.df.loc[(self.df.session==sesh)&(self.df.block==b)]
            if self.iter_type=='avgPort':
                vals = self.hex_v_iteration_pairedStateAvg(self.hexdata.loc[self.hexdata.port!=-100],gam)
            else:
                vals = self.hex_v_iteration_pairedState(self.hexdata.loc[self.hexdata.port!=-100],gam)
            if b>1:
                valmap = np.vstack([valmap,vals])
            else:
                valmap = vals
        valdat = pd.DataFrame(valmap)
        valdat['trial']= np.arange(1,len(valdat)+1)
        valdat['block']= self.df.loc[(self.df.port!=-100)&(self.df.session==sesh),'block']\
        .values[:len(valdat)]
        valdat['port']= self.df.loc[(self.df.port!=-100)&(self.df.session==sesh),'port'].\
        values[:len(valdat)]
        valdat['fromP']= np.concatenate([[3],valdat.port.values[:-1]])
        valdat['session'] = np.tile(sesh,(len(valdat)))
        return valdat

    def hex_v_iteration(self,data,gam):
        actions = [0,1,2,3,4,5]
        V = np.full((len(data),49),0.0,dtype="float16")
        if self.use_nom_rwds:
            portQs = data.loc[:,['nom_rwd_a', 'nom_rwd_b', 'nom_rwd_c']].values
        else:
            portQs = data.loc[:,['Q_a', 'Q_b', 'Q_c']].values
        portz = [0,1,2]
        self.add_bars2tmatrix()
        plt.ion()
        plt.show()
        #print('starting iteration...')
        if self.plotvals:
            plt.figure()
        for t in range(1,len(data)):
            rwdvec = np.zeros(49)
            if self.iter_type=="chosenPort":
                avail = np.array([data.iloc[t].port])
            elif self.iter_type=="otherPort":
                availPortHexes = np.setdiff1d(portz,data.iloc[t-1].port)
                avail = np.setdiff1d(availPortHexes,data.iloc[t].port)
            elif self.iter_type=='bestPort':
                availPortHexes = np.setdiff1d(portz,data.iloc[t-1].port)
                avail = np.argmax(portQs[t,availPortHexes])
            else:
                avail = np.setdiff1d(portz,data.iloc[t-1].port)
            if np.max(np.abs(portQs[t,avail]))==0:
                continue
            rwdvec[avail] = scaler.transform(portQs[t,avail].reshape(-1,1)).T[0]
            i = 0
            if self.plotvals:
                viz_value_map(V[t],i)
            oldV = V[t,:]#np.full(49,0.5,dtype="float16")#np.zeros(49)
            while True:
                allVs = []
                for a in actions:
                    allVs.append(gam*np.dot(oldV,self.bar_tmatrix[:,a].T)+rwdvec)
                newV = np.amax(allVs,axis=0)
                if self.plotvals:
                    viz_value_map(newV,i)
                if np.max(newV-oldV)<10e-4:
                    V[t] = newV
                    #set barriers to zero value
                    V[t,self.bars]=0
                    break
                oldV = newV
                i += 1
        return V

    def hex_v_iteration_pairedState(self,data,gam):
        actions = [0,1]
        V = np.full((len(data),126),0.0,dtype="float16")
        if self.use_nom_rwds:
            portQs = data.loc[:,['nom_rwd_a', 'nom_rwd_b', 'nom_rwd_c']].values
        else:
            portQs = data.loc[:,['Q_a', 'Q_b', 'Q_c']].values
        portz = [phexdf.loc[phexdf.to_state==1,'statecodes'].values[0],
        phexdf.loc[phexdf.to_state==2,'statecodes'].values[0],
         phexdf.loc[phexdf.to_state==3,'statecodes'].values[0]]
        porthexes = [0,1,2]
        self.add_bars2tmatrix_pairedState()
        plt.ion()
        plt.show()
        #print('starting iteration...')
        if self.plotvals:
            plt.figure()
        for t in range(1,len(data)):
            rwdvec = np.zeros(126)
            if self.iter_type=="chosenPort":
                avail = np.array([portz[data.iloc[t].port]])
                availPortHexes = np.array([porthexes[data.iloc[t].port]])
                avail = np.array([portz[availPortHexes[0]]])
            elif self.iter_type=="otherPort":
                availPortHexes = np.setdiff1d(porthexes,data.iloc[t-1].port)
                availPortHexes = np.setdiff1d(availPortHexes,data.iloc[t].port)
                avail = np.array([portz[availPortHexes[0]]])
            elif self.iter_type=='bestPort':
                availPortHexes = np.setdiff1d(porthexes,data.iloc[t-1].port)
                maxindex = np.argmax(portQs[t,availPortHexes])
                availPortHexes = np.array([availPortHexes[maxindex]])
                avail = np.array([portz[availPortHexes[0]]])
            elif self.iter_type=="right":
                inds = np.add(data.iloc[t-1].port,[1,2])%3
                availPortHexes = np.array([porthexes[inds[0]]])
                avail = np.array([portz[availPortHexes[0]]])
            elif self.iter_type=="left":
                inds = np.add(data.iloc[t-1].port,[1,2])%3
                availPortHexes = np.array([porthexes[inds[1]]])
                avail = np.array([portz[availPortHexes[0]]])
            else:
                avail = np.delete(portz,data.iloc[t-1].port)
                availPortHexes = np.setdiff1d(porthexes,data.iloc[t-1].port)
            #if np.max(np.abs(portQs[t,availPortHexes]))==0:
            #    continue            
            rwdvec[avail] = scaler.transform(portQs[t,availPortHexes].reshape(-1,1)).T[0]
            i = 0
            if self.plotvals:
                viz_value_map(V[t],i)
            oldV = V[t,:]#np.full(49,0.5,dtype="float16")#np.zeros(49)
            while True:
                allVs = []
                for a in actions:
                    allVs.append(gam*np.dot(oldV,self.bar_tmatrix[:,a].T)+rwdvec)
                newV = np.amax(allVs,axis=0)
                if self.plotvals:
                    viz_value_map_pairedState(newV,i)
                if np.max(newV-oldV)<10e-4:
                    V[t] = newV
                    #set barriers to zero value
                    V[t,self.paired_barstates]=np.nan
                    break
                oldV = newV
                i += 1
        return V

    def add_bars2tmatrix(self):
        self.bar_tmatrix = tmatrix.copy()
        self.bar_tmatrix[self.bars] = np.tile(emptp,(6,1))

    def convertDfHexlabelsToPairedState(self):
        self.df.loc[:,'pairedHexStates'] = -1
        seshs = self.df.session.unique()
        for s in tqdm(range(len(seshs))):
            dat = self.df.loc[self.df.session==s]
            datinds = dat.loc[dat.port!=-100].index
            for i in range(1,len(datinds)):
                tdat = dat.loc[datinds[i-1]:datinds[i]]
                hexInOrder = tdat.loc[tdat.hexlabels.diff()!=0,'hexlabels']
                hexinds = hexInOrder.index
                self.df.loc[hexinds,'pairedHexStates']=convert2pairedState(self,hexInOrder.values)
        self.df.loc[:,"pairedHexStates"] = self.df.loc[:,"pairedHexStates"].astype(object).replace(-1,method='ffill')
        self.df.loc[:,"pairedHexStates"] = self.df.loc[:,"pairedHexStates"].astype(object).replace(-1,method='bfill')

    def convert2pairedState(self,orderedHexes):
        pairedStates = []
        for i in range(1,len(orderedHexes)):
            fr,to = orderedHexes[i-1],orderedHexes[i]
            newstate = phexdf.loc[(phexdf.from_state==fr)&
            (phexdf.to_state==to),'statecodes'].values
            if len(newstate)<1:
                pairedStates.append(-1)
            else:
                pairedStates.append(newstate[0])
        return np.append([-1],pairedStates)

    def add_bars2tmatrix_pairedState(self):
        self.bar_tmatrix = paired_tmatrix.copy()
        self.paired_barstates = phexdf.loc[(np.isin(phexdf.from_state,self.bars+1))|
        (np.isin(phexdf.to_state,self.bars+1)),'statecodes'].values
        for b in self.paired_barstates:
            barloc = np.where(adj_matx==b)
            if np.shape(barloc)[1]==1:
                self.bar_tmatrix[np.array(barloc)[0,0],np.array(barloc)[1,0]]=np.zeros(126)
            else:
                for bind in range(len(np.array(barloc))):
                   self.bar_tmatrix[np.array(barloc)[0,bind],np.array(barloc)[1,bind]]=np.zeros(126)

    def plot_hex_outline(self,sesh,block,ax,size='sm'):
        sz = 300 if size=='sm' else 2000
        if self.df.loc[self.df.session==sesh,'session_type'].unique()[0]=='barrier':
            bardf = centdf.drop(self.sesh_barIDs[sesh][block-1]+1,axis=0)
        else:
            bardf = centdf.drop(self.sesh_barIDs[sesh]+1,axis=0)
        ax.scatter(bardf[0].values,bardf[1].values,marker='H',color='grey',\
            edgecolors= "darkgrey",alpha=0.4,s=sz)

    def get_availstates(self):
        self.availstates = np.setdiff1d(np.arange(0,49),self.bars)
        self.availstates = np.setdiff1d(self.availstates,[0,1,2])

    def get_cors4valOpt(self,valdat,sesh):
        da_v_val = [[],[],[],[]]
        vinds = np.concatenate([[self.df.loc[(self.df.session==sesh)].index[0]],\
            self.df.loc[(self.df.session==sesh)&(self.df.port!=-100)].index])
        smoothed_light = self.df.loc[self.df.session==sesh,'green_z_scored'].\
        rolling(window=self.bin_size).mean()
        for t in range(1,len(vinds)):
            tdat = self.df.loc[vinds[t-1]+1:vinds[t]]
            tdat = tdat.loc[tdat.lastleave==2]
            if len(tdat) < 2:
                continue
            tinds = tdat.loc[tdat.hexlabels.diff()!=0].index
            dvals = smoothed_light[tinds+self.offset]#tdat.loc[tinds+self.offset,'green_z_scored'].values
            hexpairs = tdat.loc[tinds,'hexlabels'].values
            da_v_val[2].append(tdat.loc[tinds,'vel'].values)
            da_v_val[3].append(tdat.loc[tinds,'acc'].values)
            vdat = valdat.loc[valdat.trial==t]
            hvals = vdat.mean()[:49]
            hvals[hvals==0] = np.nan
            da_v_val[0].append(dvals)
            da_v_val[1].append(hvals[hexpairs-1].values)
        dvcor = spearmanr(np.concatenate(da_v_val[0]),np.concatenate(da_v_val[1]),\
            nan_policy='omit')[0]
        self.cor_for_opt.append(dvcor)
        self.allcors.append(dvcor)
        self.allseshs.append(sesh)

    def get_cors4valOpt_pairedState(self,valdat,sesh):
        da_v_val = [[],[]]
        vinds = np.concatenate([[self.df.loc[(self.df.session==sesh)].index[0]],\
            self.df.loc[(self.df.session==sesh)&(self.df.port!=-100)].index])
        smoothed_light = self.df.loc[self.df.session==sesh,'green_z_scored'].\
        rolling(window=self.bin_size).mean()
        for t in range(1,len(valdat)):
            tdat = self.df.loc[vinds[t-1]+1:vinds[t]]
            tdat = tdat.loc[tdat.lastleave==2]
            if len(tdat) < 2:
                continue
            tinds = tdat.loc[tdat.pairedHexStates.diff()!=0].index
            dvals = smoothed_light[tinds+self.offset]#tdat.loc[tinds+self.offset,'green_z_scored'].values
            hexpairs = tdat.loc[tinds,'pairedHexStates'].values
            vdat = valdat.loc[valdat.trial==t]
            hvals = vdat.iloc[:,:126]#vdat.mean()[:126]
            #hvals[hvals==0] = np.nan
            da_v_val[0].append(dvals)
            da_v_val[1].append(hvals[hexpairs].values)
        dvcor = spearmanr(np.concatenate(da_v_val[0],axis=None),
            np.concatenate(da_v_val[1],axis=None),nan_policy='omit')[0]
        self.cor_for_opt.append(dvcor)
        self.allcors.append(dvcor)
        self.allseshs.append(sesh)

    def get_valCorFrame(self):
        da_v_val = [[],[],[],[],[],[],[],[],[],[]]
        seshs = self.df.session.unique()
        for s in tqdm(range(len(seshs))):
            sesh = seshs[s]
            vinds = np.concatenate([[self.df.loc[(self.df.session==sesh)].index[0]],\
                    self.df.loc[(self.df.session==sesh)&(self.df.port!=-100)].index])
            smoothed_light = self.df.loc[self.df.session==sesh,'green_z_scored'].\
            rolling(window=self.bin_size).mean()
            for t in range(1,len(vinds)):
                tdat = self.df.loc[vinds[t-1]+1:vinds[t]]
                tdat = tdat.loc[tdat.lastleave==2]
                if len(tdat) <2:
                    #print("didn't work for sesh ",sesh," trial ",t)
                    continue
                tinds = tdat.loc[tdat.hexlabels.diff()!=0].index
                dvals = smoothed_light[tinds+self.offset]
                hexpairs = tdat.loc[tinds,'hexlabels'].values
                vdat = self.valdata.loc[(self.valdata.session==sesh)&(self.valdata.trial==t)]
                hvals = vdat.mean()[:49]
                velvals = tdat.loc[tinds+self.offset+self.fs/2,'vel']
                accvals = tdat.loc[tinds+self.offset+self.fs/2,'acc']
                hvals[hvals==0] = np.nan
                da_v_val[0].append(dvals)
                da_v_val[1].append(hvals[hexpairs-1].values)
                da_v_val[2].append(np.full(len(dvals),sesh,dtype="int8"))
                da_v_val[3].append(np.full(len(dvals),tdat.rat.unique()[0]))
                da_v_val[4].append(np.full(len(dvals),tdat.fiberloc.unique()[0]))
                da_v_val[5].append(velvals.values)
                da_v_val[6].append(accvals.values)
                da_v_val[7].append(np.full(len(dvals),tdat.tri.unique()[0]))
                da_v_val[8].append(np.full(len(dvals),tdat.block.unique()[0]))
                da_v_val[9].append(hexpairs)
        self.corframe = pd.DataFrame({'DA':np.concatenate(da_v_val[0]),'val':np.concatenate(da_v_val[1]),\
                                'sesh':np.concatenate(da_v_val[2]),'rat':np.concatenate(da_v_val[3]),\
                                'location':np.concatenate(da_v_val[4]),'vel':np.concatenate(da_v_val[5]),\
                                'acc':np.concatenate(da_v_val[6]),'tri':np.concatenate(da_v_val[7]),\
                                'block':np.concatenate(da_v_val[8]),'hex':np.concatenate(da_v_val[9])})
        self.corframe['ratloc'] = self.corframe.rat + self.corframe.location

    def get_portVal_HexCorFrame(self):
        da_v_val = [[],[],[],[],[],[],[],[],[],[]]
        seshs = self.df.session.unique()
        for s in tqdm(range(len(seshs))):
            sesh = seshs[s]
            vinds = np.concatenate([[self.df.loc[(self.df.session==sesh)].index[0]],\
                    self.df.loc[(self.df.session==sesh)&(self.df.port!=-100)].index])
            smoothed_light = self.df.loc[self.df.session==sesh,'green_z_scored'].\
            rolling(window=self.bin_size).mean()
            for t in range(1,len(vinds)):
                tdat = self.df.loc[vinds[t-1]+1:vinds[t]]
                tdat = tdat.loc[tdat.lastleave==2]
                if len(tdat) <2:
                    continue
                if self.use_nom_rwds:
                    portValues = tdat.loc[vinds[t]-2,['nom_rwd_a', 'nom_rwd_b', 'nom_rwd_c']].values
                else:
                    portValues = tdat.loc[vinds[t]-2,['Q_a', 'Q_b', 'Q_c']].values
                    portValues = scaler.transform(portValues.reshape(-1,1)).T[0]
                chosen_val,other_val = self.get_trial_chosenAndOtherPortVal(portValues,vinds,t)
                tinds = tdat.loc[tdat.hexlabels.diff()!=0].index
                dvals = smoothed_light[tinds+self.offset]
                hexpairs = tdat.loc[tinds,'hexlabels'].values
                da_v_val[0].append(dvals)
                da_v_val[1].append(np.full(len(dvals),chosen_val,dtype="float16"))
                da_v_val[2].append(np.full(len(dvals),other_val,dtype="float16"))
                da_v_val[3].append(np.full(len(dvals),sesh,dtype="int8"))
                da_v_val[4].append(np.full(len(dvals),tdat.rat.unique()[0]))
                da_v_val[5].append(np.full(len(dvals),tdat.fiberloc.unique()[0]))
                da_v_val[6].append(np.full(len(dvals),tdat.tri.unique()[0]))
                da_v_val[7].append(np.full(len(dvals),tdat.block.unique()[0]))
                da_v_val[8].append(hexpairs)
        self.portValHexCorframe = pd.DataFrame({'DA':np.concatenate(da_v_val[0]),\
            'chosenVal':np.concatenate(da_v_val[1]),"otherVal":np.concatenate(da_v_val[2]),\
            'sesh':np.concatenate(da_v_val[3]),'rat':np.concatenate(da_v_val[4]),\
            'location':np.concatenate(da_v_val[5]),'tri':np.concatenate(da_v_val[6]),\
            'block':np.concatenate(da_v_val[7]),'hex':np.concatenate(da_v_val[8])})

    def get_trial_chosenAndOtherPortVal(self,portValues,vinds,t):
        avail = [0,1,2]
        chosen_val = portValues[self.df.loc[vinds[t],'port']]
        avail.remove(self.df.loc[vinds[t],'port'])
        avail.remove(self.df.loc[vinds[t-1],'port'])
        other_val = portValues[avail[0]]
        return(chosen_val,other_val)

    def get_valCorFrame_pairedState(self):
        da_v_val = [[],[],[],[],[],[],[],[],[],[]]
        seshs = self.df.session.unique()
        for s in tqdm(range(len(seshs))):
            sesh = seshs[s]
            vinds = np.concatenate([[self.df.loc[(self.df.session==sesh)].index[0]],\
                    self.df.loc[(self.df.session==sesh)&(self.df.port!=-100)].index])
            smoothed_light = self.df.loc[self.df.session==sesh,'green_z_scored'].\
            rolling(window=self.bin_size).mean()
            for t in range(1,len(self.valdata.loc[(self.valdata.session==sesh)])):
                tdat = self.df.loc[vinds[t-1]+1:vinds[t]]
                tdat = tdat.loc[tdat.lastleave==2]
                if len(tdat) <2:
                    continue
                tinds = tdat.loc[tdat.pairedHexStates.diff()!=0].index
                offset_inds = tinds+int(self.offset+self.fs/2)
                offset_inds = offset_inds[np.where(np.isin(offset_inds,tdat.index))]
                dvals = smoothed_light[tinds+self.offset]
                hexpairs = tdat.loc[tinds,'pairedHexStates'].values
                vdat = self.valdata.loc[(self.valdata.session==sesh)&(self.valdata.trial==t)]
                hvals = vdat.iloc[:,:126]
                velvals = tdat.loc[tinds+int(self.offset+self.fs/2),'vel']
                accvals = tdat.loc[tinds+int(self.offset+self.fs/2),'acc']
                da_v_val[0].append(dvals)
                da_v_val[1].append(hvals[hexpairs].values[0])
                da_v_val[2].append(np.full(len(dvals),sesh,dtype="int8"))
                da_v_val[3].append(np.full(len(dvals),tdat.rat.unique()[0]))
                da_v_val[4].append(np.full(len(dvals),tdat.fiberloc.unique()[0]))
                da_v_val[5].append(velvals.values)
                da_v_val[6].append(accvals.values)
                da_v_val[7].append(np.full(len(dvals),tdat.tri.unique()[0]))
                da_v_val[8].append(np.full(len(dvals),tdat.block.unique()[0]))
                da_v_val[9].append(hexpairs)
        self.corframe = pd.DataFrame({'DA':np.concatenate(da_v_val[0],axis=None),
                'val':np.concatenate(da_v_val[1],axis=None),\
                'sesh':np.concatenate(da_v_val[2],axis=None),'rat':np.concatenate(da_v_val[3],axis=None),\
                'location':np.concatenate(da_v_val[4],axis=None),'vel':np.concatenate(da_v_val[5],axis=None),\
                'acc':np.concatenate(da_v_val[6],axis=None),'tri':np.concatenate(da_v_val[7],axis=None),\
                'block':np.concatenate(da_v_val[8],axis=None),'hex':np.concatenate(da_v_val[9],axis=None)})
        self.corframe['ratloc'] = self.corframe.rat + self.corframe.location

    def get_distanceToPort(self,getDistsFromDeadEnds=True):
        ports = [0,1,2]
        portStrs = ["A","B","C"]
        self.sesh_hexDists = {"dto"+portStrs[p]:{s:[] for s in 
            self.df.session.unique()} for p in ports}
        self.sesh_arrows2goal = {"to"+portStrs[p]:{s:[] for s in 
            self.df.session.unique()} for p in ports}
        for p in ports:
            for s in self.df.session.unique():
                dat = self.df.loc[(self.df.session==s)&(self.df.port!=-100)]
                if self.df.loc[self.df.session==s,'session_type'].values[0]=='prob':
                    self.tmat = self.sesh_tmats[s]
                    self.bars = self.sesh_barIDs[s]
                    dmap = self.compute_distanceToPort(dat,p,getDistsFromDeadEnds)
                    self.sesh_hexDists["dto"+portStrs[p]][s] = dmap
                    togoal = self.find_arrows2goal(dmap,p)
                    self.sesh_arrows2goal["to"+portStrs[p]][s] = togoal
                else:
                    for b in dat.block.unique():
                        try:
                            self.tmat = self.sesh_tmats[s][int(b-1)]
                        except:
                            continue
                        self.bars = self.sesh_barIDs[s][int(b-1)]
                        dmap = self.compute_distanceToPort(dat.loc[dat.block==b],\
                            p,getDistsFromDeadEnds)
                        self.sesh_hexDists["dto"+portStrs[p]][s].append(dmap)
                        togoal = self.find_arrows2goal(dmap,p)
                        self.sesh_arrows2goal["to"+portStrs[p]][s].append(togoal)

    def compute_distanceToPort(self,data,port,getDistsFromDeadEnds=True):
        actions = [0,1]
        distmap = np.full(126,0,dtype="float16")
        portz = [phexdf.loc[phexdf.to_state==1,'statecodes'].values[0],
        phexdf.loc[phexdf.to_state==2,'statecodes'].values[0],
         phexdf.loc[phexdf.to_state==3,'statecodes'].values[0]]
        self.add_bars2tmatrix_pairedState()
        t=1
        avail = np.array([portz[port]])
        i = 0
        oldD = distmap
        portvec = distmap
        portvec[avail]=1
        dfact=0.9
        while True:
            allDs = []
            for a in actions:
                allDs.append(dfact*np.dot(oldD,self.bar_tmatrix[:,a].T)+portvec)
            newD = np.amax(allDs,axis=0)
            if np.max(newD-oldD)<10e-4:
                distmap = np.array([np.nan if newD[i]==0 else\
                    np.round(mlog(newD[i],dfact)).astype(int) for i in range(len(newD))])
                #set barriers to nan
                distmap[self.paired_barstates]=np.nan
                break
            oldD = newD
            i += 1
        if getDistsFromDeadEnds:
            return self.addSisterDistsToDistmap(distmap)
        return distmap

    def addSisterDistsToDistmap(self,distmap):
        for sCode in range(len(distmap)):
            stateHexPair = phexdf.loc[phexdf.statecodes==sCode,["from_state","to_state"]].values[0]
            sisterState = phexdf.loc[(phexdf.from_state==stateHexPair[1])&\
                    (phexdf.to_state==stateHexPair[0]),"statecodes"].values[0]
            if np.isnan(distmap[sisterState]):
                distmap[sisterState] = distmap[sCode]
        return distmap

    def find_arrows2goal(self,distmap,port):
        arrows2goal = np.copy(distmap)
        for i in arrowdf.from_state.unique():
            options = list(arrowdf.loc[arrowdf.from_state==i,"statecodes"].values)
            if all(np.isnan(distmap[options])):
                continue
            #if np.all(np.logical_not(np.isnan(distmap[options]))==
            #    np.logical_not(np.isnan(distmap[options]))[0]):
            #    continue
            #get state with shortest distance from port
            minarrow = np.nanargmin(distmap[options])
            onlyshort = options[minarrow]
            if i-1 == port:
                arrows2goal[onlyshort]=np.nan
            if not np.all(np.isnan(distmap[options])) and \
            np.nanmin(distmap[options])==distmap[onlyshort]:
                options.pop(minarrow)
            arrows2goal[options]=np.nan
        towardsgoal = np.array([not np.isnan(i) for i in arrows2goal])
        return towardsgoal

    #TODO: write cleaner "towards goal" state identificatoin for every
    #configuration


    def chunk_all(self):
        rwd = []
        unrwd = []
        for i in self.dat_visinds:
            tritrace = self.dat.loc[i+self.fs*self.plot_window[0]:i+self.plot_window[1]\
            *self.fs,self.plot_trace]
            bline = 0 #if self.plot_trace=='vel' else self.dat.loc[i+self.fs*self.plot_window[0],self.plot_trace]
            if len(tritrace)<self.fs*self.plot_window[1]-self.fs*self.plot_window[0]:
                continue
            if self.dat.loc[i,'rwd'] == 1:
                rwd.append(np.subtract(self.dat.loc[i+self.fs*self.plot_window[0]:\
                    i+self.plot_window[1]*self.fs,self.plot_trace],bline))
            elif self.dat.loc[i,'rwd'] == 0 or self.dat.loc[i,'rwd'] == -1:
                unrwd.append(np.subtract(self.dat.loc[i+self.fs*self.plot_window[0]:\
                    i+self.plot_window[1]*self.fs,self.plot_trace],bline))
        return np.array(rwd).astype(np.float16),np.array(unrwd).astype(np.float16)

    def plot_Qvals_over_sesh(self,sesh):
        xvals = np.arange(0,len(self.df.loc[(self.df.session==sesh)&(self.df.port!=-100)].index))
        fig = plt.figure(figsize=(18,10))
        plt.plot(xvals,self.df.loc[(self.df.session==sesh)&(self.df.port!=-100),'Q_a']\
                 ,color='b',label='Q(A)')
        plt.plot(xvals,self.df.loc[(self.df.session==sesh)&(self.df.port!=-100),'Q_b']\
                 ,color='orange',label='Q(B)')
        plt.plot(xvals,self.df.loc[(self.df.session==sesh)&(self.df.port!=-100),'Q_c']\
                 ,color='green',label='Q(C)')
        plt.xlabel('trial number',fontsize='large',fontweight='bold')
        plt.ylabel('Port Q Value',fontsize='large',fontweight='bold')
        plt.legend()
        fig.savefig(self.directory_prefix+"sample_Qvals_overSesh"+str(sesh)+".png")

    def plt_all(self): 
        tr = self.plot_trace
        torwd,toom = [],[]
        for s in self.df.session.unique():
            if self.plotFirstHalfBlock:
                self.dat = self.df.loc[(self.df.session==s)&(self.df.tri<30)]
            elif self.plotSecondHalfBlock:
                self.dat = self.df.loc[(self.df.session==s)&(self.df.tri>30)]
            else:
                self.dat = self.df.loc[self.df.session==s]
            self.dat_visinds = self.dat.loc[self.dat.port!=-100].index
            tor,too = self.chunk_all()
            torwd.append(np.nanmean(tor,axis=0))
            toom.append(np.nanmean(too,axis=0))
        toall = np.vstack([torwd,toom])
        xvals = np.linspace(self.fs*self.plot_window[0],self.fs*self.plot_window[1],len(torwd[0]))/self.fs
        plt.suptitle('Port Approach. '+str(len(self.visinds))+' trials.\n'+\
                    str(len(self.df.session.unique()))+' sessions.',\
                     fontsize='xx-large',fontweight='bold')
        ax1 = plt.subplot(1,1,1)
        self.plot_trace='vel'
        vr,vu = self.chunk_all()
        v = np.vstack([vr,vu])
        if self.pltvel==True:
            ax2 = ax1.twinx()
            ax2.plot(xvals,np.mean(vr,axis=0),alpha=0.6,color='green')
            ax2.plot(xvals[-self.plot_window[0]*self.fs:],np.mean(vu,axis=0)\
                [-self.plot_window[0]*self.fs:],linestyle='--',alpha=0.6,color='green')
            ax2.set_ylabel('velocity',fontsize='x-large',fontweight='bold',color='green')
            #ax2.legend()
        ax1.set_ylabel('dF/F',fontsize='x-large',fontweight='bold')
        ax1.plot(xvals[:-self.plot_window[0]*self.fs],np.nanmean(toall,axis=0)\
            [:-self.plot_window[0]*self.fs],color='darkgreen')
        ax1.fill_between(xvals[:-self.plot_window[0]*self.fs],np.nanmean(toall,axis=0)\
            [:-self.plot_window[0]*self.fs]-sem(toall)[:-self.plot_window[0]*self.fs],\
            np.nanmean(toall,axis=0)[:-self.plot_window[0]*self.fs]+sem(toall)\
            [:-self.plot_window[0]*self.fs],color='grey',alpha=0.5)
        ax1.plot(xvals[-self.plot_window[0]*self.fs:],np.nanmean(torwd,axis=0)\
            [-self.plot_window[0]*self.fs:],color='darkgreen',label='rwd')
        ax1.fill_between(xvals[-self.plot_window[0]*self.fs:],np.nanmean(torwd,axis=0)\
            [-self.plot_window[0]*self.fs:]-sem(torwd)[-self.plot_window[0]*self.fs:],\
            np.nanmean(torwd,axis=0)[-self.plot_window[0]*self.fs:]+sem(torwd)\
            [-self.plot_window[0]*self.fs:],color='grey',alpha=0.5)
        ax1.plot(xvals[-self.plot_window[0]*self.fs:],np.nanmean(toom,axis=0)\
            [-self.plot_window[0]*self.fs:],color='darkgreen',ls=':'\
                 ,label='omission')
        ax1.fill_between(xvals[-self.plot_window[0]*self.fs:],np.nanmean(toom,axis=0)\
            [-self.plot_window[0]*self.fs:]-sem(toom)[-self.plot_window[0]*self.fs:],\
            np.nanmean(toom,axis=0)[-self.plot_window[0]*self.fs:]+sem(toom)\
            [-self.plot_window[0]*self.fs:],color='grey',alpha=0.5)
        ax1.set_xlabel('time (s) from port entry')
        ax1.axvline(x=0.0,ymin=-.1,ymax=1.0,color='k',linestyle='--')
        ax1.legend()

    def plot_allRwdRespOverlay(self):
        xvals = np.arange(0,501)/250
        fig = plt.figure(figsize=(14,10))
        for i in self.visinds[:-1]:
            if self.df.loc[i,'rwd']==1:
                plt.plot(xvals,self.df.loc[i:i+500,'green_z_scored'],alpha=0.1,color="darkblue")
            else:
                plt.plot(xvals,self.df.loc[i:i+500,'green_z_scored'],alpha=0.1,color="orangered")
        i = self.visinds[-1]
        if self.df.loc[i,'rwd']==1:
            plt.plot(xvals,self.df.loc[i:i+500,'green_z_scored'],alpha=0.1,color="darkblue",label='rwd')
        else:
            plt.plot(xvals,self.df.loc[i:i+500,'green_z_scored'],alpha=0.1,color="orangered",label='omission')
        plt.ylim(-5,16)
        plt.legend()
        plt.xlabel('time from port entry (s)',fontsize='large',fontweight='bold')
        plt.ylabel("dLight (z-scored dF/F)",fontsize='large',fontweight='bold')
        fig.savefig(self.directory_prefix+'all_DA_rwdResponse.png')

    def plot_warpedCpAlignedMeansByVal(self):
        lowmeans,midmeans,highmeans = self.get_seshMeanCpAlignedTracesByVal()
        xvals = np.arange(0,self.warpLen*2)
        fig = plt.figure(figsize=(12,8))
        plt.suptitle('Warped port approach aligned to choice point and port entry from: \n'\
            +str(self.df.rat.unique())+", "+str(self.df.fiberloc.unique())+\
            ', \n'+str(len(self.df.session.unique()))+" sessions"+\
              ', and '+str(self.df.session_type.unique()[0])+' variants.'\
              ,fontsize='large',fontweight='bold')
        plt.plot(xvals,np.mean(highmeans,axis=0),color='lightblue',label='to high value port')
        plt.plot(xvals,np.mean(midmeans,axis=0),color='blue',label='to medium value port')
        plt.plot(xvals,np.mean(lowmeans,axis=0),color='darkblue',label='to low value port')
        plt.fill_between(xvals,np.mean(highmeans,axis=0)-sem(highmeans),np.mean(highmeans,axis=0)\
            +sem(highmeans),color='lightblue',alpha=.3)
        plt.fill_between(xvals,np.mean(midmeans,axis=0)-sem(midmeans),np.mean(midmeans,axis=0)\
            +sem(midmeans),color='blue',alpha=.3)
        plt.fill_between(xvals,np.mean(lowmeans,axis=0)-sem(lowmeans),np.mean(lowmeans,axis=0)\
            +sem(lowmeans),color='darkblue',alpha=.3)
        plt.axvline(self.warpLen,color='violet')
        plt.axvline(self.warpLen*2,color='k')
        plt.ylabel(self.plot_trace,fontsize='large',fontweight='bold')
        plt.xlabel('warped time from exiting last port',fontsize='large',fontweight='bold')
        plt.legend()
        plt.xticks([0,self.warpLen,self.warpLen*2],['start','choice point','port entry'])
        fig.savefig(self.directory_prefix+'warped_'+self.plot_trace+'_ApByVal.png')

    def get_seshMeanCpAlignedTracesByVal(self):
        meanlow = []
        meanmid = []
        meanhigh = []
        for s in self.df.session.unique():
            print('session ',str(s))
            self.dat = self.df.loc[self.df.session==s,:]
            self.dat_visinds = self.dat.loc[self.dat.port!=-100].index
            tolow,tomid,tohigh = self.get_warpedCpAlignedTracesByVal()
            meanlow.append(np.mean(tolow,axis=0))
            meanmid.append(np.mean(tomid,axis=0))
            meanhigh.append(np.mean(tohigh,axis=0))
        return meanlow,meanmid,meanhigh
        
    def get_warpedCpAlignedTracesByVal(self):
        low,mid,high = self.getTriIndsByTerc()
        warpedLow = self.get_warpedCpAlignedTraces(low)
        warpedMid = self.get_warpedCpAlignedTraces(mid)
        warpedHigh = self.get_warpedCpAlignedTraces(high)
        return warpedLow,warpedMid,warpedHigh

    def get_warpedCpAlignedTraces(self,inds):
        self.dat.loc[:,'smoothed'] = self.dat[self.plot_trace].rolling(window=10).mean()
        warpTraces = []
        for i in inds:
            trial = self.dat.loc[i,'tot_tri']-1
            dat = self.dat.loc[(self.dat.tot_tri==trial)&(self.dat.lastleave==2),:]
            cpInd = dat.loc[dat.critCP.diff()==1,:].index.max()
            if np.isnan(cpInd):
                continue
            preCp = resample(dat.loc[:cpInd,'smoothed'].values,self.warpLen)
            postCP = resample(dat.loc[cpInd:,'smoothed'].values,self.warpLen)
            warpTraces.append(np.concatenate([preCp,postCP]))
        return warpTraces

    def plot_warpedPortAlignedMeansByVal(self):
        lowmeans,midmeans,highmeans = self.get_seshMeanPortAlignedTracesByVal()
        xvals = np.arange(0,self.warpLen*2)
        fig = plt.figure(figsize=(12,8))
        plt.suptitle('Warped port approach aligned to choice point and port entry from: \n'\
            +str(self.df.rat.unique())+", "+str(self.df.fiberloc.unique())+\
            ', \n'+str(len(self.df.session.unique()))+" sessions"+\
              ', and '+str(self.df.session_type.unique()[0])+' variants.'\
              ,fontsize='large',fontweight='bold')
        plt.plot(xvals,np.mean(highmeans,axis=0),color='lightblue',label='to high value port')
        plt.plot(xvals,np.mean(midmeans,axis=0),color='blue',label='to medium value port')
        plt.plot(xvals,np.mean(lowmeans,axis=0),color='darkblue',label='to low value port')
        plt.fill_between(xvals,np.mean(highmeans,axis=0)-sem(highmeans),np.mean(highmeans,axis=0)\
            +sem(highmeans),color='lightblue',alpha=.3)
        plt.fill_between(xvals,np.mean(midmeans,axis=0)-sem(midmeans),np.mean(midmeans,axis=0)\
            +sem(midmeans),color='blue',alpha=.3)
        plt.fill_between(xvals,np.mean(lowmeans,axis=0)-sem(lowmeans),np.mean(lowmeans,axis=0)\
            +sem(lowmeans),color='darkblue',alpha=.3)
        plt.axvline(self.warpLen,color='k')
        plt.axvline(self.warpLen*2,color='k')
        plt.ylabel(self.plot_trace,fontsize='large',fontweight='bold')
        plt.xlabel('warped time from exiting last port',fontsize='large',fontweight='bold')
        plt.legend()
        plt.xticks([0,self.warpLen,self.warpLen*2],['start','port entry','leave port'])
        fig.savefig(self.directory_prefix+'warped_'+self.plot_trace+'_ApByVal.png')
    
    def get_seshMeanPortAlignedTracesByVal(self):
        meanlow = []
        meanmid = []
        meanhigh = []
        for s in self.df.session.unique():
            print('session ',str(s))
            self.dat = self.df.loc[self.df.session==s,:]
            self.dat_visinds = self.dat.loc[self.dat.port!=-100].index
            tolow,tomid,tohigh = self.get_warpedPortAlignedTracesByVal()
            meanlow.append(np.mean(tolow,axis=0))
            meanmid.append(np.mean(tomid,axis=0))
            meanhigh.append(np.mean(tohigh,axis=0))
        return meanlow,meanmid,meanhigh
        
    def get_warpedPortAlignedTracesByVal(self):
        low,mid,high = self.getTriIndsByTerc()
        warpedLow = self.get_warpedPortAlignedTraces(low)
        warpedMid = self.get_warpedPortAlignedTraces(mid)
        warpedHigh = self.get_warpedPortAlignedTraces(high)
        return warpedLow,warpedMid,warpedHigh
    
    def get_warpedPortAlignedTraces(self,inds):
        self.dat.loc[:,'smoothed'] = self.dat[self.plot_trace].rolling(window=10).mean()
        warpTraces = []
        for i in inds:
            trial = self.dat.loc[i,'tot_tri']-1
            dat = self.dat.loc[(self.dat.tot_tri==trial)&(self.dat.lastleave==2),'smoothed']
            dat2 = self.dat.loc[(self.dat.tot_tri==trial+1)&(self.dat.lastleave==0),'smoothed']
            try:
                #trace = resample(dat.values,self.warpLen)
                prePort = resample(dat.values,self.warpLen)
                postPort = resample(dat2.values,self.warpLen)
            except:
                print('could not warp trial ',str(trial+1))
                continue
            #warpTraces.append(trace)
            warpTraces.append(np.concatenate([prePort,postPort]))
        return warpTraces

    def getTriIndsByTerc(self,rwdtype=None):
        if rwdtype=='rwd':
            vinds = self.dat.loc[(self.dat.port!=-100)&(self.dat.rwd==1)].index
        elif rwdtype=='om':
            vinds = self.dat.loc[(self.dat.port!=-100)&(self.dat.rwd!=1)].index
        else:
            vinds = self.dat_visinds
        if 'nom_rwd_chosen' in self.pool_factor:
            low = self.dat.loc[(self.dat.port!=-100)&((self.dat.nom_rwd_chosen==20)|(self.dat.nom_rwd_chosen==10))].index
            mid = self.dat.loc[(self.dat.port!=-100)&(self.dat.nom_rwd_chosen==50)].index
            high = self.dat.loc[(self.dat.port!=-100)&((self.dat.nom_rwd_chosen==80)|(self.dat.nom_rwd_chosen==90))].index
            return low,mid,high
        if not self.poolByTerc:
            #get indices of absolute threshold groups
            poolmin = self.df.loc[self.visinds,self.pool_factor].min()
            thirds = (self.df.loc[self.visinds,self.pool_factor].max()-poolmin)/3
            thirds = [thirds+poolmin,thirds*2+poolmin,self.dat.loc[:,self.pool_factor].max()]
            low = self.dat.loc[(self.dat.port!=-100)&(self.dat[self.pool_factor]<=thirds[0])].index
            mid = self.dat.loc[(self.dat.port!=-100)&(self.dat[self.pool_factor]<=thirds[1])&
                (self.dat[self.pool_factor]>thirds[0])].index
            high = self.dat.loc[(self.dat.port!=-100)&(self.dat[self.pool_factor]>thirds[1])].index
            return low,mid,high
        approach_dat = []
        for i in range(len(vinds)):
            approach_dat.append([self.dat.loc[vinds[i],self.pool_factor],vinds[i]])
        approach_dat = np.array(approach_dat)
        sorteddat = approach_dat[np.argsort(approach_dat[:,0],axis=0)]
        sorted_tris = sorteddat[:,1]
        low = sorted_tris[:int(len(sorteddat)/3)].astype(int)
        mid = sorted_tris[int(len(sorteddat)/3):int(2*len(sorteddat)/3)].astype(int)
        high = sorted_tris[int(2*len(sorteddat)/3):].astype(int)
        return low,mid,high

    def getTracesByTerc(self):
        low,mid,high = self.getTriIndsByTerc()
        rwdhigh = []
        omhigh = []
        rwdmid = []
        ommid = []
        rwdlow = []
        omlow = []

        for i in low:
            tritrace = self.df.loc[i+self.fs*self.plot_window[0]:i+self.plot_window[1]\
            *self.fs,self.plot_trace].values
            if self.df.loc[i,'rwd'] == 1:
                rwdlow.append(tritrace)
            else:
                omlow.append(tritrace)
        for i in mid:
            tritrace = self.df.loc[i+self.fs*self.plot_window[0]:i+self.plot_window[1]\
            *self.fs,self.plot_trace].values
            if self.df.loc[i,'rwd'] == 1:
                rwdmid.append(tritrace)
            else:
                ommid.append(tritrace)
        for i in high:
            tritrace = self.df.loc[i+self.fs*self.plot_window[0]:i+self.plot_window[1]\
            *self.fs,self.plot_trace].values
            if self.df.loc[i,'rwd'] == 1:
                rwdhigh.append(tritrace)
            else:
                omhigh.append(tritrace)
        return rwdhigh,omhigh,rwdmid,ommid,rwdlow,omlow

    def getSessionTercMeans(self,secondHalf=False,useRat=True):
        rwdhigh_means = []
        omhigh_means = []
        rwdmid_means = []
        ommid_means = []
        rwdlow_means = []
        omlow_means = []
        groupLevel = "rat" if useRat else "session"
        for s in self.df.loc[:,groupLevel].unique():
            if secondHalf:
                self.dat = self.df.loc[(self.df.loc[:,groupLevel]==s)&(self.df.tri>25)]
            else:
                self.dat = self.df.loc[self.df.loc[:,groupLevel]==s]
            self.dat_visinds = self.dat.loc[self.dat.port!=-100].index
            rwdhigh,omhigh,rwdmid,ommid,rwdlow,omlow = self.getTracesByTerc()
            rs = [rwdhigh,omhigh,rwdmid,ommid,rwdlow,omlow]
            for rvec in range(len(rs)):
                if len(rs[rvec])<1:
                    rs[rvec] = np.full((10,(self.plot_window[1]-self.plot_window[0])*self.fs+1),np.nan)
                    print("rat didn't visit one port in session ",str(s),\
                        " or probability was not offered")
            rwdhigh,omhigh,rwdmid,ommid,rwdlow,omlow = rs
            rwdhigh_means.append(pd.Series(np.nanmean(rwdhigh,axis=0)).rolling(window=self.bin_size).mean())
            omhigh_means.append(pd.Series(np.nanmean(omhigh,axis=0)).rolling(window=self.bin_size).mean())
            rwdmid_means.append(pd.Series(np.nanmean(rwdmid,axis=0)).rolling(window=self.bin_size).mean())
            ommid_means.append(pd.Series(np.nanmean(ommid,axis=0)).rolling(window=self.bin_size).mean())
            rwdlow_means.append(pd.Series(np.nanmean(rwdlow,axis=0)).rolling(window=self.bin_size).mean())
            omlow_means.append(pd.Series(np.nanmean(omlow,axis=0)).rolling(window=self.bin_size).mean())
        return rwdhigh_means,omhigh_means,rwdmid_means,ommid_means,rwdlow_means,omlow_means
    
    def plotApproachByTercPooled(self):
        fig = plt.figure(figsize = (14,10))
        rwdhigh,omhigh,rwdmid,ommid,rwdlow,omlow = self.getSessionTercMeans()

        xvals = np.arange(self.fs*self.plot_window[0],self.fs*self.plot_window[1]+1)/self.fs
        plt.title(self.pool_factor)
        ax1 = plt.gca()
        ax1.plot(xvals[:-self.plot_window[0]*self.fs],np.nanmean\
            (np.vstack((rwdhigh,omhigh)),axis=0)[:-self.plot_window[0]*self.fs],color='dodgerblue')
        ax1.fill_between(xvals[:-self.plot_window[0]*self.fs],(np.nanmean\
                    (np.vstack((rwdhigh,omhigh)),axis=0)+sem(np.vstack((rwdhigh,omhigh))))\
                    [:-self.plot_window[0]*self.fs],(np.nanmean\
                    (np.vstack((rwdhigh,omhigh)),axis=0)-sem(np.vstack((rwdhigh,omhigh))))\
                    [:-self.plot_window[0]*self.fs],color='blue',alpha=.05)
        ax1.plot(xvals[-self.plot_window[0]*self.fs:],np.nanmean\
            (rwdhigh,axis=0)[-self.plot_window[0]*self.fs:],color='dodgerblue',label='high')
        ax1.fill_between(xvals[-self.plot_window[0]*self.fs:],(np.nanmean\
                    (rwdhigh,axis=0)+sem(rwdhigh))[-self.plot_window[0]*self.fs:],(np.nanmean\
                    (rwdhigh,axis=0)-sem(rwdhigh))[-self.plot_window[0]*self.fs:],color='blue',alpha=.05)
        ax1.plot(xvals[-self.plot_window[0]*self.fs:],np.nanmean\
            (omhigh,axis=0)[-self.plot_window[0]*self.fs:],color='dodgerblue',ls=':')
        ax1.set_xlabel('time (s) from port entry')
        ax1.set_ylabel('dF/F',fontsize='x-large',fontweight='bold')
        ax1.axvline(x=0.0,ymin=-.1,ymax=1.0,color='k',linestyle='--')
        ax1.plot(xvals[:-self.plot_window[0]*self.fs],np.nanmean\
            (np.vstack((rwdmid,ommid)),axis=0)[:-self.plot_window[0]*self.fs],color='dodgerblue',alpha=.5)
        ax1.fill_between(xvals[:-self.plot_window[0]*self.fs],(np.nanmean\
                    (np.vstack((rwdmid,ommid)),axis=0)+sem(np.vstack((rwdmid,ommid)),nan_policy="omit"))\
                    [:-self.plot_window[0]*self.fs],(np.nanmean\
                    (np.vstack((rwdmid,ommid)),axis=0)-sem(np.vstack((rwdmid,ommid)),nan_policy="omit"))\
                    [:-self.plot_window[0]*self.fs],color='blue',alpha=.05)
        ax1.plot(xvals[-self.plot_window[0]*self.fs:],np.nanmean\
            (rwdmid,axis=0)[-self.plot_window[0]*self.fs:],color='dodgerblue',label='medium',alpha=0.5)
        ax1.fill_between(xvals[-self.plot_window[0]*self.fs:],(np.nanmean\
                    (rwdmid,axis=0)+sem(rwdmid,nan_policy="omit"))[-self.plot_window[0]*self.fs:],(np.nanmean\
                    (rwdmid,axis=0)-sem(rwdmid,nan_policy="omit"))[-self.plot_window[0]*self.fs:],color='blue',alpha=.05)
        ax1.plot(xvals[-self.plot_window[0]*self.fs:],np.nanmean\
            (ommid,axis=0)[-self.plot_window[0]*self.fs:],color='dodgerblue',ls=':',alpha=0.5)
        ax1.axvline(x=0.0,ymin=-.1,ymax=1.0,color='k',linestyle='--')
        ax1.set_xlabel('time (s) from port entry')
        ax1.plot(xvals[:-self.plot_window[0]*self.fs],np.nanmean\
            (np.vstack((rwdlow,omlow)),axis=0)[:-self.plot_window[0]*self.fs],color='dodgerblue',alpha=0.15)
        ax1.fill_between(xvals[:-self.plot_window[0]*self.fs],(np.nanmean\
                    (np.vstack((rwdlow,omlow)),axis=0)+sem(np.vstack((rwdlow,omlow))))\
                    [:-self.plot_window[0]*self.fs],(np.nanmean\
                    (np.vstack((rwdlow,omlow)),axis=0)-sem(np.vstack((rwdlow,omlow))))\
                    [:-self.plot_window[0]*self.fs],color='blue',alpha=.3)
        ax1.plot(xvals[-self.plot_window[0]*self.fs:],np.nanmean\
            (rwdlow,axis=0)[-self.plot_window[0]*self.fs:],color='darkblue',label='low')
        ax1.fill_between(xvals[-self.plot_window[0]*self.fs:],(np.nanmean\
                    (rwdlow,axis=0)+sem(rwdlow))[-self.plot_window[0]*self.fs:],(np.nanmean\
                    (rwdlow,axis=0)-sem(rwdlow))[-self.plot_window[0]*self.fs:],color='blue',alpha=.05)
        ax1.plot(xvals[-self.plot_window[0]*self.fs:],np.nanmean\
            (omlow,axis=0)[-self.plot_window[0]*self.fs:],color='dodgerblue',alpha=0.15\
              ,ls=':')
        ax1.axvline(x=0.0,ymin=-.1,ymax=1.0,color='k',linestyle='--')
        ax1.set_xlabel('time (s) from port entry')
        ax1.legend()
        fig.savefig(self.directory_prefix+'portApproachDaBy'+self.pool_factor+'.png')

    def plotApproachByTercBySesh(self):
        rwdhighs,omhighs,rwdmids,ommids,rwdlows,omlows = self.getSessionTercMeans()
        seshs = self.df.session.unique()
        for s in range(len(seshs)):
            fig = plt.figure(figsize = (14,10))
            rwdhigh = rwdhighs[s]
            omhigh = omhighs[s]
            rwdmid = rwdmids[s]
            ommid = ommids[s]
            rwdlow = rwdlows[s]
            omlow = omlows[s]
            xvals = np.arange(self.fs*self.plot_window[0],self.fs*self.plot_window[1]+1)/self.fs
            plt.title(self.df.loc[self.df.session==seshs[s],'rat'].unique()[0]\
                +str(self.df.loc[self.df.session==seshs[s],'date'].unique()[0])+'\n'+self.pool_factor)
            ax1 = plt.gca()
            ax1.plot(xvals[:-self.plot_window[0]*self.fs],np.nanmean\
                (np.vstack((rwdhigh,omhigh)),axis=0)[:-self.plot_window[0]*self.fs],color='lightblue')
            ax1.plot(xvals[-self.plot_window[0]*self.fs:],rwdhigh\
                     [-self.plot_window[0]*self.fs:],color='lightblue',label='high')
            ax1.plot(xvals[-self.plot_window[0]*self.fs:],omhigh\
                     [-self.plot_window[0]*self.fs:],color='lightblue',ls=':')
            ax1.set_xlabel('time (s) from port entry')
            ax1.set_ylabel('dF/F',fontsize='x-large',fontweight='bold')
            ax1.axvline(x=0.0,ymin=-.1,ymax=1.0,color='k',linestyle='--')
            ax1.plot(xvals[:-self.plot_window[0]*self.fs],np.nanmean\
                (np.vstack((rwdmid,ommid)),axis=0)[:-self.plot_window[0]*self.fs],color='blue')
            ax1.plot(xvals[-self.plot_window[0]*self.fs:],rwdmid\
                [-self.plot_window[0]*self.fs:],color='blue',label='medium')
            ax1.plot(xvals[-self.plot_window[0]*self.fs:],ommid\
                     [-self.plot_window[0]*self.fs:],color='blue',ls=':')
            ax1.axvline(x=0.0,ymin=-.1,ymax=1.0,color='k',linestyle='--')
            ax1.set_xlabel('time (s) from port entry')
            ax1.plot(xvals[:-self.plot_window[0]*self.fs],np.nanmean\
                (np.vstack((rwdlow,omlow)),axis=0)[:-self.plot_window[0]*self.fs],color='darkblue')
            ax1.plot(xvals[-self.plot_window[0]*self.fs:],rwdlow\
                     [-self.plot_window[0]*self.fs:],color='darkblue',label='low')
            ax1.plot(xvals[-self.plot_window[0]*self.fs:],omlow\
                     [-self.plot_window[0]*self.fs:],color='darkblue'\
                  ,ls=':')
            ax1.axvline(x=0.0,ymin=-.1,ymax=1.0,color='k',linestyle='--')
            ax1.set_xlabel('time (s) from port entry')
            ax1.legend()
            fig.savefig(self.directory_prefix+'sesh_'+str(s)+'portApproachDaBy'+self.pool_factor+'.png')

    def set_regressionFeatures(self,featureList):
        self.regFeatures = featureList

    def plotSeshRweightsInTime(self,rweights):
        fig = plt.figure(figsize=(12,8))
        plt.title('Regression to dLight in Time from: \n'+str(self.dat.rat.unique())+\
            ", "+str(self.dat.fiberloc.unique())+\
            ', and '+str(len(self.dat.session.unique()))+"sessions"\
              ,fontsize='xx-large',fontweight='bold')
        toplt = rweights
        for r in range(len(self.regFeatures)):
            plt.plot(self.regLags/self.fs,toplt[r,:],label=self.regFeatures[r])
        plt.ylabel('Regression Weight',fontsize='large',fontweight='bold')
        plt.axhline(y=0, color="k", linestyle=":")
        plt.axvline(x=0, color="k", linestyle="--",alpha=.5)
        ax = plt.gca()
        plt.legend()
        return fig

    def calcRegWeightsInTimeFromPort(self):
        faclen = len(self.regFeatures)
        rweights = np.zeros((faclen,len(self.regLags)))
        for n in range(len(self.regLags)):
            y = self.dat.loc[self.dat_visinds+self.regLags[n],self.plot_trace]
            y = y.reset_index(drop=True)
            X = pd.DataFrame()
            for f in self.regFeatures:
                if self.reg_to_portval:
                    X[f] = self.dat.loc[self.dat_visinds,f].values
                else:
                    X[f] = self.dat.loc[self.dat_visinds+self.regLags[n],f].values
            X[self.regFeatures] = scale.fit_transform(X[self.regFeatures].values)
            X = X.drop(y.loc[y.isnull()].index,axis=0)
            y = y.drop(y.loc[y.isnull()].index,axis=0)
            try:
                y = y.drop(X.loc[X.isnull().values].index,axis=0)
                X = X.drop(X.loc[X.isnull().values].index,axis=0)
            except:
                None #print("no null values in feature space")
            modo = LR(fit_intercept=True,normalize=False).fit(X,y)
            rweights[:,n] = modo.coef_
        #fig = self.plotSeshRweightsInTime(rweights)
        #plt.xlabel('Time from port entry (s)',fontsize='x-large',fontweight='bold')
        #fig.savefig(self.directory_prefix+'s'+str(self.dat.session.unique()[0])+\
        #    'PortDaRegPlot.png')
        return rweights

    def plotRegWeightsInTimeFromPort(self):
        self.regLags = np.arange(self.fs*self.plot_window[0],self.fs*self.plot_window[1])
        rweights = []
        for s in self.df.session.unique():
            self.dat = self.df.loc[self.df.session==s]
            self.dat_visinds = self.dat.loc[self.dat.port!=-100].index
            rweights.append(self.calcRegWeightsInTimeFromPort())
        rweights = np.array(rweights)
        toplt = np.mean(rweights,axis=0)
        fig = plt.figure(figsize=(12,8))
        plt.title('Regression to dLight in Time. '+str(len(self.df.rat.unique()))+\
            ' animals, '+str(len(self.df.session.unique()))+' sessions.'\
              ,fontsize='xx-large',fontweight='bold')
        for r in range(len(self.regFeatures)):
            plt.plot(self.regLags/self.fs,toplt[r,:],label=self.regFeatures[r])
        plt.ylabel('Regression Weight',fontsize='large',fontweight='bold')
        plt.axhline(y=0, color="k", linestyle=":")
        plt.axvline(x=0, color="k", linestyle="--",alpha=.5)
        for i in range(len(self.regFeatures)):
            plt.fill_between(self.regLags/self.fs, toplt[i,:]-sem(rweights)[i,:],\
                toplt[i,:]+sem(rweights)[i,:],color="gray",alpha=0.2)
        plt.xlabel('Time from port entry (s)',fontsize='x-large',fontweight='bold')
        plt.legend()
        fig.savefig(self.directory_prefix+'portDaRegPlot.png')

    def create_hexDf(self):
        hexArray = [[] for _ in range(13)]
        Qs = np.array([0,0,0])
        nomRwds = np.array([0,0,0])
        lens = np.array([0,0,0])
        dstop = np.array([0])
        for sesh in self.df.session.unique():
            r = self.df.loc[self.df.session==sesh,'rat'].values[0]
            vinds = np.concatenate([[self.df.loc[(self.df.session==sesh)].index[0]],\
                self.df.loc[(self.df.session==sesh)&(self.df.port!=-100)].index])
            lastHex = None
            seshType =  self.df.loc[self.df.session==sesh,'session_type'].values[0]
            date =  self.df.loc[self.df.session==sesh,'date'].values[0]
            for t in range(1,len(vinds)):
                tdat = self.df.loc[vinds[t-1]+1:vinds[t]]
                tdat = tdat.loc[tdat.lastleave==2]
                if len(tdat) <2:
                    continue
                try:
                    tdat.loc[tdat.port!=-100,"port"].values[0]
                except:
                    continue
                b = tdat.block.unique()[0]
                triQs = tdat.loc[:,["Q_a","Q_b","Q_c"]].values[0]
                triNomRwds = tdat.loc[:,["nom_rwd_a","nom_rwd_b","nom_rwd_c"]].values[0]
                triPathLens = tdat.loc[:,["lenAB","lenBC","lenAC"]].values[0]
                dToP = tdat.loc[:,"dtop"].values[0]
                sesh = tdat.session.unique()[0]
                rwd = tdat.loc[tdat.port!=-100,"rwd"].values[0]
                tinds = tdat.loc[tdat.hexlabels.diff()!=0].index
                prwd = tdat.nom_rwd_chosen.unique()[0]
                hexes = tdat.loc[tinds,'hexlabels'].values
                if hexes[0] ==lastHex:#in [1,2,3]:
                    tinds = tinds[1:]
                    hexes = hexes[1:]
                if t == 1:
                    tinds = tinds[1:]
                    hexes = hexes[1:]
                lastHex = hexes[-1]
                #create new numerical column counting number of hex transitions
                tdat.loc[:,"newHex"] = -1
                cnt = 0
                for h in tinds:
                    tdat.loc[h,"newHex"] = cnt
                    cnt += 1 
                tdat.loc[:,"newHex"] = tdat.loc[:,"newHex"].replace(-1,method="ffill")
                dvals = tdat.loc[tdat.newHex!=-1,:].groupby("newHex").mean().green_z_scored.values
                velvals = tdat.loc[tdat.newHex!=-1,:].groupby("newHex").mean().vel.values
                accvals = tdat.loc[tdat.newHex!=-1,:].groupby("newHex").mean().acc.values
                hexArray[0] = hexArray[0] + list(hexes)
                hexArray[1] = hexArray[1] + list(np.full(len(hexes),-100,dtype="int8"))
                hexArray[2] = hexArray[2] + list(np.full(len(hexes),0,dtype="int8"))
                hexArray[3] = hexArray[3] + list(np.full(len(dvals),sesh,dtype="int8"))
                hexArray[4] = hexArray[4] + list(dvals)
                hexArray[5] = hexArray[5] + list(np.full(len(hexes),b,dtype="int8"))
                hexArray[6] = hexArray[6] + list(np.full(len(hexes),t-1,dtype="int16"))
                hexArray[7] = hexArray[7] + list(np.full(len(hexes),r))
                hexArray[8] = hexArray[8] + list(np.full(len(hexes),date))
                hexArray[9] = hexArray[9] + list(np.full(len(hexes),seshType))
                hexArray[10] = hexArray[10] + list(np.full(len(hexes),prwd))
                hexArray[11] = hexArray[11] + list(velvals)
                hexArray[12] = hexArray[12] + list(accvals)
                Qs = np.vstack([Qs,np.full((len(hexes),3),triQs)])
                nomRwds = np.vstack([nomRwds,np.full((len(hexes),3),triNomRwds)])
                lens = np.vstack([lens,np.full((len(hexes),3),triPathLens)])
                dstop = np.vstack([dstop,np.full((len(hexes),1),dToP)])
                hexArray[1][-1] = tdat.loc[tdat.port!=-100,"port"].values[0]
                hexArray[2][-1] = tdat.loc[tdat.port!=-100,"rwd"].values[0]
        hexArray = np.transpose(hexArray)
        hexDf = pd.DataFrame(hexArray,columns=["hexlabel","port","rwd","session","DA","block",
                               "trial","rat","date","session_type","nom_rwd_chosen","vel","acc"])
        hexDf.loc[:,"hexlabel"] = hexDf.loc[:,"hexlabel"].astype(float).astype("int8")
        hexDf.loc[:,"port"] = hexDf.loc[:,"port"].astype(float).astype("int8")
        hexDf.loc[:,"rwd"] = hexDf.loc[:,"rwd"].astype(float).astype("int8")
        hexDf.loc[:,"session"] = hexDf.loc[:,"session"].astype(float).astype("int8")
        hexDf.loc[:,"block"] = hexDf.loc[:,"block"].astype(float).astype("int8")
        hexDf.loc[:,"trial"] = hexDf.loc[:,"trial"].astype(float).astype("int16")
        hexDf.loc[:,"date"] = hexDf.loc[:,"date"].astype(float).astype(int)
        hexDf.loc[:,"DA"] = hexDf.loc[:,"DA"].astype(float)
        hexDf.loc[:,["Q_a","Q_b","Q_c"]] = Qs[1:]
        hexDf.loc[:,["lenAB","lenBC","lenAC"]] = lens[1:]
        hexDf.loc[:,["nom_rwd_a","nom_rwd_b","nom_rwd_c"]] = nomRwds[1:]
        hexDf.loc[:,"dtop"] = dstop[1:]
        print("converting hexlabels to hex + direction (pairedHexState)")
        hexDf.loc[:,"pairedHexState"] = self.convert2pairedState(hexDf.hexlabel.values)
        self.hexDf = hexDf  
