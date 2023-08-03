#Create gridworld object for triangle maze
import numpy as np

#Create gridworld object for triangle maze
class TriGrid():
    
    def __init__(self,barriers,dirstates=False):
        emptp = np.zeros(50)
        tprobs = np.identity(50)[1:]
        hexlist = [2,47,46,45,44,43,3,\
        49,42,41,40,39,48,\
        38,37,36,35,34,33,\
        32,31,30,29,28,\
        27,26,25,24,23,\
        22,21,20,19,\
        18,17,16,15,\
        14,13,12,\
        11,10,9,\
        8,7,\
        6,5,\
        4,\
        1]
        #hexlist = np.subtract(hexlist,1) #convert to index-based states
        coords = []
        cols = [7,6,6,5,5,4,4,3,3,2,2,1,1]
        maxrows = 13
        r = 0
        x = 1
        y = 1
        startr = 1
        while r < maxrows:
            maxcols = cols[r]
            c = 0
            if r%2!=0:
                startr+=1
            x=startr
            while c < maxcols:
                coords.append([x,y])
                x += 2
                c += 1
            if r%2!=0:
                y += 2
            else:
                y+=1
            r += 1
        self.state_centroids = {h: c for h,c in zip(hexlist,coords)}
        self.tmat = np.array([[emptp,emptp,emptp,tprobs[3],emptp,emptp],[emptp,tprobs[48],emptp,emptp,emptp,emptp],\
             [emptp,emptp,emptp,emptp,emptp,tprobs[47]],[tprobs[0],emptp,tprobs[4],emptp,tprobs[5],emptp],\
             [emptp,emptp,emptp,tprobs[6],emptp,tprobs[3]],[emptp,tprobs[3],emptp,tprobs[7],emptp,emptp],\
             [tprobs[4],emptp,tprobs[8],emptp,tprobs[9],emptp],[tprobs[5],emptp,tprobs[9],emptp,tprobs[10],emptp],\
             [emptp,emptp,emptp,tprobs[11],emptp,tprobs[6]],[emptp,tprobs[6],emptp,tprobs[12],emptp,tprobs[7]],\
             [emptp,tprobs[7],emptp,tprobs[13],emptp,emptp],[tprobs[8],emptp,tprobs[14],emptp,tprobs[15],emptp],\
             [tprobs[9],emptp,tprobs[15],emptp,tprobs[16],emptp],[tprobs[10],emptp,tprobs[16],emptp,tprobs[17],emptp],\
             [emptp,emptp,emptp,tprobs[18],emptp,tprobs[11]],\
             [emptp,tprobs[11],emptp,tprobs[19],emptp,tprobs[12]],[emptp,tprobs[12],emptp,tprobs[20],emptp,tprobs[13]],\
             [emptp,tprobs[13],emptp,tprobs[21],emptp,emptp],[tprobs[14],emptp,tprobs[22],emptp,tprobs[23],emptp],\
             [tprobs[15],emptp,tprobs[23],emptp,tprobs[24],emptp],[tprobs[16],emptp,tprobs[24],emptp,tprobs[25],emptp],\
             [tprobs[17],emptp,tprobs[25],emptp,tprobs[27],emptp],[emptp,emptp,emptp,tprobs[27],emptp,tprobs[18]],\
             [emptp,tprobs[18],emptp,tprobs[28],emptp,tprobs[19]],[emptp,tprobs[19],emptp,tprobs[29],emptp,tprobs[20]],\
             [emptp,tprobs[20],emptp,tprobs[30],emptp,tprobs[21]],[emptp,tprobs[21],emptp,tprobs[31],emptp,emptp],\
             [tprobs[22],emptp,tprobs[32],emptp,tprobs[33],emptp],[tprobs[23],emptp,tprobs[33],emptp,tprobs[34],emptp],\
             [tprobs[24],emptp,tprobs[34],emptp,tprobs[35],emptp],[tprobs[25],emptp,tprobs[35],emptp,tprobs[36],emptp],\
             [tprobs[26],emptp,tprobs[36],emptp,tprobs[37],emptp],[emptp,emptp,emptp,tprobs[47],emptp,tprobs[27]],\
             [emptp,tprobs[27],emptp,tprobs[38],emptp,tprobs[28]],[emptp,tprobs[28],emptp,tprobs[39],emptp,tprobs[29]],\
             [emptp,tprobs[29],emptp,tprobs[40],emptp,tprobs[30]],[emptp,tprobs[30],emptp,tprobs[41],emptp,tprobs[31]],\
             [emptp,tprobs[31],emptp,tprobs[48],emptp,emptp],[tprobs[33],emptp,tprobs[42],emptp,tprobs[43],emptp],\
             [tprobs[34],emptp,tprobs[43],emptp,tprobs[44],emptp],[tprobs[35],emptp,tprobs[44],emptp,tprobs[45],emptp],\
             [tprobs[36],emptp,tprobs[45],emptp,tprobs[46],emptp],[emptp,tprobs[47],emptp,emptp,emptp,tprobs[38]],\
             [emptp,tprobs[38],emptp,emptp,emptp,tprobs[39]],[emptp,tprobs[39],emptp,emptp,emptp,tprobs[40]],\
             [emptp,tprobs[40],emptp,emptp,emptp,tprobs[41]],[emptp,tprobs[41],emptp,emptp,emptp,tprobs[48]],\
             [tprobs[32],emptp,tprobs[2],emptp,tprobs[42],emptp],[tprobs[37],emptp,tprobs[46],emptp,tprobs[1],emptp]])
        self.to_state = [np.argmax(a,axis=1) for a in self.tmat] # 1-indexed
        self.barriers = barriers #must be 1-indexed
        self.avail_hexes = np.setdiff1d(np.arange(1,50), self.barriers)
        avail_centers = [self.state_centroids[h] for h in self.avail_hexes]
        self.state_centroids = {s:c for s,c in zip(self.avail_hexes,avail_centers)}
        #set transition probability to zero for states where barriers present
        for h in range(len(self.to_state)):
            inds = np.where(np.isin(self.to_state[h],self.barriers))[0]
            self.to_state[h][inds] = 0
            self.tmat[h][inds] = emptp
            inds0 = np.where(self.to_state[h]==0)[0]
            self.to_state[h][inds0] = 100000
        if not dirstates: #whether states incorporate direction + hexID
            self.states = self.avail_hexes
            #still allocentric, directions are map directions to keep code simple
        else:
            self.states = self.avail_hexes
            for d in range(1,3):
                self.states = np.concatenate((self.states,self.states+(49*d)))
                self.to_state = np.concatenate((self.to_state,np.add(self.to_state,(49*d))))
                for s in self.avail_hexes:
                    self.state_centroids[s+49*d] = self.state_centroids[s]
        for h in range(len(self.to_state)):
            inds0 = np.where(self.to_state[h]>=100000)[0]
            self.to_state[h][inds0] = 0
        self.aclist = [np.where(a)[0] for a in np.any(self.tmat,axis=2)] #list of all actions
        self.actions = {s:a for s,a in zip(self.states,self.aclist)}
        self.states = np.concatenate((self.states,[1000,1001,1002]))#append port states
        #generate list of actions from each state

    def set_rewards(self,portvals):
        '''portvals a dictionary with the value for each state.
        state:reward, etc. rewards will likely be zero for every state
        other than reward ports, which should be their value that trial.'''
        self.rewards = portvals

    def get_action(self,s,sprime):
        try:
            action = np.where(to_state[s]==sprime)[0][0]
        except:
            action = None
        return action

    def set_reward_outcome(self,r):
        '''r should be boolean indicating whether trial was rewarded or not
        at termination'''
        self.rwd_tri = r

    def set_state(self,s):
        self.state = s

    def current_state(self):
        return self.state

    def is_terminal(self,s):
        return s in [1000,1001,1002] #port states not in actions

    def move(self,action):
        if action in self.actions[self.state]:
            self.state = self.to_state[self.state][action]
            return self.rewards.get(self.state,0)

    def trial_over(self):
        return self.state not in self.actions

    def display_map(self,V,etrace=None):
        '''display plot of triangle maze hexes color coded by value.
        if len states > 49 create three maps, one for each hex state
        type. if etrace is not None type, display second row with eligibility trace
        for each state.'''
        plt.figure()
        port_coords = np.array([[28,15],[0,0],[16,0]])
        if len(self.states)>49: #check which state def used
            nhexes = len(self.avail_hexes)
            totr = 1
            totc = 3
            if etrace != None:
                totc = 4
                totr = 2
                for p in range(4,7):
                    ax2 = plt.subplot(totr,totc,p)
                    xcoords = list(self.state_centroids.values())[nhexes*(p-1):nhexes*p][0]
                    ycoords = list(self.state_centroids.values())[nhexes*(p-1):nhexes*p][1]
                    ax2 = plt.subplot(1,tot,p)
                    ax2.set_title('Eligibility')
                    im2 = ax2.scatter(xcoords,ycoords,c=np.array(list(etrace.values()))[:-3]\
                        [nhexes*(p-1):nhexes*p],\
                        marker='H',s=500,cmap='magma')
                    ax2.scatter(port_coords[:,0],port_coords[:,1],np.array(list(etrace.values()))[-3:],\
                        marker='H',s=500,cmap='magma')
            tot = 4 if etrace != None else 3
            for p in range(1,4):
                xcoords = list(self.state_centroids.values())[nhexes*(p-1):nhexes*p][0]
                ycoords = list(self.state_centroids.values())[nhexes*(p-1):nhexes*p][1]
                ax1 = plt.subplot(totr,totc,p)
                ax1.set_title('V(s)')
                im1 = ax1.scatter(xcoords,ycoords,c=np.array(list(V.values()))[:-3]\
                    [nhexes*(p-1):nhexes*p],\
                    marker='H',s=500,cmap='magma')
                ax1.scatter(port_coords[:,0],port_coords[:,1],np.array(list(V.values()))[-3:],\
                        marker='H',s=500,cmap='magma')
            fig.colorbar(im1, ax=ax1)
        else:
            ax1 = plt.gca()
            if etrace != None:
                ax2 = plt.subplot(122)
                ax1 = plt.subplot(121)
                ax2.set_title('Eligibility')
                im2 = ax2.scatter(xcoords,ycoords,c=etrace.values(),marker='H',\
                    s=500,cmap='magma')
                ax2.scatter(port_coords[:,0],port_coords[:,1],np.array(list(etrace.values()))[-3:],\
                        marker='H',s=500,cmap='magma')
                fig.colorbar(im2, ax=ax2)
            xcoords = list(self.state_centroids.values())[:][0]
            ycoords = list(self.state_centroids.values())[:][1]
            ax1 = plt.subplot(121)
            ax1.set_title('V(s)')
            im1 = ax1.scatter(xcoords,ycoords,c=np.array(list(V.values()))[:-3],marker='H',s=500,cmap='magma')
            ax1.scatter(port_coords[:,0],port_coords[:,1],np.array(list(V.values()))[-3:],\
                        marker='H',s=500,cmap='magma')
            fig.colorbar(im1, ax=ax1)

def standard_triangle(hexids):
    '''hexids is list of all hexes visited in static barrier session.
    Must be int.'''
    #assert isinstance(hexids[0], int), "hexids is not a list of integers!!"
    allstates = np.arange(1,50)
    availstates = np.sort(hexids)
    bars = np.setdiff1d(allstates, availstates)
    #bars = np.subtract(bars,1)
    maze = TriGrid(bars,dirstates=False)
    return maze

def port_hist_triangle(hexids):
    '''hexids is list of all hexes visited in static barrier session.
    Must be int.'''
    #assert isinstance(hexids[0], int), "hexids is not a list of integers!!"
    allstates = np.arange(1,50)
    availstates = np.sort(hexids)
    bars = np.setdiff1d(allstates, availstates)
    #bars = np.subtract(bars,1)
    maze = TriGrid(bars,dirstates=True)
    return maze