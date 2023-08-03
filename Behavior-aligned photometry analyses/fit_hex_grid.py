from shapely.geometry import Point, MultiPoint, MultiPolygon, LineString, MultiLineString
from shapely.geometry.polygon import Polygon
from shapely.affinity import rotate, translate
from shapely.affinity import scale as scaleshape
import shapely.vectorized
from math import atan
import numpy as np
from tri_gridworld import *
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
cmap3 = cm.get_cmap('viridis', 4)

from __main__ import datepath,savepath,vid

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
cents = {h: c for h,c in zip(hexlist,coords)}
centdf = pd.DataFrame(cents)
centdf = centdf.T

def plot_hex_outline(size='lg',bars=[],text=True):
	sz = 1000 if size=='lg' else 300
	bardf = centdf.drop(bars,axis=0)
	mapfig = plt.figure(figsize = (6.2,5.55))
	plt.scatter(bardf[0].values,bardf[1].values,marker='H',color='steelblue',alpha=0.9,s=sz)
	if text:
		for h in range(len(bardf)):
			plt.text(bardf.iloc[h][0]-.3,bardf.iloc[h][1]-.3,bardf.iloc[h].name,\
		    		color='whitesmoke',fontweight='demibold')
	plt.yticks([])
	plt.xticks([])
	return mapfig

emptp = np.zeros(50)
tprobs = np.identity(50)[1:]
tmat = np.array([[emptp,emptp,emptp,tprobs[3],emptp,emptp],[emptp,tprobs[48],emptp,emptp,emptp,emptp],\
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
		[tprobs[17],emptp,tprobs[25],emptp,tprobs[26],emptp],[emptp,emptp,emptp,tprobs[27],emptp,tprobs[18]],\
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
#above (tmat) is useful because contains a transition vector for each direction from each state
#tmatrix below contains a combined transition vector for each state, irrespective of direction
tmatrix = np.array([tprobs[3],tprobs[48],\
		tprobs[47],np.sum([tprobs[0],tprobs[4],tprobs[5]],axis=0),\
		np.sum([tprobs[6],tprobs[3]],axis=0),np.sum([tprobs[3],tprobs[7]],axis=0),\
		np.sum([tprobs[4],tprobs[8],tprobs[9]],axis=0),np.sum([tprobs[5],tprobs[9],tprobs[10]],axis=0),\
		np.sum([tprobs[11],tprobs[6]],axis=0),np.sum([tprobs[6],tprobs[12],tprobs[7]],axis=0),\
		np.sum([tprobs[7],tprobs[13]],axis=0),np.sum([tprobs[8],tprobs[14],tprobs[15]],axis=0),\
		np.sum([tprobs[9],tprobs[15],tprobs[16]],axis=0),np.sum([tprobs[10],tprobs[16],tprobs[17]],axis=0),\
		np.sum([tprobs[18],tprobs[11]],axis=0),np.sum([tprobs[11],tprobs[19],tprobs[12]],axis=0),\
		np.sum([tprobs[12],tprobs[20],tprobs[13]],axis=0),\
		np.sum([tprobs[13],tprobs[21]],axis=0),np.sum([tprobs[14],tprobs[22],tprobs[23]],axis=0),\
		np.sum([tprobs[15],tprobs[23],tprobs[24]],axis=0),np.sum([tprobs[16],tprobs[24],tprobs[25]],axis=0),\
		np.sum([tprobs[17],tprobs[25],tprobs[26]],axis=0),np.sum([tprobs[27],tprobs[18]],axis=0),\
		np.sum([tprobs[18],tprobs[28],tprobs[19]],axis=0),np.sum([tprobs[19],tprobs[29],tprobs[20]],axis=0),\
		np.sum([tprobs[20],tprobs[30],tprobs[21]],axis=0),np.sum([tprobs[21],tprobs[31]],axis=0),\
		np.sum([tprobs[22],tprobs[32],tprobs[33]],axis=0),np.sum([tprobs[23],tprobs[33],tprobs[34]],axis=0),\
		np.sum([tprobs[24],tprobs[34],tprobs[35]],axis=0),np.sum([tprobs[25],tprobs[35],tprobs[36]],axis=0),\
		np.sum([tprobs[26],tprobs[36],tprobs[37]],axis=0),np.sum([tprobs[47],tprobs[27]],axis=0),\
		np.sum([tprobs[27],tprobs[38],tprobs[28]],axis=0),np.sum([tprobs[28],tprobs[39],tprobs[29]],axis=0),\
		np.sum([tprobs[29],tprobs[40],tprobs[30]],axis=0),np.sum([tprobs[30],tprobs[41],tprobs[31]],axis=0),\
		np.sum([tprobs[31],tprobs[48]],axis=0),np.sum([tprobs[33],tprobs[42],tprobs[43]],axis=0),\
		np.sum([tprobs[34],tprobs[43],tprobs[44]],axis=0),np.sum([tprobs[35],tprobs[44],tprobs[45]],axis=0),\
		np.sum([tprobs[36],tprobs[45],tprobs[46]],axis=0),np.sum([tprobs[47],tprobs[38]],axis=0),\
		np.sum([tprobs[38],tprobs[39]],axis=0),np.sum([tprobs[39],tprobs[40]],axis=0),\
		np.sum([tprobs[40],tprobs[41]],axis=0),np.sum([tprobs[41],tprobs[48]],axis=0),\
		np.sum([tprobs[32],tprobs[2],tprobs[42]],axis=0),np.sum([tprobs[37],tprobs[46],tprobs[1]],axis=0)])
to_state = [[0,0,0,0,0,0]]+[np.argmax(a,axis=1) for a in tmat]
#to_state[22] = [18,  0, 26,  0, 28,  0]


class HexMap():
	def __init__(self,hexes,ids,points,sesh_type):
		self.hexes = hexes # a multipolygon object with all hexes
		self.ids = ids
		#self.point_1 = points[0] #point of first centroid in multipoint object "points"
		##numbers correspond to hex IDs
		#self.point_2 = points[1]
		#self.point_3 = points[2]
		#self.point_25 = points[3]
		#self.size_x = self.point_3.x-self.point_2.x
		#self.size_y = self.point_1.y-self.point_2.y
		#self.rotation = atan((self.point_25.y-self.point_3.y)/\
		#	(self.point_3.x-self.point_25.x))
		#self.bars = None
		self.sesh_type = sesh_type

	def new(self,new_points):
		'''save locations of landmarks on new maze, must be 
		previously marked and stored in MultiPoint object "new_points"'''
		self.new_1 = new_points[0]
		self.new_2 = new_points[1]
		self.new_3 = new_points[2]
		self.new_25 = new_points[3]
		self.new_size_x = self.new_3.x-self.new_2.x
		self.new_size_y = self.new_1.y-self.new_2.y
		self.new_rotation = atan((self.new_25.y-self.new_3.y)/\
			(self.new_3.x-self.new_25.x))

	def fit_to_new(self):
		offset_x = self.new_25.x - self.point_25.x
		offset_y = self.new_25.y - self.point_25.y
		scale_x = self.new_size_x/self.size_x
		scale_y = self.new_size_y/self.size_y
		angle_offset = self.rotation - self.new_rotation
		out = rotate(self.hexes,angle_offset,origin=self.point_25,\
			use_radians=True)
		out = translate(out,offset_x,offset_y)
		out = scaleshape(out,scale_x,scale_y,origin=self.point_25)
		self.new_hexes = out

	def define_barriers(self,bars):
		self.bars = [Polygon(b) for b in bars]

	def find_barriers(self):
		try:
			self.new_hexes
		except:
			print('new_hexes not yet defined')
			return None
		xvec = [b.centroid.x for b in self.bars]
		yvec = [b.centroid.y for b in self.bars]
		bar_inds = []
		bar_ids = []
		for h in range(len(self.new_hexes)):
			hasbar = shapely.vectorized.contains(self.new_hexes[h],xvec,yvec)
			if np.any(hasbar):
				bar_inds.append(h)
				bar_ids.append(self.ids[h])
		self.bar_inds = bar_inds
		self.bar_ids = bar_ids
		#plot and save hex layout wit barriers


	def plot_sesh_tmat(self,b='all'):
		'''Plot the transition matrix for the session, includes barrier
		info (zero transition probability).'''
		transition_matrix = np.zeros((50,50))
		for s in range(1,len(to_state)):
			if s in self.bar_ids: #skip if barrier
				continue
			next_states = to_state[s][np.where(to_state[s]>0)]
			#get rid of barriers
			if any(np.isin(next_states,self.bar_ids)):
				b_ind = np.where(np.isin(next_states,self.bar_ids))
				next_states = np.delete(next_states,b_ind)
			transition_matrix[s,next_states]=1/len(next_states)
		tfig = plt.figure(figsize=(13/2,5))
		cbar = plt.pcolormesh(transition_matrix,cmap=cmap3)
		plt.suptitle('Transition Matrix',fontsize = 'xx-large',fontweight='bold')
		plt.xlabel('Hex State ID',fontsize = 'x-large',fontweight='bold')
		plt.ylabel('Hex State ID',fontsize = 'x-large',fontweight='bold')
		cbar = plt.colorbar(ticks=[0,0.33,0.66,1])
		cbar.ax.get_yaxis().labelpad = 15
		cbar.ax.set_ylabel('p(transition)', rotation=270,fontweight='bold')
		cbar.ax.set_yticklabels(['0%', '33%','50%','100%'])
		if self.sesh_type=='barrier':
			tfig.savefig(datepath+"/transition_matrix_"+str(b)+".pdf")
		else:
			tfig.savefig(datepath+"/transition_matrix.pdf")
		return transition_matrix

	def fit_to_new_input(self,offset_x=-7,offset_y=-18,scale_x=.96,scale_y=.96,angle_offset=.092):
		out = rotate(self.hexes,angle_offset,origin=self.point_25,\
			use_radians=True)
		out = translate(out,offset_x,offset_y)
		out = scaleshape(out,scale_x,scale_y,origin=self.point_25)
		self.new_hexes = out
		fig,ax = plt.subplots()
		ax.imshow(self.vid.get_data(1000))
		for H in self.new_hexes:
		    plt.plot(*H.exterior.xy)
		plt.title('new hex map')

	def set_vid(self,vid):
		self.vid=vid

def plot_baseline_tmat():
	plt.figure()
	plt.pcolormesh(tmatrix,cmap='copper')
	plt.suptitle('Transition Matrix',fontsize = 'xx-large',fontweight='bold')
	plt.xlabel('Hex State ID',fontsize = 'x-large',fontweight='bold')
	plt.ylabel('Hex State ID',fontsize = 'x-large',fontweight='bold')

def define_crit_points(drawnlines):
    '''Return point collection object for each drawn hexbin in drawnlines'''
    #add all polygon objects to multipolygon object called polypoints
    polys = [Polygon(h) for h in drawnlines]
    points =[h.centroid for h in polys]
    polypoints = MultiPoint(points)
    return polypoints

def is_adjacent(h1,h2):
	cutoff = 5
	return abs(h1.distance(h2))<cutoff

def get_adjacent(h,polybins,hexids,barids):
	'''get distance between hex and all other hexes'''
	cutoff = 5
	dists = [abs(polybins[h].distance(p)) for p in polybins]
	adjacents = hexids[np.where(np.array(dists)<cutoff)[0]]
	adjacents = [a for a in adjacents if a not in barids]
	try:
		adjacents.remove(hexids[h])
	except:
		return adjacents
	return adjacents

def is_branch(h,polybins,hexids,barids):
	'''return whether hex is a branch point or not.
	here h is the index for a polygon object for the hex'''
	#SHOULD RE-WRITE TO IDENTIFY WHETHER CORRESPONDING TRANSITION MATRIX HAS
	#A MAX VALUE OF 1/3 IN THAT HEX'S ROW
	branch = len(get_adjacent(h,polybins,hexids,barids))==3 #and h not in [1,2,3]
	return branch

def sharedhex(x,y):
    comlist =  [v for v in to_state[x] if v in to_state[y] and v !=0]
    if len(comlist)>0:
        return comlist[0]
    else:
        return np.nan

def fill_gap(h1,h3):
	if h3 not in to_state[h1]:
		return sharedhex(h1,h3)

def add_to_seg(h,polybins,hexids,barids,seg = [],before=False):
	'''return list of adjacent hex ids that don't incluce barriers
	or branch points'''
	if hexids[h] in seg or is_branch(h,polybins,hexids,barids):
		return seg
	if hexids[h] in barids:
		return seg
	if before==False:
		seg.append(hexids[h])
	else:
		seg = [hexids[h]] + seg #append before
	adjs = get_adjacent(h,polybins,hexids,barids)
	if len(adjs)<1:
		return seg
	for a in range(len(adjs)):
		bfore = False if a==0 else True
		seg = add_to_seg(int(np.where(hexids==adjs[a])[0]),polybins,hexids,barids,seg,before=bfore)
	return seg

def get_nearest_hex(point,centroids):
	'''return hexID whose centroid is the shortest distance from point'''
	dists = []
	for c in centroids:
		dists.append(point.distance(Point(c[0])))
	if centroids[np.argmin(dists)][0] < 20:
		return centroids[np.argmin(dists)][1]
	else:
		return np.nan

def get_seg_ids(polybins,hexids,barids):
	'''Return dictionary of hexids in each segment. First
	need to create list of lists of adjacent hexes without
	branch points. These are path segments.'''
	segments = []
	branches = []
	for h in range(len(polybins)):
		if len(segments)>0 and hexids[h] in np.concatenate(segments):
			continue
		if is_branch(h,polybins,hexids,barids):
			branches.append(hexids[h])
			continue
		seg = add_to_seg(h,polybins,hexids,barids,seg=[])
		if len(seg)>0:
			segments.append(seg)
	segments = {s:l for s,l in zip(np.arange(0,len(segments)),segments)}
	return segments,branches

def define_shortest_paths(drawnlines):
	'''Return MultiLineString object of shortest paths between
	any two ports. Must be in order AC,BC,AB'''
	paths = [LineString(s) for s in drawnlines]
	multisegs = MultiLineString(segs)
	return multisegs

def hexes_in_path(polybins,multipaths,hexids):
	'''return dictionary of hexids in each segment'''
	path_hexes = {s:[] for s in np.arange(0,len(multipaths))}
	for s in range(len(multipaths)):
		#use intersects method to find which hexes are intersected by each segment
		for h in range(len(polybins)):
			if polybins[h].intersects(multipaths[s]):
				path_hexes[s].append(hexids[h])
	return pathhexes

#def path_error(x,y,):


#def order_segments(segs):
#	'''Return segment list where order of hexIDs within a segment
#	corresponds to order in maze.'''


#def add_to_path(h,p1,p2,polybins,hexids,barids,seg=[],before=False):
#	if hexids[h] in seg or hexids[h] == p1 or hexids[h] == p2:



#def get_shortest_path_lengths(p1,p2):
	'''Return shortest number of hexes between p1 and p2 (ports)'''

# Figure out cutoff distance, whether to use np.where()[0] or np.where





#def define_branch_point(drawnlines):
#	'''Return multipolygon object of all branch points'''
#    return define_crit_points(drawnlines)
#
#def define_path_segments(drawnlines):
#	'''Return MultiLineString object of path segments'''
#	segs = [LineString(s) for s in drawnlines]
#	multisegs = MultiLineString(segs)
#	return multisegs
#
#def hexes_in_seg(polybins,multisegs,hexids):
#	'''return dictionary of hexids in each segment'''
#	seghexes = {s:[] for s in np.arange(0,len(multisegs))}
#	for s in range(len(multisegs)):
#		#use intersects method to find which hexes are intersected by each segment
#		for h in range(len(polybins)):
#			if polybins[h].intersects(multisegs[s]):
#				seghexes[s].append(hexids[h])
#	return seghexes
#
#def point_to_path_seg(hexids,seghexes):
#	'''given hexids and dictionary of hexes that belong to each segment,
#	return array of seg IDs'''
#	return [list(seghexes.keys())[list(seghexes.values()).index(h)] for h in hexids]
#
#
#def mark_branch_point(branch_points,polybins,hexids):
#	'''identify which hex IDs belong to each branch point object.
#	return a dictionary of {branch point: hexid in branchpoint}'''
#	branchids = {b: None for b in np.arange(0,len(branch_points))}
#	for b in range(len(branch_points)):
#		for h in range(len(polybins))
#			if polybins[h].contains(branch_points[b]):
#				branchids[b] = hexids[h]
#	return branchids
#
##def d_to_branch(segids,points):
##	'''project point onto corresponding segment. plot distance
##	along segment. not sure how to establish direction of travel yet'''
#
#
#