#!/usr/bin/env python3

"""Functions for initial processing and plotting of photometry
signals in maze. Used in 'Photometry Analysis Pipeline-Maze.ipynb' """

__author__ = "Tim Krausz"
__email__ = "krausz.tim@gmail.com"
__status__ = "development"

from __main__ import *
import csv
from scipy.ndimage import gaussian_filter

def processAndPlotSigsWairPLS(sigs,datepath,Fs=250, smooth_win = int(250/30),lambd = 1e8):

    #define reference and signal traces for analysis
    raw_reference = sigs['ref']
    raw_green = sigs['green']
    
    #smooth signal and plot
    reference = np.array(raw_reference.rolling(window=smooth_win,min_periods=1).mean()).reshape(len(raw_reference),1)
    signal = np.array(raw_green.rolling(window=smooth_win,min_periods=1).mean()).reshape(len(raw_green),1)
    xvals = np.arange(0,len(signal))/Fs/60
    
    #find baseline for sig and ref
    porder = 1
    itermax = 50
    r_base=airPLS(raw_reference.T,lambda_=lambd,porder=porder,itermax=itermax).reshape(len(raw_reference),1)
    s_base=airPLS(raw_green.T,lambda_=lambd,porder=porder,itermax=itermax).reshape(len(raw_green),1)
    grnb = plt.figure(figsize=(16, 10))
    plt.suptitle('signal and ref with baseline overlaid')
    ax1 = grnb.add_subplot(211)
    ax1.plot(xvals,signal,color='blue',lw=1.5,label='smoothed 470')
    ax1.plot(xvals,s_base,color='black',lw=1.5,label='estimated baseline')
    ax1.legend()
    ax2 = grnb.add_subplot(212, sharex=ax1)
    ax2.plot(xvals,reference,color='purple',lw=1.5,label='smoothed reference')
    ax2.plot(xvals,r_base,color='black',lw=1.5,label='estimated baseline')
    ax2.set_xlabel('time (min)')
    ax2.legend()
    grnb.savefig(datepath+"green_w_baseline.pdf")
    
    #subtract moving baseline
    remove = 0
    reference = (reference[remove:] - r_base[remove:])
    signal = (signal[remove:] - s_base[remove:])
    
    #standardize signals and plot
    z_reference = (reference - np.median(reference)) / np.std(reference)
    z_signal = (signal - np.median(signal)) / np.std(signal)
    xvals = np.arange(0,len(z_signal))/Fs/60
    zsub = plt.figure(figsize=(16, 10))
    plt.suptitle('standardized, baseline subtracted, signal')
    ax1 = zsub.add_subplot(211)
    ax1.plot(xvals,z_signal,color='blue',lw=1.5)
    ax1.set_ylabel('z-scored signal')
    ax2 = zsub.add_subplot(212,sharex=ax1)
    ax2.plot(xvals,z_reference,color='purple',lw=1.5)
    ax2.set_ylabel('z-scored reference')
    ax2.set_xlabel('time (min)')
    zsub.savefig(datepath+"z_scored_grn_bline_sub.pdf")
    #calculate and plot dff
    lin = Lasso(alpha=0.0001,precompute=True,max_iter=1000,
                positive=True, random_state=9999, selection='random')
    lin.fit(z_reference, z_signal)
    z_reference_fitted = lin.predict(z_reference).reshape(len(z_reference),1)
    zdFF = (z_signal - z_reference_fitted)
    xvals = np.arange(0,len(zdFF))/Fs/60
    rvs = plt.figure(figsize=(16, 8))
    plt.suptitle('linear fit')
    ax1 = rvs.add_subplot(111)
    ax1.plot(z_reference,z_signal,'b.')
    ax1.plot(z_reference,z_reference_fitted, 'r--',linewidth=1.5,label='fit line')
    ax1.legend()
    ax1.set_xlabel('ref values')
    ax1.set_ylabel('signal values')
    rvs.savefig(datepath+"grn_linear_fit.jpeg")
    gdf = plt.figure(figsize=(16, 8))
    plt.suptitle('dff calculation')
    ax1 = gdf.add_subplot(211)
    ax1.plot(xvals,z_signal,color='blue',label='signal')
    ax1.plot(xvals,z_reference_fitted,color='purple',label='fitted ref')
    ax1.set_ylabel('z-scored values')
    ax2 = gdf.add_subplot(212,sharex = ax1)
    ax2.plot(xvals,zdFF,color='darkgreen')
    ax2.set_ylabel('z-scored dFF')
    ax2.set_xlabel('time (min)')
    gdf.savefig(datepath+"green_dff.pdf")
    ref_fitted1 = z_reference_fitted
    
    return zdFF,z_reference
    
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def dtg(x1,y1,xgoal,ygoal):
    '''Returns absolute distance to goal coordinates'''
    dist = np.sqrt((xgoal - x1)**2 + (ygoal - y1)**2)
    return dist
vecdtg = np.vectorize(dtg)

def get_centroid(posArray1,posArray2):
    return np.array([(posArray1[:,0]+posArray2[:,0])/2,(posArray1[:,1]+posArray2[:,1])/2]).T

def get_headAngle(posArray1,posArray2):
    dy = posArray2[:,1]-posArray1[:,1]
    dx = posArray2[:,0]-posArray1[:,0]
    return np.arctan2(dy,dx)

def get_port_times(ardtext,ardtimes):
    port_strs = ["delivered at port A","delivered at port B","delivered at port C",\
                'o Reward port A','o Reward port B','o Reward port C',\
                "rwd delivered, "]
    port_times = []
    for i in range(len(ardtext)):
        if any(s in ardtext[i] for s in port_strs):
            port_times.append(ardtimes[i])
    return np.array(port_times).astype(int)

def calculate_velocity(x, y, fps, unit_conversion=1):
    # Convert pixels to cm if required
    x = x * unit_conversion
    y = y * unit_conversion
    
    # Calculate distance
    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.sqrt(dx**2 + dy**2)

    # Calculate time
    time = 1/fps
    t = np.arange(0, len(x) - 1) * time

    # Calculate velocity
    vel = dist / time

    return vel

def calculate_acceleration(x, y, fps, pixel_to_cm=1):
    # convert pixel to cm
    x_cm = x * pixel_to_cm
    y_cm = y * pixel_to_cm

    # calculate velocity
    velocity_x = np.gradient(x_cm) * fps
    velocity_y = np.gradient(y_cm) * fps
    velocity = np.sqrt(velocity_x ** 2 + velocity_y ** 2)

    # calculate acceleration
    acceleration_x = np.gradient(velocity_x) * fps
    acceleration_y = np.gradient(velocity_y) * fps
    acceleration = np.sqrt(acceleration_x ** 2 + acceleration_y ** 2)

    return acceleration

def remove_aberrant_jumps(position_data, max_jump_distance):
    # Compute the Euclidean distance between each consecutive pair of points
    pairwise_distance = np.sqrt(np.sum(np.diff(position_data, axis=0)**2, axis=1))

    # Identify aberrant jumps as those that exceed the specified max_jump_distance (PIXELS)
    aberrant_jumps = pairwise_distance > max_jump_distance

    # Remove aberrant jumps by setting the corresponding coordinates to NaN
    position_data[1:][aberrant_jumps] = np.nan

    # Interpolate missing values
    position_data = fill_missing_gaps(position_data)

    return position_data

def fill_missing_gaps(position_data):
    # Identify missing values as NaNs
    missing_values = np.isnan(position_data[:, 0]) | np.isnan(position_data[:, 1])

    # Compute the cumulative sum of missing values to identify contiguous gaps
    cumulative_sum = np.cumsum(missing_values)
    gap_starts = np.where(np.diff(cumulative_sum) == 1)[0] + 1
    gap_ends = np.where(np.diff(cumulative_sum) == -1)[0]

    # Interpolate the missing values in each gap using linear interpolation
    for gap_start, gap_end in zip(gap_starts, gap_ends):
        if gap_start == 0 or gap_end == len(position_data) - 1:
            continue  # ignore gaps at the beginning or end of the data
        else:
            x = position_data[gap_start - 1:gap_end + 1, 0]
            y = position_data[gap_start - 1:gap_end + 1, 1]
            interp_func = interp1d(x, y, kind='linear')
            position_data[gap_start:gap_end, 0] = np.linspace(x[0], x[-1], gap_end - gap_start + 1)
            position_data[gap_start:gap_end, 1] = interp_func(position_data[gap_start:gap_end, 0])

    return position_data

def smooth_position_data(position_data, window_size=5):
    """
    Smooth position x y coordinate data taken from a 640 by 480 pixel video using a simple moving average filter.
    
    Parameters:
        x (numpy.ndarray): Position x coordinate data.
        y (numpy.ndarray): Position y coordinate data.
        window_size (int): Window size for the moving average filter.
        
    Returns:
        numpy.ndarray: Smoothed position x coordinate data.
        numpy.ndarray: Smoothed position y coordinate data.
    """
    # Pad the x and y data to handle edge cases
    x = position_data[:,0]
    y = position_data[:,1]
    x_pad = np.pad(x, (window_size//2, window_size//2), mode='edge')
    y_pad = np.pad(y, (window_size//2, window_size//2), mode='edge')
    
    # Create the convolution kernel for the moving average filter
    kernel = np.ones(window_size) / window_size
    
    # Convolve the x and y data with the kernel
    x_smoothed = np.convolve(x_pad, kernel, mode='valid')
    y_smoothed = np.convolve(y_pad, kernel, mode='valid')
    
    return np.array([x_smoothed, y_smoothed]).T

def align_pos_to_visits(Fs, visits, datepath, phot_dlc='n',
                        filecount="0", gaus_smooth=True, 
                        sigma=0.01, cutoff=0.9, use_centroid=False,
                        cam_fps=15,pixelsPerCm=3.14):
    # Import arduino behavioral data and their timestamps
    ardtext = open(datepath + f'arduinoraw{filecount}.txt', 'r').read().splitlines()
    with open(datepath + f'ArduinoStamps{filecount}.csv', 'r') as at:
        ardtimes = np.array(list(csv.reader(at)), dtype=float).ravel()
    photStart = ardtimes[1] # time when pulse was sent to R series. 0 index for sig data
    ardstamps = np.round((ardtimes - photStart) * (Fs / 1000)).astype(int) # convert ardtimes to sample number to match photometry data.

    # Load position data
    if phot_dlc == 'y':
        dlc_pos_file = 'Behav_Vid0DLC_resnet50_Triangle_Maze_Phot_RobustJun8shuffle1_350000.h5'
        pos_col = 'cap'
        dlc_pos = pd.read_hdf(datepath + dlc_pos_file).DLC_resnet50_Triangle_Maze_Phot_RobustJun8shuffle1_350000
    else:
        dlc_pos_file = 'Behav_Vid0DLC_resnet50_Triangle_Maze_EphysDec7shuffle1_800000.h5'
        pos_col = 'cap_front' if use_centroid else 'cap_back'
        dlc_pos = pd.read_hdf(datepath + dlc_pos_file).DLC_resnet50_Triangle_Maze_EphysDec7shuffle1_800000
    pos = dlc_pos[pos_col][['x', 'y']].copy()
    pos.loc[dlc_pos[pos_col].likelihood < cutoff, ['x', 'y']] = np.nan

    # Import times of each position point, equal to the time of each camera frame recording
    frametimes = np.array(list(csv.reader(open(datepath + f'testvidtimes{filecount}.csv', 'r'))), dtype=float).ravel()
    frametimes = (frametimes - photStart)# * (Fs / 1000) # make photometry start time zero

    # Align with photometry visit indices
    porttimes = get_port_times(ardtext, ardtimes) - photStart
    framestamps = []
    inds = (frametimes<=porttimes[0])
    framestamps.append(frametimes[inds]*(Fs/1000))
    for p in range(1,len(porttimes)):
        inds = (frametimes>porttimes[p-1])&(frametimes<=porttimes[p])
        framestamps.append((frametimes[inds]-porttimes[p-1])*(Fs/1000) + visits[p-1])
    inds = (frametimes>=porttimes[-1])
    framestamps.append((frametimes[inds]-porttimes[-1])*(Fs/1000) + visits[-1])
    framestamps = np.concatenate(framestamps).astype(int)

    #TODO ADD IN CODE TO REMOVE ABERRANT JUMPS AND THEN FILL IN MISSING GAPS
    pixelJumpCutoff = 30 * pixelsPerCm
    pos.loc[:,['x','y']] = remove_aberrant_jumps(position_data=pos.loc[:,['x','y']].values,\
        max_jump_distance=pixelJumpCutoff)
    pos.loc[:,['x','y']] = fill_missing_gaps(pos.loc[:,['x','y']].values)
    pos.loc[:,['x','y']] = smooth_position_data(pos.loc[:,['x','y']].values,window_size=3)
    # Smooth with gaussian kernel
    #if gaus_smooth:
    #    pos = gaussian_filter(pos, sigma=(0, sigma))
    #else:
    #    pos.loc[:,['x','y']] = smooth_position_data(pos.loc[:,['x','y']].values,window_size=3)
        
    #vel = np.linalg.norm(np.diff(pos, axis=0), axis=1) * 15 * 1 / 3.14 # velocity in cm/s
    vel = calculate_velocity(pos['x'].values, pos['y'].values,fps=cam_fps,unit_conversion=pixelsPerCm)
    vel = np.append([0],vel)
    acc = calculate_acceleration(pos['x'].values, pos['y'].values,fps=cam_fps,pixel_to_cm=pixelsPerCm)
    return pos['x'].values, pos['y'].values, vel, framestamps, acc, dlc_pos
        
#def align_pos_to_visits(Fs,visits,datepath,phot_dlc='n',
#    filecount="0",gaus_smooth=True,
#    sigma=0.01,cutoff=0.9,use_centroid=False):
#
#    #import arduino behavioral data and their timestamps
#    a = open(datepath+'arduinoraw'+"0"+'.txt','r')
#    ardtext= a.read().splitlines()
#    a.close
#    with open(datepath+'ArduinoStamps'+str("0")+'.csv','r') as at:
#        reader = csv.reader(at)
#        ardtimes = list(reader)
#    ardtimes = [float(val) for sublist in ardtimes for val in sublist]
#    photStart = ardtimes[1] #time when pulse was sent to R series. 0 index for sig data
#    ardtimes = np.subtract(ardtimes,photStart) #make photometry start time zero
#    #convert ardtimes to sample number to match photometry data. 
#    #Target rate = 250/sec from 10 KHz initial sample rate (same as photometry)
#    ardstamps = np.round(ardtimes*(Fs/1000)).astype(int)
#    if phot_dlc=='y':
#        #dlc_pos = pd.read_hdf(datepath+'Behav_Vid0DLC_resnet50_Triangle_'+
#        #    'Maze_Phot_RobustJun8shuffle1_350000.h5').\
#        #DLC_resnet50_Triangle_Maze_Phot_RobustJun8shuffle1_350000
#        dlc_pos = pd.read_hdf(datepath+'Behav_Vid0DLC_'+\
#            'resnet50_Maze_UpdatedAug21shuffle1_300000.h5').\
#            DLC_resnet50_Maze_UpdatedAug21shuffle1_300000
#        pos = dlc_pos.cap.loc[:,['x','y']]
#        pos.loc[dlc_pos.cap.likelihood<.99]=np.nan
#        x = pos.values[:,0]
#        y = pos.values[:,1]
#    else:
#        dlc_pos = pd.read_hdf(datepath+'Behav_Vid0'+\
#            'DLC_resnet50_Triangle_Maze_EphysDec7shuffle1_800000.h5').\
#        DLC_resnet50_Triangle_Maze_EphysDec7shuffle1_800000
#        if use_centroid:
#            posFront = dlc_pos.cap_front.loc[:,['x','y']]
#            posFront.loc[dlc_pos.cap_front.likelihood<.9]=np.nan
#            posFront.loc[posFront.x.isnull(),['x']]=posBack.loc[posFront.x.isnull(),['x']]
#            posFront.loc[posFront.y.isnull(),['y']]=posBack.loc[posFront.y.isnull(),['y']]
#            posBack.loc[posBack.x.isnull(),['x']]=posFront.loc[posBack.x.isnull(),['x']]
#            posBack.loc[posBack.y.isnull(),['y']]=posFront.loc[posBack.y.isnull(),['y']]
#            pos_centroids = get_centroid(posFront.values,posBack.values)
#            x = pos_centroids[:,0]
#            y = pos_centroids[:,1]
#        else:
#            posBack = dlc_pos.cap_back.loc[:,['x','y']]
#            posBack.loc[dlc_pos.cap_back.likelihood<cutoff]=np.nan
#            x = posBack.loc[:,'x']
#            y = posBack.loc[:,'y']
#    #Import times of each position point, equal to the time of each camera frame recording
#    with open(datepath+'testvidtimes'+str("0")+'.csv','r') as v:
#        reader = csv.reader(v)
#        frametimes = list(reader)
#    frametimes = [float(val) for sublist in frametimes for val in sublist]
#    frametimes = np.subtract(frametimes,photStart) #make photometry start time zero
#    #align with photometry visit indices (from )
#    porttimes = get_port_times(ardtext,ardtimes)#np.concatenate([[0],get_port_times(ardtext,ardtimes)])
#    framestamps = []
#    inds = (frametimes<=porttimes[0])
#    framestamps.append(frametimes[inds]*(Fs/1000))
#    for p in range(1,len(porttimes)):
#        inds = (frametimes>porttimes[p-1])&(frametimes<=porttimes[p])
#        framestamps.append((frametimes[inds]-porttimes[p-1])*(Fs/1000) + visits[p-1])
#    inds = (frametimes>=porttimes[-1])
#    framestamps.append((frametimes[inds]-porttimes[-1])*(Fs/1000) + visits[-1])
#    framestamps = np.concatenate(framestamps).astype(int)
#    #smooth with gaussian kernel
#    if gaus_smooth:
#        x = gaussian_filter(x,sigma=sigma+sigma/6)
#        y = gaussian_filter(y,sigma=sigma)
#        
#    vel = []
#    for p in range(len(x)-1):
#        vel.append(dtg(x[p],y[p],x[p+1],y[p+1]))
#        
#    vel = np.multiply(vel,15)*1/3.14 #velocity in cm/s
#    vel = np.append(vel,[0])
#    acc = np.diff(vel)
#    acc = np.append([0],acc)
#    return x,y,vel,framestamps,acc,dlc_pos


