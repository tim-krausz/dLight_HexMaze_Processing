#only need implant, date, and rec_folder specification to run

from __main__ import *

def BitsToVolts(Data, ChInfo, Unit,good_chan):
    print('Converting to uV... ', end='')

    if Unit.lower() == 'uv': U = 1
    elif Unit.lower() == 'mv': U = 10**-3

    for C in range(len(good_chan)):
        Data[C] = Data[C] * ChInfo[good_chan[C]]['bit_volts'] * U
    return(Data)


class EphysVisitGetter():
    Unit = 'uv'
    nchan = 264#276
    adc_start = 256
    basePath = "/Volumes/Tim-1/Ephys/"
    fs = 30000

    def __init__(self,implant,date,rec_folder,new_fs):
        self.implant = implant
        self.date = date
        self.rec_folder = rec_folder
        self.new_fs = new_fs
    
    def prepData(self):
        self.loadpath = self.basePath+self.implant+'/'+self.date+'/'
        self.rec_folder = self.loadpath+self.rec_folder+"/experiment1/recording1/"
        self.filepath = self.rec_folder + "continuous/Rhythm_FPGA-100.0/"
        self.contfile = self.filepath+'continuous.dat'
        self.extracted = np.memmap(self.contfile, dtype='int16',mode='c')
        self.nsamples = len(self.extracted)/self.nchan

    def getSeshDurationInS(self):
        return self.nsamples/self.fs

    def get_pulse_times(self,ADC_pulse_chan):
        pulse_min_time = self.new_fs/1000*(6.66)
        #identify points above threshold
        pulse_above_thold = np.where(ADC_pulse_chan>10000)[0]
        #identify periods above threshold longer than 5ms (fs/1000*5)
        pulse_durs = np.diff(np.where(np.diff(np.append(np.concatenate([[0],\
        pulse_above_thold]),0))!=1)[0])
        pulse_starts = pulse_above_thold[np.concatenate([[0],\
            np.cumsum(pulse_durs[:-1])+1])]
        #find timepoint of threshold crossing
        pulse_starts = pulse_starts[np.where(pulse_durs>pulse_min_time)]
        return pulse_starts

    def get_visit_times(self):
        try:
            pulse_chan = np.load(self.loadpath+"pulse_chan.npy")
            return self.get_pulse_times(pulse_chan)
        except:
            print("pulse channel not yet extracted. Extracting now...")
        #iteratively extract relevant data streams (good channels and ADC pulse channel)
        self.extracted = np.reshape(self.extracted,(int(self.nsamples),self.nchan))
        i = 1
        pulse_chan = [-20000]
        
        while True:
            try:
                self.extracted[self.fs*20*i,self.adc_start]
            except:
                print(self.fs*20*i,' is not an extractable index')
                break
            ADC_pulse_chan = self.extracted[self.fs*20*(i-1):self.fs*20*i,self.adc_start]
            ADC_pulse_chan = ADC_pulse_chan.T[::int(self.fs/self.new_fs)]
            pulse_chan = np.concatenate((pulse_chan,ADC_pulse_chan))
            i += 1
        
        port_entries = self.get_pulse_times(pulse_chan)
        return port_entries

