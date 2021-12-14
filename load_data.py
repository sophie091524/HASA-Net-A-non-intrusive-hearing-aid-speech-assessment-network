import os
import time
import librosa
import random
import numpy as np
import pandas as pd
import collections
from scipy.io import wavfile
from scipy.signal import hamming
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import math
maxv = np.iinfo(np.int16).max 

def make_spectrum(sig, feature_type=None, mode=None):   
    n_fft=512
    hop_length=256
    win_length=512
    
    #sig = wavfile.read(path)[-1].astype(float)/maxv

    signal = librosa.util.fix_length(sig, len(sig) + n_fft//2)
    F = librosa.stft(signal,n_fft=512,hop_length=256,win_length=512,window=hamming)    
    Lp=np.abs(F)
    phase = np.exp(1j * np.angle(F))
    
    if feature_type == 'logmag':
        Sxx = np.log1p(Lp)
    elif feature_type == 'lps':
        Sxx = np.log10(Lp**2)
    else:
        Sxx = Lp
    #print(f'initial:{Sxx.shape}')
    if mode == 'mean_std':
        mean = np.mean(Sxx, axis=-1).reshape(Sxx.shape[0], 1)
        std = np.std(Sxx, axis=-1).reshape(Sxx.shape[0], 1)+1e-12
        Sxx = (Sxx-mean)/std 
        #print(f'shape:{Sxx.shape}')
    elif mode == 'minmax':
        Sxx = 2 * (Sxx - _min)/(_max - _min) - 1
    #print(f'after:{Sxx.shape}')
    return Sxx


def SPL_dB(signal):
    rms= np.sqrt(np.mean(signal**2))
    Pa_ref = 2.0e-5 # 20 uPa
    ans=10*math.log10(pow((rms/ Pa_ref),2))
    return ans

def mulnum(signal, init_spldb):
    #init_power = np.sum(signal**2)
    Pa_ref = 2.0e-5 # 20 uPa
    new_rms = Pa_ref*10**(65/20)
    init_rms = np.sqrt(np.mean(signal**2))
    #print(new_rms, init_rms)
    ans = round(new_rms/init_rms,2)
    return ans

class HASQI_train(Dataset):  
    def __init__(self, filepath):
        self.data_list = filepath
        self.ref = self.data_list['ref'].to_numpy()
        self.data = self.data_list['data'].to_numpy()
        str2array = lambda x: np.fromstring(x.replace('\n','').replace('[','').replace(']','').replace('  ',' '), sep=' ')
        self.hl = self.data_list['HL'].apply(str2array).to_numpy()
        self.HASQIscore = self.data_list['HASQI'].astype('float32').to_numpy()
                      
    def __getitem__(self, idx):
        ref = wavfile.read(self.ref[idx])[-1].astype('float32')/maxv
        data = wavfile.read(self.data[idx])[-1].astype('float32')/maxv
        SPL_dB_ref, SPL_dB_data = SPL_dB(ref), SPL_dB(data)
        multiple, multiple_data =  mulnum(ref, SPL_dB_ref), mulnum(data, SPL_dB_data)
        new_ref, new_data = multiple*ref, multiple_data*data # calibrate to 65 dB SPL
        Sxx_ref, Sxx_data = make_spectrum(new_ref), make_spectrum(new_data)
        hl = self.hl[idx]
        score = self.HASQIscore[idx]
        
        return Sxx_ref, Sxx_data,\
               torch.from_numpy(hl).float(), torch.from_numpy(np.asarray(score)).float()
        
    def __len__(self):
        return len(self.data_list)
    
class HASQI_test(Dataset):  
    def __init__(self, filepath):
        self.data_list = filepath
        self.ref = self.data_list['ref'].to_numpy()
        self.data = self.data_list['data'].to_numpy()
        str2array = lambda x: np.fromstring(x.replace('\n','').replace('[','').replace(']','').replace('  ',' '), sep=' ')
        self.hl = self.data_list['HL'].apply(str2array).to_numpy()
        self.HASQIscore = self.data_list['HASQI'].astype('float32').to_numpy()
        self.hltype = self.data_list['HLType'].to_numpy()
    
    def __getitem__(self, idx):
        data_name = self.data[idx]
        ref = wavfile.read(self.ref[idx])[-1].astype('float32')/maxv
        data = wavfile.read(self.data[idx])[-1].astype('float32')/maxv
        SPL_dB_ref, SPL_dB_data = SPL_dB(ref), SPL_dB(data)
        multiple, multiple_data =  mulnum(ref, SPL_dB_ref), mulnum(data, SPL_dB_data)
        new_ref, new_data = multiple*ref, multiple_data*data # calibrate to 65 dB SPL
        Sxx_ref, Sxx_data = make_spectrum(new_ref), make_spectrum(new_data)
        hl = self.hl[idx]
        score = self.HASQIscore[idx]
        hltype = self.hltype[idx]
        
        return data_name, Sxx_ref, Sxx_data,\
               torch.from_numpy(hl).float(), torch.from_numpy(np.asarray(score)).float(), hltype 
    
    def __len__(self):
        return len(self.data_list)
    
###########################################################  
class HASPI_train(Dataset):  
    def __init__(self, filepath):
        self.data_list = filepath
        self.ref = self.data_list['ref'].to_numpy()
        self.data = self.data_list['data'].to_numpy()
        str2array = lambda x: np.fromstring(x.replace('\n','').replace('[','').replace(']','').replace('  ',' '), sep=' ')
        self.hl = self.data_list['HL'].apply(str2array).to_numpy()
        self.HASPIscore = self.data_list['HASPI'].astype('float32').to_numpy()
                      
    def __getitem__(self, idx):
        ref = wavfile.read(self.ref[idx])[-1].astype('float32')/maxv
        data = wavfile.read(self.data[idx])[-1].astype('float32')/maxv
        SPL_dB_ref, SPL_dB_data = SPL_dB(ref), SPL_dB(data)
        multiple, multiple_data =  mulnum(ref, SPL_dB_ref), mulnum(data, SPL_dB_data)
        new_ref, new_data = multiple*ref, multiple_data*data # calibrate to 65 dB SPL
        Sxx_ref, Sxx_data = make_spectrum(new_ref), make_spectrum(new_data)
        hl = self.hl[idx]
        score = self.HASPIscore[idx]
        
        return Sxx_ref, Sxx_data,\
               torch.from_numpy(hl).float(), torch.from_numpy(np.asarray(score)).float()
        
    def __len__(self):
        return len(self.data_list)
    
class HASPI_test(Dataset):  
    def __init__(self, filepath):
        self.data_list = filepath
        self.ref = self.data_list['ref'].to_numpy()
        self.data = self.data_list['data'].to_numpy()
        str2array = lambda x: np.fromstring(x.replace('\n','').replace('[','').replace(']','').replace('  ',' '), sep=' ')
        self.hl = self.data_list['HL'].apply(str2array).to_numpy()
        self.HASPIscore = self.data_list['HASPI'].astype('float32').to_numpy()
        self.hltype = self.data_list['HLType'].to_numpy()
    
    def __getitem__(self, idx):
        data_name = self.data[idx]
        ref = wavfile.read(self.ref[idx])[-1].astype('float32')/maxv
        data = wavfile.read(self.data[idx])[-1].astype('float32')/maxv
        SPL_dB_ref, SPL_dB_data = SPL_dB(ref), SPL_dB(data)
        multiple, multiple_data =  mulnum(ref, SPL_dB_ref), mulnum(data, SPL_dB_data)
        new_ref, new_data = multiple*ref, multiple_data*data # calibrate to 65 dB SPL
        Sxx_ref, Sxx_data = make_spectrum(new_ref), make_spectrum(new_data)
        hl = self.hl[idx]
        score = self.HASPIscore[idx]
        hltype = self.hltype[idx]
        
        return data_name, Sxx_ref, Sxx_data,\
               torch.from_numpy(hl).float(), torch.from_numpy(np.asarray(score)).float(), hltype 
      
    def __len__(self):
        return len(self.data_list)
    
###########################################################
class Dataset_train(Dataset):  
    def __init__(self, filepath):
        self.data_list = filepath
        self.ref = self.data_list['ref'].to_numpy()
        self.data = self.data_list['data'].to_numpy()
        str2array = lambda x: np.fromstring(x.replace('\n','').replace('[','').replace(']','').replace('  ',' '), sep=' ')
        self.hl = self.data_list['HL'].apply(str2array).to_numpy()
        self.HASQIscore = self.data_list['HASQI'].astype('float32').to_numpy()
        self.HASPIscore = self.data_list['HASPI'].astype('float32').to_numpy()
                      
    def __getitem__(self, idx):
        ref = wavfile.read(self.ref[idx])[-1].astype('float32')/maxv
        data = wavfile.read(self.data[idx])[-1].astype('float32')/maxv
        SPL_dB_ref, SPL_dB_data = SPL_dB(ref), SPL_dB(data)
        multiple, multiple_data =  mulnum(ref, SPL_dB_ref), mulnum(data, SPL_dB_data)
        new_ref, new_data = multiple*ref, multiple_data*data # calibrate to 65 dB SPL
        Sxx_ref, Sxx_data = make_spectrum(new_ref), make_spectrum(new_data)
        hl = self.hl[idx]
        hasqi = self.HASQIscore[idx]
        haspi = self.HASPIscore[idx]
        
        return Sxx_ref, Sxx_data, \
               torch.from_numpy(hl).float(), torch.from_numpy(np.asarray(hasqi)).float(),\
               torch.from_numpy(np.asarray(haspi)).float()
        
    def __len__(self):
        return len(self.data_list)
    
class Dataset_test(Dataset):  
    def __init__(self, filepath):
        self.data_list = filepath
        self.ref = self.data_list['ref'].to_numpy()
        self.data = self.data_list['data'].to_numpy()
        str2array = lambda x: np.fromstring(x.replace('\n','').replace('[','').replace(']','').replace('  ',' '), sep=' ')
        self.hl = self.data_list['HL'].apply(str2array).to_numpy()
        self.HASQIscore = self.data_list['HASQI'].astype('float32').to_numpy()
        self.HASPIscore = self.data_list['HASPI'].astype('float32').to_numpy()
        self.hltype = self.data_list['HLType'].to_numpy()
    
    def __getitem__(self, idx):
        data_name = self.data[idx]
        ref = wavfile.read(self.ref[idx])[-1].astype('float32')/maxv
        data = wavfile.read(self.data[idx])[-1].astype('float32')/maxv
        SPL_dB_ref, SPL_dB_data = SPL_dB(ref), SPL_dB(data)
        multiple, multiple_data =  mulnum(ref, SPL_dB_ref), mulnum(data, SPL_dB_data)
        new_ref, new_data = multiple*ref, multiple_data*data # calibrate to 65 dB SPL
        Sxx_ref, Sxx_data = make_spectrum(new_ref), make_spectrum(new_data)
        hl = self.hl[idx]
        hasqi = self.HASQIscore[idx]
        haspi = self.HASPIscore[idx]
        hltype = self.hltype[idx]
        return data_name, Sxx_ref, Sxx_data,\
               torch.from_numpy(hl).float(), torch.from_numpy(np.asarray(hasqi)).float(), \
               torch.from_numpy(np.asarray(haspi)).float(), hltype      
    
    def __len__(self):
        return len(self.data_list)
    
if __name__ == '__main__':     
    batch_size = 1
    num_workers = 8
    import pandas as pd
    import torch
    from torch_stft import STFT
    valid_filepath = '../Merge/train_rising.csv'
    
    df = pd.read_csv(valid_filepath)
    df = df.iloc[0:5]
    print(len(df))
    valid_data = HASQI_train(df)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
     
    for step , (ref, data, hl, score) in enumerate(valid_loader):
        stft = STFT(filter_length=512, hop_length=256, win_length=512, window='hamming')
        mag_data, phase_data = stft.transform(ref)
        print(mag_data.size(),hl.size(), score.size()) #(B,F,T), (B,6), (B)
        #p.ndarray [shape=(d, t)] or None
        #librosa.feature.rms(y=None, S=None, frame_length=2048, hop_length=512, center=True, pad_mode='reflect')
        spetrum = mag_data.squeeze(0).cpu().numpy()
        #rms = librosa.feature.rms(S=spetrum,frame_length=512, hop_length=256)[0]
        #print(rms)
        #print(len(spetrum[0]))
        #print(len(rms[0]))
        #times = librosa.frames_to_time(np.arange(len(rms)))

        #frame_mask = torch.zeros(mag_data.size(0), mag_data.size(2))
        #print(name, hltype)
        