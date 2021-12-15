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
        Sxx_ref, Sxx_data = make_spectrum(ref), make_spectrum(data)
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
        Sxx_ref, Sxx_data = make_spectrum(ref), make_spectrum(data)
        hl = self.hl[idx]
        hasqi = self.HASQIscore[idx]
        haspi = self.HASPIscore[idx]
        hltype = self.hltype[idx]
        return data_name, Sxx_ref, Sxx_data,\
               torch.from_numpy(hl).float(), torch.from_numpy(np.asarray(hasqi)).float(), \
               torch.from_numpy(np.asarray(haspi)).float(), hltype      
    
    def __len__(self):
        return len(self.data_list)
    
