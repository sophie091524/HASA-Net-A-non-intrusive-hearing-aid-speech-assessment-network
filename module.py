import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb 

def get_act_fn(act_fn):
    if act_fn == 'relu':
        return nn.ReLU()
    elif act_fn == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.3)
    elif act_fn == 'elu':
        return nn.ELU()
    elif act_fn == 'sigmoid':
        return nn.Sigmoid()
    elif act_fn == 'softplus':
        return nn.Softplus()
    else:
        raise ValueError('Invalid argument for act_fn') 

class BLSTM_frame_sig_att(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, linear_output, act_fn):
        super().__init__()
        self.blstm = nn.LSTM(input_size = input_size, 
                             hidden_size = hidden_size, 
                             num_layers = num_layers, 
                             dropout = dropout, 
                             bidirectional = True, 
                             batch_first = True)
        self.linear1 = nn.Linear(hidden_size*2, linear_output, bias=True)
        self.act_fn = get_act_fn(act_fn)
        self.dropout = nn.Dropout(p=0.3)
        self.hasqiAtt_layer = nn.MultiheadAttention(linear_output, num_heads=8)
        self.haspiAtt_layer = nn.MultiheadAttention(linear_output, num_heads=8)
        
        self.hasqiframe_score = nn.Linear(linear_output, 1, bias=True)
        self.haspiframe_score = nn.Linear(linear_output, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.hasqiaverage_score = nn.AdaptiveAvgPool1d(1)  
        self.haspiaverage_score = nn.AdaptiveAvgPool1d(1)  
            
    def forward(self, x, hl): #hl:(B,6)
        B, Freq, T = x.size()
        x = x.permute(0,2,1) #(B, 257, T_length)->(B, T_length, 257) 
        #print(f'x:{x.size()}')
        hl = hl.unsqueeze(1) #hl:(B,1,6)
        #print(f'hl:{hl.size()}')
        hl_repeat = hl.repeat(1,T,1)
        #print(f'after repeat, hl:{hl_repeat.size()}')
        x_concate = torch.cat((x,hl_repeat), 2)
        #print(f'concatenate:{x_concate.size()}')
        
        out, _ = self.blstm(x_concate) #(B,T, 2*hidden)
        #print(f'before att:{out.size()}')
        out = self.dropout(self.act_fn(self.linear1(out))).transpose(0,1) #(T_length, B,  128) 
        hasqi, _ = self.hasqiAtt_layer(out,out,out)
        haspi, _ = self.haspiAtt_layer(out,out,out) 
        hasqi, haspi = hasqi.transpose(0,1), haspi.transpose(0,1) #(B, T_length, 128), #(B, T_length, 128)  
        hasqi, haspi = self.hasqiframe_score(hasqi), self.haspiframe_score(haspi) #(B, T_length, 1) 
        hasqi, haspi = self.sigmoid(hasqi), self.sigmoid(haspi) #pass a sigmoid
        hasqi_fram, haspi_fram = hasqi.permute(0,2,1), haspi.permute(0,2,1) #(B, 1, T_length) 
        #print(f'before average:{out.size()}')
        hasqi_avg, haspi_avg = self.hasqiaverage_score(hasqi_fram), self.haspiaverage_score(haspi_fram)  #(B,1,1)
        #print(f'after average:{avg_score.size()}')
        
        return hasqi_fram, haspi_fram, hasqi_avg.squeeze(1), haspi_avg.squeeze(1) #(B, 1, T_length) (B,1) 
       
