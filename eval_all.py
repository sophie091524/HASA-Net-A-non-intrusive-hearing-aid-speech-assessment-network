import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from sklearn.metrics import mean_squared_error
import scipy.stats

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str)
args = parser.parse_args()

#if args.hl_type is None:
scenario = ['flat','sloping','rising','cookiebite','noisenotched','highfrequency']
#scenario_dict = { 0:'flat',1:'sloping',2:'rising',3:'cookiebite',4:'noisenotch',5:'highfrequency'}

for mode in ['Seen', 'Unseen']:   
    # all 
    filename = os.path.join(args.out_dir, f'result_{mode.lower()}.csv')  
    print(os.path.join(args.out_dir, f'result_{mode.lower()}.csv')) 
    df = pd.read_csv(filename)
    srcc, pvalue = scipy.stats.spearmanr(df['TrueHASQI'].to_numpy(),df['PredictHASQI'].to_numpy())
    Pearson_cc, p_value = scipy.stats.pearsonr(df['TrueHASQI'].to_numpy(),df['PredictHASQI'].to_numpy())
    mse = mean_squared_error(df['TrueHASQI'].to_numpy(), df['PredictHASQI'].to_numpy())
    print(f'HASQI Avg:{mode},MSE:{mse:.4f},LCC:{Pearson_cc:.4f},SRCC:{srcc:.4f}')
    #plt.scatter(df['TrueHASQI'].to_numpy(), df['PredictHASQI'].to_numpy())
    #plt.xlabel("True HASQI")
    #plt.ylabel("Predicted HASQI")
    #plt.title(f'{mode} LCC:{Pearson_cc:.5f},SRCC:{srcc:.5f}')
    #plt.savefig(f'{args.out_dir}scatterall_{mode.lower()}.png')
    #plt.close()
    # seperate each hl_type    
    df = pd.read_csv(filename)   
    
    fig, axs = plt.subplots(2, 3, figsize=(20,8))
    
    for i in scenario:  
        df_temp = df.loc[df['HL'] == i]
        print(i, len(df_temp))
        srcc, pvalue = scipy.stats.spearmanr(df_temp['TrueHASQI'].to_numpy(),df_temp['PredictHASQI'].to_numpy())
        Pearson_cc, p_value = scipy.stats.pearsonr(df_temp['TrueHASQI'].to_numpy(),df_temp['PredictHASQI'].to_numpy()) 
        mse = mean_squared_error(df_temp['TrueHASQI'].to_numpy(), df_temp['PredictHASQI'].to_numpy())
        print(f'HASQI,MSE:{mse:.4f},LCC:{Pearson_cc:.4f},SRCC:{srcc:.4f}')
        idx = scenario.index(i)
        if idx<3:
            row,col = 0,idx
        else:
            row,col = 1,idx-3
        
        axs[row,col].plot(df_temp['TrueHASQI'].to_numpy(),df_temp['PredictHASQI'].to_numpy(),'o', ms=3)
        axs[row,col].tick_params(labelsize=8)
        axs[row,col].set_xlim([0, 1])
        axs[row,col].set_ylim([0, 1])
        axs[row,col].set_title(f'{i}\n LCC:{Pearson_cc:.7f}\n SRCC:{srcc:.7f}')

    #for ax in axs.flat:
    #    ax.set(xlabel="True HASQI", ylabel="Predicted HASQI")
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    #for ax in axs.flat:
    #    ax.label_outer()
    
    # If you don't do tight_layout() you'll have weird overlaps
    plt.tight_layout()
    plt.savefig(f'{args.out_dir}HASQIscatter_{mode.lower()}_separate.png')
    plt.close()
    
for mode in ['Seen', 'Unseen']:   
    # all 
    filename = os.path.join(args.out_dir, f'result_{mode.lower()}.csv')  
    print(os.path.join(args.out_dir, f'result_{mode.lower()}.csv')) 
    df = pd.read_csv(filename)
    srcc, pvalue = scipy.stats.spearmanr(df['TrueHASPI'].to_numpy(),df['PredictHASPI'].to_numpy())
    Pearson_cc, p_value = scipy.stats.pearsonr(df['TrueHASPI'].to_numpy(),df['PredictHASPI'].to_numpy())
    mse = mean_squared_error(df['TrueHASPI'].to_numpy(), df['PredictHASPI'].to_numpy())
    print(f'HASPI Avg:{mode},MSE:{mse:.4f},LCC:{Pearson_cc:.4f},SRCC:{srcc:.4f}')
    #plt.scatter(df['TrueHASQI'].to_numpy(), df['Predict'].to_numpy())
    #plt.xlabel("True HASQI")
    #plt.ylabel("Predicted HASQI")
    #plt.title(f'{mode} LCC:{Pearson_cc:.7f},SRCC:{srcc:.7f}')
    #plt.savefig(f'{args.out_dir}scatterall_{mode.lower()}.png')
    #plt.close()
    # seperate each hl_type    
    df = pd.read_csv(filename)   
    
    fig, axs = plt.subplots(2, 3, figsize=(12,8))
    
    for i in scenario:  
        df_temp = df.loc[df['HL'] == i]
        print(i, len(df_temp))
        srcc, pvalue = scipy.stats.spearmanr(df_temp['TrueHASPI'].to_numpy(),df_temp['PredictHASPI'].to_numpy())
        Pearson_cc, p_value = scipy.stats.pearsonr(df_temp['TrueHASPI'].to_numpy(),df_temp['PredictHASPI'].to_numpy())
        mse = mean_squared_error(df_temp['TrueHASPI'].to_numpy(), df_temp['PredictHASPI'].to_numpy())
        print(f'HASPI,MSE:{mse:.4f},LCC:{Pearson_cc:.4f},SRCC:{srcc:.4f}')
        #print(f'LCC:{Pearson_cc:.7f}, SRCC:{srcc:.7f}')
        idx = scenario.index(i)
        if idx<3:
            row,col = 0,idx
        else:
            row,col = 1,idx-3
        
        axs[row,col].plot(df_temp['TrueHASPI'].to_numpy(),\
                        df_temp['PredictHASPI'].apply(lambda x: round(x,5)).to_numpy(),\
                        'o', ms=3)
        axs[row,col].tick_params(labelsize=8)
        axs[row,col].set_xlim([0, 1])
        axs[row,col].set_ylim([0, 1])
        axs[row,col].set_title(f'{i}\n LCC:{Pearson_cc:.7f}\n SRCC:{srcc:.7f}')

    #for ax in axs.flat:
    #    ax.set(xlabel="True HASQI", ylabel="Predicted HASQI")
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    #for ax in axs.flat:
    #    ax.label_outer()
    
    # If you don't do tight_layout() you'll have weird overlaps
    plt.tight_layout()
    plt.savefig(f'{args.out_dir}HASPIscatter_{mode.lower()}_separate.png')
    plt.close()