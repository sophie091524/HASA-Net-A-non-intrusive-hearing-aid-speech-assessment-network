import os
import gc
import yaml
import random
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from getpath import get_trainfile, get_validfile, get_testfile 
from load_data import Dataset_train, Dataset_test
import module 
from trainer import train, test
import pdb
from sklearn.model_selection import train_test_split

def yaml_config_hook(config_file):
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)
    if "defaults" in cfg.keys():
        del cfg["defaults"]
    return cfg

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main(config_path):   
    # Arguments
    parser = argparse.ArgumentParser(description="Combine_Net")
    config = yaml_config_hook(config_path)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    print(f'save dic:{args.train_checkpoint_dir}')
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(args.seed)
    
    if args.Train:           
        df_train, df_valid = get_trainfile(), get_validfile()
        #df_train, df_valid = df_train.iloc[0:500], df_valid.iloc[0:500]
        print(f'Train length:{len(df_train)}, Valid length:{len(df_valid)}')
        train_data = Dataset_train(df_train)
        valid_data = Dataset_test(df_valid)
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, drop_last=True, 
                                  num_workers=args.num_workers, shuffle=True, pin_memory=True)
        valid_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size, drop_last=True,
                                  num_workers=args.num_workers, shuffle=True, pin_memory=True) 
        
        model = getattr(module, args.model)(args.input_size,args.hidden_size,args.num_layers,args.dropout,\
                                            args.linear_output,args.act_fn)       
        model = model.cuda()        
        train(model, train_loader, valid_loader, args) 
        
    else:        
        df_seen, df_unseen = get_testfile()
        print(f'Seen length:{len(df_seen)}, Unseen length:{len(df_unseen)}')
        # seen
        data_seen = Dataset_test(df_seen)
        seen_loader = DataLoader(data_seen, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        model = getattr(module, args.model)(args.input_size,args.hidden_size,args.num_layers,args.dropout,\
                                            args.linear_output,args.act_fn)        
        model = model.cuda()
        print(f'Loading the model from training dic:{args.train_checkpoint_dir}')   
        ckpt = torch.load(args.train_checkpoint_dir+f'best_loss.pth')['model']
        model.load_state_dict(ckpt) 
        print("==> Testing for Seen HL...")   
        test(model, seen_loader, 'seen', args)
    
        # unseen
        data_unseen = Dataset_test(df_unseen)
        unseen_loader = DataLoader(data_unseen, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        model = getattr(module, args.model)(args.input_size,args.hidden_size,args.num_layers,args.dropout,\
                                            args.linear_output,args.act_fn)      
        model = model.cuda()
        print(f'Loading the model from training dic:{args.train_checkpoint_dir}')   
        ckpt = torch.load(args.train_checkpoint_dir+f'best_loss.pth')['model']
        model.load_state_dict(ckpt) 
        print("==> Testing for Unseen HL...")   
        test(model, unseen_loader, 'unseen', args)

if __name__ == "__main__":    
    config_path = "hyper.yaml"
    print(config_path)
    main(config_path)