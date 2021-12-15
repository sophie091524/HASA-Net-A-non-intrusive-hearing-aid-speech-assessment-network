import pandas as pd

def load_csvfile(csvfile, mode):
    list_of_dataframes = []
    for filename in csvfile:
        f = pd.read_csv(filename)
        if mode=='train':
            new_f = f
        elif mode =='valid':
            new_f = f
        else:
            new_f = f.iloc[100:]
        list_of_dataframes.append(new_f)
    merged_df = pd.concat(list_of_dataframes)
    print(len(merged_df))
    return merged_df

def get_trainfile():
    filepath = ['../Merge_new/train_flat.csv','../Merge_new/train_sloping.csv','../Merge_new/train_rising.csv',
                '../Merge_new/train_cookiebite.csv','../Merge_new/train_noisenotched.csv','../Merge_new/train_highfrequency.csv']
    df = load_csvfile(filepath, 'train')
    return df 
    
def get_validfile():
    filepath = ['../Merge_new/valid_seen.csv',
                '../Merge_new/valid_unseen.csv']    
    df = load_csvfile(filepath, 'valid')
    return df 

def get_testfile():    
    seen_filepath = ['../Merge/test_flat_seen_0.csv',
                '../Merge/test_flat_seen_1.csv',
                '../Merge/test_sloping_seen_0.csv',
                '../Merge/test_sloping_seen_1.csv',
                '../Merge/test_rising_seen_0.csv',
                '../Merge/test_rising_seen_1.csv',
                '../Merge/test_cookiebite_seen_0.csv',
                '../Merge/test_cookiebite_seen_1.csv',     
                '../Merge/test_noisenotched_seen_0.csv',
                '../Merge/test_noisenotched_seen_1.csv',
                '../Merge/test_highfrequency_seen_0.csv', 
                '../Merge/test_highfrequency_seen_1.csv']
                
    unseen_filepath = ['../Merge/test_flat_unseen_0.csv', 
                '../Merge/test_flat_unseen_1.csv',
                '../Merge/test_sloping_unseen_0.csv',
                '../Merge/test_sloping_unseen_1.csv',
                '../Merge/test_rising_unseen_0.csv',
                '../Merge/test_rising_unseen_1.csv',
                '../Merge/test_cookiebite_unseen_0.csv', 
                '../Merge/test_cookiebite_unseen_1.csv', 
                '../Merge/test_noisenotched_unseen_0.csv',  
                '../Merge/test_noisenotched_unseen_1.csv',
                '../Merge/test_highfrequency_unseen_0.csv',
                '../Merge/test_highfrequency_unseen_1.csv']
    
    df_seen, df_unseen = load_csvfile(seen_filepath, 'test'), load_csvfile(unseen_filepath, 'test')   
    return df_seen, df_unseen 
