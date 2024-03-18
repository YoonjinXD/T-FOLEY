params = {

    # --- Data --- : provide lists of folders that contain .wav files
    'train_dirs': ['./DCASE_2023_Challenge_Task_7_Dataset/train.txt'],
    'test_dirs': ['./DCASE_2023_Challenge_Task_7_Dataset/eval.txt'],
    'sample_rate': 22050,
    'audio_length': 88200,  # traning data seconds * sample_rate
    'num_workers': 4,
    
    # --- Model ---
    'model_dir':                   'logs/',
    'sequential':                  'lstm',
    'factors':                     [2,2,3,3,5,5,7],
    'dims':                        [32,64,128,128,256,256,512,512],
    
    # --- Condition ---
    'time_emb_dim':                512,
    'class_emb_dim':               512,
    'mid_dim':                     512,
    'film_type':                   'block', # {None, temporal, block}
    'block_nums':                  [49,49,49,49,49,49,14],
    'event_type':                  'rms', # {rms, power, onset}
    'event_dims':                  {'rms': 690, 'power': 88200, 'onset': 88200},
    'cond_prob':                   [0.1, 0.1], # [class prob, event prob]
    
    # --- Training ---
    'lr':                          1e-4,
    'batch_size':                  16,
    'ema_rate':                    0.999,
    'scheduler_patience_epoch':    25,
    'scheduler_factor':            0.8,
    'scheduler_threshold':         0.01,
    'restore':                     False,
    
    # --- Logging ---
    'checkpoint_id':               None,
    'num_epochs_to_save':          10, 
    'num_steps_to_test':           250,
    'n_bins':                      5,
    
}
