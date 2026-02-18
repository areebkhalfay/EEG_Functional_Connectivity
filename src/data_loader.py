import os
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
import src.config as cfg

def get_subject_nwb_path(subj_num):
    """
    Returns full path to the NWB file for a subject.

    Args:
        subj_num (int): integer like 41, 42, ..., 62
    Returns:
        string: full path to the NWB file for that subject.
    """
    subj_str = f"CS{subj_num}"
    folder = f"sub-{subj_str}"
    fname = f"sub-{subj_str}_ses-P{subj_num}CSR1_behavior+ecephys.nwb"
    return os.path.join(cfg.DATA_ROOT, folder, fname)

def load_cuts_with_scene_change(csv_path):
    """
    Parses the scene cut CSV.

    Args:
        csv_path (str): Scene cuts csv path.
    Returns:
        df (Pandas Dataframe): dataframe of scene cut data.
        movie_duration (int): total length of movie.
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values("shot_start_t").reset_index(drop=True)

    scene_ids = df["scene_id"].values
    is_scene_change = np.zeros(len(df), dtype=bool)
    is_scene_change[0] = True
    is_scene_change[1:] = scene_ids[1:] != scene_ids[:-1]

    df["t_cut"] = df["shot_start_t"].astype(float)
    df["is_scene_change"] = is_scene_change
    
    df["shot_end_t"] = df["shot_start_t"] + df["shot_dur_t"]
    movie_duration = df["shot_end_t"].max()

    return df, movie_duration

def make_one_window_per_cut(eeg, fs, cuts_df, eeg_movie_start=10.0, window_size_sec=1, normalize=True):
    """
    Extracts EEG windows around scene cuts.

    Args:
        eeg (Numpy Array): Raw eeg subject data.
        fs (int): Sampling rate.
        cuts_df (Pandas dataframe): Scene cut dataframe.
        eeg_movie_start (int): Starting time point of eeg data collection given movie.
        window_size_sec (int): Window time point granularity.
        normalize (boolean): To normalize or not to normalize the data.
    Returns:
        X(Numpy Array): Data features derived from EEG data.
        Y(Numpy Array): Labels derived from scene cut data.
        meta (list): Appropriate metadata.
    """
    T, C = eeg.shape
    window_size = int(window_size_sec * fs)
    half_window = window_size // 2

    # Per-subject normalization
    eeg_proc = eeg.astype(np.float32)
    if normalize:
        mean = eeg_proc.mean(axis=0, keepdims=True)
        std = eeg_proc.std(axis=0, keepdims=True) + 1e-8
        eeg_proc = (eeg_proc - mean) / std

    cut_times = cuts_df["t_cut"].values.astype(float)
    is_scene_change = cuts_df["is_scene_change"].values

    X_list, y_list, meta = [], [], []

    for cut_idx, (t_cut, scene_change_flag) in enumerate(zip(cut_times, is_scene_change)):
        center_eeg_time = t_cut + eeg_movie_start
        center_sample = int(round(center_eeg_time * fs))
        start = center_sample - half_window
        end = start + window_size

        if start < 0 or end > T:
            continue

        window = eeg_proc[start:end, :]
        X_list.append(window)
        y_list.append(1 if scene_change_flag else 0)
        meta.append({
            "cut_idx": int(cut_idx),
            "t_cut_movie": float(t_cut),
            "center_eeg_time": float(center_eeg_time),
            "center_sample": int(center_sample),
            "is_scene_change": bool(scene_change_flag),
        })

    if len(X_list) == 0:
        X = np.zeros((0, window_size, C), dtype=np.float32)
        y = np.zeros((0,), dtype=np.int64)
    else:
        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.int64)

    return X, y, meta

def load_all_subjects_data():
    """
    Loops through all subjects and loads their data into memory.

    Returns:
        per_subject_data (dictionary/hashmap): Dictionary of subject eeg features (X array), scene cut labels, metadata and brain regions per channel.
    """
    per_subject_data = {}
    cuts_df, _ = load_cuts_with_scene_change(cfg.SCENE_CUTS_CSV)

    for subject in cfg.SUBJECT_IDS:
        try:
            nwb_path = get_subject_nwb_path(subject)
            if not os.path.exists(nwb_path):
                print(f"Skipping Subject {subject}: File not found.")
                continue
                
            print(f"Processing subject: {subject}")
            io = NWBHDF5IO(nwb_path, 'r', load_namespaces=True)
            nwbfile = io.read()
            
            # Get brain regions
            elec_df = nwbfile.electrodes.to_dataframe()
            brainRegionsPerChannel = elec_df[elec_df['origchannel'].str.startswith('macro')]['location'].to_numpy()
            
            # Get LFP Macro data
            ecephys_mod = nwbfile.processing['ecephys']
            lfp_macro = ecephys_mod['LFP_macro']
            esMacro = lfp_macro.electrical_series['ElectricalSeries']
            lfp_macro_data = np.array(esMacro.data)
            
            # Validation to ensure consistent subject data
            if lfp_macro_data.shape[1] != 96:
                print(f"Patient {subject} has {lfp_macro_data.shape[1]} channels (expected 96). Skipping.")
                continue
                
            lfp_macro_data = lfp_macro_data[:, :96]

            # Preprocessing
            X_s, y_s, meta_s = make_one_window_per_cut(
                eeg=lfp_macro_data,
                fs=cfg.FS,
                cuts_df=cuts_df,
                eeg_movie_start=cfg.EEG_MOVIE_START,
                window_size_sec=cfg.WINDOW_SIZE_SEC,
                normalize=True
            )
            
            per_subject_data[subject] = {
                "X": X_s, 
                "y": y_s, 
                "meta": meta_s, 
                "Brain Regions": brainRegionsPerChannel
            }
            io.close()
            
        except Exception as e:
            print(f"Error processing subject {subject}: {e}")
            continue
            
    return per_subject_data