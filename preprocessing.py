import mne
import numpy as np
import os
import time
from constants import *

def source_reconstruction(epochs,inverse_operator,method='sLORETA',lambda2=None,snr=3.0,ori='normal') -> mne.SourceEstimate:
    if lambda2 is None:
        lambda2 = 1.0 / snr ** 2
    stc = mne.minimum_norm.apply_inverse_epochs(
        epochs,
        inverse_operator,
        lambda2=lambda2,
        method=method,
        pick_ori = ori,
        verbose=False
    )
    return stc

def sample_to_epoch(sample,timestamps,inlet_info,apple_position) -> mne.Epochs:

    raw = mne.io.RawArray(sample, inlet_info,verbose=False)    
    raw.filter(F_LOW, F_HIGH, verbose=False)
    raw.notch_filter(F_NOTCH, trans_bandwidth=2, verbose=False)
    raw.set_eeg_reference('average', projection=True, verbose=False)
  

    event_timestep = int(BEFORE_MARKER_TIME*inlet_info['sfreq'])-1
    print(event_timestep)

    if apple_position>SCREEN_WIDTH/2:
        event_id = {"MI_right": 1}
        event = np.array([[event_timestep,0,1]])
    else:
        event_id = {"MI_left": 0}
        event = np.array([[event_timestep,0,0]])
    
    tmin = -BEFORE_MARKER_TIME+0.05 
    tmax = MARKER_TIME 
    print(f"tmin: {tmin}, tmax: {tmax}")
    
    epochs = mne.Epochs(
        raw,
        event,
        tmin=tmin,
        tmax=tmax,
        baseline=None,  
        event_id=event_id,    
        preload=True,
        verbose=False,
    )

    return epochs

def create_inverse_operator(info) -> mne.minimum_norm.InverseOperator:
    """This function is strongly based on the function generic_inverse created by
    Viktor Naas. Link: https://github.com/wavesresearch/double_step_optimization/blob/2686d518257a58023d16135d1a1c1b06cdff30ab/project_utils/utils.py#L325"""

    subjects_dir = os.getcwd()+'/mne_data/MNE-sample-data/subjects'
    mne.set_config('SUBJECTS_DIR',subjects_dir)
    mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir,verbose=False)

    noise_cov = mne.make_ad_hoc_cov(info,verbose = False)
    bem = mne.read_bem_solution(f'{subjects_dir}/fsaverage/bem/fsaverage-5120-5120-5120-bem-sol.fif',verbose=False)
    src = mne.setup_source_space('fsaverage',spacing='oct6',add_dist='patch',subjects_dir=subjects_dir,verbose=False)

    fwd = mne.make_forward_solution(
        info,
        trans="fsaverage",
        src=src,
        bem=bem,
        meg=False,
        eeg=True,
        mindist=5.0,
        n_jobs=None,
        verbose=False
    )
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        info,
        fwd,
        noise_cov,
        loose = 0.2,
        depth = 0.8,
        verbose=False
    )
    return inverse_operator

def preprocess_giga_data(epochs_left,epochs_right):
    epochs = mne.concatenate_epochs([epochs_left,epochs_right],verbose=False)
    epochs.apply_proj(verbose=False)
    epochs.apply_baseline((None,-0.2),verbose=False)
    info = epochs.info
    return epochs,info

def extract_features(epochs,inverse_operator,tmin=-0.5,tmax=1.0,decimation_factor=1,frequencies=F_BANDS):
    X = []
    for f_low,f_high in frequencies:
        epoch_filtered = epochs.copy().filter(f_low,f_high,verbose=False)
        epoch_filtered = epoch_filtered.decimate(decimation_factor,verbose=False)
        epoch_filtered = epoch_filtered.crop(tmin=tmin,tmax=tmax,verbose=False)
        stcs = source_reconstruction(epoch_filtered,inverse_operator,method = 'sLORETA',snr=3.0)
        x = np.array([np.mean(np.abs(stc.data)**2,axis=1) for stc in stcs])
        X.append(x)
    X = np.concatenate(X,axis=1)
    return X

 

 