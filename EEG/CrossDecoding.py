# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:34:06 2022

@author: U987219
"""

import os
import numpy as np
import  matplotlib.pyplot as plt
import scipy.stats as st

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection as models
from sklearn.svm import SVC
from scipy.ndimage.filters import uniform_filter1d



import mne
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)
from mne.stats import (spatio_temporal_cluster_1samp_test, permutation_cluster_1samp_test)

## harded coded params ##
#paths paths
pth_experiment = 'C:\\Users\\Yuanfang\\Documents\\work\\projects\\Timingproj\\EEG\\'
pth_fig = os.path.join(pth_experiment, 'data\\figs\\')
pth_epoch = os.path.join(pth_experiment, 'data\\epoched\\')

# posteior channels
chs_post = ['O1','O2','Oz',
           'P1','P2','P3','P4','P5','P6','P7','P8',
           'PO10','PO3','PO4','PO7','PO8','PO9','POz','Pz']

# mark ID
ID = {'building'  : [101, 102, 103, 104, 105, 106, 107, 108],
      'box'       : [111, 112, 113, 114, 115, 116, 117, 118],
      'tool'      : [121, 122, 123, 124, 125, 126, 127, 128],
      'instrument': [131, 132, 133, 134, 135, 136, 137, 138],
      'scene'     : [141, 142, 143, 144, 145, 146, 147, 148],
      'hand'      : [151, 152, 153, 154, 155, 156, 157, 158],
      'chair'     : [161, 162, 163, 164, 165, 166, 167, 168]}

p_channel_flg = 0

# read **epoched** subject data - one file per subject
epoch_all_ls = sorted([f for f in os.listdir(pth_epoch) if "ds125_ICA_epo.fif" in f])   # folder contains only "usable" Ps
n_subj = len(epoch_all_ls)     # number subjects

data_container_time = np.zeros([n_subj, 2, 125])  # shape: n_subk, "category" decoding (i.e., 0=building/box; 1=tool/instrument), # times

## p loop
for p in range(n_subj):
    
    # pnum
    p_num = epoch_all_ls[p][0:4]
    p_file = epoch_all_ls[p]
    print('** running participant %s,\nfile: %s **' % (p_num, p_file))
    
    # load stored epochs 
    epoch = mne.read_epochs(os.path.join(pth_epoch, p_file), preload=True)
    
    # subset channels 
    chn_lbl = ['allChannels','postChannels'][p_channel_flg]
    if p_channel_flg==1:  # 0=all channels (default);   1=posterior channels
          epoch = epoch.pick_channels(chs_post)
    print('** channels setup for decoding: %s **' % chn_lbl)
    
    # loop across x-decode conditions
    for xDecode_flg in range(2):
        if xDecode_flg==0:
              # extract epochs
              epochX = epoch['building', 'box', 'scene', 'chair']
              # assign labels
              events = epochX.events
              y_events = mne.merge_events(events, ID['building']+ID['scene'], 1)
              y_events = mne.merge_events(y_events, ID['box']+ID['chair'], 2)
              # assign cv groups
              grp_events = events
              grp_events = mne.merge_events(grp_events, ID['building']+ID['box'], -1) # always training
              grp_events = mne.merge_events(grp_events, ID['scene']+ID['chair'], 0)
              # labels 
              cond_lbl_fn = 'buildingBox_cross-decoding_sceneChair'
              cond_lbl_figTitle = 'building|box > scene|chair'
        if xDecode_flg==1:
              # extract epochs
              epochX = epoch['tool', 'instrument', 'hand', 'chair']
              # assign labels
              events = epochX.events
              y_events = mne.merge_events(events, ID['tool']+ID['hand'], 1)
              y_events = mne.merge_events(y_events, ID['instrument']+ID['chair'], 2)
              # assign cv groups
              grp_events = events
              grp_events = mne.merge_events(grp_events, ID['hand']+ID['chair'], -1) # always training
              grp_events = mne.merge_events(grp_events, ID['tool']+ID['instrument'], 0)
              # labels
              cond_lbl_fn = 'toolInstrument_cross-decoding_handChair'
              cond_lbl_figTitle = 'tool|instrument > hand|chair'
        
        # output message of which dimensions are xDecodedyou t 
        print('** running: %s **' % (cond_lbl_figTitle))
         
        # do within-pair decoding
        # prepare the data
        X = epochX.get_data()
        y = y_events[:,2]
        grp = grp_events[:,2]
        # the pipeline
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        # the sliding estimator
        time_decod = SlidingEstimator(clf, n_jobs=8, scoring='roc_auc', verbose=True)
        #time_decod = GeneralizingEstimator(clf, n_jobs=16, scoring='roc_auc', verbose=True)
        # score from cv scheme
        scores = cross_val_multiscore(time_decod, X, y, cv=models.PredefinedSplit(test_fold=grp), n_jobs=16)
        # as just 1 fold, so not need to do average 
        score_time_avg = np.squeeze(scores)
        
        # rescale
        times = 1e3 * epochX.times # to have times in ms

        # store data for later group plots and statistics              
        data_container_time[p,xDecode_flg,:] = uniform_filter1d(score_time_avg, 9) # smooth w/ 9-point running average
# save data for later group analysis
out_pth_tmp = os.path.join(pth_experiment, 'data', 'exported', 'CrossDecoding')
if not os.path.exists(out_pth_tmp):
    os.makedirs(out_pth_tmp)
np.save(os.path.join(out_pth_tmp, 'PsxDecodeFlgxTimes_SVC.npy'), data_container_time)

#%% ############################# group plot ##############################
for xDecode_flg in range(2):
     
     if xDecode_flg==0:
        cond_lbl_fn = 'buildingBox_cross-decoding_sceneChair'
        cond_lbl_figTitle = 'building|box > scene|chair'
     if xDecode_flg==1:
        cond_lbl_fn = 'toolInstrument_cross-decoding_toolChair'
        cond_lbl_figTitle = 'tool|instrument > hand|chair'
    
     pair_grp = data_container_time[:, xDecode_flg, :]
    
     # statistics inference (TFCE corrected)
     # Calculate statistical thresholds
     pair_grp_1samp = pair_grp - 0.5 # compare with 0.5
     pair_grp_1samp = pair_grp_1samp[..., np.newaxis]
     #pair_grp_1samp = np.log2(pair_grp_1samp+1)
     
     print('Clustering...')
     threshold = -1 * st.distributions.t.ppf(q=0.05, df=len(pair_grp_1samp) - 1)
     #threshold = dict(start=-2, step=0.01) 
     tvals, clusters, pvals, H0 = spatio_temporal_cluster_1samp_test(pair_grp_1samp, threshold=threshold, adjacency=None, n_jobs=8, tail=1, n_permutations=10000)
     
     
     # average across participants
     score_time_avg = np.average(pair_grp, 0)
     ci = st.t.interval(alpha=0.95, df=len(pair_grp)-1, loc=np.mean(pair_grp,0), scale=st.sem(pair_grp,0)) 
     ci = (ci[1] - ci[0])/2
     
     ci = st.sem(pair_grp, 0)
    
     ## group plot 
     plt.plot(times, score_time_avg, color='#0072B2', linewidth=1, label="Classif. score")
     plt.axhline(0.5, color='k', linestyle='-', linewidth=0.5, label="Chance level")
    # plt.axvline(0, color='k', label='stim onset')
     plt.legend() 
     plt.fill_between(times, (score_time_avg-ci), (score_time_avg+ci), facecolor='#0072B2', alpha=0.2)
     # shading significant temporal clusters
     print('** %s **\n' % (cond_lbl_figTitle))
     clusters = [[int(y) for y in x[0]] for x in clusters]   
     for c, p_val in zip(clusters, pvals):
         if p_val <= 0.05:
             plt.axvspan(times[c[0]], times[c[-1]], ymin=0.9, ymax=0.9, color='red', alpha=0.5)
             print('Cluster from %f ms to %f ms, p-value %f\n' % (times[c[0]], times[c[-1]], p_val))
    
     plt.xlabel('Times (ms)')
     plt.ylabel('AUC score')
     plt.ylim([0.45, 0.65])
     plt.title('%s\n%s; group' %(cond_lbl_figTitle, chn_lbl))
     plt.gca().spines["top"].set_visible(False)
     plt.gca().spines["right"].set_visible(False)
     plt.gca().spines["bottom"].set_visible(False)
     # save fig
     fig_pth_tmp = '%s\\CrossDecoding_grp\\%s\\' % \
     (pth_fig,chn_lbl)
     if not os.path.exists(fig_pth_tmp):
         os.makedirs(fig_pth_tmp)
     fig_fm_tmp = '%s_filt0.05-100_%s_SVC_125hz_auc.svg' % \
     (cond_lbl_fn, chn_lbl)
     plt.savefig(fig_pth_tmp + fig_fm_tmp)
     
     tocsv = np.stack((score_time_avg, ci), axis=-1)
     np.savetxt((fig_pth_tmp+'%s_bb2sc.csv' % cond_lbl_fn), tocsv, delimiter=",")

     plt.close('all')