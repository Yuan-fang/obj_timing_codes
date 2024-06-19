# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 15:28:36 2022

@author: U987219
"""
import os
import numpy as np
import  matplotlib.pyplot as plt
import scipy.stats as st

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (LogisticRegression, RidgeClassifier)
import sklearn.model_selection as models
from sklearn.svm import SVC
from scipy.ndimage.filters import uniform_filter1d

import mne
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)
from mne.stats import spatio_temporal_cluster_1samp_test

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
    
    # down sample to 100 Hz
    #epoch = epoch.resample(100)
    
    # subset channels 
    chn_lbl = ['allChannels','postChannels'][p_channel_flg]
    if p_channel_flg==1:  # 0=all channels (default);   1=posterior channels
          epoch = epoch.pick_channels(chs_post)
    print('** channels setup for decoding: %s **' % chn_lbl)
    
    
    # loop across x-decode conditions
    for xDecode_flg in range(2):
        if xDecode_flg==0:
              # extract epochs
              epochX = epoch['building', 'box']
              # assign labels
              events = epochX.events
              y_events = mne.merge_events(events, ID['building'], 1)
              y_events = mne.merge_events(y_events, ID['box'], 2)
              # assign cv groups
              grp_events = events
              for i in range(8):
                     grp_events = mne.merge_events(grp_events, [101+i, 111+i], 1+i)
              # grp_events = mne.merge_events(grp_events, [105, 106, 107, 108, 115, 116, 117, 118], 0)
              # grp_events = mne.merge_events(grp_events, [101, 102, 103, 104, 111, 112, 113, 114], 1)
              
        if xDecode_flg==1:
              # extract epochs
              epochX = epoch['tool', 'instrument']
              # assign labels
              events = epochX.events
              y_events = mne.merge_events(events, ID['tool'], 1)
              y_events = mne.merge_events(y_events, ID['instrument'], 2)
              # assign cv groups
              grp_events = events
              for i in range(8):
                    grp_events = mne.merge_events(grp_events, [121+i, 131+i], 1+i)
              # grp_events = mne.merge_events(grp_events, [123, 127, 128, 133, 137, 138], 0)
              # grp_events = mne.merge_events(grp_events, [121, 122, 124, 125, 126, 131, 132, 134, 135, 136], 1)
        
        # do within-pair decoding
        # prepare the data
        X = epochX.get_data()
        y = y_events[:,2]
        grp = grp_events[:,2]
        # the pipeline
        clf = make_pipeline(StandardScaler(),  SVC())
        # the sliding estimator
        time_decod = SlidingEstimator(clf, n_jobs=16, scoring='roc_auc', verbose=True)
        # score from cv scheme
        scores = cross_val_multiscore(time_decod, X, y, grp, cv=models.LeaveOneGroupOut(), n_jobs=16)
        # average across
        score_time_avg = np.average(scores,0)
        
        # rescale
        times = 1e3 * epochX.times # to have times in ms

        # store data for later group plots and statistics              
        data_container_time[p,xDecode_flg,:] = uniform_filter1d(score_time_avg, 9) # smooth w/ 9-point running average

# save data for later group analysis
out_pth_tmp = os.path.join(pth_experiment, 'data', 'exported', 'WithinDecoding')
if not os.path.exists(out_pth_tmp):
    os.makedirs(out_pth_tmp)
np.save(os.path.join(out_pth_tmp, 'PsxDecodeFlgxTimes_SVC.npy'), data_container_time)

#%% ###### group plot ########
for xDecode_flg in range(2):
     
     if xDecode_flg==0:
         cond_lbl_figTitle = 'leave-one-out: building|box' 
         cond_lbl_fn = 'buildingbox_leaveoneOut' 
     if xDecode_flg==1:
        cond_lbl_figTitle = 'leave-one-out: tool|instrument'  
        cond_lbl_fn = 'toolinstrument_leaveoneOut'
    
     pair_grp = data_container_time[:, xDecode_flg, :]
    
     # statistics inference (TFCE corrected)
     # Calculate statistical thresholds
     pair_grp_1samp = pair_grp - 0.5 # compare with 0.5
     pair_grp_1samp = pair_grp_1samp[..., np.newaxis]
     #pair_grp_1samp = np.log2(pair_grp_1samp+1)
     
     print('Clustering...')
     #threshold = -1 * st.distributions.t.ppf(q=0.05 / 2, df=len(pair_grp_1samp) - 1)
     #threshold = dict(start=-2, step=0.01) 
     thresh = -1 * st.distributions.t.ppf(q=0.05, df=pair_grp_1samp.shape[0] - 1) # single-side
     tvals, clusters, pvals, H0 = spatio_temporal_cluster_1samp_test(pair_grp_1samp, threshold=thresh, adjacency=None, tail=1, \
                                                                     max_step=9, n_jobs=8, n_permutations=10000)
     
     
     # average across participants
     score_time_avg = np.average(pair_grp, 0)
     ci = st.t.interval(alpha=0.95, df=len(pair_grp)-1, loc=np.mean(pair_grp,0), scale=st.sem(pair_grp,0)) 
     ci = (ci[1] - ci[0])/2
     
     #ci = st.sem(pair_grp, 0)
    
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
             plt.axvspan(times[c[0]], times[c[-1]], ymin=0.7, ymax=0.7, color='red', alpha=0.5)
             print('Cluster from %f ms to %f ms, p-value %f\n' % (times[c[0]], times[c[-1]], p_val))
    
     plt.xlabel('Times (ms)')
     plt.ylabel('AUC score')
     plt.ylim([0.4, 0.7])
     plt.title('%s\n%s; group' %(cond_lbl_figTitle, chn_lbl))
     plt.gca().spines["top"].set_visible(False)
     plt.gca().spines["right"].set_visible(False)
     plt.gca().spines["bottom"].set_visible(False)
     # save fig
     fig_pth_tmp = '%s\\WithinDecoding_grp\\%s\\' % \
     (pth_fig,chn_lbl)
     if not os.path.exists(fig_pth_tmp):
         os.makedirs(fig_pth_tmp)
     fig_fm_tmp = '%s_filt0.05-100_%s_SVC_125hz_auc.svg' % \
     (cond_lbl_fn, chn_lbl)
     plt.savefig(fig_pth_tmp + fig_fm_tmp)
     
     tocsv = np.stack((score_time_avg, ci), axis=-1)
     np.savetxt((fig_pth_tmp+'%s_bb2sc.csv' % cond_lbl_fn), tocsv, delimiter=",")

     plt.close('all')







#%% ###### permutation ########
N = 10000 # number of permutations

# categories to permute
perm_category = ['building', 'box']

# labels to permute
perm_labels = np.array([ID[i] for i in perm_category])

data_container_null = np.zeros([N, n_subj, 125])
for n in range(N):
    ## p loop
    for p in range(n_subj):
        
        # pnum
        p_num = epoch_all_ls[p][0:4]
        p_file = epoch_all_ls[p]
        print('** running participant %s,\nfile: %s **' % (p_num, p_file))
        
        # load stored epochs 
        epoch = mne.read_epochs(os.path.join(pth_epoch, p_file), preload=True)
        
        # channels
        chn_lbl = ['allChannels','postChannels'][p_channel_flg]
        if p_channel_flg==1:  # 0=all channels (default);   1=posterior channels
              epoch = epoch.pick_channels(chs_post)
        print('** channels setup for decoding: %s **' % chn_lbl)
        
        # permute at trials level
        for i in range(perm_labels.shape[1]):
            permed = np.random.permutation(epoch.events[((epoch.events[:,2] == perm_labels[:,i][0]) | (epoch.events[:,2] == perm_labels[:,i][1]))][:,2])
            epoch.events[((epoch.events[:,2] == perm_labels[:,i][0]) | (epoch.events[:,2] == perm_labels[:,i][1])), 2] = permed 
        
        # loop across x-decode conditions
        epochX = epoch[perm_category]   
        
        # assign labels
        events = epochX.events
        y_events = mne.merge_events(events, ID['building'], 1)
        y_events = mne.merge_events(y_events, ID['box'], 2)
        # assign cv groups
        grp_events = events
        for i in range(8):
               grp_events = mne.merge_events(grp_events, [101+i, 111+i], 1+i)
        
        # do within-pair decoding
        # prepare the data
        X = epochX.get_data()
        y = y_events[:,2]
        grp = grp_events[:,2]
        # the pipeline
        clf = make_pipeline(StandardScaler(),  SVC())
        # the sliding estimator
        time_decod = SlidingEstimator(clf, n_jobs=16, scoring='roc_auc', verbose=True)
        # score from cv scheme
        scores = cross_val_multiscore(time_decod, X, y, grp, cv=models.LeaveOneGroupOut(), n_jobs=16)
        # average across
        score_time_avg = np.average(scores,0)
        # store data for later group plots and statistics              
        data_container_null[n, p, :] = uniform_filter1d(score_time_avg, 9) # smooth w/ 9-point running average
        

# save data for later group analysis
out_pth_tmp = os.path.join(pth_experiment, 'data', 'exported', 'null_data_container')
if not os.path.exists(out_pth_tmp):
    os.makedirs(out_pth_tmp)
np.save(os.path.join(out_pth_tmp, 'within_svc.npy'), data_container_null)

#%% ###### t-mass estimate #######
# get null distribution
Nulldistribution = np.zeros(N)
for n in range(N):
    data_null = data_container_null[n, :, :] - 0.5 # compare with 0.5
    
    start_del = np.arange(25) # first -200 ms
    end_del = np.arange(125-4, 125) # last 32 ms
    to_del = np.concatenate((start_del, end_del))
    data_null = np.delete(data_null, to_del, axis=1) # delete begin and end
    
    data_null = data_null[..., np.newaxis]
    thresh = -1 * st.distributions.t.ppf(q=0.05, df=data_null.shape[0] - 1) # single-side
    tvals, clusters, pvals, H0 = spatio_temporal_cluster_1samp_test(data_null, threshold=thresh, \
                                                                    adjacency=None, \
                                                                    tail=1, n_permutations=1)
    # find tmass
    tmass = []
    if len(clusters) == 0:
        tmass = [0]
    else:
        for cluster in clusters:
            tmass.append(tvals[cluster[0]].sum())
    
    # null distribution of tmass
    Nulldistribution[n] = max(tmass)
    
# get cluster-level critical t-mass at 95% 1-sided
tmass_critical = np.percentile(Nulldistribution, 95)

#%% ######## significance test #######
# get significance of each cluster
xDecode_flg = 0
     
if xDecode_flg==0:
    cond_lbl_figTitle = 'RSA: building|box' 
    cond_lbl_fn = 'buildingbox_RSA' 
if xDecode_flg==1:
   cond_lbl_figTitle = 'RSA: tool|instrument'  
   cond_lbl_fn = 'toolinstrument_RSA'
   
pair_grp = data_container_time[:, xDecode_flg, :]
   
# statistics inference (TFCE corrected)
# Calculate statistical thresholds
pair_grp_1samp = pair_grp - 0.5 # compare with 0.5

start_del = np.arange(25) # first -200 ms
end_del = np.arange(125-4, 125) # last 32 ms
to_del = np.concatenate((start_del, end_del))
pair_grp_1samp = np.delete(pair_grp_1samp, to_del, axis=1) # delete begin and end

pair_grp_1samp = pair_grp_1samp[..., np.newaxis]
#pair_grp_1samp = np.log2(pair_grp_1samp+1)

print('Clustering...')
tvals_obs, clusters_obs, pvals_obs, H0_obs = spatio_temporal_cluster_1samp_test(pair_grp_1samp, threshold=thresh, \
                                                                adjacency=None, \
                                                                tail=1, n_permutations=1)
# compute t-mass and p-value
tmasses_obs = []
pvals_obs = []
if len(clusters_obs) == []:
    print("################## No cluster survived at thresh %.3f" % thresh)
else:
    for k, cluster_obs in enumerate(clusters_obs):
        tmasses_obs.append(tvals_obs[cluster_obs[0]].sum())
        pvalue = np.sum(Nulldistribution >= tmasses_obs[k])/N
        pvals_obs.append(pvalue)
        print("#### cluster: {}, tmass: {}, p-value: {}".format(cluster_obs, \
                                                                  tmasses_obs[k], \
                                                                  pvals_obs[k]))

#%% ###### group plot #######
if xDecode_flg==0:
    cond_lbl_figTitle = 'Classification: building|box' 
    cond_lbl_fn = 'buildingbox_classification' 
if xDecode_flg==1:
   cond_lbl_figTitle = 'RSA: tool|instrument'  
   cond_lbl_fn = 'toolinstrument_RSA'

# average across participants
score_time_avg = np.average(pair_grp, 0)
ci = st.t.interval(alpha=0.95, df=len(pair_grp)-1, loc=np.mean(pair_grp,0), scale=st.sem(pair_grp,0)) 
ci = (ci[1] - ci[0])/2

   
## group plot 
plt.plot(times, score_time_avg, label="Classifi. score")
plt.axhline(0.5, color='k', linestyle='--', label="Chance level")
plt.axvline(0, color='k', label='stim onset')
plt.legend() 
plt.fill_between(times, (score_time_avg-ci), (score_time_avg+ci), color='b', alpha=0.5)
# shading significant temporal clusters
print('** %s **\n' % (cond_lbl_figTitle))
clusters = [[int(y) for y in x[0]] for x in clusters_obs]   

for k, c in enumerate(clusters):
    if pvals_obs[k] <= 0.05:
        plt.axvspan(times[c[0]+25], times[c[-1]+25], color='red', alpha=0.5) 
        print('Cluster from %f ms to %f ms, p-value %f\n' % (times[c[0]+25], times[c[-1]+25], pvals_obs[k]))
   
plt.xlabel('Times (ms)')
plt.ylabel('AUC score')
plt.ylim([0.2, 0.8])
plt.title('%s\n%s; group' %(cond_lbl_figTitle, chn_lbl))
# save fig
fig_pth_tmp = '%s\\WithinDecoding_grp\\%s\\' % \
(pth_fig,chn_lbl)
if not os.path.exists(fig_pth_tmp):
    os.makedirs(fig_pth_tmp)
fig_fm_tmp = '%s_filt0.05-100_%s_125hz_auc.png' % \
(cond_lbl_fn, chn_lbl)
plt.savefig(fig_pth_tmp + fig_fm_tmp, dpi=900)

    
    


    
    
    
    