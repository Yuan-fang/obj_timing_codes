# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 21:18:28 2022

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
from scipy.ndimage import uniform_filter
from scipy.stats import permutation_test


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

data_container_time = np.zeros([n_subj, 125, 125])  # shape: n_subk, "category" decoding (i.e., 0=building/box; 1=tool/instrument), # times

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

    # extract epochs
    epochX = epoch['building', 'scene', 'box', 'chair']
    # loop through event 
    # erp = []
    # exemps = []
    # for i in epochX.event_id:
    #     erp.append(epochX[i].average().data)
    #     exemps.append(epochX.event_id[i])     
    # events = np.zeros((len(exemps), 3))
    # events[:, 2] = exemps
    # X = np.array(erp)              
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
    time_decod = GeneralizingEstimator(clf, n_jobs=16, scoring='roc_auc', verbose=True)
    # score from cv scheme
    scores = cross_val_multiscore(time_decod, X, y, grp, cv=models.PredefinedSplit(test_fold=grp), n_jobs=16)
    # as just 1 fold, so not need to do average 
    score_time_avg = np.squeeze(scores)
    
    # rescale
    times = 1e3 * epochX.times # to have times in ms
    
    # store data for later group plots and statistics              
    data_container_time[p, :, :] = uniform_filter(score_time_avg, 9)

# save data for later group analysis
out_pth_tmp = os.path.join(pth_experiment, 'data', 'exported', 'CrossDecoding')
if not os.path.exists(out_pth_tmp):
    os.makedirs(out_pth_tmp)
np.save(os.path.join(out_pth_tmp, 'PsxDecodeFlgxTimesxTimes_SVC_{}.npy'.format(cond_lbl_fn)), data_container_time)

#%% ########################### group plot (temporal generalization) ###################
cond_lbl_fn = 'buildingBox_cross-decoding_sceneChair'
cond_lbl_figTitle = 'building|Box > scene|chair (temporal generalization)'
   
pair_grp = data_container_time

# statistics inference (TFCE corrected)
# Calculate statistical thresholds
pair_grp_1samp = pair_grp - 0.5 # compare with 0.5

print('Clustering...')
threshold = -1 * st.distributions.t.ppf(q=0.05/2, df=len(pair_grp_1samp) - 1)
#threshold = dict(start=-2, step=0.01) 
tvals, clusters, pvals, H0 = permutation_cluster_1samp_test(pair_grp_1samp, threshold=threshold, adjacency=None, n_jobs=8, tail=1, n_permutations=10000, out_type='mask')

# Create new stats image with only significant clusters
T_obs_plot = np.nan * np.ones_like(tvals) 
for c, p_val in zip(clusters, pvals): 
    if p_val <= 0.05: 
        T_obs_plot[c] = tvals[c]
        
# edge detection for significant cluster (for later symmetrical testing)
index = np.where(~np.isnan(T_obs_plot))
x0 = index[0].min() 
x1 = index[0].max()
y0 = index[1].min()
y1 = index[1].max()
fov = np.arange(min(x0,y0), max(x1,y1)+1, 1)

# constructing symmetricity measures
data_temp = np.moveaxis(pair_grp, 0, 2)
symData = data_temp[np.ix_(fov, fov)]
symData = np.moveaxis(symData, 2, 0)
upper = np.tril(symData, k=-1)
lower = np.triu(symData, k=1) 
lower_new = np.moveaxis(lower, [2, 1], [1, 2])
diffscore = upper - lower_new

# contruct the upper diagonal matrix mask for statistical testing
temp_mas = np.triu(np.ones((diffscore.shape[1], diffscore.shape[2]), dtype=bool))
mas = np.broadcast_to(temp_mas, diffscore.shape)


print('Clustering...')
threshold_sym = -1 * st.distributions.t.ppf(q=0.05, df=len(diffscore) - 1)
#threshold = dict(start=-2, step=0.01) 
tvals_sym, clusters_sym, pvals_sym, H0_sym = permutation_cluster_1samp_test(diffscore, threshold=threshold_sym, adjacency=None, n_jobs=8, \
                                                                            tail=1, n_permutations=10000, out_type='mask', exclude=mas[0])

# average across participants
score_time_avg = np.average(pair_grp, 0)

# compute t-mass and p-value
if len(clusters_sym) == 0:
   print("################## No cluster survived at thresh %.3f" % threshold_sym)
else:
   for k, cluster_sym in enumerate(clusters_sym):
       print("#### cluster: {}, p-value: {}".format(cluster_sym, \
                                                    pvals_sym[k])) 
    # plot
    
   cluster_plot = np.full(score_time_avg.shape, False)
   mask = np.full(cluster_sym.shape, False)
   for k, c in enumerate(clusters_sym):
       if pvals_sym[k] <= 0.05:
           mask = mask + clusters_sym[k]
   cluster_plot[np.ix_(fov, fov)] = mask
    
   fig, ax = plt.subplots()
    
   score_time_diff = score_time_avg - np.rot90(np.fliplr(score_time_avg))
    
   main = ax.imshow(score_time_diff, interpolation='lanczos', origin='lower', cmap='RdBu_r', extent=times[[0, -1, 0, -1]], vmin=-0.05, vmax=0.05) 
   ax.contour(cluster_plot, [0], colors='red', origin='lower', linewidths=1, extent=times[[0, -1, 0, -1]])            
   ax.set_xlabel('Testing Time (s)')
   ax.set_ylabel('Training Time (s)')
   ax.set_title('Temporal generalization') 
   ax.axvline(0, color='k') 
   ax.axhline(0, color='k') 
   ax.plot(times[[0,-1]],times[[0,-1]], color='k')
   cbar = ax.figure.colorbar(main) 
   cbar.set_label('AUC')
   ax.set_title('%s\n%s; group' %(cond_lbl_figTitle, chn_lbl))
   # save fig 
   # save fig
   fig_pth_tmp = '%s\\CrossDecoding_grp\\%s\\' % \
   (pth_fig,chn_lbl)
   if not os.path.exists(fig_pth_tmp):
       os.makedirs(fig_pth_tmp)
   fig_fm_tmp = '%s_filt0.05-100_%s_125hz_SVC_generalization_symmetricity1.svg' % \
   (cond_lbl_fn, chn_lbl)
   fig.savefig(fig_pth_tmp + fig_fm_tmp)
     
   fig.clf()
   
   # compute centroid of the cluster
   cluster_coords = np.where(cluster_plot)
   row_ind = cluster_coords[0]
   col_ind = cluster_coords[1]
   centroid_row = np.mean(times[row_ind])
   centroid_col = np.mean(times[col_ind])
   print(f"Centroid of the cluster: row = {centroid_row} ms; column = {centroid_col} ms")
   print(f"Difference between row and column is {centroid_row - centroid_col} ms")



#%% ############################# group plot (temporal generalization) ##############################
pair_grp = data_container_time

# statistics inference (TFCE corrected)
# Calculate statistical thresholds
pair_grp_1samp = pair_grp - 0.5 # compare with 0.5

print('Clustering...')
threshold = -1 * st.distributions.t.ppf(q=0.05, df=len(pair_grp_1samp) - 1)
#threshold = dict(start=-2, step=0.01) 
tvals, clusters, pvals, H0 = permutation_cluster_1samp_test(pair_grp_1samp, threshold=threshold, adjacency=None, n_jobs=8, tail=1, n_permutations=10000, out_type='mask')



# Create new stats image with only significant clusters
T_obs_plot = np.nan * np.ones_like(tvals) 
for c, p_val in zip(clusters, pvals): 
    if p_val <= 0.05: 
        T_obs_plot[c] = tvals[c]
        
# edge detection for significant cluster (for later symmetrical testing)
index = np.where(~np.isnan(T_obs_plot))
x0 = index[0].min() 
x1 = index[0].max()
y0 = index[1].min()
y1 = index[1].max()
fov = np.arange(min(x0,y0), max(x1,y1)+1, 1)


cluster_plot = np.full(score_time_avg.shape, False)
mask = np.full(score_time_avg.shape, False)
for k, c in enumerate(clusters):
    if pvals[k] <= 0.05:
        mask = mask + clusters[k]
#cluster_plot[np.ix_(fov, fov)] = mask

# constructing symmetricity measures
data_temp = np.moveaxis(pair_grp, 0, 2)
symData = data_temp[np.ix_(fov, fov)]
symData = np.moveaxis(symData, 2, 0)
upper = np.tril(symData, k=-1).sum(axis=(1,2))
lower = np.triu(symData, k=1).sum(axis=(1,2))     
        
# permutation test on symmetrcity
def statistic(x, y):
   return np.mean(x) - np.mean(y)

res = permutation_test((upper,lower), statistic, permutation_type='samples', n_resamples=10000, alternative='two-sided')
print('#################The difference between upper tri and lower tri for {}:{}'.format(cond_lbl_fn, res.statistic))
print('#################The p value between upper tri and lower tri for {}:{}'.format(cond_lbl_fn, res.pvalue))        

# average across participants
score_time_avg = np.average(pair_grp, 0)

## group plot 
fig, ax = plt.subplots()

main = ax.imshow(score_time_avg, interpolation='lanczos', origin='lower', cmap='RdBu_r', extent=times[[0, -1, 0, -1]], vmin=0.45, vmax=0.55)
ax.contour(mask, [0], colors='k', origin='lower', linewidths=0.5, linestyles='dashed', extent=times[[0, -1, 0, -1]]) 
#plt.imshow(T_obs_plot, origin='lower', cmap=plt.cm.RdBu_r, extent=times[[0, -1, 0, -1]], vmin=0.4, vmax=0.65, alpha=0.2)
#rectangle = plt.Rectangle((times[fov[0]], times[fov[0]]), times[fov[-1]]-times[fov[0]], times[fov[-1]]-times[fov[0]], fill=False, ec="red") 
#ax.add_patch(rectangle)
ax.set_xlabel('Testing Time (ms)')
ax.set_ylabel('Training Time (ms)')
ax.set_title('Temporal generalization') 
#ax.axvline(0, color='k') 
#ax.axhline(0, color='k') 
#ax.plot(times[[0,-1]],times[[0,-1]], color='white', linewidth=0.5, linestyle='dashdot')
cbar = ax.figure.colorbar(main) 
cbar.set_label('AUC')
ax.set_title('%s\n%s; group' %(cond_lbl_figTitle, chn_lbl))
# save fig
fig_pth_tmp = '%s\\CrossDecoding_grp\\%s\\' % \
(pth_fig,chn_lbl)
if not os.path.exists(fig_pth_tmp):
    os.makedirs(fig_pth_tmp)
fig_fm_tmp = '%s_filt0.05-100_%s_125hz_SVC_generalization.svg' % \
(cond_lbl_fn, chn_lbl)
plt.savefig(fig_pth_tmp + fig_fm_tmp)

plt.close('all')

#%% permutation test 

# point by point testing
def statistic_timepoint(x, y):
    return x - y
   
# construct null distribution
N = 10000 # number of permutations
null_container_sub = np.zeros((symData.shape[0], int(symData.shape[1]*(symData.shape[2] - 1)/2), N))
test_container_sub = np.zeros((symData.shape[0], int(symData.shape[1]*(symData.shape[2] - 1)/2)))
for sub in range(len(symData)):
   print('######Permuting participant: {}#####'.format(sub))
   sym_i = symData[sub,:,:]
   upper_i = sym_i[np.tril_indices(len(sym_i), k=-1)]
   sym_i_diagflip = np.rot90(np.fliplr(sym_i))
   lower_i = sym_i_diagflip[np.tril_indices(len(sym_i_diagflip), k=-1)]
   res_p = permutation_test((upper_i,lower_i), statistic_timepoint, permutation_type='samples', n_resamples=N, alternative='two-sided')
   null_container_sub[sub,:,:] = res_p.null_distribution.transpose()
   test_container_sub[sub, :] = res_p.statistic
   
## t-mass estimate
Nulldistribution = np.zeros(N)
for n in range(N):
   print('##### permutation n = {}'.format(n))
   null_sub = null_container_sub[:, :, n]
   null_sub_2d = np.zeros((symData.shape[0], symData.shape[1], symData.shape[2]))
   for sub in range(symData.shape[0]):
       sub_1d = null_sub[sub, :]
       sub_2d = np.zeros((symData.shape[1], symData.shape[2]))
       sub_2d[np.tril_indices(sub_2d.shape[0], k = -1)] = sub_1d
       null_sub_2d[sub, :, :] = uniform_filter(sub_2d, 9)
     
   thresh = -1 * st.distributions.t.ppf(q=0.05/2, df=null_sub_2d.shape[0] - 1) # two-sided
   tvals, clusters, pvals, H0 = spatio_temporal_cluster_1samp_test(null_sub_2d, threshold=thresh, \
                                                               adjacency=None, \
                                                               tail=0, n_permutations=1)
   # find tmass
   tmass = []
   if len(clusters) == 0:
       tmass = [0]
   else:
       for cluster in clusters:
           tmass.append(abs(tvals[cluster].sum()))
           
   # null distribution of tmass
   Nulldistribution[n] = max(tmass)
    
# get cluster-level critical t-mass at 95% 1-sided
tmass_critical = np.percentile(Nulldistribution, 95)
       
# actuall laterization index
test_sub_2d = np.zeros((symData.shape[0], symData.shape[1], symData.shape[2]))
for sub in range(symData.shape[0]):
   sub_1d = test_container_sub[sub, :] 
   sub_2d = np.zeros((symData.shape[1], symData.shape[2]))
   sub_2d[np.tril_indices(sub_2d.shape[0], k = -1)] = sub_1d
   test_sub_2d[sub, :, :] = sub_2d
     
tvals_obs, clusters_obs, pvals_obs, H0_obs = spatio_temporal_cluster_1samp_test(test_sub_2d, threshold=thresh, \
                                                                               adjacency=None, \
                                                                               tail=1, n_jobs=8, n_permutations=10000, out_type='mask') 
   
# compute t-mass and p-value
tmasses_obs = []
pvals_obs = []
if len(clusters_obs) == 0:
   print("################## No cluster survived at thresh %.3f" % thresh)
else:
   for k, cluster_obs in enumerate(clusters_obs):
       tmasses_obs.append(tvals_obs[cluster_obs].sum())
       pvalue = np.sum(Nulldistribution >= tmasses_obs[k])/N
       pvals_obs.append(pvalue)
       print("#### cluster: {}, tmass: {}, p-value: {}".format(cluster_obs, \
                                                                 tmasses_obs[k], \
                                                                 pvals_obs[k])) 
    # plot
    
   cluster_plot = np.full(score_time_avg.shape, False)
   mask = np.full(cluster_obs.shape, False)
   for k, c in enumerate(clusters_obs):
       if pvals_obs[k] <= 0.05:
           mask = mask + clusters_obs[k]
   cluster_plot[np.ix_(fov, fov)] = mask
    
   fig, ax = plt.subplots()
    
   score_time_diff = score_time_avg - np.rot90(np.fliplr(score_time_avg))
    
   main = ax.imshow(score_time_diff, interpolation='lanczos', origin='lower', cmap='RdBu_r', extent=times[[0, -1, 0, -1]], vmin=-0.05, vmax=0.05) 
   ax.contour(cluster_plot, [0], colors='red', origin='lower', linewidths=1, extent=times[[0, -1, 0, -1]])            
   ax.set_xlabel('Testing Time (s)')
   ax.set_ylabel('Training Time (s)')
   ax.set_title('Temporal generalization') 
   ax.axvline(0, color='k') 
   ax.axhline(0, color='k') 
   ax.plot(times[[0,-1]],times[[0,-1]], color='k')
   cbar = ax.figure.colorbar(main) 
   cbar.set_label('AUC')
   ax.set_title('%s\n%s; group' %(cond_lbl_figTitle, chn_lbl))
   # save fig 
   # save fig
   fig_pth_tmp = '%s\\CrossDecoding_grp\\%s\\' % \
   (pth_fig,chn_lbl)
   if not os.path.exists(fig_pth_tmp):
       os.makedirs(fig_pth_tmp)
   fig_fm_tmp = '%s_filt0.05-100_%s_125hz_SVC_generalization_symmetricity.png' % \
   (cond_lbl_fn, chn_lbl)
   fig.savefig(fig_pth_tmp + fig_fm_tmp, dpi=300)
     
   fig.clf()