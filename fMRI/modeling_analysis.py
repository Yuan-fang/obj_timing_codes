#!/usr/bin/env python3
"""
This script averaged the HRF across different runs for each participant.
It also creates plots for each roi in each participant.

@author: Yuanfang Zhao
Created on Sun Mar 21 15:37:56 2021
Revised on Jun 17, 2024
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from hrf import *


# define the fitting model
def single_gamma_hrf(t, A=4, W=3.9, K=1.0):

    """
    A: peak latency
    W: full width at half maximum
    K: scaling factor
    """

    A = float(A)
    W = float(W)
    K = float(K)

    return \
            K * (t / A) ** ((A ** 2) / (W ** 2) * 8.0 * np.log(2.0)) \
            * np.e ** ((t - A) / -((W ** 2) / A / 8.0 / np.log(2.0)))


# define the model-fitting function
def fit_model_and_extract_params(xv, mean_data):
    """
    xv: the timescore vector
    mean_data: the group averaged hrf

    """
    ind = (xv >= 0)
    xv_pos = xv[ind]
    tc = mean_data[ind]
    best_vals, covar = curve_fit(single_gamma_hrf, xv_pos, tc,
                                 bounds=((3.8, 1.8, 0), (8.6, 6, np.inf)), maxfev=50000)
    # get the residual unexplained by the model
    residuals = tc - single_gamma_hrf(xv_pos, *best_vals)
    # compute r-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((tc - np.mean(tc)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return best_vals, r_squared

# define the statistics for inference
def statistic(x, y, type='TTP'):
    """
    x, y: the to-be compared individual hrf under two conditions
    type: the parameter of hrf
    """
    mean_x = np.mean(x, axis=0)
    mean_y = np.mean(y, axis=0)
    # apply model-fitting to x
    best_vals_x, r_squared_x = fit_model_and_extract_params(xv, mean_x)
    # assign estimates of x to different labels
    TTP_x = best_vals_x[0]
    FWHM_x = best_vals_x[1]
    Amp_x = best_vals_x[2]

    # apply model-fitting to y
    best_vals_y, r_squared_y = fit_model_and_extract_params(xv, mean_y)
    # assign estimates of y to different labels
    TTP_y = best_vals_y[0]
    FWHM_y = best_vals_y[1]
    Amp_y = best_vals_y[2]

    # compute statistic
    if type == 'TTP':
        return TTP_x - TTP_y
    elif type == 'FWHM':
        return FWHM_x - FWHM_y
    elif type == 'Amp':
        return Amp_x - Amp_y
    elif type == 'Gof':
        return r_squared_x - r_squared_y


# define project dir
basedir = '/project/3018040.06/nii'

# get subject ID list
sublist = '/project/3018040.06/scripts/subjinfo/sessid'
f = open(sublist)
subjs = [line.strip() for line in f]

# define the data folder
sessstr = 'obj'

# suffix of the to-be-processed file
suffix = 'runsaverage_WL0_average_N30'

# define the roi
rois = ['lPPA']

# define the condition file
cond_f = '/project/3018040.06/scripts/analysis/design/obj.par'

# define the output folder
savepath = os.path.join(basedir, 'grpResults')
os.makedirs(savepath, exist_ok='true')


# loop through each roi
for roi in rois:
    # placeholder for grouping:
    # coi (condition of interest, i.e., building);
    # con (condition of non-interest, i.e., box);
    # coc (condition used for control: i.e., scene)
    # cha (chair)
    grp_coi = {}
    grp_con = {}
    grp_coc = {}
    grp_cha = {}
    # iterate across all subjects
    for k, sub in enumerate(subjs):
        # path of individual hrf file
        f = os.path.join(basedir, sub, sessstr, 'ROIrunsAVG', '{}_{}.csv'.format(roi, suffix))
        # read individual hrf data
        roi_avg = np.genfromtxt(f, delimiter=',')

        # define the timescale for later plot purpose
        hrfObj = HrfRetrieval(roi_avg[0, :], cond_f, 0.14, 0.14, [-4, 15])
        xv = hrfObj.tscale

        # conditional waveform: 0-building, 1-box, 2-tool, 3-instrument, 4-scene, 5-hand
        if roi == 'lLO':
            coi = roi_avg[2, :]
            con = roi_avg[3, :]
            coc = roi_avg[5, :]
            cha = roi_avg[6, :]
        else:
            coi = roi_avg[0, :]
            con = roi_avg[1, :]
            coc = roi_avg[4, :]
            cha = roi_avg[6, :]

        # dict for each subject for each condition
        grp_coi["{0}".format(sub)] = coi
        grp_con["{0}".format(sub)] = con
        grp_coc["{0}".format(sub)] = coc
        grp_cha["{0}".format(sub)] = cha

    # from dict to array
    grp_coi = np.array(list(grp_coi.values()))
    grp_con = np.array(list(grp_con.values()))
    grp_coc = np.array(list(grp_coc.values()))
    grp_cha = np.array(list(grp_cha.values()))
    # grp mean and standard deviation
    mean_coi = np.mean(grp_coi, axis=0)
    error_coi = grp_coi.std(axis=0) / np.sqrt(grp_coi.shape[0])
    mean_con = np.mean(grp_con, axis=0)
    error_con = grp_con.std(axis=0) / np.sqrt(grp_con.shape[0])
    mean_coc = np.mean(grp_coc, axis=0)
    error_coc = grp_coc.std(axis=0) / np.sqrt(grp_coc.shape[0])
    mean_cha = np.mean(grp_cha, axis=0)
    error_cha = grp_cha.std(axis=0) / np.sqrt(grp_cha.shape[0])
    # difference wave for building vs. box
    grp_diffI = grp_coi - grp_con
    # difference wave for scene vs. chair
    grp_diffC = grp_coc - grp_cha
    # grp diff for building vs. box
    mean_diffI = np.mean(grp_diffI, axis=0)
    error_diffI = grp_diffI.std(axis=0) / np.sqrt(grp_diffI.shape[0])
    # grp diff for scene vs. chair
    mean_diffC = np.mean(grp_diffC, axis=0)
    error_diffC = grp_diffC.std(axis=0) / np.sqrt(grp_diffC.shape[0])

    # model-fitting
    # dict for all the conditions
    conditions = {
        'Condition of interest': mean_coi,
        'Condition of not interest': mean_con,
        'Condition of control': mean_coc,
        'Condition of chair': mean_cha,
        'Condition of coi - con': mean_diffI,
        'Condition of coc - cha': mean_diffC
    }

    # get the estimates for each condition and store it into a dict
    results = {}
    for condition_name, mean_data in conditions.items():
        best_vals, r_squared = fit_model_and_extract_params(xv, mean_data)
        results[condition_name] = {
            'TTP': best_vals[0],
            'FWHM': best_vals[1],
            'Amp': best_vals[2],
            'r_squared': r_squared,
            'best_vals': best_vals
        }
        # print the results out
        print(f'{condition_name} explained: {r_squared}')
        print(f'{condition_name} amplitude: {best_vals[2]}')
        print(f'{condition_name} latency: {best_vals[0]}')
        print(f'{condition_name} FWHM: {best_vals[1]}')

    # do permutation
    # construct null distribution
    # number of permutations
    n = 1000
    # placeholder for statistic under zero hypothesis
    stats_dist = np.zeros(n)
    # specify conditions to be compared (tested)
    A = grp_coi
    B = grp_con
    # specify the parameter of hrf ('TTP', 'FWHM', 'Amp')
    meas = 'Amp'
    # iterate to compute the statistic
    for i in range(n):
        # contruct 2 random shuffled variables
        grp_3d = np.dstack((A, B))
        grp_3d = np.moveaxis(grp_3d, 2, 0)
        for c in range(grp_3d.shape[1]):
            np.random.shuffle(grp_3d[:, c, :])
        grp_A_Samp = grp_3d[0, :, :]
        grp_B_Samp = grp_3d[1, :, :]
        # compute the statistic
        stats_dist[i] = statistic(grp_A_Samp, grp_B_Samp, type=meas)
    # compute p-value (2-sided)
    stats = statistic(A, B, type=meas)
    pval = np.sum(np.abs(stats_dist) >= np.abs(stats)) / n
    print('ROI: {0}, type: {1}, permutation p value:{2}'.format(roi, meas, pval))


    # plot
    # get index of positive time points
    ind = (xv >= 0)
    xv_pos = xv[ind]
    # get parameter estimates of parameters for different conditions
    best_vals_coi = results['Condition of interest']['best_vals']
    best_vals_con = results['Condition of not interest']['best_vals']
    best_vals_coc = results['Condition of control']['best_vals']
    best_vals_cha = results['Condition of chair']['best_vals']
    # plot the fitted hrf response and raw hrf response
    plt.scatter(xv, mean_coi, facecolors='#D26027', edgecolors='none', alpha=0.4, s=10, label='building')
    plt.plot(xv_pos, single_gamma_hrf(xv_pos, *best_vals_coi), color='#D26027', alpha=1, linestyle='-')
    plt.fill_between(xv, mean_coi - error_coi, mean_coi + error_coi, alpha=0.2, facecolor='#D26027')

    plt.scatter(xv, mean_con, facecolors='#59B3E4', edgecolors='none', alpha=0.4, s=10, label='box')
    plt.plot(xv_pos, single_gamma_hrf(xv_pos, *best_vals_con), color='#59B3E4', alpha=1, linestyle='-')
    plt.fill_between(xv, mean_con - error_con, mean_con + error_con, alpha=0.2, facecolor='#59B3E4')

    plt.scatter(xv, mean_coc, facecolors='g', edgecolors='none', alpha=0.4, s=10, label='scene')
    plt.plot(xv_pos, single_gamma_hrf(xv_pos, *best_vals_coc), color='g', alpha=1, linestyle='-')
    plt.fill_between(xv, mean_coc - error_coc, mean_coc + error_coc, alpha=0.2, facecolor='g')

    plt.scatter(xv, mean_cha, facecolors='y', edgecolors='none', alpha=0.4, s=10, label='chair')
    plt.plot(xv_pos, single_gamma_hrf(xv_pos, *best_vals_cha), color='y', alpha=1, linestyle='-')
    plt.fill_between(xv, mean_cha - error_cha, mean_cha + error_cha, alpha=0.2, facecolor='y')

    plt.axhline(0, color='k', linestyle='-', linewidth=0.5)
    plt.xlabel('time(s)')
    plt.ylabel('BOLD')
    plt.title('Averaged response of different conditions for {0} (N={1})'.format(roi, grp_coi.shape[0]))
    plt.legend()
    ax = plt.gca()
    ax.set_ylim([-0.002, 0.009])
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.savefig(os.path.join(savepath, '{}_3conditions_{}.svg'.format(roi, suffix)))
    plt.clf()

    # plot the fitted difference wave and raw difference wave
    plt.scatter(xv, mean_diffI, facecolors='r', edgecolors='none', alpha=0.4, s=10, label='building-box')
    plt.axhline(0, color='k', linestyle='-', linewidth=0.5)
    plt.plot(xv_pos, single_gamma_hrf(xv_pos, *best_vals_diffI), color='r', linestyle='-')
    plt.fill_between(xv, mean_diffI - error_diffI, mean_diffI + error_diffI, alpha=0.2, facecolor='r')

    plt.scatter(xv, mean_diffC, facecolors='b', edgecolors='none', alpha=0.4, s=10, label='scene-chair')
    plt.plot(xv_pos, single_gamma_hrf(xv_pos, *best_vals_diffC), color='b', linestyle='-')
    plt.fill_between(xv, mean_diffC - error_diffC, mean_diffC + error_diffC, alpha=0.2, facecolor='b')

    plt.xlabel('time(s)')
    plt.ylabel('BOLD')
    plt.title('Averaged response of different conditions for {0} (N={1})'.format(roi, grp_coi.shape[0]))
    plt.legend()
    ax = plt.gca()
    ax.set_ylim([-0.002, 0.009])
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.savefig(os.path.join(savepath, '{}_diffconditions_{}.svg'.format(roi, suffix)))
    plt.clf()
