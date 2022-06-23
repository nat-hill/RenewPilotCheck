#!/usr/bin/python3
"""
 plot_hdf5.py

 Plotting from HDF5 file
 Script to analyze recorded hdf5 file from channel sounding (see Sounder/).
 Usage format is:
    ./plot_hdf5.py <hdf5_file_name>

 Example:
    ./plot_hdf5.py ../Sounder/logs/test-hdf5.py


---------------------------------------------------------------------
 Copyright © 2018-2020. Rice University.
 RENEW OPEN SOURCE LICENSE: http://renew-wireless.org/license
---------------------------------------------------------------------
"""

import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import collections
import time
from find_lts import *
from optparse import OptionParser
from channel_analysis import *
import hdf5_lib
from hdf5_lib import *
import matplotlib
#matplotlib.use("Agg")


def verify_hdf5(hdf5, frame_i=100, cell_i=0, ofdm_sym_i=0, ant_i =0,
                user_i=0, ul_sf_i=0, subcarrier_i=10, offset=-1,
                dn_calib_offset=0, up_calib_offset=0, thresh=0.001,
                deep_inspect=False, corr_thresh=0.00, exclude_bs_nodes=[],
                demodulate=False):
    """Plot data in the hdf5 file to verify contents.

    Args:
        hdf5: An hdf5_lib object.
        frame_i: The index of the frame to be plotted.
        cell_i: The index of the hub where base station is connected.
        ofdm_sym_i: The index of the reference ofdm symbol in a pilot.
        ant_i: The index of the reference base station antenna.
        user_i: The index of the reference user.
    """
    plt.close("all")

    # Retrieve attributes
    n_frm_end = hdf5.n_frm_end
    n_frm_st = hdf5.n_frm_st
    metadata = hdf5.metadata
    if 'SYMBOL_LEN' in metadata: # to support older datasets
        samps_per_slot = int(metadata['SYMBOL_LEN'])
    elif 'SLOT_SAMP_LEN' in metadata:
        samps_per_slot = int(metadata['SLOT_SAMP_LEN'])
    num_pilots = int(metadata['PILOT_NUM'])
    num_cl = int(metadata['CL_NUM'])
    prefix_len = int(metadata['PREFIX_LEN'])
    postfix_len = int(metadata['POSTFIX_LEN'])
    z_padding = prefix_len + postfix_len
    if offset < 0: # if no offset is given use prefix from HDF5
        offset = int(prefix_len)
    fft_size = int(metadata['FFT_SIZE'])
    cp = int(metadata['CP_LEN'])
    rate = int(metadata['RATE'])
    pilot_type = metadata['PILOT_SEQ_TYPE'].astype(str)[0]
    nonzero_sc_size = fft_size
    if 'DATA_SUBCARRIER_NUM' in metadata:
        nonzero_sc_size = metadata['DATA_SUBCARRIER_NUM']
    ofdm_pilot = np.array(metadata['OFDM_PILOT'])
    if "OFDM_PILOT_F" in metadata.keys():
        ofdm_pilot_f = np.array(metadata['OFDM_PILOT_F'])
    else:
        if pilot_type == 'zadoff-chu':
            _, pilot_f = generate_training_seq(preamble_type='zadoff-chu', seq_length=nonzero_sc_size, cp=cp, upsample=1, reps=1)
        else:
            _, pilot_f = generate_training_seq(preamble_type='lts', cp=32, upsample=1)
        ofdm_pilot_f = pilot_f
    reciprocal_calib = np.array(metadata['RECIPROCAL_CALIB'])
    samps_per_slot_no_pad = samps_per_slot - z_padding
    symbol_per_slot = ((samps_per_slot_no_pad) // (cp + fft_size))
    if 'UL_SYMS' in metadata:
        ul_slot_num = int(metadata['UL_SYMS'])
    elif 'UL_SLOTS' in metadata:
        ul_slot_num = int(metadata['UL_SLOTS'])
    n_ue = num_cl

    all_bs_nodes = []
    plot_bs_nodes = []
    num_frames = 0
    pilot_data_avail = len(hdf5.pilot_samples) > 0
    if pilot_data_avail:
        all_bs_nodes = set(range(hdf5.pilot_samples.shape[3]))
        plot_bs_nodes = list(all_bs_nodes - set(exclude_bs_nodes))
        pilot_samples = hdf5.pilot_samples[:, :, :, plot_bs_nodes, :]

        frm_plt = min(frame_i, pilot_samples.shape[0] + n_frm_st)
        # Verify frame_i does not exceed max number of collected frames
        ref_frame = min(frame_i - n_frm_st, pilot_samples.shape[0])

    ul_data_avail = len(hdf5.uplink_samples) > 0
    if ul_data_avail:
        uplink_samples = hdf5.uplink_samples[:, :, :, plot_bs_nodes, :]
    noise_avail = len(hdf5.noise_samples) > 0
    if noise_avail:
        noise_samples = hdf5.noise_samples[:, :, :, plot_bs_nodes, :]
    dl_data_avail = len(hdf5.downlink_samples) > 0
    if dl_data_avail:
        downlink_samples = hdf5.downlink_samples
        if not pilot_data_avail:
            frm_plt = min(frame_i, downlink_samples.shape[0] + n_frm_st)
            # Verify frame_i does not exceed max number of collected frames
            ref_frame = min(frame_i - n_frm_st, downlink_samples.shape[0])

    if deep_inspect:


        num_cl_tmp = num_pilots  # number of UEs to plot data for
        num_frames = pilot_samples.shape[0]
        num_cells = pilot_samples.shape[1]
        num_bs_ants = pilot_samples.shape[3]

        samps_mat = np.reshape(
                pilot_samples, (num_frames, num_cells, num_cl_tmp, num_bs_ants, samps_per_slot, 2))
        samps = (samps_mat[:, :, :, :, :, 0] +
                samps_mat[:, :, :, :, :, 1]*1j)*2**-15


        filter_pilots_start = time.time()
        match_filt, seq_num, seq_len, cmpx_pilots, seq_orig = hdf5_lib.filter_pilots(samps, z_padding, fft_size, cp, pilot_type, nonzero_sc_size)
        filter_pilots_end = time.time()

        frame_sanity_start = time.time()
        match_filt_clr, frame_map, f_st, peak_map = hdf5_lib.frame_sanity(match_filt, seq_num, seq_len, n_frm_st, frame_to_plot=frame_i, plt_ant=ant_i, cp = cp)
        frame_sanity_end = time.time()
        print(">>>> filter_pilots time: %f \n" % ( filter_pilots_end - filter_pilots_start) )
        print(">>>> frame_sanity time: %f \n" % ( frame_sanity_end - frame_sanity_start) )
