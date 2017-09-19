"""
.. _tut_compute_inverse_erf:

========================================
Compute inverse solution for evoked data
========================================

Here we'll use our knowledge from the other examples and tutorials
to compute an inverse solution and apply it on event related fields.
"""
# Author: Denis A. Engemann
# License: BSD 3 clause

import os.path as op
import mne
import hcp

from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              write_inverse_operator, read_inverse_operator)

from src.utils import utils

links_dir = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(links_dir, 'subjects', 'SUBJECTS_DIR')
MMVT_DIR = utils.get_link_dir(links_dir, 'mmvt')
HCP_DIR = utils.get_link_dir(links_dir, 'hcp')
MEG_DIR = utils.get_link_dir(links_dir, 'meg')


# from hcp import preprocessing as preproc

##############################################################################
# we assume our data is inside a designated folder under $HOME
# storage_dir = op.expanduser('~/mne-hcp-data')
recordings_path = op.join(HCP_DIR, 'hcp-meg')
subject = '100307'  # our test subject
task = 'task_working_memory'
meg_sub_fol = op.join(MEG_DIR, subject)
fwd_fname = op.join(meg_sub_fol, '{}-fwd.fif'.format(subject))
inv_fname = op.join(meg_sub_fol, '{}-inv.fif'.format(subject))
noise_cov_fname = op.join(meg_sub_fol, '{}-noise-cov.fif'.format(subject, task))
stc_fname = op.join(meg_sub_fol, '{}-{}.stc'.format(subject, task))

n_jobs = 4
run_index = 0

##############################################################################
# We're reading the evoked data.
# These are the same as in :ref:`tut_plot_evoked`

hcp_evokeds = hcp.read_evokeds(onset='stim', subject=subject,
                                   data_type=task, HCP_DIR=HCP_DIR)
for evoked in hcp_evokeds:
    if not evoked.comment == 'Wrkmem_LM-TIM-face_BT-diff_MODE-mag':
        continue

##############################################################################
# We'll now use a convenience function to get our forward and source models
# instead of computing them by hand.

# src_outputs = hcp.anatomy.compute_forward_stack(
#     subject=subject, subjects_dir=subjects_dir,
#     HCP_DIR=HCP_DIR, recordings_path=recordings_path,
#     # speed up computations here. Setting `add_dist` to True may improve the
#     # accuracy.
#     src_params=dict(add_dist=False),
#     info_from=dict(data_type=task, run_index=run_index))
#
# fwd = src_outputs['fwd']

fwd = mne.read_forward_solution(fwd_fname)

##############################################################################
# Now we can compute the noise covariance. For this purpose we will apply
# the same filtering as was used for the computations of the ERF in the first
# place. See also :ref:`tut_reproduce_erf`.

if not op.isfile(noise_cov_fname):
    raw_noise = hcp.read_raw(subject=subject, HCP_DIR=HCP_DIR,
                             data_type='noise_empty_room')
    raw_noise.load_data()

    # apply ref channel correction and drop ref channels
    hcp.preprocessing.apply_ref_correction(raw_noise)

    # Note: MNE complains on Python 2.7
    raw_noise.filter(0.50, None, method='iir',
                     iir_params=dict(order=4, ftype='butter'), n_jobs=n_jobs)
    raw_noise.filter(None, 60, method='iir',
                     iir_params=dict(order=4, ftype='butter'), n_jobs=n_jobs)

    ##############################################################################
    # Note that using the empty room noise covariance will inflate the SNR of the
    # evkoked and renders comparisons  to `baseline` rather uninformative.
    noise_cov = mne.compute_raw_covariance(raw_noise, method='empirical')
    mne.write_cov(noise_cov_fname, noise_cov)
else:
    noise_cov = mne.read_cov(noise_cov_fname)

##############################################################################
# Now we assemble the inverse operator, project the data and show the results
# on the `fsaverage` surface, the freesurfer average brain.

if not op.isfile(inv_fname):
    inv_op = mne.minimum_norm.make_inverse_operator(
        evoked.info, fwd, noise_cov=noise_cov)
    write_inverse_operator(inv_fname, inv_op)
else:
    inv_op = read_inverse_operator(inv_fname)


if not op.isfile(stc_fname):
    stc = mne.minimum_norm.apply_inverse(  # these data have a pretty high SNR and
        evoked, inv_op, method='MNE', lambda2=1./9.**2)  # 9 is a lovely number.
    stc.save(stc_fname[:-4])
else:
    stc = mne.read_source_estimate(stc_fname, subject)

print(stc.shape)
print('Done!')

# stc = stc.to_original_src(
#     src_outputs['src_fsaverage'], subjects_dir=subjects_dir)
#
# brain = stc.plot(subject='fsaverage', subjects_dir=subjects_dir, hemi='both')
# brain.set_time(145)  # we take the peak seen in :ref:`tut_plot_evoked` and
# brain.show_view('caudal')  # admire wide spread visual activation.
