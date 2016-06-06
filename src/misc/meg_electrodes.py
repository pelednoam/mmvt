import os.path as op
import glob
import numpy as np
import re
import mne
import matplotlib.pyplot as plt
from src.utils import utils
from src.preproc import meg_preproc

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
FREE_SURFER_HOME = utils.get_link_dir(LINKS_DIR, 'freesurfer', 'FREESURFER_HOME')
BLENDER_ROOT_DIR = op.join(LINKS_DIR, 'mmvt')
SUBJECTS_MEG_DIR = op.join(LINKS_DIR, 'meg')


def meg_recon_to_electrodes(subject, atlas, bipolar, error_radius=3, elec_length=4, vertices_num_threshold=0):
    elecs_probs, probs_fname = utils.get_electrodes_labeling(
        subject, BLENDER_ROOT_DIR, atlas, bipolar, error_radius, elec_length)
    elecs_probs_fol = op.split(probs_fname)[0]
    elecs_probs_data_fname = op.join(elecs_probs_fol, '{}_meg.pkl'.format(utils.namebase(probs_fname)))
    elecs_probs = get_meg(subject, elecs_probs, bipolar)
    utils.save(elecs_probs, elecs_probs_data_fname)
    # for elec_probs in elecs_probs:
    #     if len(elec_probs['cortical_indices']) > vertices_num_threshold and elec_probs['approx'] == 3:
    #         print(elec_probs['name'], elec_probs['hemi'], len(elec_probs['cortical_indices']))
    #         # meg = get_meg(subject, elec_probs['cortical_indices'], elec_probs['hemi'])


def get_meg(subject, elecs_probs, bipolar, vertices_num_threshold=30):
    # meg_files = glob.glob(op.join(BLENDER_ROOT_DIR, subject, 'activity_map_{}'.format(hemi), '*.npy'))
    # meg_data = np.zeros((len(meg_files), len(vertices)))
    # for meg_file in meg_files:
    #     t = int(re.findall('\d+', utils.namebase(meg_file))[0])
    #     meg_data_t = np.load(meg_file)
    #     meg_data[t] = meg_data_t[vertices, 0]
    #     print("sdf")
    f = np.load(op.join(BLENDER_ROOT_DIR, subject, 'electrodes', 'electrodes_{}data_diff.npz'.format(
        'bipolar_' if bipolar else '')))
    conds = np.array([cond.decode() for cond in f['conditions'] if isinstance(cond, np.bytes_)])
    names = np.array([name.decode() for name in f['names'] if isinstance(name, np.bytes_)])
    figs_fol = op.join(BLENDER_ROOT_DIR, subject, 'figs')
    utils.make_dir(figs_fol)
    subject = 'ep001'
    task = 'MSIT'
    fname_format, events_id, event_digit = meg_preproc.get_fname_format(task)
    raw_cleaning_method = 'nTSSS'
    constrast = 'interference'
    inverse_method = 'dSPM'
    meg_preproc.init_globals(subject, fname_format=fname_format, raw_cleaning_method=raw_cleaning_method,
                             subjects_meg_dir=SUBJECTS_MEG_DIR, task=task, subjects_mri_dir=SUBJECTS_DIR,
                             BLENDER_ROOT_DIR=BLENDER_ROOT_DIR, files_includes_cond=True, fwd_no_cond=True,
                             constrast=constrast)

    for elec_probs in elecs_probs:
        elec_probs['data'] = {cond:None for cond in events_id.keys()}
        # if len(elec_probs['cortical_indices_dists']) > 0:
        #     print(elec_probs['name'], len(elec_probs['cortical_indices']), np.min(elec_probs['cortical_indices_dists']))

    msit_keys_dict = {'neutral':'noninterference', 'interference':'interference'}
    meg_elecs = {}
    for cond in events_id.keys():
        meg_elecs[cond] = []
        cond_ind = np.where(msit_keys_dict[cond] == conds)[0][0]
        stc_fname = meg_preproc.STC_HEMI_SMOOTH.format(cond=cond, method=inverse_method, hemi='rh')
        stc = mne.read_source_estimate(stc_fname)

        for elec_probs in elecs_probs:
            # len(elec_probs['cortical_indices']) > vertices_num_threshold
            if len(elec_probs['cortical_indices_dists']) > 0 and np.min(elec_probs['cortical_indices_dists']) < 1 and elec_probs['approx'] == 3:
                # print(elec_probs['name'], elec_probs['hemi'], len(elec_probs['cortical_indices']))
                elec_ind = np.where(elec_probs['name'] == names)[0][0]
                data = stc.rh_data if elec_probs['hemi'] == 'rh' else stc.lh_data
                T = data.shape[1]
                elec_meg_data = data[elec_probs['cortical_indices']]
                dists = elec_probs['cortical_indices_dists']
                norm_dists = dists / np.sum(dists)
                norm_dists = np.reshape(norm_dists, (1, len(norm_dists)))
                elec_meg_data = np.dot(norm_dists, elec_meg_data)
                elec_meg_data = np.reshape(elec_meg_data, ((T)))
                elec_data = f['data'][elec_ind, :, cond_ind]
                elec_data_diff = np.max(elec_data) - np.min(elec_data)
                elec_meg_data *= elec_data_diff / (np.max(elec_meg_data) - np.min(elec_meg_data))
                elec_meg_data += elec_data[0] - elec_meg_data[0]
                elec_probs['data'][cond] = elec_meg_data
                plt.figure()
                plt.plot(elec_meg_data, label='pred')
                plt.plot(elec_data, label='elec')
                plt.legend()
                plt.title('{}-{}'.format(elec_probs['name'], cond))
                plt.savefig(op.join(figs_fol, '{}-{}.jpg'.format(elec_probs['name'], cond)))
                plt.close()
    return elecs_probs


if __name__ == '__main__':
    subject = 'mg78'
    atlas = 'laus250'
    bipolar = True
    meg_recon_to_electrodes(subject, atlas, bipolar)
    print('finish!')