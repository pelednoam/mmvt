import os
import os.path as op
import glob
from scipy.spatial.distance import cdist
import numpy as np
import fnmatch
import mne
import nibabel as nib

from src.utils import utils
from src.utils import args_utils as au
from src.utils import labels_utils as lu

links_dir = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(links_dir, 'subjects', 'SUBJECTS_DIR')
MMVT_DIR = utils.get_link_dir(links_dir, 'mmvt')
FMRI_DIR = utils.get_link_dir(links_dir, 'fMRI')


def get_new_label(subject, atlas, regex, new_label_name, new_label_r=5, overwrite=False, n_jobs=6):
    new_label_fname = op.join(SUBJECTS_DIR, subject, 'label', '{}.label'.format(new_label_name))
    if op.isfile(new_label_fname) and not overwrite:
        new_label = mne.read_label(new_label_fname)
        return new_label, new_label.hemi

    labels = lu.read_labels(subject, SUBJECTS_DIR, atlas)
    selected_labels = [l for l in labels if fnmatch.fnmatch(l.name, '*{}*'.format(regex))]
    hemis = set([lu.get_label_hemi(l.name) for l in selected_labels])
    if len(hemis) > 1:
        raise Exception('The selected labels belong to more than one hemi!')
    selected_hemi = list(hemis)[0]
    # centers_of_mass = lu.calc_center_of_mass(selected_labels)
    selected_labels_pos = np.array(utils.flat_list([l.pos for l in selected_labels]))
    center_of_mass = np.mean(selected_labels_pos, 0)
    hemi_verts = np.load(op.join(MMVT_DIR, subject, 'surf', '{}.pial.npz'.format(selected_hemi)))['verts']
    dists = cdist(hemi_verts, [center_of_mass])
    vertice_indice = np.argmin(dists)
    new_label = mne.grow_labels(subject, vertice_indice, new_label_r, 0 if selected_hemi == 'lh' else 1, SUBJECTS_DIR,
                                n_jobs, names=new_label_name, surface='pial')[0]
    utils.make_dir(op.join(SUBJECTS_DIR, subject))
    utils.make_dir(op.join(SUBJECTS_DIR, subject, 'label'))
    new_label.save(new_label_fname)
    return new_label, selected_hemi


def load_fmri_data(fmri_surf_fname):
    file_type = utils.file_type(fmri_surf_fname)
    if file_type in ['nii', 'nii.gz', 'mgz', 'mgh']:
        x = nib.load(fmri_surf_fname).get_data()
    elif file_type == 'npy':
        x = np.load(fmri_surf_fname)
    else:
        raise Exception('fMRI file format is not supported!')
    return x


def get_fmri_data(subject, fmri_surf_fname):

    def get_fmri_data(fmri_surf_fname):
        if not op.isfile(fmri_surf_fname):
            fmri_surf_fname = op.join(MMVT_DIR, subject, 'fmri', fmri_surf_fname)
        if not op.isfile(fmri_surf_fname):
            fmri_surf_fname = op.join(FMRI_DIR, subject, fmri_surf_fname)
        if not op.isfile(fmri_surf_fname):
            raise Exception('fMRI does not exist! {}'.format(fmri_surf_fname))
        x = load_fmri_data(fmri_surf_fname)
        return fmri_surf_fname, x

    def get_fmri_other_hemi_data(fmri_surf_fname):
        file_type = utils.file_type(fmri_surf_fname)
        fmri_other_hemi_file_name = lu.get_other_hemi_label_name(utils.namebase(fmri_surf_fname))
        fmri_other_hemi_surf_fname = op.join(utils.get_parent_fol(fmri_surf_fname),
                                             '{}.{}'.format(fmri_other_hemi_file_name, file_type))
        if not op.isfile(fmri_other_hemi_surf_fname):
            raise Exception("Can't find {}!".format(fmri_other_hemi_surf_fname))
        return load_fmri_data(fmri_other_hemi_surf_fname)

    hemi = lu.get_label_hemi(utils.namebase(fmri_surf_fname))
    other_hemi = 'lh' if hemi == 'rh' else 'rh'
    x = {}
    fmri_surf_fname, x[hemi] = get_fmri_data(fmri_surf_fname)
    x[other_hemi] = get_fmri_other_hemi_data(fmri_surf_fname)
    return x


def calc_label_corr(subject, x, label, hemi, label_name, overwrite=False, n_jobs=6):
    output_fname_template = op.join(MMVT_DIR, subject, 'fmri', 'fmri_seed_{}_{}.npy'.format(label_name, '{hemi}'))
    minmax_fname = op.join(MMVT_DIR, subject, 'fmri', 'seed_{}_minmax.pkl'.format(label_name))
    if utils.both_hemi_files_exist(output_fname_template) and op.isfile(minmax_fname) and not overwrite:
        return True
    label_ts = np.mean(x[hemi][label.vertices, :], 0)
    corr_min, corr_max = 0 , 0
    for hemi in utils.HEMIS:
        verts_num = x[hemi].shape[0]
        corr_vals = np.zeros((verts_num))
        indices = np.array_split(np.arange(verts_num), n_jobs)
        chunks = [(x[hemi][indices_chunk], indices_chunk, label_ts, thread) for thread, indices_chunk in enumerate(indices)]
        results = utils.run_parallel(_calc_label_corr, chunks, n_jobs)
        for hemi_corr_chunk, indices_chunk in results:
            corr_vals[indices_chunk] = hemi_corr_chunk
        corr_vals[np.isnan(corr_vals)] = 0
        np.save(output_fname_template.format(hemi=hemi), corr_vals)
        corr_min = min(corr_min, np.min(corr_vals))
        corr_max = max(corr_max, np.max(corr_vals))
    corr_minmax = utils.get_max_abs(corr_max, corr_min)
    corr_min, corr_max = -corr_minmax, corr_minmax
    utils.save((corr_min, corr_max), minmax_fname)
    return utils.both_hemi_files_exist(output_fname_template) and op.isfile(minmax_fname)


def _calc_label_corr(p):
    import time
    hemi_data, indices_chunk, label_ts, thread = p
    hemi_corr = np.zeros((len(indices_chunk)))
    now = time.time()
    N = len(indices_chunk)
    for ind, v_ts in enumerate(hemi_data):
        utils.time_to_go(now, ind, N, runs_num_to_print=1000, thread=thread)
        hemi_corr[ind] = np.corrcoef(label_ts, v_ts)[0, 1]
    return hemi_corr, indices_chunk


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True)
    parser.add_argument('-a', '--atlas', help='atlas', required=False, default='laus125')
    parser.add_argument('-r', '--regex', help='labels regex', required=False, default='post*cingulate*rh')
    parser.add_argument('-l', '--new_label_name', help='', required=False, default='posterior_cingulate_rh')
    parser.add_argument('--new_label_r', help='', required=False, default=5, type=int)
    parser.add_argument('--fmri_surf_fname', help='', required=False, default='fmri_hesheng_rh.npy')
    parser.add_argument('--overwrite', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--n_jobs', help='n_jobs', required=False, default=6, type=int)
    args = utils.Bag(au.parse_parser(parser))

    new_label, hemi = get_new_label(args.subject, args.atlas, args.regex, args.new_label_name, args.overwrite, args.n_jobs)
    x = get_fmri_data(args.subject, args.fmri_surf_fname)
    calc_label_corr(args.subject, x, new_label, hemi, args.new_label_name, args.overwrite, args.n_jobs)
