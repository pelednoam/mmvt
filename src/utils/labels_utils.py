import mne.surface
from scipy.spatial.distance import cdist
import time
import os.path as op
import numpy as np
import os
import shutil
import glob
from src.utils import utils

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
HEMIS = ['rh', 'lh']


def solve_labels_collision(subject, subjects_dir, atlas, backup_atlas, n_jobs=1):
    now = time.time()
    print('Read labels')
    labels = utils.read_labels_parallel(subject, subjects_dir, atlas, n_jobs)
    backup_labels_fol = op.join(subjects_dir, subject, 'label', backup_atlas)
    labels_fol = op.join(subjects_dir, subject, 'label', atlas)
    if op.isdir(backup_labels_fol):
        shutil.rmtree(backup_labels_fol)
    os.rename(labels_fol, backup_labels_fol)
    utils.make_dir(labels_fol)
    hemis_verts, labels_hemi, pia_verts = {}, {}, {}
    print('Read surface ({:.2f}s)'.format(time.time() - now))
    for hemi in HEMIS:
        surf_fname = op.join(subjects_dir, subject, 'surf', '{}.pial'.format(hemi))
        hemis_verts[hemi], _ = mne.surface.read_surface(surf_fname)
        labels_hemi[hemi] = [l for l in labels if l.hemi == hemi]
    print('Calc centroids ({:.2f}s)'.format(time.time() - now))
    centroids = calc_labels_centroids(labels_hemi, hemis_verts)
    for hemi in HEMIS:
        print('Calc vertices labeling for {} ({:.2f}s)'.format(hemi, time.time() - now))
        hemi_centroids_dist = cdist(hemis_verts[hemi], centroids[hemi])
        vertices_labels_indices = np.argmin(hemi_centroids_dist, axis=1)
        labels_hemi_chunks = utils.chunks(list(enumerate(labels_hemi[hemi])), len(labels_hemi[hemi]) / n_jobs)
        params = [(labels_hemi_chunk, atlas, vertices_labels_indices, hemis_verts, labels_fol) for labels_hemi_chunk in labels_hemi_chunks]
        print('Save labels for {} ({:.2f}s)'.format(hemi, time.time() - now))
        utils.run_parallel(_save_new_labels_parallel, params, n_jobs)


def _save_new_labels_parallel(params_chunk):
    labels_hemi_chunk, atlas, vertices_labels_indices, hemis_verts, labels_fol = params_chunk
    for ind, label in labels_hemi_chunk:
        vertices = np.where(vertices_labels_indices == ind)[0]
        pos = hemis_verts[label.hemi][vertices]
        new_label = mne.Label(vertices, pos, hemi=label.hemi, name=label.name, filename=None,
            subject=label.subject, color=label.color, verbose=None)
        if not op.isfile(op.join(labels_fol, new_label.name)):
            new_label.save(op.join(labels_fol, new_label.name))


def calc_labels_centroids(labels_hemi, hemis_verts):
    centroids = {}
    for hemi in HEMIS:
        centroids[hemi] = np.zeros((len(labels_hemi[hemi]), 3))
        for ind, label in enumerate(labels_hemi[hemi]):
            coo = hemis_verts[label.hemi][label.vertices]
            centroids[label.hemi][ind, :] = np.mean(coo, axis=0)
    return centroids


def backup_annotation_files(subject, subjects_dic, aparc_name, backup_str='backup'):
    # Backup annotation files
    for hemi in HEMIS:
        annot_fname = op.join(subjects_dic, subject, 'label', '{}.{}.annot'.format(hemi, aparc_name))
        if op.isfile(annot_fname):
            shutil.copyfile(op.join(subjects_dic, subject, 'label', '{}.{}.annot'.format(hemi, aparc_name)),
                            op.join(subjects_dic, subject, 'label', '{}.{}.{}.annot'.format(hemi, aparc_name, backup_str)),)


def get_atlas_labels_names(subject, atlas, delim='-', pos='end', return_flat_labels_list=False, include_unknown=True, n_jobs=1):
    annot_fname_hemi = op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format('{hemi}', atlas))
    labels_names_hemis = dict(lh=[], rh=[])
    all_labels = []
    if utils.both_hemi_files_exist(annot_fname_hemi):
        for hemi in ['rh', 'lh']:
            annot_fname = op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format(hemi, atlas))
            _, _, labels_names = mne.label._read_annot(annot_fname)
            labels_names = fix_labels_names(labels_names, hemi, delim, pos)
            all_labels.extend(labels_names)
            labels_names_hemis[hemi] = labels_names
    else:
        all_labels = utils.read_labels_parallel(subject, SUBJECTS_DIR, atlas, n_jobs)
        for label in all_labels:
            labels_names_hemis[label.hemi].append(label.name)
    if len(labels_names_hemis['rh']) == 0 or len(labels_names_hemis['lh']) == 0:
        raise Exception("Can't read {} labels for atlas {}".format(subject, atlas))
    if return_flat_labels_list:
        if not include_unknown:
            all_labels = [l for l in all_labels if 'unknown' not in l]
        return all_labels
    else:
        if not include_unknown:
            for hemi in HEMIS:
                labels_names_hemis[hemi] = [l for l in labels_names_hemis[hemi] if 'unknown' not in l]
        return labels_names_hemis


def fix_labels_names(labels_names, hemi, delim='-', pos='end'):
    fixed_labels_names = []
    for label_name in labels_names:
        if isinstance(label_name, bytes):
            label_name = label_name.decode('utf-8')
        if not '{}-'.format(hemi) in label_name or \
            not '{}.'.format(hemi) in label_name or \
            not '-{}'.format(hemi) in label_name or \
            not '.{}'.format(hemi) in label_name:
                if pos == 'end':
                    label_name = '{}{}{}'.format(label_name, delim, hemi)
                elif pos == 'start':
                    label_name = '{}{}{}'.format(hemi, delim, label_name)
                else:
                    raise Exception("pos can be 'end' or 'start'")
        fixed_labels_names.append(label_name)
    return fixed_labels_names


def get_hemi_delim_and_pos(label_name):
    for hemi in ['rh', 'lh']:
        if label_name.startswith('{}-'.format(hemi)):
            delim, pos, label = '-', 'start', label_name[3:]
            break
        if label_name.startswith('{}.'.format(hemi)):
            delim, pos, label = '.', 'start', label_name[3:]
            break
        if label_name.endswith('-{}'.format(hemi)):
            delim, pos, label = '-', 'end', label_name[:-3]
            break
        if label_name.endswith('.{}'.format(hemi)):
            delim, pos, label = '.', 'end', label_name[:-3]
            break
    return delim, pos, label


def get_label_hemi_invariant_name(label_name):
    _, _, label_inv_name = get_hemi_delim_and_pos(label_name)
    return label_inv_name


def get_hemi_from_name(label_name):
    label_hemi = ''
    for hemi in ['rh', 'lh']:
        if label_name.startswith('{}-'.format(hemi)) or label_name.startswith('{}.'.format(hemi)) or \
                label_name.endswith('-{}'.format(hemi)) or label_name.endswith('.{}'.format(hemi)):
            label_hemi = hemi
            break
    if label_hemi == '':
        raise Exception("Can't find hemi in {}".format(label_name))
    return label_hemi


def read_labels(subject, subjects_dir, atlas, try_first_from_annotation=True, only_names=False,
                output_fname='', n_jobs=1):
    if try_first_from_annotation:
        try:
            labels = mne.read_labels_from_annot(subject, atlas)
        except:
            labels = read_labels_parallel(subject, subjects_dir, atlas, n_jobs)
    else:
        labels = read_labels_parallel(subject, subjects_dir, atlas, n_jobs)
    if output_fname != '':
        with open(output_fname, 'w') as output_file:
            for label in labels:
                output_file.write('{}\n'.format(label.name))
    if only_names:
        labels = [l.name for l in labels]
    return labels


def read_labels_parallel(subject, subjects_dir, atlas, n_jobs):
    labels_files = glob.glob(op.join(subjects_dir, subject, 'label', atlas, '*.label'))
    files_chunks = utils.chunks(labels_files, len(labels_files) / n_jobs)
    results = utils.run_parallel(_read_labels_parallel, files_chunks, n_jobs)
    labels = []
    for labels_chunk in results:
        labels.extend(labels_chunk)
    return labels


def _read_labels_parallel(files_chunk):
    labels = []
    for label_fname in files_chunk:
        label = mne.read_label(label_fname)
        labels.append(label)
    return labels


if __name__ == '__main__':
    subject = 'mg96'
    atlas = 'laus250'
    label_name = 'bankssts_1-lh'
    n_jobs = 6
    # check_labels(subject, SUBJECTS_DIR, atlas, label_name)
    # solve_labels_collision(subject, SUBJECTS_DIR, '{}_orig'.format(atlas), atlas, n_jobs)