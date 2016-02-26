import mne.surface
from scipy.spatial.distance import cdist
import time
import os.path as op
import numpy as np
import os
import shutil
from src import utils

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
            shutil.copyfile(op.join(subjects_dic, subject, 'label', '{}.{}.{}.annot'.format(hemi, aparc_name, backup_str)))


if __name__ == '__main__':
    subject = 'mg96'
    atlas = 'laus250'
    label_name = 'bankssts_1-lh'
    n_jobs = 6
    # check_labels(subject, SUBJECTS_DIR, atlas, label_name)
    solve_labels_collision(subject, SUBJECTS_DIR, '{}_orig'.format(atlas), atlas, n_jobs)