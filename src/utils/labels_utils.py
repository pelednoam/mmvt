import mne.surface
from scipy.spatial.distance import cdist
import time
import os.path as op
import numpy as np
import os
import shutil
import glob
import traceback

from src.utils import utils
from src.utils import preproc_utils as pu
from src.mmvt_addon import colors_utils as cu

SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()
HEMIS = ['rh', 'lh']


def morph_labels_from_fsaverage(subject, subjects_dir, mmvt_dir, aparc_name='aparc250', fs_labels_fol='',
            sub_labels_fol='', n_jobs=6, fsaverage='fsaverage', overwrite=False):
    subject_dir = op.join(subjects_dir, subject)
    labels_fol = op.join(subjects_dir, fsaverage, 'label', aparc_name) if fs_labels_fol=='' else fs_labels_fol
    sub_labels_fol = op.join(subject_dir, 'label', aparc_name) if sub_labels_fol=='' else sub_labels_fol
    if not op.isdir(sub_labels_fol):
        os.makedirs(sub_labels_fol)
    fsaverage_annot_files_exist = utils.both_hemi_files_exist(op.join(subjects_dir, fsaverage, 'label', '{}.{}.annot'.format(
        '{hemi}', aparc_name)))
    if fsaverage_annot_files_exist:
        labels = read_labels(fsaverage, subjects_dir, aparc_name, n_jobs=n_jobs)
    else:
        print("The annot files doesn't found ({}), trying to read the lablels files".format(
            op.join(subjects_dir, fsaverage, 'label', '{}.{}.annot'.format('{hemi}', aparc_name))))
        labels = read_labels(fsaverage, subjects_dir, aparc_name, n_jobs=n_jobs)
    if len(labels) == 0:
        print('morph_labels_from_fsaverage: No labels files found!')
        return False
    if subject == fsaverage:
        return True
    surf_loaded = False
    for fs_label in labels:
        label_file = op.join(labels_fol, '{}.label'.format(fs_label.name))
        local_label_name = op.join(sub_labels_fol, '{}.label'.format(op.splitext(op.split(label_file)[1])[0]))
        if not op.isfile(local_label_name) or overwrite:
            # fs_label = mne.read_label(label_file)
            fs_label.values.fill(1.0)
            sub_label = fs_label.morph(fsaverage, subject, grade=None, n_jobs=n_jobs, subjects_dir=subjects_dir)
            if np.all(sub_label.pos == 0):
                if not surf_loaded:
                    verts = {}
                    for hemi in HEMIS:
                        if op.isfile(op.join(mmvt_dir, subject, 'surf', '{}.pial.npz'.format(hemi))):
                            hemi_verts, _ = utils.read_pial_npz(subject, mmvt_dir, hemi)
                        elif op.isfile(op.join(subjects_dir, subject, 'surf', '{}.pial.ply'.format(hemi))):
                            hemis_verts, _ = utils.read_ply_file(
                                op.join(subjects_dir, subject, 'surf', '{}.pial.ply'.format(hemi)))
                        else:
                            print("Can't find {} pial ply/npz files!".format(hemi))
                            return False
                        verts[hemi] = hemi_verts
                    surf_loaded = True
                sub_label.pos = verts[sub_label.hemi][sub_label.vertices]
            sub_label.save(local_label_name)
    return True


def solve_labels_collision(subject, subjects_dir, atlas, backup_atlas, n_jobs=1):
    now = time.time()
    print('Read labels')
    # utils.read_labels_parallel(subject, subjects_dir, atlas, labels_fol='', n_jobs=n_jobs)
    labels = read_labels(subject, subjects_dir, atlas, n_jobs=n_jobs)
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


def get_atlas_labels_names(subject, atlas, subjects_dir, delim='-', pos='end', return_flat_labels_list=False, include_unknown=False,
                           include_corpuscallosum=False, n_jobs=1):
    annot_fname_hemi = op.join(subjects_dir, subject, 'label', '{}.{}.annot'.format('{hemi}', atlas))
    labels_names_hemis = dict(lh=[], rh=[])
    all_labels = []
    if utils.both_hemi_files_exist(annot_fname_hemi):
        for hemi in ['rh', 'lh']:
            annot_fname = op.join(subjects_dir, subject, 'label', '{}.{}.annot'.format(hemi, atlas))
            _, _, labels_names = mne.label._read_annot(annot_fname)
            labels_names = fix_labels_names(labels_names, hemi, delim, pos)
            all_labels.extend(labels_names)
            labels_names_hemis[hemi] = labels_names
    else:
        all_labels = read_labels_parallel(subject, subjects_dir, atlas, labels_fol='' , n_jobs=n_jobs)
        for label in all_labels:
            labels_names_hemis[label.hemi].append(label.name)
    if len(labels_names_hemis['rh']) == 0 or len(labels_names_hemis['lh']) == 0:
        raise Exception("Can't read {} labels for atlas {}".format(subject, atlas))
    if return_flat_labels_list:
        if not include_unknown:
            all_labels = [l for l in all_labels if 'unknown' not in l]
        if not include_corpuscallosum:
            all_labels = [l for l in all_labels if 'corpuscallosum' not in l]
        return all_labels
    else:
        if not include_unknown:
            for hemi in HEMIS:
                labels_names_hemis[hemi] = [l for l in labels_names_hemis[hemi] if 'unknown' not in l]
        if not include_corpuscallosum:
            for hemi in HEMIS:
                labels_names_hemis[hemi] = [l for l in labels_names_hemis[hemi] if 'corpuscallosum' not in l]
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
                output_fname='', exclude=[], rh_then_lh=False, sorted_according_to_annot_file=False,
                hemi='both', surf_name='pial', labels_fol='', n_jobs=1):
    try:
        if try_first_from_annotation:
            try:
                labels = mne.read_labels_from_annot(subject, atlas, subjects_dir=subjects_dir, surf_name=surf_name,
                                                    hemi=hemi)
            except:
                print("read_labels_from_annot failed! subject {} atlas {} surf name {} hemi {}. Trying to read labels files".format(
                    subject, atlas, surf_name, hemi))
                labels = read_labels_parallel(subject, subjects_dir, atlas, labels_fol, n_jobs=n_jobs)
        else:
            labels = read_labels_parallel(subject, subjects_dir, atlas, labels_fol, n_jobs=n_jobs)
        labels = [l for l in labels if not np.any([e in l.name for e in exclude])]
        if rh_then_lh:
            rh_labels = [l for l in labels if l.hemi == 'rh']
            lh_labels = [l for l in labels if l.hemi == 'lh']
            labels = rh_labels + lh_labels
        if sorted_according_to_annot_file:
            annot_labels = get_atlas_labels_names(subject, atlas, subjects_dir, return_flat_labels_list=True)
            labels.sort(key=lambda x: annot_labels.index(x.name))
        if output_fname != '':
            with open(output_fname, 'w') as output_file:
                for label in labels:
                    output_file.write('{}\n'.format(label.name))
        if only_names:
            labels = [l.name for l in labels]
        return labels
    except:
        print(traceback.format_exc())
        return []


def read_labels_parallel(subject, subjects_dir, atlas, hemi='', labels_fol='', n_jobs=1):
    try:
        labels_fol = op.join(subjects_dir, subject, 'label', atlas) if labels_fol == '' else labels_fol
        if hemi != '':
            labels_files = glob.glob(op.join(labels_fol, '*{}.label'.format(hemi)))
            labels_files.extend(glob.glob(op.join(labels_fol, '{}.*label'.format(hemi))))
        else:
            labels_files = glob.glob(op.join(labels_fol, '*.label'))
        files_chunks = utils.chunks(labels_files, len(labels_files) / n_jobs)
        results = utils.run_parallel(_read_labels_parallel, files_chunks, njobs=n_jobs)
        labels = []
        for labels_chunk in results:
            labels.extend(labels_chunk)
        return labels
    except:
        print(traceback.format_exc())
        return []


def _read_labels_parallel(files_chunk):
    labels = []
    for label_fname in files_chunk:
        label = mne.read_label(label_fname)
        labels.append(label)
    return labels


def read_hemi_labels(subject, subjects_dir, atlas, hemi, surf_name='pial', labels_fol=''):
    # todo: replace with labels utils read labels function
    labels_fol = op.join(subjects_dir, subject, 'label', atlas) if labels_fol=='' else labels_fol
    annot_fname_template = op.join(subjects_dir, subject, 'label', '{}.{}.annot'.format('{hemi}', atlas))
    if utils.both_hemi_files_exist(annot_fname_template):
        labels = mne.read_labels_from_annot(subject, atlas, hemi, surf_name)
        if len(labels) == 0:
            raise Exception('No labels were found in the {} annot file!'.format(annot_fname_template))
    else:
        labels = []
        for label_file in glob.glob(op.join(labels_fol, '*{}.label'.format(hemi))):
            label = mne.read_label(label_file)
            labels.append(label)
        if len(labels) == 0:
            raise Exception('No labels were found in {}!'.format(labels_fol))
    return labels


def calc_center_of_mass(labels, ret_mat=False):
    center_of_mass = np.zeros((len(labels), 3)) if ret_mat else {}
    for ind, label in enumerate(labels):
        if ret_mat:
            center_of_mass[ind] = np.mean(label.pos, 0)
        else:
            center_of_mass[label.name] = np.mean(label.pos, 0)
    return center_of_mass


def label_is_excluded(label_name, compiled_excludes):
    return not compiled_excludes.search(label_name) is None


def calc_time_series_per_label(x, labels, measure, excludes=(),
                               figures_dir='', do_plot=False, do_plot_all_vertices=False):
    import sklearn.decomposition as deco
    from functools import partial
    import re
    import matplotlib.pyplot as plt

    _label_is_excluded = partial(label_is_excluded, compiled_excludes=re.compile('|'.join(excludes)))
    labels = [l for l in labels if not _label_is_excluded(l.name)]
    labels_data = np.zeros((len(labels), x.shape[-1]))
    labels_names = []
    if do_plot_all_vertices:
        all_vertices_plots_dir = op.join(figures_dir, 'all_vertices')
        utils.make_dir(all_vertices_plots_dir)
    if do_plot:
        measure_plots_dir = op.join(figures_dir, measure)
        utils.make_dir(measure_plots_dir)
    for ind, label in enumerate(labels):
        if measure == 'mean':
            labels_data[ind, :] = np.mean(x[label.vertices, 0, 0, :], 0)
        elif measure == 'PCA':
            print(label)
            _x = x[label.vertices, 0, 0, :].T
            remove_cols = np.where(np.all(_x == np.mean(_x, 0), 0))[0]
            _x = np.delete(_x, remove_cols, 1)
            _x = (_x - np.mean(_x, 0)) / np.std(_x, 0)
            pca = deco.PCA(1)
            x_r = pca.fit(_x).transform(_x)
            labels_data[ind, :] = x_r.ravel()
        elif measure == 'cv': #''coef_of_variation':
            label_mean = np.mean(x[label.vertices, 0, 0, :], 0)
            label_std = np.std(x[label.vertices, 0, 0, :], 0)
            labels_data[ind, :] = label_std / label_mean
        labels_names.append(label.name)
        if do_plot_all_vertices:
            plt.figure()
            plt.plot(x[label.vertices, 0, 0, :].T)
            plt.savefig(op.join(all_vertices_plots_dir, '{}.jpg'.format(label.name)))
            plt.close()
        if do_plot:
            plt.figure()
            plt.plot(labels_data[ind, :])
            plt.savefig(op.join(measure_plots_dir, '{}_{}.jpg'.format(measure, label.name)))
            plt.close()

    return labels_data, labels_names


def morph_labels(morph_from_subject, morph_to_subject, atlas, hemi, n_jobs=1):
    labels_fol = op.join(SUBJECTS_DIR, morph_to_subject, 'label')
    labels_fname = op.join(labels_fol, '{}.{}.pkl'.format(hemi, atlas,morph_from_subject))
    annot_file = op.join(SUBJECTS_DIR, morph_from_subject, 'label', '{}.{}.annot'.format(hemi, atlas))
    if not op.isfile(annot_file):
        print("Can't find the annot file in {}!".format(annot_file))
        return []
    if not op.isfile(labels_fname):
        labels = mne.read_labels_from_annot(morph_from_subject, atlas, subjects_dir=SUBJECTS_DIR, hemi=hemi)
        if morph_from_subject != morph_to_subject:
            morphed_labels = []
            for label in labels:
                label.values.fill(1.0)
                morphed_label = label.morph(morph_from_subject, morph_to_subject, 5, None, SUBJECTS_DIR, n_jobs)
                morphed_labels.append(morphed_label)
            labels = morphed_labels
        utils.save(labels, labels_fname)
    else:
        labels = utils.load(labels_fname)
    return labels


def create_atlas_coloring(subject, atlas, n_jobs=-1):
    ret = False
    coloring_dir = op.join(MMVT_DIR, subject, 'coloring')
    utils.make_dir(coloring_dir)
    coloring_fname = op.join(coloring_dir, 'labels_{}_coloring.csv'.format(atlas))
    coloring_names_fname = op.join(coloring_dir, 'labels_{}_colors_names.txt'.format(atlas))
    try:
        labels = read_labels(subject, SUBJECTS_DIR, atlas, n_jobs=n_jobs)
        colors_rgb_and_names = cu.get_distinct_colors_and_names()
        labels_colors_rgb, labels_colors_names = {}, {}
        for label in labels:
            label_inv_name = get_label_hemi_invariant_name(label.name)
            if label_inv_name not in labels_colors_rgb:
                labels_colors_rgb[label_inv_name], labels_colors_names[label_inv_name] = next(colors_rgb_and_names)
        with open(coloring_fname, 'w') as colors_file, open(coloring_names_fname, 'w') as col_names_file:
            for label in labels:
                label_inv_name = get_label_hemi_invariant_name(label.name)
                color_rgb = labels_colors_rgb[label_inv_name]
                color_name = labels_colors_names[label_inv_name]
                colors_file.write('{},{},{},{}\n'.format(label.name, *color_rgb))
                col_names_file.write('{},{}\n'.format(label.name, color_name))
        ret = op.isfile(coloring_fname)
    except:
        print('Error in save_labels_coloring!')
        print(traceback.format_exc())
    return ret


def create_labels_coloring(subject, labels_names, labels_values, coloring_name, norm_percs=(3, 99),
                           norm_by_percentile=True, colors_map='jet'):
    coloring_dir = op.join(MMVT_DIR, subject, 'coloring')
    utils.make_dir(coloring_dir)
    coloring_fname = op.join(coloring_dir, '{}.csv'.format(coloring_name))
    ret = False
    try:
        labels_colors = utils.arr_to_colors(
            labels_values, norm_percs=norm_percs, norm_by_percentile=norm_by_percentile, colors_map=colors_map)
        with open(coloring_fname, 'w') as colors_file:
            for label_name, label_color, label_value in zip(labels_names, labels_colors, labels_values):
                colors_file.write('{},{},{},{},{}\n'.format(label_name, *label_color[:3], label_value))
        ret = op.isfile(coloring_fname)
    except:
        print('Error in create_labels_coloring!')
        print(traceback.format_exc())
    return ret


if __name__ == '__main__':
    # subject = 'mg96'
    # atlas = 'laus250'
    # label_name = 'bankssts_1-lh'
    # n_jobs = 6
    # check_labels(subject, SUBJECTS_DIR, atlas, label_name)
    # solve_labels_collision(subject, SUBJECTS_DIR, '{}_orig'.format(atlas), atlas, n_jobs)
    pass
