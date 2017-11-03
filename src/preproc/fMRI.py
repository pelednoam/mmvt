import os
import os.path as op
import mne
import mne.stats.cluster_level as mne_clusters
import nibabel as nib
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import shutil
import glob
import traceback
from collections import defaultdict
import mne.label

from src.utils import utils
from src.utils import freesurfer_utils as fu
from src.preproc import meg as meg
from src.utils import preproc_utils as pu
from src.utils import labels_utils as lu

try:
    from sklearn.neighbors import BallTree
except:
    print('No sklearn!')

try:
    from surfer import Brain
    from surfer import viz
    # from surfer import project_volume_data
    SURFER = True
except:
    SURFER = False


SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()
SUBJECTS_MEG_DIR = utils.get_link_dir(utils.get_links_dir(), 'meg')
FMRI_DIR = utils.get_link_dir(utils.get_links_dir(), 'fMRI')

FSAVG_VERTS = 10242
FSAVG5_VERTS = 163842
COLIN27_VERTS = dict(lh=166836, rh=165685)

_bbregister = 'bbregister --mov {fsl_input}.nii --bold --s {subject} --init-fsl --lta register.lta'
_mri_robust_register = 'mri_robust_register --mov {fsl_input}.nii --dst $SUBJECTS_DIR/colin27/mri/orig.mgz' +\
                       ' --lta register.lta --satit --vox2vox --cost mi --mapmov {subject}_reg_mi.mgz'


def get_hemi_data(subject, hemi, source, surf_name='pial', name=None, sign="abs", min=None, max=None):
    brain = Brain(subject, hemi, surf_name, curv=False, offscreen=True)
    print('Brain {} verts: {}'.format(hemi, brain.geo[hemi].coords.shape[0]))
    hemi = brain._check_hemi(hemi)
    # load data here
    scalar_data, name = brain._read_scalar_data(source, hemi, name=name)
    print('fMRI contrast map vertices: {}'.format(len(scalar_data)))
    min, max = brain._get_display_range(scalar_data, min, max, sign)
    if sign not in ["abs", "pos", "neg"]:
        raise ValueError("Overlay sign must be 'abs', 'pos', or 'neg'")
    surf = brain.geo[hemi]
    old = viz.OverlayData(scalar_data, surf, min, max, sign)
    return old, brain


def build_fmri_contrast_file_template(subject, fmri_contrast_file_template='', template_brain='', remote_fmri_dir=''):
    remote_fmri_dir = op.join(FMRI_DIR, subject) if remote_fmri_dir == '' else remote_fmri_dir
    if fmri_contrast_file_template == '':
        fmri_contrast_file_template = '*{hemi}*'
    if '?h' in fmri_contrast_file_template:
        fmri_contrast_file_template = fmri_contrast_file_template.replace('?h', '{hemi}')
    if '{hemi}' not in fmri_contrast_file_template:
        print('build_fmri_contrast_file_template: no {hemi} in fmri_contrast_file_template!')
        fmri_contrast_file_template = '{}*{}*'.format(fmri_contrast_file_template, '{hemi}')
    fmri_contrast_file_template = fmri_contrast_file_template.format(
        subject=subject, morph_to_subject=template_brain, hemi='{hemi}')
    full_fmri_contrast_file_template = op.join(remote_fmri_dir, fmri_contrast_file_template)
    fmri_contrast_file_template = find_hemi_files_from_template(full_fmri_contrast_file_template)
    return fmri_contrast_file_template, full_fmri_contrast_file_template


def calc_fmri_min_max(subject, fmri_contrast_file_template, task='', norm_percs=(3, 97),
                      symetric_colors=True, contrast_name='', new_name='', remote_fmri_dir='', template_brain=''):
    data = None
    fmri_contrast_template_files = []
    for hemi in utils.HEMIS:
        if isinstance(fmri_contrast_file_template, dict):
            hemi_fname = fmri_contrast_file_template[hemi]
        elif isinstance(fmri_contrast_file_template, str):
            hemi_fname = fmri_contrast_file_template.format(hemi=hemi)
            if not op.isfile(hemi_fname):
                fmri_contrast_template_files, _ = build_fmri_contrast_file_template(
                    subject, fmri_contrast_file_template, template_brain, remote_fmri_dir)
                hemi_fname = fmri_contrast_template_files[0].format(hemi=hemi)
            # hemi_fname = fmri_contrast_file_template.format(hemi=hemi)
        else:
            raise Exception('Wrong type of template!')
        x = load_fmri_data(hemi_fname)
        verts, _ = utils.read_ply_file(op.join(MMVT_DIR, subject, 'surf', '{}.pial.ply'.format(hemi)))
        if x.shape[0] != verts.shape[0]:
            if x.shape[0] in [FSAVG5_VERTS, FSAVG_VERTS]:
                temp_barin = 'fsaverage5' if x.shape[0] == FSAVG5_VERTS else 'fsaverage'
                raise Exception(
                    "It seems that the fMRI contrast was made on {}, and not on the subject.\n".format(temp_barin) +
                    "You can run the fMRI preproc on the template brain, or morph the fMRI contrast map to the subject.")
            else:
                raise Exception("fMRI contrast map ({}) and the {} pial surface ".format(len(x), hemi) +
                                "({}) doesn't have the same vertices number!".format(verts.shape[0]))
        x_ravel = x.ravel()
        data = x_ravel if data is None else np.hstack((x_ravel, data))
    data_min, data_max = utils.calc_min_max(data, norm_percs=norm_percs)
    if data_min == 0 and data_max == 0:
        print('Both min and max values are 0!!!')
        return False
    print('calc_fmri_min_max: min: {}, max: {}'.format(data_min, data_max))
    data_minmax = utils.get_max_abs(data_max, data_min)
    if symetric_colors and np.sign(data_max) != np.sign(data_min) and data_min != 0:
        data_max, data_min = data_minmax, -data_minmax
    # todo: the output_fname was changed, check where it's being used!
    new_name = calc_new_name(new_name, task, contrast_name, fmri_contrast_template_files, fmri_contrast_file_template)
    output_fname = op.join(MMVT_DIR, subject, 'fmri', '{}_minmax.pkl'.format(new_name))
    print('Saving {}'.format(output_fname))
    utils.make_dir(op.join(MMVT_DIR, subject, 'fmri'))
    utils.save((data_min, data_max), output_fname)
    return op.isfile(output_fname)


def calc_new_name(new_name, task, contrast_name, fmri_contrast_template_files, fmri_contrast_file_template):
    if new_name != '':
        new_name = new_name
    else:
        if task != '' or contrast_name != '':
            new_name = '{}{}'.format('{}_'.format(task) if task != '' else '', contrast_name)
        else:
            if len(fmri_contrast_template_files) > 0:
                new_name = utils.namebase(fmri_contrast_template_files[0]).replace('{hemi}', '')
            else:
                new_name = utils.namebase(fmri_contrast_file_template).replace('{hemi}', '')
            if new_name[-1] in ['_', '-', '.']:
                new_name = new_name[:-1]
    if new_name.startswith('fmri_'):
        new_name = new_name[len('fmri_'):]
    return new_name


def save_fmri_hemi_data(subject, hemi, contrast_name, fmri_fname, task, output_fol=''):
    if not op.isfile(fmri_fname):
        print('No such file {}!'.format(fmri_fname))
        return
    morphed_fmri_fname = '{0}_morphed_to_{2}{1}'.format(*op.splitext(fmri_fname), subject)
    # if not op.isfile(morphed_fmri_fname):
    fmri = nib.load(fmri_fname)
    x = fmri.get_data().ravel()
    morph_from_subject = check_vertices_num(subject, hemi, x)
    org_subject_fname = ''
    if subject != morph_from_subject:
        # Save for data for the morph_from_subject
        org_subject_fname = 'fmri_{}_{}_{}_{}.npy'.format(subject, task, contrast_name, hemi)
        _save_fmri_hemi_data(morph_from_subject, hemi, x, contrast_name, task, output_fol, org_subject_fname)
        if not op.isfile(morphed_fmri_fname):
            fu.surf2surf(morph_from_subject, subject, hemi, fmri_fname, morphed_fmri_fname, cwd=None, print_only=False)
            fmri = nib.load(morphed_fmri_fname)
            x = fmri.get_data().ravel()
        else:
            fmri = nib.load(morphed_fmri_fname)
            x = fmri.get_data().ravel()

    subject_fname =  _save_fmri_hemi_data(subject, hemi, x, contrast_name, task, output_fol)
    org_subject_fname = op.join(MMVT_DIR, morph_from_subject, 'fmri', org_subject_fname)
    return subject_fname, org_subject_fname, morph_from_subject


def _save_fmri_hemi_data(subject, hemi, x, contrast_name, task, output_fol='', output_name=''):
    verts, faces = utils.read_pial(subject, MMVT_DIR, hemi)
    if len(verts) != x.shape[0]:
        raise Exception('Wrong number of vertices!')
    if output_fol == '':
        output_fol = op.join(MMVT_DIR, subject, 'fmri')
    utils.make_dir(output_fol)
    if output_name == '':
        output_name = 'fmri_{}_{}_{}.npy'.format(task, contrast_name, hemi)
    output_name = op.join(output_fol, output_name)
    print('Saving {}'.format(output_name))
    np.save(output_name, x)
    return output_name


def init_clusters(subject, input_fname):
    contrast_per_hemi, verts_per_hemi = {}, {}
    for hemi in utils.HEMIS:
        fmri_fname = input_fname.format(hemi=hemi)
        if utils.file_type(input_fname) == 'npy':
            x = np.load(fmri_fname)
            contrast_per_hemi[hemi] = x #[:, 0]
        else:
            # try nibabel
            x = nib.load(fmri_fname)
            contrast_per_hemi[hemi] = x.get_data().ravel()
        pial_npz_fname = op.join(MMVT_DIR, subject, 'surf', '{}.pial.npz'.format(hemi))
        if not op.isfile(pial_npz_fname):
            print('No pial npz file (), creating one'.format(pial_npz_fname))
            verts, faces = utils.read_ply_file(op.join(MMVT_DIR, subject, 'surf', '{}.pial.ply'.format(hemi)))
            np.savez(pial_npz_fname[:-4], verts=verts, faces=faces)
        d = np.load(pial_npz_fname)
        verts_per_hemi[hemi] = d['verts']
    connectivity_fname = op.join(MMVT_DIR, subject, 'spatial_connectivity.pkl')
    if not op.isfile(connectivity_fname):
        from src.preproc import anatomy
        anatomy.create_spatial_connectivity(subject)
    connectivity_per_hemi = utils.load(connectivity_fname)
    return contrast_per_hemi, connectivity_per_hemi, verts_per_hemi


def find_clusters(subject, surf_template_fname, t_val, atlas, task, n_jobs=1):
    # contrast_name = contrast_name if volume_name == '' else volume_name
    # volume_name = volume_name if volume_name != '' else contrast_name
    # if input_fol == '':
    #     input_fol = op.join(MMVT_DIR, subject, 'fmri')
    # input_fname = op.join(input_fol, 'fmri_{}_{}_{}.npy'.format(task, contrast_name, '{hemi}'))

    surf_full_input_fname = op.join(MMVT_DIR, subject, 'fmri', surf_template_fname)
    surf_full_input_fnames = find_hemi_files_from_template(surf_full_input_fname)
    if len(surf_full_input_fnames) == 0:
        print('No hemi files were found from the template {}'.format(surf_full_input_fname))
        return False
    surf_full_input_fname = utils.select_one_file(surf_full_input_fnames, surf_full_input_fname, 'fMRI surf')
    contrast, connectivity, verts = init_clusters(subject, surf_full_input_fname)
    clusters_labels = utils.Bag(dict(threshold=t_val, values=[]))
    for hemi in utils.HEMIS:
        clusters, _ = mne_clusters._find_clusters(contrast[hemi], t_val, connectivity=connectivity[hemi])
        # blobs_output_fname = op.join(input_fol, 'blobs_{}_{}.npy'.format(contrast_name, hemi))
        # print('Saving blobs: {}'.format(blobs_output_fname))
        # save_clusters_for_blender(clusters, contrast[hemi], blobs_output_fname)
        clusters_labels_hemi = lu.find_clusters_overlapped_labeles(
            subject, clusters, contrast[hemi], atlas, hemi, verts[hemi], n_jobs)
        if clusters_labels_hemi is None:
            print("Can't find clusters in {}!".format(hemi))
        else:
            clusters_labels.values.extend(clusters_labels_hemi)

    name = utils.namebase(surf_full_input_fname).replace('_{hemi}', '').replace('fmri_', '')
    if task != '':
        name = '{}_{}'.format(name, task)
    clusters_labels_output_fname = op.join(
        MMVT_DIR, subject, 'fmri', 'clusters_labels_{}.pkl'.format(name, atlas))
    print('Saving clusters labels: {}'.format(clusters_labels_output_fname))
    utils.save(clusters_labels, clusters_labels_output_fname)
    return op.isfile(clusters_labels_output_fname)


# def find_clusters_tval_hist(subject, contrast_name, output_fol, input_fol='', n_jobs=1):
#     contrast, connectivity, _ = init_clusters(subject, contrast_name, input_fol)
#     clusters = {}
#     tval_values = np.arange(2, 20, 0.1)
#     now = time.time()
#     for ind, tval in enumerate(tval_values):
#         try:
#             # utils.time_to_go(now, ind, len(tval_values), 5)
#             clusters[tval] = {}
#             for hemi in utils.HEMIS:
#                 clusters[tval][hemi], _ = mne_clusters._find_clusters(
#                     contrast[hemi], tval, connectivity=connectivity[hemi])
#             print('tval: {:.2f}, len rh: {}, lh: {}'.format(tval, max(map(len, clusters[tval]['rh'])),
#                                                         max(map(len, clusters[tval]['rh']))))
#         except:
#             print('error with tval {}'.format(tval))
#     utils.save(clusters, op.join(output_fol, 'clusters_tval_hist.pkl'))


def load_clusters_tval_hist(input_fol):
    from itertools import chain
    clusters = utils.load(op.join(input_fol, 'clusters_tval_hist.pkl'))
    res = []
    for t_val, clusters_tval in clusters.items():
        tval = float('{:.2f}'.format(t_val))
        max_size = max([max([len(c) for c in clusters_tval[hemi]]) for hemi in utils.HEMIS])
        avg_size = np.mean(list(chain.from_iterable(([[len(c) for c in clusters_tval[hemi]] for hemi in utils.HEMIS]))))
        clusters_num = sum(map(len, [clusters_tval[hemi] for hemi in utils.HEMIS]))
        res.append((tval, max_size, avg_size, clusters_num))
    res = sorted(res)
    # res = sorted([(t_val, max([len(c) for c in [c_tval[hemi] for hemi in utils.HEMIS]])) for t_val, c_tval in clusters.items()])
    # tvals = [float('{:.2f}'.format(t_val)) for t_val, c_tval in clusters.items()]
    max_sizes = [r[1] for r in res]
    avg_sizes = [r[2] for r in res]
    tvals = [float('{:.2f}'.format(r[0])) for r in res]
    clusters_num = [r[3] for r in res]
    fig, ax1 = plt.subplots()
    ax1.plot(tvals, max_sizes, 'b')
    ax1.set_ylabel('max size', color='b')
    ax2 = ax1.twinx()
    ax2.plot(tvals, clusters_num, 'r')
    # ax2.plot(tvals, avg_sizes, 'g')
    ax2.set_ylabel('#clusters', color='r')
    plt.show()
    print('sdfsd')


def save_clusters_for_blender(clusters, contrast, output_file):
    vertices_num = len(contrast)
    data = np.ones((vertices_num, 4)) * -1
    colors = utils.get_spaced_colors(len(clusters))
    for ind, (cluster, color) in enumerate(zip(clusters, colors)):
        x = contrast[cluster]
        cluster_max = max([abs(np.min(x)), abs(np.max(x))])
        cluster_data = np.ones((len(cluster), 1)) * cluster_max
        cluster_color = np.tile(color, (len(cluster), 1))
        data[cluster, :] = np.hstack((cluster_data, cluster_color))
    np.save(output_file, data)


def create_functional_rois(subject, contrast_name, clusters_labels_fname='', func_rois_folder=''):
    if clusters_labels_fname == '':
        clusters_labels = utils.load(op.join(
            MMVT_DIR, subject, 'fmri', 'clusters_labels_{}.npy'.format(contrast_name)))
    if func_rois_folder == '':
        func_rois_folder = op.join(SUBJECTS_DIR, subject, 'mmvt', 'fmri', 'functional_rois', '{}_labels'.format(contrast_name))
    utils.delete_folder_files(func_rois_folder)
    for cl in clusters_labels:
        cl_name = 'fmri_{}_{:.2f}'.format(cl['name'], cl['max'])
        new_label = mne.Label(cl['vertices'], cl['coordinates'], hemi=cl['hemi'], name=cl_name,
            filename=None, subject=subject, verbose=None)
        new_label.save(op.join(func_rois_folder, cl_name))


def show_fMRI_using_pysurfer(subject, input_file, hemi='both'):
    brain = Brain(subject, hemi, "pial", curv=False, offscreen=False)
    brain.toggle_toolbars(True)
    if hemi=='both':
        for hemi in ['rh', 'lh']:
            print('adding {}'.format(input_file.format(hemi=hemi)))
            brain.add_overlay(input_file.format(hemi=hemi), hemi=hemi)
    else:
        print('adding {}'.format(input_file.format(hemi=hemi)))
        brain.add_overlay(input_file.format(hemi=hemi), hemi=hemi)


def mri_convert_hemis(contrast_file_template, contrasts=None, existing_format='nii.gz'):
    for hemi in utils.HEMIS:
        if contrasts is None:
            contrasts = ['']
        for contrast in contrasts:
            if '{contrast}' in contrast_file_template:
                contrast_fname = contrast_file_template.format(hemi=hemi, contrast=contrast, format='{format}')
            else:
                contrast_fname = contrast_file_template.format(hemi=hemi, format='{format}')
            if not op.isfile(contrast_fname.format(format='mgz')):
                convert_fmri_file(contrast_fname, existing_format, 'mgz')


# def mri_convert(volume_fname, from_format='nii.gz', to_format='mgz'):
#     try:
#         print('convert {} to {}'.format(volume_fname.format(format=from_format), volume_fname.format(format=to_format)))
#         utils.run_script('mri_convert {} {}'.format(volume_fname.format(format=from_format),
#                                                     volume_fname.format(format=to_format)))
#     except:
#         print('Error running mri_convert!')


def convert_fmri_file(input_fname_template, from_format='nii.gz', to_format='mgz'):
    try:
        output_fname = input_fname_template.format(format=to_format)
        intput_fname = input_fname_template.format(format=from_format)
        output_files = glob.glob(output_fname)
        if len(output_files) == 0:
            inputs_files = glob.glob(intput_fname)
            if len(inputs_files) == 1:
                intput_fname = inputs_files[0]
                if not op.isfile(output_fname):
                    utils.run_script('mri_convert {} {}'.format(intput_fname, output_fname))
                return output_fname
            elif len(inputs_files) == 0:
                print('No imput file was found! {}'.format(intput_fname))
                return ''
            else:
                print('Too many input files were found! {}'.format(intput_fname))
                return ''
        else:
            return output_files[0]
    except:
        print('Error running mri_convert!')
        return ''


def calc_subs_surface_activity(subject, fmri_file_template, template_brains, threshold=2, subcortical_codes_fname='',
        aseg_stats_file_name='', method='max', k_points=100, format='mgz', do_plot=False):
    # todo: Should fix:
    # 1) morph the data to subject's space / read vertices from the template brain
    # 2) Solve issues if the data has time dim
    volume_fname = find_fmri_fname_template(
        subject, fmri_file_template, template_brains, only_volumes=True, format=format)
    x = nib.load(volume_fname)
    x_data = x.get_data()
    seg_labels = get_subs_names(subcortical_codes_fname, aseg_stats_file_name)

    if do_plot:
        fig = plt.figure()
        ax = Axes3D(fig)

    sig_subs = []
    # Find the segmentation file
    aseg = morph_aseg(subject, x_data, volume_fname)
    out_folder = op.join(MMVT_DIR, subject, 'fmri', 'subcortical_fmri_activity')
    if not op.isdir(out_folder):
        os.mkdir(out_folder)
    sub_cortical_generator = utils.sub_cortical_voxels_generator(aseg, seg_labels, spacing=5, use_grid=False)
    for pts, seg_name, seg_id in sub_cortical_generator:
        print(seg_name)
        verts, _ = utils.read_ply_file(op.join(MMVT_DIR, subject, 'subcortical', '{}.npz'.format(seg_name)))
        vals = np.array([x_data[i, j, k] for i, j, k in pts])
        is_sig = np.max(np.abs(vals)) >= threshold
        print(seg_name, seg_id, np.mean(vals), is_sig)
        pts = utils.transform_voxels_to_RAS(aseg.header, pts)
        # plot_points(verts,pts)
        verts_vals = calc_vert_vals(verts, pts, vals, method=method, k_points=k_points)
        print('verts vals: {}+-{}'.format(verts_vals.mean(), verts_vals.std()))
        if sum(abs(verts_vals) > threshold) > 0:
            sig_subs.append(seg_name)
        verts_colors = utils.arr_to_colors_two_colors_maps(verts_vals, threshold=2)
        verts_data = np.hstack((np.reshape(verts_vals, (len(verts_vals), 1)), verts_colors))
        np.save(op.join(out_folder, seg_name), verts_data)
        if do_plot:
            plot_points(verts, colors=verts_colors, fig_name=seg_name, ax=ax)
        # print(pts)
    if do_plot:
        plt.savefig(op.join(MMVT_DIR, subject, 'fmri', 'subcorticals_surface_activity.png'))
        plt.show()


def morph_aseg(subject, x_data, volume_fname, aseg_fname='', new_aseg_fname=''):
    if aseg_fname == '':
        aseg_fname = op.join(SUBJECTS_DIR, subject, 'mri', 'aseg.mgz')
    aseg = nib.load(aseg_fname)
    if np.any(x_data.shape[:3] != aseg.shape):
        if new_aseg_fname == '':
            new_aseg_fname = op.join(FMRI_DIR, subject, 'aseg.mgz')
        if op.isfile(new_aseg_fname):
            aseg = nib.load(new_aseg_fname)
        if np.any(x_data.shape[:3] != aseg.shape):
            utils.remove_file(new_aseg_fname)
            fu.vol2vol(subject, aseg_fname, volume_fname, new_aseg_fname)
        aseg = nib.load(new_aseg_fname)
    return aseg


def get_subs_names(subcortical_codes_fname, aseg_stats_file_name=''):
    if subcortical_codes_fname != '':
        subcortical_codes = np.genfromtxt(subcortical_codes_fname, dtype=str, delimiter=',')
        seg_labels = list(map(str, subcortical_codes[:, 0]))
    elif aseg_stats_file_name != '':
        aseg_stats = np.genfromtxt(aseg_stats_file_name, dtype=str, delimiter=',', skip_header=1)
        seg_labels = list(map(str, aseg_stats[:, 0]))
    else:
        raise Exception('No segmentation file!')
    return seg_labels


def calc_subs_activity(subject, fmri_file_template, measures=['mean'], subcortical_codes_fname='', overwrite=False):
    volume_fname = get_fmri_fname(subject, fmri_file_template, only_volumes=True, raise_exception=False)
    x = nib.load(volume_fname)
    x_data = x.get_data()
    seg_labels = get_subs_names(subcortical_codes_fname)

    # Find the segmentation file
    out_folder = op.join(MMVT_DIR, subject, 'fmri')
    if not op.isdir(out_folder):
        os.mkdir(out_folder)
    aseg = morph_aseg(subject, x_data, volume_fname)
    out_fnames = []
    if isinstance(measures, str):
        measures = [measures]
    for measure in measures:
        labels_data, seg_names = [], []
        out_fname = op.join(out_folder, 'subcorticals_{}.npz'.format(measure))
        out_fnames.append(out_fname)
        if op.isfile(out_fname) and not overwrite:
            continue
        sub_cortical_generator = utils.sub_cortical_voxels_generator(aseg, seg_labels, use_grid=False)
        for pts, seg_name, seg_id in sub_cortical_generator:
            seg_names.append(seg_name)
            x = np.array([x_data[i, j, k] for i, j, k in pts])
            if measure == 'mean':
                labels_data.append(np.mean(x, 0))
            elif measure.startswith('pca'):
                comps_num = 1 if '_' not in measure else int(measure.split('_')[1])
                labels_data.append(utils.pca(x, comps_num))
        labels_data = np.array(labels_data, dtype=np.float64)
        np.savez(out_fname, data=labels_data, names=seg_names)
        print('Writing to {}, {}'.format(out_fname, labels_data.shape))
    return all([op.isfile(o) for o in out_fnames])


def calc_vert_vals(verts, pts, vals, method='max', k_points=100):
    ball_tree = BallTree(pts)
    dists, pts_inds = ball_tree.query(verts, k=k_points, return_distance=True)
    near_vals = vals[pts_inds]
    # sig_dists = dists[np.where(abs(near_vals)>2)]
    cover = len(np.unique(pts_inds.ravel()))/float(len(pts))
    print('{}% of the points are covered'.format(cover*100))
    if method=='dist':
        n_dists = 1/(dists**2)
        norm = 1/np.sum(n_dists, 1)
        norm = np.reshape(norm, (len(norm), 1))
        n_dists = norm * n_dists
        verts_vals = np.sum(near_vals * n_dists, 1)
    elif method=='max':
        verts_vals = near_vals[range(near_vals.shape[0]), np.argmax(abs(near_vals), 1)]
    return verts_vals


def plot_points(subject, verts, pts=None, colors=None, fig_name='', ax=None):
    if ax is None:
        fig = plt.figure()
        ax = Axes3D(fig)
    colors = 'tomato' if colors is None else colors
    # ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], 'o', color=colors, label='verts')
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=20, c=colors, label='verts')
    if pts is not None:
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'o', color='blue', label='voxels')
        plt.legend()
    if ax is None:
        plt.savefig(op.join(MMVT_DIR, subject, 'fmri', '{}.jpg'.format(fig_name)))
        plt.close()


def project_on_surface(subject, volume_file, surf_output_fname,
                       target_subject=None, overwrite_surf_data=False, is_pet=False):
    if target_subject is None:
        target_subject = subject
    utils.make_dir(op.join(MMVT_DIR, subject, 'fmri'))
    for hemi in utils.HEMIS:
        if not op.isfile(surf_output_fname.format(hemi=hemi)) or overwrite_surf_data:
            print('project {} to {}'.format(volume_file, hemi))
            if not is_pet:
                surf_data = fu.project_volume_data(volume_file, hemi, subject_id=subject, surf="pial", smooth_fwhm=3,
                    target_subject=target_subject, output_fname=surf_output_fname.format(hemi=hemi))
            else:
                surf_data = fu.project_pet_volume_data(subject, volume_file, hemi, surf_output_fname.format(hemi=hemi))
            nans = np.sum(np.isnan(surf_data))
            if nans > 0:
                print('there are {} nans in {} surf data!'.format(nans, hemi))
        surf_data = np.squeeze(nib.load(surf_output_fname.format(hemi=hemi)).get_data())
        output_fname = op.join(MMVT_DIR, subject, 'fmri', 'fmri_{}'.format(op.basename(surf_output_fname.format(hemi=hemi))))
        npy_output_fname = op.splitext(output_fname)[0]
        if not op.isfile('{}.npy'.format(npy_output_fname)) or overwrite_surf_data:
            print('Saving surf data in {}.npy'.format(npy_output_fname))
            np.save(npy_output_fname, surf_data)


def load_surf_files(subject, surf_template_fname, overwrite_surf_data=False):
    utils.make_dir(op.join(MMVT_DIR, subject, 'fmri'))
    surf_full_output_fname = op.join(FMRI_DIR, subject, surf_template_fname).replace('{subject}', subject)
    surf_full_output_fnames = find_hemi_files_from_template(surf_full_output_fname)
    if len(surf_full_output_fnames) == 0:
        surf_full_output_fname = op.join(MMVT_DIR, subject, 'fmri', surf_template_fname)
        surf_full_output_fnames = find_hemi_files_from_template(surf_full_output_fname)
        if len(surf_full_output_fnames) == 0:
            print('No hemi files were found from the template {}'.format(surf_full_output_fname))
            return False, ''
    surf_full_output_fname = surf_full_output_fnames[0]
    output_fname_template = op.join(MMVT_DIR, subject, 'fmri', 'fmri_{}'.format(op.basename(
        surf_full_output_fname)))
    npy_output_fname_template = '{}.npy'.format(op.splitext(output_fname_template)[0])
    if utils.both_hemi_files_exist(npy_output_fname_template) and not overwrite_surf_data:
        return True, npy_output_fname_template
    for hemi in utils.HEMIS:
        fmri_fname = surf_full_output_fname.format(hemi=hemi)
        x = np.squeeze(nib.load(fmri_fname).get_data())
        morph_from_subject = check_vertices_num(subject, hemi, x)
        if subject != morph_from_subject:
            morphed_fmri_fname = '{0}_morphed_to_{2}{1}'.format(*op.splitext(fmri_fname), subject)
            if not op.isfile(morphed_fmri_fname):
                fu.surf2surf(morph_from_subject, subject, hemi, fmri_fname, morphed_fmri_fname)
            x = np.squeeze(nib.load(morphed_fmri_fname).get_data())
        npy_output_fname = npy_output_fname_template.format(hemi=hemi)
        if not op.isfile(npy_output_fname) or overwrite_surf_data:
            print('Saving surf data in {}'.format(npy_output_fname))
            np.save(npy_output_fname, x)
    return utils.both_hemi_files_exist(npy_output_fname_template), npy_output_fname_template


def calc_files_diff(subject, surf_template_fname, overwrite_surf_data=False):
    surf_template_fnames = surf_template_fname.split(',')
    if len(surf_template_fnames) != 2:
        print('calc_files_diff: surf_template_fname should be 2 names seperated with a comma.')
        return False, ''
    both_files_exist = True
    npy_output_fname_template = op.join(MMVT_DIR, subject, 'fmri', 'fmri_{}minus_{}'.format(
        *[tmp.replace('*', '').replace('{hemi}', '') for tmp in surf_template_fnames]) + '{hemi}.npy')
    for hemi in utils.HEMIS:
        surfs_data = []
        for ind, fname_template in enumerate(surf_template_fnames):
            surf_full_output_fname = op.join(MMVT_DIR, subject, 'fmri', fname_template)
            surf_full_output_fnames = find_hemi_files_from_template(surf_full_output_fname, 'npy')
            if len(surf_full_output_fnames) == 0:
                print("Cant find {}!".format(surf_full_output_fname))
                return False, ''
            surf_full_output_fname = surf_full_output_fnames[0]
            x = np.load(surf_full_output_fname.format(hemi=hemi))
            surfs_data.append(x)
        shapes = [surfs_data[0].shape, surfs_data[1].shape]
        if shapes[0] != shapes[1]:
            print('calc_files_diff: Files have different shapes! {}'.format(shapes))
            if shapes[0][0] == shapes[1][0] and shapes[0][1] != shapes[1][1]:
                argmax = 0 if shapes[0][1] > shapes[1][1] else 1
                argmin = 1 - argmax
                nskip = shapes[argmax][1] - shapes[argmin][1]
                print('Skipping {} first frames in {}'.format(nskip, surf_template_fnames[argmax]))
                surfs_data[argmax] = surfs_data[argmax][:, nskip:]
            else:
                return False, ''
        surfs_diff = surfs_data[0] - surfs_data[1]
        npy_output_fname = npy_output_fname_template.format(hemi=hemi)
        if not op.isfile(npy_output_fname) or overwrite_surf_data:
            print('Saving surf data in {}'.format(npy_output_fname))
            np.save(npy_output_fname, surfs_diff)
        both_files_exist = both_files_exist and op.isfile(npy_output_fname)
    return both_files_exist, npy_output_fname_template


def load_images_file(image_fname):
    for hemi in ['rh', 'lh']:
        x = nib.load(image_fname.format(hemi=hemi))
        nans = np.sum(np.isnan(np.array(x.dataobj)))
        if nans > 0:
            print('there are {} nans in {} image!'.format(nans, hemi))


def mask_volume(volume, mask, masked_volume):
    vol_nib = nib.load(volume)
    vol_data = vol_nib.get_data()
    mask_nib = nib.load(mask)
    mask_data = mask_nib.get_data().astype(np.bool)
    vol_data[mask_data] = 0
    vol_nib.data = vol_data
    nib.save(vol_nib, masked_volume)


def load_and_show_npy(subject, npy_file, hemi):
    x = np.load(npy_file)
    brain = Brain(subject, hemi, "pial", curv=False, offscreen=False)
    brain.toggle_toolbars(True)
    brain.add_overlay(x[:, 0], hemi=hemi)


def copy_volume_to_blender(subject, volume_fname_template, contrast='', overwrite_volume_mgz=True):
    if op.isfile(volume_fname_template.format(format='mgh')) and \
            (not op.isfile(volume_fname_template.format(format='mgz')) or overwrite_volume_mgz):
        fu.mri_convert(volume_fname_template, 'mgh', 'mgz')
        format = 'mgz'
    else:
        # volume_files = glob.glob(op.join(volume_fname_template.replace('{format}', '*')))
        volume_files = find_volume_files_from_template(volume_fname_template.replace('{format}', '*'))
        if len(volume_files) == 0:
            print('No volume file! Should be in {}'.format(volume_fname_template.replace('{format}', '*')))
            return ''
        if len(volume_files) > 1:
            print('Too many volume files!')
            return ''
        else:
            format = utils.file_type(volume_files[0])
    volume_fname = volume_fname_template.format(format=format)
    blender_volume_fname = op.basename(volume_fname) if contrast=='' else '{}.{}'.format(contrast, format)
    utils.make_dir(op.join(MMVT_DIR, subject, 'freeview'))
    shutil.copyfile(volume_fname, op.join(MMVT_DIR, subject, 'freeview', blender_volume_fname))
    return volume_fname


# def project_volume_to_surface_get_files(subject, remote_subject_dir, args):
#     necessary_files = {'mri': ['orig.mgz'],
#                        'surf': ['lh.pial', 'rh.pial', 'lh.thickness', 'rh.thickness']}
#     return utils.prepare_subject_folder(
#         necessary_files, subject, remote_subject_dir, SUBJECTS_DIR,
#         args.sftp, args.sftp_username, args.sftp_domain, args.sftp_password,
#         args.overwrite_fs_files, args.print_traceback, args.sftp_port)


@utils.check_for_freesurfer
@utils.files_needed({'mri': ['orig.mgz'], 'surf': ['lh.pial', 'rh.pial', 'lh.thickness', 'rh.thickness']})
def project_volume_to_surface(subject, volume_fname_template, overwrite_surf_data=True,
                              target_subject='', remote_fmri_dir='', is_pet=False):
    if target_subject == '':
        target_subject = subject
    utils.make_dir(op.join(MMVT_DIR, subject, 'freeview'))
    volume_fname, surf_output_fname, npy_surf_fname = get_volume_and_surf_fnames(
        subject, volume_fname_template, target_subject, remote_fmri_dir)
    if volume_fname == '':
        return False, ''
    if not utils.both_hemi_files_exist(npy_surf_fname) or overwrite_surf_data:
        project_on_surface(subject, volume_fname, surf_output_fname,
                       target_subject, overwrite_surf_data=overwrite_surf_data, is_pet=is_pet)
    freeview_volume_fname = op.join(MMVT_DIR, subject, 'freeview', op.basename(volume_fname))
    if not op.isfile(freeview_volume_fname):
        shutil.copy(volume_fname, freeview_volume_fname)
    return utils.both_hemi_files_exist(npy_surf_fname) and op.isfile(freeview_volume_fname), npy_surf_fname


def get_volume_and_surf_fnames(subject, volume_fname_template, target_subject='', remote_fmri_dir=''):
    remote_fmri_dir = op.join(FMRI_DIR, subject) if remote_fmri_dir == '' else remote_fmri_dir
    volume_fname_template = volume_fname_template if volume_fname_template != '' else '*'
    full_input_fname_template = op.join(remote_fmri_dir, volume_fname_template)
    full_input_fname_template = full_input_fname_template.replace('{format}', '*')
    full_input_fname_template = full_input_fname_template.format(subject=subject)
    print('input_fname_template: {}'.format(full_input_fname_template))
    volume_fname = utils.look_for_one_file(full_input_fname_template, 'fMRI volume files', pick_the_first_one=False,
                                           search_func=find_volume_files_from_template)
    if volume_fname is None:
        print("Can't find the input file! {}".format(full_input_fname_template))
        return '', '', ''

    utils.make_dir(op.join(FMRI_DIR, subject))
    local_fname = op.join(FMRI_DIR, subject, utils.namesbase_with_ext(volume_fname))
    if not op.isfile(local_fname):
        shutil.copy(volume_fname, local_fname)
    volume_fname = local_fname

    target_subject_prefix = '_{}'.format(target_subject) if subject != target_subject else ''
    surf_output_fname = op.join(utils.get_parent_fol(volume_fname), '{}{}_{}.mgz'.format(
        utils.namebase(volume_fname), target_subject_prefix, '{hemi}'))
    npy_surf_fname = op.join(MMVT_DIR, subject, 'fmri',
                             'fmri_{}.npy'.format(utils.namebase(surf_output_fname.format(hemi='{hemi}'))))
    return volume_fname, surf_output_fname, npy_surf_fname


def calc_meg_activity_for_functional_rois(subject, meg_subject, atlas, task, contrast_name, contrast, inverse_method):
    fname_format, fname_format_cond, events_id, event_digit = meg.get_fname_format(task)
    raw_cleaning_method = 'tsss' # 'nTSSS'
    files_includes_cond = True
    meg.init_globals(meg_subject, subject, fname_format, fname_format_cond, files_includes_cond, raw_cleaning_method, contrast_name,
        SUBJECTS_MEG_DIR, task, SUBJECTS_DIR, MMVT_DIR)
    root_fol = op.join(SUBJECTS_DIR, subject, 'mmvt', 'fmri', 'functional_rois')
    labels_fol = op.join(root_fol, '{}_labels'.format(contrast))
    labels_output_fname = op.join(root_fol, '{}_labels_data_{}'.format(contrast, '{hemi}'))
    # src = meg.create_smooth_src(subject)
    for hemi in ['rh', 'lh']:
        meg.calc_labels_avg_per_condition(atlas, hemi, 'pial', events_id, labels_from_annot=False,
            labels_fol=labels_fol, stcs=None,
            labels_output_fname_template=labels_output_fname)


def copy_volumes(subject, contrast_file_template, contrast, volume_fol, volume_name):
    contrast_format = 'mgz'
    volume_type = 'mni305'
    volume_file = contrast_file_template.format(contrast=contrast, hemi=volume_type, format='{format}')
    if not op.isfile(volume_file.format(format=contrast_format)):
        fu.mri_convert(volume_file, 'nii.gz', contrast_format)
    volume_fname = volume_file.format(format=contrast_format)
    subject_volume_fname = op.join(volume_fol, '{}_{}'.format(subject, volume_name))
    if not op.isfile(subject_volume_fname):
        volume_fol, volume_name = op.split(volume_fname)
        fu.transform_mni_to_subject(subject, volume_fol, volume_name, '{}_{}'.format(subject, volume_name))
    blender_volume_fname = op.join(MMVT_DIR, subject, 'freeview', '{}.{}'.format(contrast, contrast_format))
    if not op.isfile(blender_volume_fname):
        print('copy {} to {}'.format(subject_volume_fname, blender_volume_fname))
        shutil.copyfile(subject_volume_fname, blender_volume_fname)


def analyze_4d_data(subject, atlas, input_fname_template, measures=['mean'], template_brain='', norm_percs=(1,99),
                          overwrite=False, remote_fmri_dir='', do_plot=False, do_plot_all_vertices=False,
                          excludes=('corpuscallosum', 'unknown'), input_format='nii.gz'):
    files_exist = all([utils.both_hemi_files_exist(op.join(
        MMVT_DIR, subject, 'fmri', 'labels_data_{}_{}_{}.npz'.format(atlas, em, '{hemi}'))) for em in measures])
    minmax_fname_template = op.join(MMVT_DIR, subject, 'fmri', 'labels_data_{}_{}_minmax.pkl'.format(atlas, '{em}'))
    minmax_exist = all([op.isfile(minmax_fname_template.format(em=em)) for em in measures])
    if files_exist and minmax_exist and not overwrite:
        return True
    utils.make_dir(op.join(MMVT_DIR, subject, 'fmri'))
    morph_from_subject = subject if template_brain == '' else template_brain
    figures_dir = op.join(remote_fmri_dir, subject, 'figures')
    input_fname_template_file = get_fmri_fname(subject, input_fname_template, only_volumes=False, raise_exception=False)
    # input_fname_template_file = find_4d_fmri_file(subject, input_fname_template, template_brain, remote_fmri_dir)
    labels_minmax = defaultdict(list)
    for hemi in utils.HEMIS:
        fmri_fname = input_fname_template_file.format(hemi=hemi)
        fmri_fname = convert_fmri_file(fmri_fname, from_format=input_format)
        print('loading {} ({})'.format(fmri_fname, utils.file_modification_time(fmri_fname)))
        x = nib.load(fmri_fname).get_data()
        morph_from_subject = check_vertices_num(subject, hemi, x, morph_from_subject)
        # print(max([max(label.vertices) for label in labels]))
        labels = []
        for em in measures:
            output_fname = op.join(MMVT_DIR, subject, 'fmri', 'labels_data_{}_{}_{}.npz'.format(atlas, em, hemi))
            if op.isfile(output_fname) and not overwrite:
                print('{} already exist'.format(output_fname))
                if not op.isfile(minmax_fname_template.format(em=em)) or overwrite:
                    labels_data = np.load(output_fname)['data']
                    labels_minmax[em].append(utils.calc_min_max(labels_data, norm_percs=norm_percs))
                continue
            if len(labels) == 0:
                # labels = lu.read_hemi_labels(morph_from_subject, SUBJECTS_DIR, atlas, hemi)
                labels = lu.read_labels(morph_from_subject, SUBJECTS_DIR, atlas, hemi=hemi)
                if len(labels) == 0:
                    print('No {} {} labels were found!'.format(morph_from_subject, atlas))
                    return False
            labels_data, labels_names = lu.calc_time_series_per_label(
                x, labels, em, excludes, figures_dir, do_plot, do_plot_all_vertices)
            np.savez(output_fname, data=labels_data, names=labels_names)
            labels_minmax[em].append(utils.calc_min_max(labels_data, norm_percs=norm_percs))
            print('{} was saved'.format(output_fname))

    for em in measures:
        if not op.isfile(minmax_fname_template.format(em=em)) or overwrite:
            data_min, data_max = utils.calc_minmax_from_arr(labels_minmax[em])
            utils.save((data_min, data_max), minmax_fname_template.format(em=em))

    files_exist = all([utils.both_hemi_files_exist(op.join(
        MMVT_DIR, subject, 'fmri', 'labels_data_{}_{}_{}.npz'.format(atlas, em, '{hemi}'))) for em in measures])
    minmax_exist = all([op.isfile(minmax_fname_template.format(em=em)) for em in measures])
    return files_exist and minmax_exist


# def find_4d_fmri_file(subject, input_fname_template, template_brain='', remote_fmri_dir=''):
#     input_fname_template_files, full_input_fname_template = build_fmri_contrast_file_template(
#         subject, input_fname_template, template_brain, remote_fmri_dir)
#     if len(input_fname_template_files) > 1:
#         print('More the one file was found! {}'.format(full_input_fname_template))
#         print(input_fname_template_files)
#         return ''
#     elif len(input_fname_template_files) == 0:
#         print("Can't find template files! {}".format(full_input_fname_template))
#         print(subject, input_fname_template, template_brain, remote_fmri_dir)
#         return ''
#     return input_fname_template_files[0]


@utils.tryit()
def calc_labels_mean_freesurfer_get_files(
        args, remote_subject_dir, subject, atlas, input_fname_template, templates_brain='', target_subject='',
        remote_fmri_dir=''):
    input_fname_template_file = find_fmri_fname_template(subject, input_fname_template, templates_brain, False, format)
    fmri_fname = input_fname_template_file.format(hemi='rh')
    if not op.isfile(fmri_fname):
        target_subject = subject
    if target_subject == '':
        x = nib.load(fmri_fname)
        if x.shape[0] in [FSAVG5_VERTS, FSAVG_VERTS]:
            target_subject = 'fsaverage5' if x.shape[0] == FSAVG5_VERTS else 'fsaverage'
        else:
            target_subject = subject

    annot_template_fname = op.join(SUBJECTS_DIR, target_subject, 'label', '{}.{}.annot'.format('{hemi}', atlas))
    if not utils.both_hemi_files_exist(annot_template_fname):
        print('{} does not exist!'.format(annot_template_fname))
        necessary_files = {'label': ['lh.{}.annot'.format(atlas), 'rh.{}.annot'.format(atlas)]}
        utils.prepare_subject_folder(
            necessary_files, target_subject, remote_subject_dir, SUBJECTS_DIR,
            args.sftp, args.sftp_username, args.sftp_domain, args.sftp_password,
            args.overwrite_fs_files, args.print_traceback, args.sftp_port)
    return utils.both_hemi_files_exist(annot_template_fname)


@utils.tryit()
def calc_labels_mean_freesurfer(
        subject, atlas, input_fname_template, templates_brain='', target_subject='',
        remote_fmri_dir='', overwrite=True, excludes=('corpuscallosum', 'unknown'),
        overwrite_mri_segstat=False, norm_percs=(1,99)):
    # input_fname_template_file = find_4d_fmri_file(subject, input_fname_template, template_brain, remote_fmri_dir)
    output_fname_hemi = op.join(MMVT_DIR, subject, 'fmri', 'labels_data_{}_mean_{}.npz'.format(atlas, '{hemi}'))
    minmax_fname_template = op.join(MMVT_DIR, subject, 'fmri', 'labels_data_{}_mean_minmax.pkl'.format(atlas))
    if utils.both_hemi_files_exist(output_fname_hemi) and op.isfile(minmax_fname_template) and not overwrite:
        return True
    input_fname_template_file = find_fmri_fname_template(subject, input_fname_template, templates_brain, False, format)
    res_dir = op.join(FMRI_DIR, subject)
    utils.make_dir(res_dir)
    labels_minmax = []
    for hemi in utils.HEMIS:
        output_fname = output_fname_hemi.format(hemi=hemi)
        fmri_fname = input_fname_template_file.format(hemi=hemi)
        if op.isfile(output_fname) and not overwrite:
            continue
        if target_subject == '':
            x = nib.load(fmri_fname)
            if x.shape[0] in [FSAVG5_VERTS, FSAVG_VERTS]:
                target_subject = 'fsaverage5' if x.shape[0] == FSAVG5_VERTS else 'fsaverage'
            else:
                target_subject = subject
        labels_data, labels_names = fu.calc_labels_avg(
            target_subject, hemi, atlas, fmri_fname, res_dir, SUBJECTS_DIR, overwrite)
        labels_names, labels_data = lu.remove_exclude_labels_and_data(labels_names, labels_data, excludes)
        np.savez(output_fname, data=labels_data, names=labels_names)
        print('{} was saved ({} labels)'.format(output_fname, len(labels_names)))
        labels_minmax.append(utils.calc_min_max(labels_data, norm_percs=norm_percs))

    data_min, data_max = utils.calc_minmax_from_arr(labels_minmax)
    utils.save((data_min, data_max), minmax_fname_template)
    return utils.both_hemi_files_exist(output_fname_hemi) and op.isfile(minmax_fname_template)


@utils.check_for_freesurfer
def calc_volumetric_labels_mean(subject, atlas, fmri_file_template, measures=['mean'],
                                overwrite_aseg_file=False, norm_percs=(1, 99), print_only=False, args={}):
    output_fname_hemi = op.join(
        MMVT_DIR, subject, 'fmri', 'labels_data_{}_volume_{}_{}.npz'.format(atlas, '{measure}', '{hemi}'))
    minmax_fname_template = op.join(
        MMVT_DIR, subject, 'fmri', 'labels_data_{}_volume_{}_minmax.pkl'.format(atlas, '{measure}'))
    volume_fname = get_fmri_fname(subject, fmri_file_template, only_volumes=True, raise_exception=False)
    if volume_fname == '':
        return False
    ret, aparc_aseg_fname = fu.create_aparc_aseg_file(
        subject, atlas, SUBJECTS_DIR, overwrite_aseg_file, print_only, mmvt_args=args)
    if not ret:
        return False
    ret = True
    new_aseg_fname = op.join(MMVT_DIR, subject, 'fmri', utils.namesbase_with_ext(aparc_aseg_fname))
    x_data = nib.load(volume_fname).get_data()
    aparc_aseg = morph_aseg(subject, x_data, volume_fname, aparc_aseg_fname, new_aseg_fname)
    aparc_aseg_data = aparc_aseg.get_data()
    for measure in measures:
        if utils.both_hemi_files_exist(output_fname_hemi.format(hemi='{hemi', measure=measure)) and \
                op.isfile(minmax_fname_template.format(measure=measure)) and not overwrite_aseg_file:
            continue
        labels_minmax = []
        for hemi, offset in zip(['lh', 'rh'], [1000, 2000]):
            labels_names, labels_data = [], []
            _, _, names = mne.label._read_annot(
                op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format(hemi, atlas)))
            names = [name.astype(str) for name in names]
            for label_index, label_name in enumerate(names):
                label_id = label_index + offset + 1
                pts = utils.calc_label_voxels(label_id, aparc_aseg_data)
                if len(pts) == 0:
                    continue
                labels_names.append(label_name)
                x = np.array([x_data[i, j, k] for i, j, k in pts])
                if measure == 'mean':
                    labels_data.append(np.mean(x, 0))
                elif measure.startswith('pca'):
                    comps_num = 1 if '_' not in measure else int(measure.split('_')[1])
                    labels_data.append(utils.pca(x, comps_num))
            labels_data = np.array(labels_data, dtype=np.float64)
            labels_minmax.append(utils.calc_min_max(labels_data, norm_percs=norm_percs))
            output_fname = output_fname_hemi.format(hemi=hemi, measure=measure)
            np.savez(output_fname, data=labels_data, names=labels_names)
            print('Writing to {}, {}'.format(output_fname, labels_data.shape))
            ret = ret and op.isfile(output_fname)

        data_min, data_max = utils.calc_minmax_from_arr(labels_minmax)
        minmax_fname = minmax_fname_template.format(measure=measure)
        utils.save((data_min, data_max), minmax_fname_template)
        ret = ret and op.isfile(minmax_fname)

    return ret


def load_labels_ts(subject, atlas, labels_order_fname, st_template='*{atlas}*.txt', extract_measure='mean',
                   excludes=('corpuscallosum', 'unknown'), indices_to_remove_from_data=(0,4,113,117),
                   backup_existing_files=True, pick_the_first_one=False):
    if isinstance(extract_measure, list):
        if len(extract_measure) == 1:
            extract_measure = extract_measure[0]
        elif len(extract_measure) == 0:
            print('extract_measure is an empty list!')
            return False
        elif len(extract_measure) > 1:
            print('The load_labels_ts can get only one extract_measure!')
            return False
    st_template = op.join(FMRI_DIR, subject, st_template.format(atlas=atlas, subject=subject))
    st_file = utils.look_for_one_file(st_template, 'st', pick_the_first_one)
    if st_file is None:
        return False
    labels_data = np.genfromtxt(st_file).T
    ret = save_labels_data(
        subject, atlas, labels_data, labels_order_fname, extract_measure, excludes,
        indices_to_remove_from_data, backup_existing_files)
    return ret


def save_labels_data(
        subject, atlas, labels_data, labels_order_fname, extract_measure='mean', excludes=('corpuscallosum', 'unknown'),
        indices_to_remove_from_data=(0, 4, 113, 117), backup_existing_files=True):
    if len(indices_to_remove_from_data) > 0:
        labels_data = np.delete(labels_data, indices_to_remove_from_data, 0)
    labels = utils.read_list_from_file(labels_order_fname)
    labels, indices = lu.remove_exclude_labels(labels, excludes)
    remove_indices = list(set(range(len(labels))) - set(indices))
    labels_data = np.delete(labels_data, remove_indices, 0)
    if len(labels) != labels_data.shape[0]:
        print('len(labels_order) ({}) != fmri_data.shape[0] {}!'.format(len(labels), labels_data.shape[0]))
        return False
    else:
        print('len(labels_order) = fmri_data.shape[0] = {}'.format(len(labels)))
    indices = lu.get_lh_rh_indices(labels)
    labels = np.array(labels)
    output_fname_hemi = op.join(MMVT_DIR, subject, 'fmri', 'labels_data_{}_{}_{}.npz'.format(
        atlas, extract_measure, '{hemi}'))
    for hemi in utils.HEMIS:
        output_fname = output_fname_hemi.format(hemi=hemi)
        if backup_existing_files and op.isfile(output_fname):
            backup_fname = utils.add_str_to_file_name(output_fname, '_backup')
            shutil.copy(output_fname, backup_fname.format(hemi=hemi))
        labels_data_hemi = labels_data[indices[hemi]]
        labels_names_hemi = labels[indices[hemi]]
        np.savez(output_fname, data=labels_data_hemi, names=labels_names_hemi)
        print('{} was saved ({} labels)'.format(output_fname, len(labels_names_hemi)))
    return utils.both_hemi_files_exist(output_fname_hemi)


def check_vertices_num(subject, hemi, x, morph_from_subject=''):
    if x.shape[0] == FSAVG_VERTS:
        morph_from_subject = 'fsaverage'
    elif x.shape[0] == FSAVG5_VERTS:
        morph_from_subject = 'fsaverage5'
    elif x.shape[0] == COLIN27_VERTS[hemi]:
        morph_from_subject = 'colin27'
    else:
        verts, faces = utils.read_pial(subject, MMVT_DIR, hemi)
        if x.shape[0] == verts.shape[0]:
            morph_from_subject = subject
        else:
            if morph_from_subject != '':
                verts, faces = utils.read_pial(morph_from_subject, MMVT_DIR, hemi)
                if x.shape[0] != verts.shape[0]:
                    raise Exception("Can't find the subject to morph from!")
            else:
                raise Exception("Can't find the subject to morph from!")
    return morph_from_subject


def calc_labels_minmax(subject, atlas, extract_modes):
    for em in extract_modes:
        min_max_output_fname = op.join(MMVT_DIR, subject, 'fmri', 'labels_data_{}_{}_minmax.npy'.format(atlas, em))
        template = op.join(MMVT_DIR, subject, 'fmri', op.basename('labels_data_{}_{}_{}.npz'.format(atlas, em, '{hemi}')))
        if utils.both_hemi_files_exist(template):
            labels_data = [np.load(template.format(hemi=hemi)) for hemi in utils.HEMIS]
            np.save(min_max_output_fname, [min([np.min(d['data']) for d in labels_data]),
                                           max([np.max(d['data']) for d in labels_data])])
        else:
            print("Can't find {}!".format(template))
    return np.all([op.isfile(op.join(MMVT_DIR, subject, 'fmri', 'labels_data_{}_{}_minmax.npy'.format(atlas, em)))
                   for em in extract_modes])


def find_fmri_fname_template(subject, fmri_file_template, template_brains, only_volumes=False, format='mgz'):
    if isinstance(template_brains, str):
        template_brains = [template_brains]
    if fmri_file_template == '':
        fmri_file_template = '*{hemi}*{format}'
    # todo: should do something more clever than just taking the first template brain
    input_fname_template = fmri_file_template.format(
        subject=subject, morph_to_subject=template_brains[0], hemi='{hemi}', format=format)
    fmri_fname_template = get_fmri_fname(subject, '*morphed*{}'.format(fmri_file_template), raise_exception=False)
    if not utils.both_hemi_files_exist(fmri_fname_template):
        fmri_fname_template = get_fmri_fname(subject, input_fname_template, only_volumes=only_volumes)
    return fmri_fname_template


def save_dynamic_activity_map(subject, fmri_file_template='', template_brains='fsaverage', format='mgz',
                              norm_percs=(1, 99), overwrite=False):
    minmax_fname = op.join(MMVT_DIR, subject, 'fmri', 'activity_map_minmax.npy')
    fmri_fname_template = find_fmri_fname_template(subject, fmri_file_template, template_brains, False, format)
    hemi_minmax = []
    for hemi in utils.HEMIS:
        fol = op.join(MMVT_DIR, subject, 'fmri', 'activity_map_{}'.format(hemi))
        fmri_fname = fmri_fname_template.format(hemi=hemi)
        # Check if there is a morphed file
        data = nib.load(fmri_fname).get_data().squeeze()
        T = data.shape[1]
        if not overwrite and len(glob.glob(op.join(fol, '*.npy'))) == T:
            hemi_minmax.append(utils.calc_min_max(data, norm_percs=norm_percs))
            continue
        verts, faces = utils.read_pial(subject, MMVT_DIR, hemi)
        file_verts_num, subject_verts_num = data.shape[0], verts.shape[0]
        if file_verts_num != subject_verts_num:
            if file_verts_num == FSAVG_VERTS:
                target_subject = 'fsaverage'
            elif file_verts_num == FSAVG5_VERTS:
                target_subject = 'fsaverage5'
            else:
                raise Exception('save_activity_map: wrong number of vertices!')
            sp = fmri_fname.split(hemi)
            sep = '.' if '.' in utils.namebase(fmri_fname) else '_'
            target_fname = '{}{}{}{}{}'.format(sp[0], 'morphed{}to{}{}'.format(sep, sep, subject), sep, hemi, sp[1])
            print('Morphing data from {} to {} -> {}'.format(target_subject, subject, target_fname))
            fu.surf2surf(target_subject, subject, hemi, fmri_fname, target_fname, cwd=None, print_only=False)
            if op.isfile(target_fname):
                fmri_fname = target_fname
            else:
                raise Exception('surf2surf: Target file was not created!')
            data = nib.load(fmri_fname).get_data().squeeze()
        assert (data.shape[0] == subject_verts_num)
        hemi_minmax.append(utils.calc_min_max(data, norm_percs=norm_percs))
        utils.delete_folder_files(fol)
        now = time.time()
        T = data.shape[1]
        for t in range(T):
            utils.time_to_go(now, t, T, runs_num_to_print=10)
            np.save(op.join(fol, 't{}'.format(t)), data[:, t])

    data_min, data_max = utils.calc_minmax_from_arr(hemi_minmax)
    print('save_dynamic_activity_map minmax: {},{}'.format(data_min, data_max))
    np.save(minmax_fname, (data_min, data_max))
    return np.all([len(glob.glob(op.join(MMVT_DIR, subject, 'fmri', 'activity_map_{}'.format(hemi), '*.npy'))) == T
                   for hemi in utils.HEMIS])


def find_template_files(template_fname, file_types=('mgz', 'nii.gz', 'nii', 'npy')):
    def find_files(template_fname):
        recursive = '**' in set(template_fname.split(op.sep))
        return [f for f in glob.glob(template_fname, recursive=recursive) if op.isfile(f) and utils.file_type(f) in file_types]

    files = find_files(template_fname)
    if len(files) == 0:
        print('Adding * to the end of the template_fname')
        files = find_files('{}*'.format(template_fname))
    print('find_template_files: {}, template: {}'.format(files, template_fname))
    return files


def find_hemi_files_from_template(template_fname, file_types=('mgz', 'nii.gz', 'nii', 'npy')):
    try:
        if isinstance(file_types, str):
            file_types = [file_types]
        # if '{subject}' in template_fname:
        #     template_fname = template_fname.replace('{subject}', subject)
        template_files = find_template_files(template_fname.replace('{hemi}', '?h'), file_types)
        return find_hemi_files(template_files)
    except:
        print('Error in find_hemi_files_from_template: {}'.format(template_fname))
        print(traceback.format_exc())
        return []


def find_hemi_files(files):
    if len(files) < 2:
        print('len(files) should be >= 2!')
        print(files)
        return []
    file_types = set([utils.file_type(f) for f in files])
    if len(set(['nii', 'nii.gz', 'mgz', 'mgh']) & file_types) > 0:
        files = get_unique_files_into_mgz(files)
    hemis_files = []
    rh_files = [f for f in files if lu.get_hemi_from_name(utils.namebase(f)) == 'rh'] #  '_rh' in utils.namebase(f) or '.rh' in utils.namebase(f)]
    parent_fol = utils.get_parent_fol(rh_files[0])
    for rh_file in rh_files:
        lh_file = lu.get_other_hemi_label_name(utils.namebase(rh_file)) # rh_file.replace('_rh', '_lh').replace('.rh', '.lh')
        lh_file = op.join(parent_fol, '{}.{}'.format(lh_file, utils.file_type(rh_file)))
        if op.isfile(lh_file):
            hemis_files.append(rh_file.replace('rh', '{hemi}'))
    print('find_hemi_files return {}'.format(hemis_files))
    return hemis_files


def find_volume_files(files):
    # if convert_to_mgz:
    #     files = get_unique_files_into_mgz(files)
    def hemi_in_fname(fname):
        return ('_rh' in fname or '_lh' in fname or
                '.rh' in fname or '.lh' in fname or
                '-rh' in fname or '-lh' in fname or
                'rh_' in fname or 'lh_' in fname or
                'rh.' in fname or 'lh.' in fname or
                'rh-' in fname or 'lh-' in fname)
    volume_files = [f for f in files if not hemi_in_fname(utils.namesbase_with_ext(f))]
    if len(files) > 0 and len(volume_files) == 0:
        print('find_volume_files: No volume files were found! hemi was found in all the given files! {}'.format(files))
    return volume_files


def find_volume_files_from_template(template_fname):
    return find_volume_files(find_template_files(template_fname))


def get_fmri_fname(subject, fmri_file_template, no_files_were_found_func=lambda:'', only_volumes=False,
                   raise_exception=True):
    fmri_fname = ''
    if '{subject}' in fmri_file_template:
        fmri_file_template = fmri_file_template.replace('{subject}', subject)
    full_fmri_file_template = op.join(FMRI_DIR, subject, fmri_file_template)
    if only_volumes:
        files = find_volume_files_from_template(full_fmri_file_template)
    else:
        files = find_hemi_files_from_template(full_fmri_file_template)
    if len(files) == 0:
        full_fmri_file_template = op.join(FMRI_DIR, subject, '**', fmri_file_template)
        if only_volumes:
            files = find_volume_files_from_template(full_fmri_file_template)
        else:
            files = find_hemi_files_from_template(full_fmri_file_template)
    files_num = len(set([utils.namebase(f) for f in files]))
    if files_num == 1:
        fmri_fname = files[0]
    elif files_num == 0:
        if raise_exception:
            raise Exception("Can't find any file in {}!".format(fmri_file_template))
        else:
            print("Can't find any file in {}!".format(fmri_file_template))
            return no_files_were_found_func()
    elif files_num > 1:
        fmri_fname = utils.select_one_file(files, fmri_file_template, 'fMRI')
        # if raise_exception:
        #     raise Exception("More than one file can be found in {}! {}".format(full_fmri_file_template, files))
    return fmri_fname


@utils.check_for_freesurfer
def clean_4d_data(subject, atlas, fmri_file_template, trg_subject='fsaverage5', fsd='rest', only_preproc=False,
                             fwhm=6, lfp=0.08, nskip=4, remote_fmri_dir='', overwrite=False, print_only=False):
    # fsd: functional subdirectory
    def no_files_were_found():
        print('Trying to find remote files in {}'.format(op.join(remote_fmri_dir, fsd, '001', fmri_file_template)))
        files = find_volume_files_from_template(op.join(remote_fmri_dir, fsd, '001', fmri_file_template)) + \
                find_volume_files_from_template(op.join(remote_fmri_dir, fmri_file_template))
        print('files: {}'.format(files))
        files_num = len(set([utils.namebase(f) for f in files]))
        if files_num == 1:
            fmri_fname = op.join(FMRI_DIR, subject, files[0].split(op.sep)[-1])
            utils.make_dir(op.join(FMRI_DIR, subject))
            shutil.copy(files[0], fmri_fname)
        else:
            print("Can't find any file in {}!".format(fmri_file_template))
            return ''
            # raise Exception("Can't find any file in {}!".format(fmri_file_template))


    def create_folders_tree(fmri_fname):
        # Fisrt it's needed to create the freesurfer folders tree for the preproc-sess
        fol = utils.make_dir(op.join(FMRI_DIR, subject, fsd, '001'))
        if not op.isfile(op.join(fol, 'f.nii.gz')):
            if utils.file_type(fmri_fname) == 'mgz':
                fmri_fname = fu.mgz_to_nii_gz(fmri_fname)
            shutil.copy(fmri_fname, op.join(fol, 'f.nii.gz'))
        if not op.isfile(op.join(FMRI_DIR, subject, 'subjectname')):
            with open(op.join(FMRI_DIR, subject, 'subjectname'), 'w') as sub_file:
                sub_file.write(subject)

    def create_analysis_info_file(fsd, trg_subject, tr, fwhm=6, lfp=0.08, nskip=4):
        rs = utils.partial_run_script(locals(), cwd=FMRI_DIR, print_only=print_only)
        for hemi in utils.HEMIS:
            rs('mkanalysis-sess -analysis {fsd}_{hemi} -notask -TR {tr} -surface {trg_subject} {hemi} -fsd {fsd}' +
               ' -per-run -nuisreg global.waveform.dat 1 -nuisreg wm.dat 1 -nuisreg vcsf.dat 1 -lpf {lfp} -mcextreg' +
               ' -fwhm {fwhm} -nskip {nskip} -stc up -force', hemi=hemi)

    def find_trg_subject(trg_subject):
        if not op.isdir(op.join(SUBJECTS_DIR, trg_subject)):
            if op.isdir(op.join(FREESURFER_HOME, 'subjects', trg_subject)):
                os.symlink(op.join(FREESURFER_HOME, 'subjects', trg_subject),
                           op.join(SUBJECTS_DIR, trg_subject))
            else:
                raise Exception("The target subject {} doesn't exist!".format(trg_subject))

    def copy_output_files():
        new_fname_template = op.join(FMRI_DIR, subject, '{}.sm{}.{}.{}.mgz'.format(
            fsd, int(fwhm), trg_subject, '{hemi}'))
        for hemi in utils.HEMIS:
            new_fname = new_fname_template.format(hemi=hemi)
            if not op.isfile(new_fname):
                res_fname = op.join(FMRI_DIR, subject, fsd, '{}_{}'.format(fsd, hemi), 'res', 'res-001.nii.gz')
                if op.isfile(res_fname):
                    fu.nii_gz_to_mgz(res_fname)
                    res_fname = utils.change_fname_extension(res_fname, 'mgz')
                    shutil.copy(res_fname, new_fname)
        for hemi in utils.HEMIS:
            utils.make_link(new_fname_template.format(hemi=hemi), op.join(
                MMVT_DIR, subject, 'fmri', utils.namesbase_with_ext(new_fname_template.format(hemi=hemi))))
        return utils.both_hemi_files_exist(new_fname_template)

    def copy_preproc_sess_outputs():
        hemi_file_name = 'fmcpr.sm6.{}.{}.{}'.format(subject, '{hemi}', '{format}')
        for hemi in utils.HEMIS:
            res_fname = op.join(FMRI_DIR, subject, fsd, '001', hemi_file_name.format(hemi=hemi, format='nii.gz'))
            new_fname = op.join(FMRI_DIR, subject, hemi_file_name.format(hemi=hemi, format='mgz'))
            if not op.isfile(new_fname) or overwrite:
                res_fname = fu.nii_gz_to_mgz(res_fname)
                os.link(res_fname, new_fname)
        volume_fname = op.join(FMRI_DIR, subject, fsd, '001', 'fmcpr.sm6.mni305.2mm.nii.gz')
        volume_new_fname = op.join(FMRI_DIR, subject, 'fmcpr.sm6.mni305.2mm.mgz')
        volume_fname = fu.nii_gz_to_mgz(volume_fname)
        os.link(volume_fname, volume_new_fname)
        return utils.both_hemi_files_exist(op.join(FMRI_DIR, subject, hemi_file_name.format(
            hemi='{hemi}', format='mgz'))) and op.isfile(volume_new_fname)

    def no_output(*args):
        return not op.isfile(op.join(FMRI_DIR, subject, fsd, *args))

    def run(cmd, *output_args, **kargs):
        if no_output(*output_args) or overwrite or print_only:
            rs(cmd, **kargs)
            if not print_only and no_output(*output_args):
                raise Exception('{}\nNo output created in {}!!\n\n'.format(
                    cmd, op.join(FMRI_DIR, subject, fsd, *output_args)))

    trg_subject = subject if trg_subject == '' else trg_subject
    new_fname_template = op.join(FMRI_DIR, subject, '{}.sm{}.{}.{}.mgz'.format(
        fsd, int(fwhm), trg_subject, '{hemi}'))
    if utils.both_hemi_files_exist(new_fname_template) and not overwrite:
        return True

    find_trg_subject(trg_subject)
    if fmri_file_template == '':
        fmri_file_template = '*'
    fmri_fname = get_fmri_fname(
        subject, fmri_file_template, no_files_were_found, only_volumes=True, raise_exception=False)
    if fmri_fname == '':
        return False
    output_files_exist = copy_output_files()
    if output_files_exist:
        return True
    create_folders_tree(fmri_fname)
    rs = utils.partial_run_script(locals(), cwd=FMRI_DIR, print_only=print_only)
    run('preproc-sess -surface {trg_subject} lhrh -s {subject} -fwhm {fwhm} -fsd {fsd} -mni305 -per-run',
        '001', 'fmcpr.sm{}.mni305.2mm.nii.gz'.format(int(fwhm)))
    if only_preproc:
        return copy_preproc_sess_outputs()
    run('plot-twf-sess -s {subject} -dat f.nii.gz -mc -fsd {fsd} && killall display', 'fmcpr.mcdat.png')
    run('plot-twf-sess -s {subject} -dat f.nii.gz -fsd {fsd} -meantwf && killall display', 'global.waveform.dat.png')

    # registration
    run('tkregister-sess -s {subject} -per-run -fsd {fsd} -bbr-sum > {subject}/{fsd}/reg_quality.txt',
        'reg_quality.txt')

    # Computes seeds (regressors) that can be used for functional connectivity analysis or for use as nuisance regressors.
    if no_output('001', 'wm.dat'):
        rs('fcseed-config -wm -overwrite -fcname wm.dat -fsd {fsd} -cfg {subject}/wm_{fsd}.cfg')
        run('fcseed-sess -s {subject} -cfg {subject}/wm_{fsd}.cfg', '001', 'wm.dat')
    if no_output('001', 'vcsf.dat'):
        rs('fcseed-config -vcsf -overwrite -fcname vcsf.dat -fsd {fsd} -mean -cfg {subject}/vcsf_{fsd}.cfg')
        run('fcseed-sess -s {subject} -cfg {subject}/vcsf_{fsd}.cfg', '001', 'vcsf.dat')

    tr = get_tr(fmri_fname)
    create_analysis_info_file(fsd, trg_subject, tr, fwhm, lfp, nskip)
    for hemi in utils.HEMIS:
        # computes the average signal intensity maps
        run('selxavg3-sess -s {subject} -a {fsd}_{hemi} -svres -no-con-ok',
            '{}_{}'.format(fsd, hemi), 'res', 'res-001.nii.gz', hemi=hemi)

    return copy_output_files() if not print_only else True


# def functional_connectivity_freesurfer(subject, fsd='rest', measure='mean', seg_id=1010, fcname='L_Posteriorcingulate',
#                                        overwrite=False, print_only=False):
#
#     def no_output(*args):
#         return not op.isfile(op.join(FMRI_DIR, subject, fsd, *args))
#
#     def run(cmd, *output_args, **kargs):
#         if no_output(*output_args) or overwrite or print_only:
#             rs(cmd, **kargs)
#             if not print_only and no_output(*output_args):
#                 raise Exception('{}\nNo output created in {}!!\n\n'.format(
#                     cmd, op.join(FMRI_DIR, subject, fsd, *output_args)))
#
#     # http://surfer.nmr.mgh.harvard.edu/fswiki/FsFastFunctionalConnectivityWalkthrough
#     rs = utils.partial_run_script(locals(), cwd=FMRI_DIR, print_only=print_only)
#     rs('fcseed-config -segid {seg_id} -fcname {fcname}.dat -fsd {fsd} -{measure} -cfg {measure}.{fcname}.config')
#     rs('fcseed-sess -s {subject} -cfg {fcname}.config')
#
#     if no_output('001', 'wm.dat'):
#         run('fcseed-config -wm -fcname wm.dat -fsd bold -pca -cfg wm.config')
#     'fcseed-sess -s sessionid -cfg wm.config'

def get_tr(fmri_fname):
    try:
        tr_fname = utils.add_str_to_file_name(fmri_fname, '_tr', 'pkl')
        if op.isfile(tr_fname):
            return utils.load(tr_fname)
        # if utils.is_file_type(fmri_fname, 'nii.gz'):
        #     fmri_fname = fu.nii_gz_to_mgz(fmri_fname)
        #     # old_fmri_fname = fmri_fname
        #     # fmri_fname = '{}mgz'.format(fmri_fname[:-len('nii.gz')])
        #     # if not op.isfile(fmri_fname):
        #     #     fu.mri_convert(old_fmri_fname, fmri_fname)
        # elif utils.is_file_type(fmri_fname, 'nii'):
        #     fmri_fname = fu.nii_to_mgz(fmri_fname)
        # if utils.is_file_type(fmri_fname, 'mgz'):
        #     fmri_fname = op.join(FMRI_DIR, subject, fmri_fname)
        tr = fu.get_tr(fmri_fname)
            # print('fMRI fname: {}'.format(fmri_fname))
        print('tr: {}'.format(tr))
        utils.save(tr, tr_fname)
        return tr
        # else:
        #     print('file format not supported!')
        #     return None
    except:
        print(traceback.format_exc())
        return None


def fmri_pipeline(subject, atlas, contrast_file_template, task='', contrast='', fsfast=True, t_val=2,
         fmri_files_fol='', load_labels_from_annotation=True, n_jobs=2):
    '''

    Parameters
    ----------
    subject: subject's name
    atlas: pacellation name
    contrast_file_template: template for the contrast file name. To get a full name the user should run:
          contrast_file_template.format(hemi=hemi, constrast=constrast, format=format)
    t_val: tval cutt off for finding clusters
    surface_name: Just for output name
    contrast_format: The contrast format (mgz, nii, nii.gz, ...)
    existing_format: The exsiting format (mgz, nii, nii.gz, ...)
    fmri_files_fol: The fmri files output folder
    load_labels_from_annotation: For finding the intersected labels, if True the function tries to read the labels from
        the annotation file, if False it tries to read the labels files.
    Returns
    -------

    '''
    fol = op.join(FMRI_DIR, args.task, subject)
    if not op.isdir(fol):
        raise Exception('You should first put the fMRI contrast files in {}'.format(fol))
    contrasts_files = {}
    if fsfast and op.isdir(op.join(fol, 'bold')):
        # todo: What to do with group-avg in fsfast?
        contrasts = set([utils.namebase(f) for f in glob.glob(op.join(fol, 'bold', '*'))])
        for contrast in contrasts:
            contrast_files = glob.glob(op.join(fol, 'bold', '*{}*'.format(contrast), 'sig.*'), recursive=True)
            contrasts_files[contrast] = dict(
                volume_files=find_volume_files(contrast_files),
                hemis_files=find_hemi_files(contrast_files))
    else:
        contrast = contrast if contrast != '' else contrast_file_template.replace('*', '').replace('?', '')
        contrasts_files[contrast] = dict(
            volume_files=find_volume_files_from_template(op.join(fol, contrast_file_template)),
            hemis_files=find_hemi_files_from_template(op.join(fol, contrast_file_template)))
        if not contrasts_files[contrast]['hemis_files']:
            raise Exception('No contrast maps projected to the hemispheres were found in {}'.format(
                op.join(fol, contrast_file_template)))

    utils.make_dir(op.join(MMVT_DIR, subject, 'freeview'))
    for contrast, contrast_dict in contrasts_files.items():
        volume_files, hemis_files_templates = contrast_dict['volume_files'], contrast_dict['hemis_files']
        for volume_file in volume_files:
            fu.mri_convert_to(volume_file, 'mgz')
            shutil.copyfile(volume_file, op.join(MMVT_DIR, subject, 'freeview', '{}.{}'.format(contrast, format)))
        hemis_files_templates = [t for t in hemis_files_templates if not t.endswith('_morphed_to_{}.mgz'.format(subject))]
        for hemis_files_teamplate in hemis_files_templates:
            new_hemis_fname, new_hemis_org_subject_fname = {}, {}
            for hemi in utils.HEMIS:
                new_hemi_fname = fu.mri_convert_to(hemis_files_teamplate.format(hemi=hemi), 'mgz')
                new_hemis_fname[hemi], new_hemis_org_subject_fname[hemi], morphed_from_subject = \
                    save_fmri_hemi_data(subject, hemi, contrast, new_hemi_fname, task, output_fol=fmri_files_fol)
            calc_fmri_min_max(
                subject, new_hemis_fname, task=task, norm_percs=args.norm_percs, contrast_name=contrast,
                symetric_colors=args.symetric_colors)
            if morphed_from_subject != subject:
                calc_fmri_min_max(
                    morphed_from_subject, new_hemis_org_subject_fname, task=task, norm_percs=args.norm_percs,
                    symetric_colors=args.symetric_colors,
                    contrast_name=contrast)
        # todo: save clusters also for morphed_from_subject
        # todo: we should send a template for the surf fname instead of contrast
        find_clusters(subject, contrast, t_val, atlas, task, '', fmri_files_fol, n_jobs)
    # todo: check what to return
    return True


def fmri_pipeline_all(subject, atlas, task='*', contrast='*', filter_dic=None, new_name='',
                      norm_by_percentile=False, norm_percs=None, symetric_colors=True):

    def remove_dups(all_names):
        all_names = list(set(all_names))
        all_names = [t for t in all_names if not ('-and-' in t and all([tt in all_names for tt in t.split('-and-')]))]
        return '-and-'.join(sorted(all_names))

    def change_cluster_values_names(cluster, uid):
        for blob in cluster['values']:
            blob['name'] = '{}-{}'.format(uid, blob['name'])

    hemi_all_data = {}
    file_names = [utils.namebase(f) for f in glob.glob(
        op.join(MMVT_DIR, subject, 'fmri', 'fmri_{}_{}_rh.npy'.format(task, contrast)))]
    all_tasks = remove_dups([f.split('_')[1] for f in file_names])
    all_contrasts = remove_dups([f.split('_')[2] for f in file_names])
    new_name = new_name if new_name != '' else '{}_{}'.format(all_tasks, all_contrasts)
    for hemi in utils.HEMIS:
        hemi_fnames = glob.glob(op.join(MMVT_DIR, subject, 'fmri', 'fmri_{}_{}_{}.npy'.format(task, contrast, hemi)))
        hemi_all_data[hemi] = np.load(hemi_fnames[0])
        for hemi_fname in hemi_fnames[1:]:
            hemi_data = np.load(hemi_fname)
            hemi_all_data[hemi] = [x1 if abs(x1) > abs(x2) else x2 for x1,x2 in zip(hemi_data, hemi_all_data[hemi])]
        output_name = 'fmri_{}_{}.npy'.format(new_name, hemi)
        np.save(op.join(MMVT_DIR, subject, 'fmri', output_name), hemi_all_data[hemi])
    new_hemis_fname = op.join(MMVT_DIR, subject, 'fmri', 'fmri_{}_{}.npy'.format(new_name, '{hemi}'))
    calc_fmri_min_max(
        subject, new_hemis_fname, task=all_tasks, norm_percs=norm_percs, contrast_name=all_contrasts,
        symetric_colors=symetric_colors, new_name=new_name)
    all_clusters_fnames = glob.glob(op.join(MMVT_DIR, subject, 'fmri', 'clusters_labels_*_{}.pkl'.format(atlas)))
    all_clusters_fnames = [f for f in all_clusters_fnames if '-and-' not in utils.namebase(f)]
    all_clusters_uids = ['-'.join(n.split('_')[:2]) for n in
                         [utils.namebase(f)[len('clusters_labels_'):] for f in all_clusters_fnames]]
    all_clusters = utils.load(all_clusters_fnames[0])
    change_cluster_values_names(all_clusters, all_clusters_uids[0])
    all_clusters = filter_clusters(all_clusters, filter_dic)
    for cluster_fname, cluster_uid in zip(all_clusters_fnames[1:], all_clusters_uids[1:]):
        cluster = utils.load(cluster_fname)
        change_cluster_values_names(cluster, cluster_uid)
        cluster = filter_clusters(cluster, filter_dic)
        if all_clusters['threshold'] != cluster['threshold']:
            print("Not all the cluster have the same threshold, can't join them!")
            return False
        all_clusters['values'] += cluster['values']
    utils.save(all_clusters, op.join(MMVT_DIR, subject, 'fmri', 'clusters_labels_{}_{}.pkl'.format(
        new_name, atlas)))


def filter_clusters(clusters, filter_dic):
    if filter_dic is None:
        return clusters
    new_cluster = dict()
    new_cluster['threshold'] = clusters['threshold']
    new_cluster['values'] = []
    uid = '-'.join(clusters['values'][0]['name'].split('-')[:2])
    for cluster in clusters['values']:
        if uid not in filter_dic:
            continue
        # for roi in ['dACC', 'OFC', 'dmPFC', 'vlPFC']:
        #     if roi in cluster['name']:
        #         print(cluster['name'], '{0:.2f}'.format(cluster['max']))
        for val in filter_dic[uid]:
            _tval = '{0:.2f}'.format(cluster['max']) == '{0:.2f}'.format(val['tval'])
            _name = cluster['name'] == '{}-{}-{}'.format(uid, val['name'], val['hemi'])
            if _tval and _name:
                print('Cluster found! {}'.format(cluster['name']))
                if 'new_name' in val:
                    cluster['name'] = '{}-{}-{}'.format(uid, val['new_name'], val['hemi'])
                new_cluster['values'].append(cluster)
    return new_cluster


def get_unique_files_into_mgz(files):
    contrast_files_dic = defaultdict(list)
    for contrast_file in files:
        ft = utils.file_type(contrast_file)
        contrast_files_dic[contrast_file[:-len(ft) - 1]].append(ft)
    for contrast_file, fts in contrast_files_dic.items():
        if 'mgz' not in fts and 'nii.gz' in fts:
            # fu.mri_convert_to('{}.{}'.format(contrast_file, fts[0]), 'mgz')
            fu.nii_gz_to_mgz('{}.nii.gz'.format(contrast_file))
    files = ['{}.mgz'.format(contrast_file) for contrast_file in contrast_files_dic.keys()]
    print('get_unique_files_into_mgz: {}'.format(files))
    return files


def load_fmri_data(fmri_surf_fname):
    if not op.isfile(fmri_surf_fname):
        raise Exception("load_fmri_data: Can't find {}".format(fmri_surf_fname))
    file_type = utils.file_type(fmri_surf_fname)
    if file_type in ['nii', 'nii.gz', 'mgz', 'mgh']:
        x = nib.load(fmri_surf_fname).get_data().squeeze()
    elif file_type == 'npy':
        x = np.load(fmri_surf_fname)
    else:
        raise Exception('fMRI file format is not supported!')
    return x


def load_fmri_data_for_both_hemis(subject, surf_name):
    surf_name = surf_name if surf_name != '' else '{}_'.format(surf_name)
    fname_temp = op.join(MMVT_DIR, subject, 'fmri', 'fmri_{}{}.npy'.format(surf_name, '{hemi}'))
    if utils.both_hemi_files_exist(fname_temp):
        return {hemi:load_fmri_data(op.join(MMVT_DIR, subject, 'fmri', 'fmri_{}_{}.npy'.format(surf_name, hemi)))
                for hemi in utils.HEMIS}
    else:
        fmri_files = get_all_fmri_files(subject)
        if len(fmri_files) > 0:
            if len(fmri_files) > 1:
                print("Can't find {}. Do you want to pick a different one?".format(fname_temp))
                fname = utils.select_one_file(fmri_files, print_title=False)
            else:
                fname = fmri_files[0]
            fname = op.join(MMVT_DIR, subject, 'fmri', fname)
            return {hemi: load_fmri_data(fname.format(hemi=hemi)) for hemi in utils.HEMIS}
        else:
            raise Exception("Can't find {} or any other fMRI files in {}!".format(
                'fmri_{}_{}.npy'.format(surf_name, '{hemi}'), op.join(MMVT_DIR, subject, 'fmri')))


def get_all_fmri_files(subject):
    files = []
    for fol in [op.join(MMVT_DIR, subject, 'fmri'), op.join(FMRI_DIR, subject)]:
        for template in ['fmri_*.npy', '*.mgz', '*.nii', '*.nii.gz']:
            files.extend(glob.glob(op.join(fol, template)))
    files = list(set(['{}.{}'.format(lu.get_template_hemi_label_name(utils.namebase_sep(f)), utils.file_type_sep(f))
                      for f in files if lu.get_label_hemi(utils.namebase_sep(f)) != '']))
    for fname in files:
        if not utils.both_hemi_files_exist(op.join(MMVT_DIR, subject, 'fmri', fname)) and \
                utils.both_hemi_files_exist(op.join(FMRI_DIR, subject, fname)):
            for hemi in utils.HEMIS:
                utils.make_link(op.join(FMRI_DIR, subject, fname.format(hemi=hemi)),
                                op.join(MMVT_DIR, subject, 'fmri', fname.format(hemi=hemi)))
    return files


def misc(args):
    contrast_name = 'interference'
    contrasts = {'non-interference-v-base': '-a 1', 'interference-v-base': '-a 2',
                 'non-interference-v-interference': '-a 1 -c 2', 'task.avg-v-base': '-a 1 -a 2'}
    fol = op.join(FMRI_DIR, args.task, args.subject[0])
    contrast_file_template = op.join(fol, 'bold',
        '{contrast_name}.sm05.{hemi}'.format(contrast_name=contrast_name, hemi='{hemi}'), '{contrast}', 'sig.{format}')
    # contrast_file_template = op.join(fol, 'sig.{hemi}.{format}')


    contrast_name = 'group-avg'
    # main(subject, atlas, None, contrast_file_template, t_val=14, surface_name='pial', existing_format='mgh')
    # find_clusters_tval_hist(subject, contrast_name, fol, input_fol='', n_jobs=1)
    # load_clusters_tval_hist(fol)

    # contrast = 'non-interference-v-interference'
    inverse_method = 'dSPM'
    # meg_subject = 'ep001'

    # overwrite_volume_mgz = False
    # data_fol = op.join(FMRI_DIR, task, 'healthy_group')
    # contrast = 'pp003_vs_healthy'
    # contrast = 'pp009_ARC_High_Risk_Linear_Reward_contrast'
    # contrast = 'pp009_ARC_PPI_highrisk_L_VLPFC'

    # create_functional_rois(subject, contrast, data_fol)

    # # todo: find the TR automatiaclly
    # TR = 1.75

    # show_fMRI_using_pysurfer(subject, '/homes/5/npeled/space3/fMRI/ECR/hc004/bold/congruence.sm05.lh/congruent-v-incongruent/sig.mgz', 'rh')

    # fsfast.run(subject, root_dir=ROOT_DIR, par_file = 'msit.par', contrast_name=contrast_name, tr=TR, contrasts=contrasts, print_only=False)
    # fsfast.plot_contrast(subject, ROOT_DIR, contrast_name, contrasts, hemi='rh')
    # mri_convert_hemis(contrast_file_template, list(contrasts.keys())


    # show_fMRI_using_pysurfer(subject, input_file=contrast_file, hemi='lh')
    # root = op.join('/autofs/space/franklin_003/users/npeled/fMRI/MSIT/pp003')
    # volume_file = op.join(root, 'sig.anat.mgz')
    # mask_file = op.join(root, 'VLPFC.mask.mgz')
    # masked_file = op.join(root, 'sig.anat.masked.mgz')
    # contrast_file = op.join(root, 'sig.{hemi}.mgz')
    # contrast_masked_file = op.join(root, 'sig.masked.{hemi}.mgz')

    # for hemi in ['rh', 'lh']:
    #     save_fmri_colors(subject, hemi, contrast_masked_file.format(hemi=hemi), 'pial', threshold=2)
    # Show the fRMI in pysurfer
    # show_fMRI_using_pysurfer(subject, input_file=contrast_masked_file, hemi='both')

    # load_and_show_npy(subject, '/homes/5/npeled/space3/visualization_blender/mg79/fmri_lh.npy', 'lh')

    # mask_volume(volume_file, mask_file, masked_file)
    # show_fMRI_using_pysurfer(subject, input_file='/autofs/space/franklin_003/users/npeled/fMRI/MSIT/pp003/sig.{hemi}.masked.mgz', hemi='both')
    # calculate_subcorticals_activity(subject, '/homes/5/npeled/space3/MSIT/mg78/bold/interference.sm05.mni305/non-interference-v-interference/sig.anat.mgh',
    #              '/autofs/space/franklin_003/users/npeled/MSIT/mg78/aseg_stats.csv')
    # calculate_subcorticals_activity(subject, '/home/noam/fMRI/MSIT/mg78/bold/interference.sm05.mni305/non-interference-v-interference/sig.anat.mgh',
    #              '/home/noam/fMRI/MSIT/mg78/aseg_stats.csv')
    # volume_file = nib.load('/autofs/space/franklin_003/users/npeled/fMRI/MSIT/mg78/bold/interference.sm05.mni305/non-interference-v-interference/sig_subject.mgz')
    # vol_data, vol_header = volume_file.get_data(), volume_file.get_header()

    # contrast_file=contrast_file_template.format(
    #     contrast='non-interference-v-interference', hemi='mni305', format='mgz')
    # calculate_subcorticals_activity(subject, volume_file, subcortical_codes_file=op.join(BLENDER_DIR, 'sub_cortical_codes.txt'),
    #     method='dist')

    # SPM_ROOT = '/homes/5/npeled/space3/spm_subjects'
    # for subject_fol in utils.get_subfolders(SPM_ROOT):
    #     subject = utils.namebase(subject_fol)
    #     print(subject)
    #     contrast_masked_file = op.join(subject_fol, '{}_VLPFC_{}.mgz'.format(subject, '{hemi}'))
    #     show_fMRI_using_pysurfer(subject, input_file=contrast_masked_file, hemi='rh')
    # brain = Brain('fsaverage', 'both', "pial", curv=False, offscreen=False)


def calc_also_minmax(ret_flag, fmri_contrast_file_template, args):
    if ret_flag and 'calc_fmri_min_max' not in args.function:
        args.function.append('calc_fmri_min_max')
    return fmri_contrast_file_template, args


def main(subject, remote_subject_dir, args, flags):
    volume_name = args.volume_name if args.volume_name != '' else subject
    fol = op.join(FMRI_DIR, args.task, subject)
    remote_fmri_dir = op.join(FMRI_DIR, subject) if args.remote_fmri_dir == '' else \
        utils.build_remote_subject_dir(args.remote_fmri_dir, subject)
    print('remote_fmri_dir: {}'.format(remote_fmri_dir))
    if args.contrast_template == '':
        if args.fsfast:
            fmri_contrast_file_template = op.join(fol, 'bold', '{contrast_name}.sm05.{hemi}'.format(
                contrast_name=args.contrast_name, hemi='{hemi}'), '{contrast}', 'sig.{format}')
        else:
            fmri_contrast_file_template = op.join(fol, '{}_{}.mgz'.format(volume_name, '{hemi}'))
    else:
        fmri_contrast_file_template = args.contrast_template

    # todo: should find automatically the existing_format
    if 'fmri_pipeline' in args.function:
        flags['fmri_pipeline'] = fmri_pipeline(
            subject, args.atlas, fmri_contrast_file_template, args.task, args.contrast, args.fsfast,
            args.threshold, n_jobs=args.n_jobs)

    if utils.should_run(args, 'project_volume_to_surface'):
        flags['project_volume_to_surface'] = project_volume_to_surface(
            subject, args.fmri_file_template, overwrite_surf_data=args.overwrite_surf_data, target_subject=args.target_subject,
            is_pet=args.is_pet, remote_fmri_dir=remote_fmri_dir, mmvt_args=args)
        flags['project_volume_to_surface'], surf_output_fname = pu.check_func_output(flags['project_volume_to_surface'])
        fmri_contrast_file_template, args = calc_also_minmax(
            flags['project_volume_to_surface'], surf_output_fname, args)

    if utils.should_run(args, 'load_surf_files'):
        flags['load_surf_files'], output_fname_template = load_surf_files(
            subject, args.fmri_file_template, args.overwrite_surf_data)
        fmri_contrast_file_template, args = calc_also_minmax(flags['load_surf_files'], output_fname_template, args)

    if 'calc_files_diff' in args.function:
        flags['calc_files_diff'], output_fname_template = calc_files_diff(subject, args.fmri_file_template, args.overwrite_surf_data)
        fmri_contrast_file_template, args = calc_also_minmax(flags['calc_files_diff'], output_fname_template, args)

    if utils.should_run(args, 'calc_fmri_min_max'):
        flags['calc_fmri_min_max'] = calc_fmri_min_max(
            subject, fmri_contrast_file_template, task=args.task, norm_percs=args.norm_percs,
            symetric_colors=args.symetric_colors, contrast_name=args.contrast_name,
            remote_fmri_dir=remote_fmri_dir, template_brain=args.target_subject)

    if utils.should_run(args, 'find_clusters'):
        flags['find_clusters'] = find_clusters(
            subject, args.fmri_file_template, args.threshold, args.atlas, args.task, args.n_jobs)

    if 'fmri_pipeline_all' in args.function:
        flags['fmri_pipeline_all'] = fmri_pipeline_all(subject, args.atlas, filter_dic=None)

    if 'analyze_4d_data' in args.function:
        flags['analyze_4d_data'] = analyze_4d_data(
            subject, args.atlas, args.fmri_file_template, args.labels_extract_mode, args.template_brain,
            args.norm_percs, args.overwrite_labels_data, remote_fmri_dir, args.resting_state_plot,
            args.resting_state_plot_all_vertices, args.excluded_labels, args.input_format)

    if 'calc_labels_minmax' in args.function:
        flags['calc_labels_minmax'] = calc_labels_minmax(subject, args.atlas, args.labels_extract_mode)

    if 'save_dynamic_activity_map' in args.function:
        flags['save_dynamic_activity_map'] = save_dynamic_activity_map(
            subject, args.fmri_file_template, template_brains=args.template_brain,
            norm_percs=args.norm_percs, overwrite=args.overwrite_activity_data)

    if 'calc_subs_surface_activity' in args.function:
        flags['calc_subs_surface_activity'] = calc_subs_surface_activity(
            subject, args.fmri_file_template, args.template_brain, args.subs_threshold, args.subcortical_codes_file,
            args.aseg_stats_fname, method=args.calc_subs_surface_method, k_points=args.calc_subs_surface_points,
            format='mgz', do_plot=False)

    if 'clean_4d_data' in args.function:
        flags['clean_4d_data'] = clean_4d_data(
            subject, args.atlas, args.fmri_file_template, args.template_brain, args.fsd, args.only_preproc,
            args.fwhm, args.lfp, args.nskip, remote_fmri_dir, args.overwrite_4d_preproc, args.print_only)

    if 'calc_meg_activity' in args.function:
        meg_subject = args.meg_subject
        if meg_subject == '':
            print('You must set MEG subject (--meg_subject) to run calc_meg_activity function!')
        else:
            flags['calc_meg_activity'] = calc_meg_activity_for_functional_rois(
                subject, meg_subject, args.atlas, args.task, args.contrast_name, args.contrast, args.inverse_method)

    if 'calc_subs_activity' in args.function:
        flags['calc_subs_activity'] = calc_subs_activity(
            subject, args.fmri_sub_file_template, args.labels_extract_mode, args.subcortical_codes_file,
            args.overwrite_subs_data)

    if 'copy_volumes' in args.function:
        flags['copy_volumes'] = copy_volumes(subject, fmri_contrast_file_template)

    if 'get_tr' in args.function:
        tr = get_tr(args.fmri_fname)
        flags['get_tr'] = not tr is None

    if 'load_labels_ts' in args.function:
        flags['load_labels_ts'] = load_labels_ts(
            subject, args.atlas, args.labels_order_fname, args.st_template, args.labels_extract_mode,
            args.excluded_labels, args.labels_indices_to_remove_from_data, args.backup_existing_files,
            args.pick_the_first_one)

    if 'calc_labels_mean_freesurfer' in args.function:
        ret = calc_labels_mean_freesurfer_get_files(
            args, remote_subject_dir, subject, args.atlas, args.fmri_file_template, args.template_brain,
            args.target_subject, remote_fmri_dir)
        if not ret:
            print('Not all the necessary files exist!')
            flags['calc_labels_mean_freesurfer'] = False
        else:
            flags['calc_labels_mean_freesurfer'] = calc_labels_mean_freesurfer(
                subject, args.atlas, args.fmri_file_template, args.template_brain,
                args.target_subject, remote_fmri_dir, args.overwrite_labels_data, args.excluded_labels,
                args.overwrite_mri_segstat, args.norm_percs)

    if 'calc_volumetric_labels_mean' in args.function:
        flags['calc_volumetric_labels_mean'] = calc_volumetric_labels_mean(
            subject, args.atlas, args.fmri_file_template, args.labels_extract_mode, args.overwrite_parc_aseg_file,
            args.norm_percs, args.print_only, args=args)

    return flags


def read_cmd_args(argv=None):
    import argparse
    from src.utils import args_utils as au

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-c', '--contrast', help='contrast map', required=False, default='')
    parser.add_argument('-n', '--contrast_name', help='contrast map', required=False, default='')
    parser.add_argument('-t', '--task', help='task', required=False, default='', type=au.str_arr_type)
    parser.add_argument('--threshold', help='clustering threshold', required=False, default=2, type=float)
    parser.add_argument('--fsfast', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--is_pet', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--contrast_template', help='', required=False, default='')
    parser.add_argument('--existing_format', help='existing format', required=False, default='mgz')
    parser.add_argument('--input_format', help='input format', required=False, default='nii.gz')
    parser.add_argument('--volume_type', help='volume type', required=False, default='mni305')
    parser.add_argument('--volume_name', help='volume file name', required=False, default='')
    parser.add_argument('--surface_name', help='surface_name', required=False, default='pial')
    parser.add_argument('--meg_subject', help='meg_subject', required=False, default='')
    parser.add_argument('--inverse_method', help='inverse method', required=False, default='dSPM')
    parser.add_argument('--remote_fmri_dir', help='remote fMRI folder', required=False, default='')

    parser.add_argument('--overwrite_surf_data', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_colors_file', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_volume', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_subs_data', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_parc_aseg_file', help='', required=False, default=0, type=au.is_true)

    parser.add_argument('--norm_by_percentile', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--norm_percs', help='', required=False, default='1,99', type=au.int_arr_type)
    parser.add_argument('--symetric_colors', help='', required=False, default=1, type=au.is_true)

    # Resting state flags
    parser.add_argument('--fmri_file_template', help='', required=False, default='')
    parser.add_argument('--fmri_sub_file_template', help='', required=False, default='')
    parser.add_argument('--fsd', help='functional subdirectory', required=False, default='rest')
    parser.add_argument('--only_preproc', help='run only only_preproc', required=False, default=0, type=au.is_true)
    parser.add_argument('--labels_extract_mode', help='', required=False, default='mean', type=au.str_arr_type)
    parser.add_argument('--morph_labels_from_subject', help='', required=False, default='fsaverage')
    parser.add_argument('--morph_labels_to_subject', help='', required=False, default='')
    parser.add_argument('--resting_state_plot', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--resting_state_plot_all_vertices', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--excluded_labels', help='', required=False, default='corpuscallosum,unknown', type=au.str_arr_type)
    parser.add_argument('--st_template', help='', required=False, default='*{subject}_{atlas}*.txt')
    parser.add_argument('--overwrite_labels_data', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_activity_data', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_mri_segstat', help='', required=False, default=0, type=au.is_true)
    # parser.add_argument('--raw_fwhm', help='Raw Full Width at Half Maximum for Spatial Smoothing', required=False, default=5, type=float)
    parser.add_argument('--template_brain', help='', required=False, default='')
    parser.add_argument('--target_subject', help='', required=False, default='')
    # parser.add_argument('--fsd', help='functional subdirectory', required=False, default='rest')
    parser.add_argument('--fwhm', help='', required=False, default=6, type=float)
    parser.add_argument('--lfp', help='', required=False, default=0.08, type=float)
    parser.add_argument('--nskip', help='', required=False, default=4, type=int)
    parser.add_argument('--print_only', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_4d_preproc', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--backup_existing_files', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--pick_the_first_one', help='', required=False, default=0, type=au.is_true)

    # Misc flags
    parser.add_argument('--fmri_fname', help='', required=False, default='')
    parser.add_argument('--labels_order_fname', help='', required=False, default='')
    parser.add_argument('--labels_indices_to_remove_from_data', help='', required=False, default='', type=au.int_arr_type)
    parser.add_argument('--subcortical_codes_file', help='', required=False, default='sub_cortical_codes.txt')
    parser.add_argument('--aseg_stats_fname', help='', required=False, default='')
    parser.add_argument('--calc_subs_surface_method', help='', required=False, default='max')
    parser.add_argument('--calc_subs_surface_points', help='', required=False, default=100, type=int)
    parser.add_argument('--subs_threshold', help='', required=False, default=2, type=float)

    pu.add_common_args(parser)
    args = utils.Bag(au.parse_parser(parser, argv))
    args.necessary_files = {'surf': ['lh.sphere.reg', 'rh.sphere.reg']}
    if 'clean_4d_data' in args.function or args.function == 'prepare_subject_folder':
        args.necessary_files = {'surf': ['rh.thickness', 'lh.thickness', 'rh.white', 'lh.white', 'lh.sphere.reg', 'rh.sphere.reg'],
                                'mri': ['brainmask.mgz', 'orig.mgz', 'aparc+aseg.mgz'],
                                'mri:transforms': ['talairach.xfm'],
                                'label': ['lh.cortex.label', 'rh.cortex.label']}
        # 'label': ['lh.cortex.label', 'rh.cortex.label']
    if args.is_pet:
        args.fsfast = False
    # print(args)
    for sub in args.subject:
        if '*' in sub:
            args.subject.remove(sub)
            args.subject.extend([fol.split(op.sep)[-1] for fol in glob.glob(op.join(FMRI_DIR, sub))])
    args.subcortical_codes_file = op.join(MMVT_DIR, args.subcortical_codes_file)
    if 'rest' in args.function:
        args.function.extend(['project_volume_to_surface', 'analyze_4d_data', 'save_dynamic_activity_map',
                              'calc_subs_activity'])
    return args


if __name__ == '__main__':
    args = read_cmd_args()
    pu.run_on_subjects(args, main)
    print('finish!')
