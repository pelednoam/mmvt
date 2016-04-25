import os
import shutil
import numpy as np
from collections import defaultdict, OrderedDict
import itertools
import time
import re
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg
import matplotlib.cm as cmx
import subprocess
from functools import partial
import glob
import mne
import colorsys
import math
import os.path as op
import types
from sklearn.datasets.base import Bunch
import traceback
import multiprocessing
import scipy.io as sio

try:
    import cPickle as pickle
except:
    import pickle
import uuid

PLY_HEADER = 'ply\nformat ascii 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nelement face {}\nproperty list uchar int vertex_index\nend_header\n'
STAT_AVG, STAT_DIFF = range(2)
HEMIS = ['lh', 'rh']

class Bag( dict ):
    """ a dict with d.key short for d["key"]
        d = Bag( k=v ... / **dict / dict.items() / [(k,v) ...] )  just like dict
    """
        # aka Dotdict

    def __init__(self, *args, **kwargs):
        dict.__init__( self, *args, **kwargs )
        self.__dict__ = self

    def __getnewargs__(self):  # for cPickle.dump( d, file, protocol=-1)
        return tuple(self)


def get_exisiting_dir(dirs):
    ex_dirs = [d for d in dirs if os.path.isdir(d)]
    if len(ex_dirs)==0:
        raise Exception('No exisiting dir!')
    else:
        return ex_dirs[0]


def get_exisiting_file(dirs):
    ex_files = [d for d in dirs if os.path.isfile(d)]
    if len(ex_files)==0:
        raise Exception('No exisiting file!')
    else:
        return ex_files[0]


def delete_folder_files(fol):
    if os.path.isdir(fol):
        shutil.rmtree(fol)
    os.makedirs(fol)


def get_scalar_map(x_min, x_max, color_map='jet'):
    cm = plt.get_cmap(color_map)
    cNorm = matplotlib.colors.Normalize(vmin=x_min, vmax=x_max)
    return cmx.ScalarMappable(norm=cNorm, cmap=cm)


def arr_to_colors(x, x_min=None, x_max=None, colors_map='jet', scalar_map=None):
    if scalar_map is None:
        x_min, x_max = check_min_max(x, x_min, x_max)
        scalar_map = get_scalar_map(x_min, x_max, colors_map)
    return scalar_map.to_rgba(x)


def mat_to_colors(x, x_min=None, x_max=None, colorsMap='jet', scalar_map=None, flip_cm=False):
    if flip_cm:
        x = -x
        x_min = np.min(x) if x_max is None else -x_max
        x_max = np.max(x) if x_min is None else -x_min

    x_min, x_max = check_min_max(x, x_min, x_max)
    colors = arr_to_colors(x, x_min, x_max, colorsMap, scalar_map)
    return colors[:, :, :3]


def check_min_max(x, x_min, x_max):
    if x_min is None:
        x_min = np.min(x)
    if x_max is None:
        x_max = np.max(x)
    return x_min, x_max


def arr_to_colors_two_colors_maps(x, x_min=None, x_max=None, cm_big='YlOrRd', cm_small='PuBu', threshold=0, default_val=0,
                                  scalar_map_big=None, scalar_map_small=None, flip_cm_big=False, flip_cm_small=False):
    colors = np.ones((len(x), 3)) * default_val
    x_min, x_max = check_min_max(x, x_min, x_max)

    if np.sum(x >= threshold) > 0:
        if not flip_cm_big:
            big_colors = arr_to_colors(x[x>=threshold], threshold, x_max, cm_big, scalar_map_big)[:, :3]
        else:
            big_colors = arr_to_colors(-x[x>=threshold], -x_max, -threshold, cm_big, scalar_map_big)[:, :3]
        colors[x>=threshold, :] = big_colors
    if np.sum(x <= -threshold) > 0:
        if not flip_cm_small:
            small_colors = arr_to_colors(x[x<=-threshold], x_min, -threshold, cm_small, scalar_map_small)[:, :3]
        else:
            small_colors = arr_to_colors(-x[x<=-threshold], threshold, -x_min, cm_small, scalar_map_small)[:, :3]
        colors[x<=-threshold, :] = small_colors
    return colors


def mat_to_colors_two_colors_maps(x, x_min=None, x_max=None, cm_big='YlOrRd', cm_small='PuBu', threshold=0, default_val=0,
        scalar_map_big=None, scalar_map_small=None, flip_cm_big=False, flip_cm_small=False):
    colors = np.ones((x.shape[0],x.shape[1], 3)) * default_val
    x_min, x_max = check_min_max(x, x_min, x_max)
    # scalar_map_pos = get_scalar_map(threshold, x_max, cm_big)
    # scalar_map_neg = get_scalar_map(x_min, -threshold, cm_small)
    # todo: calculate the scaler map before the loop to speed up
    scalar_map_pos, scalar_map_neg = None, None
    for ind in range(x.shape[0]):
        colors[ind] = arr_to_colors_two_colors_maps(x[ind], x_min, x_max, cm_big, cm_small, threshold,
            default_val, scalar_map_pos, scalar_map_neg, flip_cm_big, flip_cm_small)
    return np.array(colors)


def read_srf_file(srf_file):
    with open(srf_file, 'r') as f:
        lines = f.readlines()
        verts_num, faces_num = map(int, lines[1].strip().split(' '))
        sep = '  ' if len(lines[2].split('  ')) > 1 else ' '
        verts = np.array([list(map(float, l.strip().split(sep))) for l in lines[2:verts_num+2]])[:,:-1]
        faces = np.array([list(map(int, l.strip().split(' '))) for l in lines[verts_num+2:]])[:,:-1]
    return verts, faces, verts_num, faces_num


def read_ply_file(ply_file):
    with open(ply_file, 'r') as f:
        lines = f.readlines()
        verts_num = int(lines[2].split(' ')[-1])
        faces_num = int(lines[6].split(' ')[-1])
        verts_lines = lines[9:9 + verts_num]
        faces_lines = lines[9 + verts_num:]
        verts = np.array([list(map(float, l.strip().split(' '))) for l in verts_lines])
        faces = np.array([list(map(int, l.strip().split(' '))) for l in faces_lines])[:,1:]
    return verts, faces


def write_ply_file(verts, faces, ply_file_name):
    verts_num = verts.shape[0]
    faces_num = faces.shape[0]
    faces = np.hstack((np.ones((faces_num, 1)) * 3, faces))
    with open(ply_file_name, 'w') as f:
        f.write(PLY_HEADER.format(verts_num, faces_num))
    with open(ply_file_name, 'ab') as f:
        np.savetxt(f, verts, fmt='%.5f', delimiter=' ')
        np.savetxt(f, faces, fmt='%d', delimiter=' ')


def read_obj_file(obj_file):
    with open(obj_file, 'r') as f:
        lines = f.readlines()
        verts = np.array([[float(v) for v in l.strip().split(' ')[1:]] for l in lines if l[0]=='v'])
        faces = np.array([[int(v) for v in l.strip().split(' ')[1:]] for l in lines if l[0]=='f'])
    faces -= 1
    return verts, faces


def srf2ply(srf_file, ply_file):
    # print('convert {} to {}'.format(namebase(srf_file), namebase(ply_file)))
    verts, faces, verts_num, faces_num = read_srf_file(srf_file)
    write_ply_file(verts, faces, ply_file)
    return ply_file


def obj2ply(obj_file, ply_file):
    verts, faces = read_obj_file(obj_file)
    write_ply_file(verts, faces, ply_file)


def convert_srf_files_to_ply(srf_folder, overwrite=True):
    srf_files = glob.glob(os.path.join(srf_folder, '*.srf'))
    for srf_file in srf_files:
        ply_file = '{}.ply'.format(srf_file[:-4])
        if overwrite or not op.isfile(ply_file):
            srf2ply(srf_file, ply_file)


def get_ply_vertices_num(ply_file_template):
    if os.path.isfile(ply_file_template.format('rh')) and os.path.isfile(ply_file_template.format('lh')):
        rh_vertices, _ = read_ply_file(ply_file_template.format('rh'))
        lh_vertices, _ = read_ply_file(ply_file_template.format('lh'))
        return {'rh':rh_vertices.shape[0], 'lh':lh_vertices.shape[0]}
    else:
        print('No surface ply files!')
        return None


def check_hemi(hemi):
    if hemi in HEMIS:
        hemi = [hemi]
    elif hemi=='both':
        hemi = HEMIS
    else:
        raise ValueError('wrong hemi value!')
    return hemi


def get_data_max_min(data, norm_by_percentile, norm_percs=None, data_per_hemi=False, hemis = HEMIS):
    if data_per_hemi:
        if norm_by_percentile:
            data_max = max([np.percentile(data[hemi], norm_percs[1]) for hemi in hemis])
            data_min = min([np.percentile(data[hemi], norm_percs[0]) for hemi in hemis])
        else:
            data_max = max([np.max(data[hemi]) for hemi in hemis])
            data_min = min([np.min(data[hemi]) for hemi in hemis])
    else:
        if norm_by_percentile:
            data_max = np.percentile(data, norm_percs[1])
            data_min = np.percentile(data, norm_percs[0])
        else:
            data_max = np.max(data)
            data_min = np.min(data)
    return data_max, data_min


def get_max_abs(data_max, data_min):
    return max(map(abs, [data_max, data_min]))


def normalize_data(data, norm_by_percentile, norm_percs=None):
    data_max, data_min = get_data_max_min(data, norm_by_percentile, norm_percs)
    max_abs = get_max_abs(data_max, data_min)
    norm_data = data / max_abs
    return norm_data


def calc_stat_data(data, stat, axis=2):
    if stat == STAT_AVG:
        stat_data = np.squeeze(np.mean(data, axis=axis))
    elif stat == STAT_DIFF:
        stat_data = np.squeeze(np.diff(data, axis=axis))
    else:
        raise Exception('Wonrg stat value!')
    return stat_data


def read_freesurfer_lookup_table(freesurfer_home='', get_colors=False):
    freesurfer_home = get_environ_dir('FREESURFER_HOME', freesurfer_home)
    lut_fname = os.path.join(freesurfer_home, 'FreeSurferColorLUT.txt')
    if get_colors:
        lut = np.genfromtxt(lut_fname, dtype=None, usecols=(0, 1, 2, 3, 4, 5), names=['id', 'name', 'r', 'g', 'b', 'a'])
    else:
        lut = np.genfromtxt(lut_fname, dtype=None, usecols=(0, 1), names=['id', 'name'])
    return lut


def get_environ_dir(var_name, default_val=''):
    ret_val = os.environ.get(var_name) if default_val == '' else default_val
    if not os.path.isdir(ret_val):
        raise Exception('get_environ_dir: No existing dir!')
    return ret_val


def get_link_dir(links_dir, link_name, var_name='', default_val='', throw_exception=False):
    val = os.path.join(links_dir, link_name)
    if not os.path.isdir(val) and default_val != '':
        val = default_val
    if not os.path.isdir(val):
        val = os.environ.get(var_name, '')
    if not os.path.isdir(val):
        if throw_exception:
            raise Exception('No {} dir!'.format(link_name))
        else:
            print('No {} dir!'.format(link_name))
    return val


def get_links_dir():
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    proj_dir = os.path.split(curr_dir)[0]
    code_dir = os.path.split(proj_dir)[0]
    links_dir = os.path.join(code_dir, 'links')
    return links_dir


def get_electrodes_labeling(subject, atlas, bipolar=False, error_radius=3, elec_length=4):
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    proj_dir = os.path.split(curr_dir)[0]
    code_dir = os.path.split(proj_dir)[0]
    electrode_labeling_fname = op.join(code_dir, 'electrodes_rois', 'electrodes',
        '{}_{}_electrodes_all_rois_cigar_r_{}_l_{}{}.pkl'.format(subject, atlas, error_radius, elec_length,
        '_bipolar_stretch' if bipolar else ''))
    if op.isfile(electrode_labeling_fname):
        return load(electrode_labeling_fname)
    else:
        return None

# def read_sub_cortical_lookup_table(lookup_table_file_name):
#     names = {}
#     with open(lookup_table_file_name, 'r') as f:
#         for line in f.readlines():
#             lines = line.strip().split('\t')
#             if len(lines) > 1:
#                 name, code = lines[0].strip(), int(lines[1])
#                 names[code] = name
#     return names


def get_numeric_index_to_label(label, lut=None, free_surfer_home=''):
    if lut is None:
        if free_surfer_home == '':
            free_surfer_home = os.environ['FREE_SURFER_HOME']
        lut = read_freesurfer_lookup_table(free_surfer_home)
    if type(label) == str:
        seg_name = label
        seg_id = lut['id'][lut['name'] == seg_name][0]
    elif type(label) == int:
        seg_id = label
        seg_name = lut['name'][lut['id'] == seg_id][0]
    return seg_name, int(seg_id)


def lut_labels_to_indices(regions, lut):
    sub_corticals = []
    for reg in regions:
        name, id = get_numeric_index_to_label(reg, lut)
        sub_corticals.append(id)
    return sub_corticals

def how_many_curlies(str):
    return len(re.findall('\{*\}', str))


def run_script(cmd, verbose=False):
    if verbose:
        print('running: {}'.format(cmd))
    output = subprocess.check_output('{} | tee /dev/stderr'.format(cmd),
                                     shell=True)
    print(output)
    return output


# def partial_run_script(vars, more_vars=None):
#     return partial(lambda cmd,v:run_script(cmd.format(**v)), v=vars)

def partial_run_script(vars, more_vars=None, print_only=False):
    return partial(_run_script_wrapper, vars=vars, print_only=print_only)


def _run_script_wrapper(cmd, vars, print_only=False, **kwargs):
    for k,v in kwargs.items():
        vars[k] = v
    print(cmd.format(**vars))
    if not print_only:
        run_script(cmd.format(**vars))


def sub_cortical_voxels_generator(aseg, seg_labels, spacing=5, use_grid=True, freesurfer_home=''):
    if freesurfer_home=='':
        freesurfer_home = os.environ['FREE_SURFER_HOME']

    # Read the segmentation data using nibabel
    aseg_data = aseg.get_data()

    # Read the freesurfer lookup table
    lut = read_freesurfer_lookup_table(freesurfer_home)

    # Generate a grid using spacing
    grid = None
    if use_grid:
        grid = generate_grid_using_spacing(spacing, aseg_data.shape)

    # Get the indices to the desired labels
    for label in seg_labels:
        seg_name, seg_id = get_numeric_index_to_label(label, lut)
        pts = calc_label_voxels(seg_id, aseg_data, grid)
        yield pts, seg_name, seg_id


def generate_grid_using_spacing(spacing, shp):
    # Generate a grid using spacing
    kernel = np.zeros((int(spacing), int(spacing), int(spacing)))
    kernel[0, 0, 0] = 1
    sx, sy, sz = shp
    nx, ny, nz = np.ceil((sx/spacing, sy/spacing, sz/spacing))
    grid = np.tile(kernel, (nx, ny, nz))
    grid = grid[:sx, :sy, :sz]
    grid = grid.astype('bool')
    return grid


def calc_label_voxels(seg_id, aseg_data, grid=None):
    # Get indices to label
    ix = aseg_data == seg_id
    if grid is not None:
        ix *= grid  # downsample to grid
    pts = np.array(np.where(ix)).T
    return pts


def transform_voxels_to_RAS(aseg_hdr, pts):
    from mne.transforms import apply_trans

    # Transform data to RAS coordinates
    trans = aseg_hdr.get_vox2ras_tkr()
    pts = apply_trans(trans, pts)

    return pts


def transform_RAS_to_voxels(pts, aseg_hdr=None, subject_mri_dir=''):
    from mne.transforms import apply_trans, invert_transform

    if aseg_hdr is None:
        aseg_hdr = get_aseg_header(subject_mri_dir)
    trans = aseg_hdr.get_vox2ras_tkr()
    trans = invert_transform(trans)
    pts = apply_trans(trans, pts)
    return pts


def get_aseg_header(subject_mri_dir):
    import  nibabel as nib
    aseg_fname = os.path.join(subject_mri_dir, 'mri', 'aseg.mgz')
    aseg = nib.load(aseg_fname)
    aseg_hdr = aseg.get_header()
    return aseg_hdr


def namebase(file_name):
    return os.path.splitext(os.path.basename(file_name))[0]


def file_type(file_name):
    return os.path.splitext(os.path.basename(file_name))[1][1:]


def morph_labels_from_fsaverage(subject, subjects_dir='', aparc_name='aparc250', fs_labels_fol='',
            sub_labels_fol='', n_jobs=6, fsaverage='fsaverage', overwrite=False):
    if subjects_dir=='':
        subjects_dir = os.environ['SUBJECTS_DIR']
    subject_dir = os.path.join(subjects_dir, subject)
    labels_fol = os.path.join(subjects_dir, fsaverage, 'label', aparc_name) if fs_labels_fol=='' else fs_labels_fol
    sub_labels_fol = os.path.join(subject_dir, 'label', aparc_name) if sub_labels_fol=='' else sub_labels_fol
    if not os.path.isdir(sub_labels_fol):
        os.makedirs(sub_labels_fol)
    labels_files = glob.glob(os.path.join(labels_fol, '*.label'))
    if len(labels_files) == 0:
        raise Exception('morph_labels_from_fsaverage: No labels files found in {}!'.format(labels_fol))
    surf_loaded = False
    for label_file in labels_files:
        local_label_name = os.path.join(sub_labels_fol, '{}.label'.format(os.path.splitext(os.path.split(label_file)[1])[0]))
        if not os.path.isfile(local_label_name) or overwrite:
            fs_label = mne.read_label(label_file)
            fs_label.values.fill(1.0)
            sub_label = fs_label.morph(fsaverage, subject, grade=None, n_jobs=n_jobs, subjects_dir=subjects_dir)
            if np.all(sub_label.pos == 0):
                if not surf_loaded:
                    verts = {}
                    for hemi in HEMIS:
                        d = np.load(op.join(subjects_dir, subject, 'mmvt', '{}.pial.npz'.format(hemi)))
                        verts[hemi] = d['verts']
                    surf_loaded = True
                sub_label.pos = verts[sub_label.hemi][sub_label.vertices]
            sub_label.save(local_label_name)


def labels_to_annot(subject, subjects_dir='', aparc_name='aparc250', labels_fol='', overwrite=True):
    if subjects_dir == '':
        subjects_dir = os.environ['SUBJECTS_DIR']
    subject_dir = os.path.join(subjects_dir, subject)
    labels_fol = os.path.join(subject_dir, 'label', aparc_name) if labels_fol=='' else labels_fol
    labels = []
    if overwrite:
        for hemi in HEMIS:
            remove_file(os.path.join(subject_dir, 'label', '{}.{}.annot'.format(hemi, aparc_name)))
    labels_files = glob.glob(os.path.join(labels_fol, '*.label'))
    if len(labels_files) == 0:
        raise Exception('labels_to_annot: No labels files!')
    for label_file in labels_files:
        label = mne.read_label(label_file)
        # print(label.name)
        labels.append(label)

    labels.sort(key=lambda l: l.name)
    mne.write_labels_to_annot(subject=subject, labels=labels, parc=aparc_name, overwrite=overwrite,
                              subjects_dir=subjects_dir)


def remove_file(fname, raise_error_if_does_not_exist=False):
    try:
        if os.path.isfile(fname):
            os.remove(fname)
    except:
        if raise_error_if_does_not_exist:
            raise Exception(traceback.format_exc())
        else:
            print(traceback.format_exc())

def get_hemis(hemi):
    return HEMIS if hemi == 'both' else [hemi]


def rmtree(fol):
    if os.path.isdir(fol):
        shutil.rmtree(fol)

def make_dir(fol):
    if not os.path.isdir(fol):
        os.makedirs(fol)
    return fol


def get_subfolders(fol):
    return [os.path.join(fol,subfol) for subfol in os.listdir(fol) if os.path.isdir(os.path.join(fol,subfol))]


def get_spaced_colors(n):
    if n <= 7:
        colors = ['r', 'g', 'c', 'm', 'y', 'b', 'k'][:n]
    else:
        HSV_tuples = [(x*1.0/n, 0.5, 0.5) for x in range(n)]
        colors = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return colors


def downsample_2d(x, R):
    return x.reshape(x.shape[0],-1,R).mean(2)


def read_sub_corticals_code_file(sub_corticals_codes_file, read_also_names=False):
    if os.path.isfile(sub_corticals_codes_file):
        codes = np.genfromtxt(sub_corticals_codes_file, usecols=(1), delimiter=',', dtype=int)
        codes = map(int, codes)
        if read_also_names:
            names = np.genfromtxt(sub_corticals_codes_file, usecols=(0), delimiter=',', dtype=str)
            names = map(str, names)
            sub_corticals = {code:name for code, name in zip(codes, names)}
        else:
            sub_corticals = list(codes)
    else:
        sub_corticals = []
    return sub_corticals


def convert_stcs_to_h5(root, folds):
    for fol in folds:
        stcs_files = glob.glob(op.join(root, fol, '*-rh.stc'))
        for stc_rh_file in stcs_files:
            stc_rh = mne.read_source_estimate(stc_rh_file)
            stc_lh_file = '{}-lh.stc'.format(stc_rh_file[:-len('-lh.stc')])
            stc_lh = mne.read_source_estimate(stc_lh_file)
            if np.all(stc_rh.data==stc_lh.data) and np.all(stc_rh.lh_data==stc_lh.lh_data) and np.all(stc_rh.rh_data==stc_lh.rh_data):
                if not op.isfile('{}-stc.h5'.format(stc_rh_file[:-len('-lh.stc')])):
                    stc_rh.save(stc_rh_file[:-len('-rh.stc')], ftype='h5')
                    stc_h5 = mne.read_source_estimate('{}-stc.h5'.format(stc_rh_file[:-len('-lh.stc')]))
                    if np.all(stc_h5.data==stc_rh.data) and np.all(stc_h5.rh_data==stc_rh.rh_data) and np.all(stc_h5.lh_data==stc_lh.lh_data):
                        print('delete {} and {}'.format(stc_rh_file, stc_lh_file))
                        os.remove(stc_rh_file)
                        os.remove(stc_lh_file)


def get_activity_max_min(stc, norm_by_percentile=False, norm_percs=None, threshold=None, hemis=HEMIS):
    if isinstance(stc, dict):
        if norm_by_percentile:
            data_max = max([np.percentile(stc[hemi], norm_percs[1]) for hemi in hemis])
            data_min = min([np.percentile(stc[hemi], norm_percs[0]) for hemi in hemis])
        else:
            data_max = max([np.max(stc[hemi]) for hemi in hemis])
            data_min = min([np.min(stc[hemi]) for hemi in hemis])
    else:
        if norm_by_percentile:
            data_max = np.percentile(stc.data, norm_percs[1])
            data_min = np.percentile(stc.data, norm_percs[0])
        else:
            data_max = np.max(stc.data)
            data_min = np.min(stc.data)

    if threshold is not None:
        if threshold > data_max:
            data_max = threshold * 1.1
        if -threshold < data_min:
            data_min = -threshold * 1.1

    return data_max, data_min


def get_max_min(data, threshold=None):
    ret = np.zeros((data.shape[1], 2))
    if threshold is None:
        ret[:, 0], ret[:, 1] = np.max(data, 0), np.min(data, 0)
    else:
        ret[:, 0] = max(np.max(data, 0), threshold)
        ret[:, 1] = min(np.min(data, 0), -threshold)
    return ret


def get_abs_max(data):
    ret = np.zeros((data.shape[1], 2))
    ret[:, 0], ret[:, 1] = np.max(data, 0), np.min(data, 0)
    return [r[0] if abs(r[0])>abs(r[1]) else r[1] for r in ret]


def copy_file(src, dst):
    shutil.copyfile(src, dst)


def get_labels_vertices(labels, vertno):
    nvert = [len(vn) for vn in vertno]
    label_vertidx, labels_names = [], []
    for label in labels:
        print('calculating vertices for {}'.format(label.name))
        if label.hemi == 'both':
            # handle BiHemiLabel
            sub_labels = [label.lh, label.rh]
        else:
            sub_labels = [label]
        this_vertidx = list()
        for slabel in sub_labels:
            if slabel.hemi == 'lh':
                this_vertno = np.intersect1d(vertno[0], slabel.vertices)
                vertidx = np.searchsorted(vertno[0], this_vertno)
            elif slabel.hemi == 'rh':
                this_vertno = np.intersect1d(vertno[1], slabel.vertices)
                vertidx = nvert[0] + np.searchsorted(vertno[1], this_vertno)
            else:
                raise ValueError('label %s has invalid hemi' % label.name)
            this_vertidx.append(vertidx)

        # convert it to an array
        this_vertidx = np.concatenate(this_vertidx)
        if len(this_vertidx) == 0:
            print('source space does not contain any vertices for label {}'.format(label.name))
            this_vertidx = None  # to later check if label is empty
        label_vertidx.append(this_vertidx)
        labels_names.append(label.name)
    return label_vertidx, labels_names


def read_labels(labels_fol, hemi='both'):
    hemis = [hemi] if hemi != 'both' else HEMIS
    labels = []
    for hemi in hemis:
        for label_file in glob.glob(os.path.join(labels_fol, '*{}.label'.format(hemi))):
            print('read label from {}'.format(label_file))
            label = mne.read_label(label_file)
            labels.append(label)
    return labels


def dic2bunch(dic):
    return Bunch(**dic)


def check_stc_vertices(stc, hemi, ply_file):
    verts, faces = read_ply_file(ply_file)
    data = stc_hemi_data(stc, hemi)
    if verts.shape[0]!=data.shape[0]:
        raise Exception('save_activity_map: wrong number of vertices!')
    else:
        print('Both {}.pial.ply and the stc file have {} vertices'.format(hemi, data.shape[0]))


def stc_hemi_data(stc, hemi):
    return stc.rh_data if hemi=='rh' else stc.lh_data


def parallel_run(pool, func, params, n_jobs):
    return pool.map(func, params) if n_jobs > 1 else [func(p) for p in params]


def fsaverage_vertices():
    return [np.arange(10242), np.arange(10242)]


def prepare_local_subjects_folder(neccesary_files, subject, remote_subject_dir, local_subjects_dir, print_traceback=False):
    local_subject_dir = os.path.join(local_subjects_dir, subject)
    for fol, files in neccesary_files.items():
        if not os.path.isdir(os.path.join(local_subject_dir, fol)):
            os.makedirs(os.path.join(local_subject_dir, fol))
        for file_name in files:
            try:
                if not os.path.isfile(os.path.join(local_subject_dir, fol, file_name)):
                    shutil.copyfile(os.path.join(remote_subject_dir, subject, fol, file_name),
                                os.path.join(local_subject_dir, fol, file_name))
            except:
                if print_traceback:
                    print(traceback.format_exc())
    all_files_exists = True
    for fol, files in neccesary_files.items():
        for file_name in files:
            if not os.path.isfile(os.path.join(local_subject_dir, fol, file_name)):
                print("The file {} doesn't exist in the local subjects folder!!!".format(file_name))
                all_files_exists = False
    if not all_files_exists:
        # raise Exception('Not all files exist in the local subject folder!!!')
        return False


def to_ras(points, round_coo=False):
    RAS_AFF = np.array([[-1, 0, 0, 128],
        [0, 0, -1, 128],
        [0, 1, 0, 128],
        [0, 0, 0, 1]])
    ras = [np.dot(RAS_AFF, np.append(p, 1))[:3] for p in points]
    if round_coo:
        ras = np.array([np.around(p) for p in ras], dtype=np.int16)
    return np.array(ras)


def check_for_necessary_files(neccesary_files, root_fol):
    for fol, files in neccesary_files.items():
        for file in files:
            full_path = os.path.join(root_fol, fol, file)
            if not os.path.isfile(full_path):
                raise Exception('{} does not exist!'.format(full_path))


def run_parallel(func, params, njobs=1):
    if njobs == 1:
        results = [func(p) for p in params]
    else:
        pool = multiprocessing.Pool(processes=njobs)
        results = pool.map(func, params)
        pool.close()
    return results


def get_parent_fol():
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.split(curr_dir)[0]


def get_figs_fol():
    return os.path.join(get_parent_fol(), 'figs')


def get_files_fol():
    return os.path.join(get_parent_fol(), 'pkls')


def save(obj, fname):
    with open(fname, 'wb') as fp:
        # protocol=2 so we'll be able to load in python 2.7
        pickle.dump(obj, fp)


def load(fname):
    with open(fname, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def fwd_vertno(fwd):
    return sum(map(len, [src['vertno'] for src in fwd['src']]))


def plot_3d_PCA(X, names=None, n_components=3):
    X_PCs = calc_PCA(X, n_components)
    plot_3d_scatter(X_PCs, names)


def calc_PCA(X, n_components=3):
    from sklearn import decomposition
    X = (X - np.mean(X, 0)) / np.std(X, 0) # You need to normalize your data first
    pca = decomposition.PCA(n_components=n_components)
    X = pca.fit(X).transform(X)
    print ('explained variance (first %d components): %.2f'%(n_components, sum(pca.explained_variance_ratio_)))
    return X


def plot_3d_scatter(X, names=None, labels=None, classifier=None):
    from mpl_toolkits.mplot3d import Axes3D, proj3d
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2])

    if not names is None:
        if not labels is None:
            for label in labels:
                ind = names.index(label)
                add_annotation(ax, label, X[ind, 0], X[ind, 1], X[ind, 2])
        else:
            for x,y,z,name in zip(X[:, 0], X[:, 1], X[:, 2], names):
                add_annotation(ax, name, x, y, z)

    if not classifier is None:
        make_ellipses(classifier, ax)

    plt.show()


def plot_2d_scatter(X, names=None, labels=None, classifier=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(X[:, 0], X[:, 1])

    if not names is None:
        if not labels is None:
            for label in labels:
                ind = names.index(label)
                add_annotation(ax, label, X[ind, 0], X[ind, 1])
        else:
            for x, y, name in zip(X[:, 0], X[:, 1], names):
                add_annotation(ax, name, x, y)

    if not classifier is None:
        make_ellipses(classifier, ax)

    plt.show()


def add_annotation(ax, text, x, y, z=None):
    from mpl_toolkits.mplot3d import proj3d
    import pylab
    if not z is None:
        x2, y2, _ = proj3d.proj_transform(x,y,z, ax.get_proj())
    else:
        x2, y2 = x, y
    pylab.annotate(
        text, xy = (x2, y2), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))


def calc_clusters_bic(X, n_components=0, do_plot=True):
    from sklearn import mixture
    import itertools

    lowest_bic = np.infty
    bic = []
    if n_components==0:
        n_components = X.shape[0]
    n_components_range = range(1, n_components)
    cv_types = ['spherical', 'diag']#, 'tied'] # 'full'
    res = defaultdict(dict)
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a mixture of Gaussians with EM
            gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            res[cv_type][n_components] = gmm
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)

    if do_plot:
        # Plot the BIC scores
        color_iter = itertools.cycle(['k', 'r', 'g', 'b', 'c', 'm', 'y'])
        bars = []
        spl = plt.subplot(1, 1, 1)
        for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
            xpos = np.array(n_components_range) + .2 * (i - 2)
            bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                          (i + 1) * len(n_components_range)],
                                width=.2, color=color))
        plt.xticks(n_components_range)
        plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
        plt.title('BIC score per model')
        xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
            .2 * np.floor(bic.argmin() / len(n_components_range))
        plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
        spl.set_xlabel('Number of components')
        spl.legend([b[0] for b in bars], cv_types)
        plt.show()
    return res, best_gmm, bic


def make_ellipses(gmm, ax):
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import proj3d

    for n, color in enumerate('rgb'):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        x, y, z = gmm.means_[n, :3]
        x2, y2, _ = proj3d.proj_transform(x,y,z, ax.get_proj())
        ell = mpl.patches.Ellipse([x2, y2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)


def find_subsets(l, k):
    sl, used = set(l), set()
    picks = []
    while len(sl-used) >= k:
        pick = np.random.choice(list(sl-used), k, replace=False).tolist()
        picks.append(pick)
        used = used | set(pick)
    if len(sl-used) > 0:
        picks.append(list(sl-used))
    return picks

def flat_list_of_sets(l):
    from operator import or_
    return reduce(or_, l)


def flat_list_of_lists(l):
    return sum(l, [])


def how_many_cores():
    return multiprocessing.cpu_count()


def rand_letters(num):
    return str(uuid.uuid4())[:num]


def how_many_subplots(pics_num):
    if pics_num < 4:
        return pics_num, 1
    dims = [(k**2, k, k) for k in range(1,9)]
    for max_pics_num, x, y in dims:
        if pics_num <= max_pics_num:
            return x, y
    return 10, 10


def chunks(l, n):
    # todo: change the code to use np.array_split
    n = max(1, int(n))
    return [l[i:i + n] for i in range(0, len(l), n)]

def powerset(iterable):
    from itertools import chain, combinations
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def subsets(s):
    return map(set, powerset(s))


def stack(arr, stack_type='v'):
    '''
    :param arr: array input
    :param stack_type: v for vstack, h for hstack
    :return: numpy array
    '''
    if stack_type == 'v':
        stack_func = np.vstack
    elif stack_type == 'h':
        stack_func = np.hstack
    else:
        raise Exception('Wrong stack type! {}'.format(stack_type))

    X = []
    for item in arr:
        X = item if len(X)==0 else stack_func((X, item))
    return X


def elec_group_number(elec_name, bipolar=False):
    if isinstance(elec_name, bytes):
        elec_name = elec_name.decode('utf-8')
    if bipolar:
        elec_name2, elec_name1 = elec_name.split('-')
        group, num1 = elec_group_number(elec_name1, False)
        _, num2 = elec_group_number(elec_name2, False)
        return group, num1, num2
    else:
        ind = np.where([int(s.isdigit()) for s in elec_name])[-1][0]
        num = int(elec_name[ind:])
        group = elec_name[:ind]
        return group, num


def elec_group(elec_name, bipolar):
    if bipolar:
        group, _, _ = elec_group_number(elec_name, bipolar)
    else:
        group, _ = elec_group_number(elec_name, bipolar)
    return group


def max_min_diff(x):
    return max(x) - min(x)


def diff_4pc(y, dx=1):
    '''
    http://gilgamesh.cheme.cmu.edu/doc/software/jacapo/9-numerics/9.1-numpy/9.2-integration.html#numerical-differentiation
    calculate dy by 4-point center differencing using array slices

    \frac{y[i-2] - 8y[i-1] + 8[i+1] - y[i+2]}{12h}

    y[0] and y[1] must be defined by lower order methods
    and y[-1] and y[-2] must be defined by lower order methods

    :param y: the signal
    :param dx: np.diff(x): Assumes the points are evenely spaced!
    :return: The derivatives
    '''
    dy = np.zeros(y.shape,np.float)
    dy[2:-2] = (y[0:-4] - 8*y[1:-3] + 8*y[3:-1] - y[4:])/(12.*dx)
    dy[0] = (y[1]-y[0])/dx
    dy[1] = (y[2]-y[1])/dx
    dy[-2] = (y[-2] - y[-3])/dx
    dy[-1] = (y[-1] - y[-2])/dx
    return dy


def sort_dict_by_values(dic):
    return OrderedDict(sorted(dic.items()))


def first_key(dic):
    rev_fic = {v:k for k,v in dic.items()}
    first_item = sorted(dic.values())[0]
    return rev_fic[first_item]


def superset(x):
    return itertools.chain.from_iterable(itertools.combinations(x, n) for n in range(1, len(x)+1))
    # all_sets = set()
    # for l in range(1, len(arr)+1):
    #     for subset in itertools.combinations(arr, l):
    #         all_sets.add(subset)
    # return all_sets

def params_suffix(optimization_params):
    return ''.join(['_{}_{}'.format(param_key, param_val) for param_key, param_val in
        sorted(optimization_params.items())])


def time_to_go(now, run, runs_num, runs_num_to_print=10):
    if run % runs_num_to_print == 0 and run != 0:
        time_took = time.time() - now
        more_time = time_took / run * (runs_num - run)
        print('{}/{}, {:.2f}s, {:.2f}s to go!'.format(run, runs_num, time_took, more_time))


def lower_rec_indices(m):
    for i in range(m):
        for j in range(i):
            yield (i, j)


def lower_rec_to_arr(x):
    M = x.shape[0]
    L = int((M*M+M)/2-M)
    ret = np.zeros((L))
    for ind, (i,j) in enumerate(lower_rec_indices(M)):
        ret[ind] = x[i, j]
    return ret


def find_list_items_in_list(l_new, l_org):
    indices = []
    for item in l_new:
        indices.append(l_org.index(item) if item in l_org else -1)
    return indices


def moving_avg(x, window):
    # import pandas as pd
    # return pd.rolling_mean(x, window)#[:, window-1:]
    weights = np.repeat(1.0, window)/window
    sma = np.zeros((x.shape[0], x.shape[1] - window + 1))
    for ind in range(x.shape[0]):
        sma[ind] = np.convolve(x[ind], weights, 'valid')
    return sma


def is_exe(fpath):
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)


def set_exe_permissions(fpath):
    os.chmod(fpath, 0o744)


def both_hemi_files_exist(file_template):
    return op.isfile(file_template.format(hemi='rh')) and op.isfile(file_template.format(hemi='lh'))


def csv_from_excel(xlsx_fname, csv_fname):
    import xlrd
    import csv
    wb = xlrd.open_workbook(xlsx_fname)
    sh = wb.sheet_by_name('Sheet1')
    csv_file = open(csv_fname, 'w')
    wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)

    for rownum in range(sh.nrows):
        wr.writerow([val for val in sh.row_values(rownum)])

    csv_file.close()


def get_all_subjects(subjects_dir, prefix, exclude_substr):
    subjects = []
    folders = [namebase(fol) for fol in get_subfolders(subjects_dir)]
    for subject_fol in folders:
        if subject_fol[:len(prefix)].lower() == prefix and exclude_substr not in subject_fol:
            subjects.append(subject_fol)
    return subjects


def read_labels_parallel(subject, subjects_dir, atlas, n_jobs):
    labels_files = glob.glob(op.join(subjects_dir, subject, 'label', atlas, '*.label'))
    files_chunks = chunks(labels_files, len(labels_files) / n_jobs)
    results = run_parallel(_read_labels_parallel, files_chunks, n_jobs)
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


def merge_two_dics(dic1, dic2):
    # Only for python >= 3.5
    # return {**dic1, **dic2}
    ret = dic1.copy()
    ret.update(dic2)
    return ret


def color_name_to_rgb(color_name):
    try:
        import webcolors
        return webcolors.name_to_rgb(color_name)
    except:
        print('No webcolors!')
        return None


def color_name_to_rgb(rgb):
    try:
        import webcolors
        return webcolors.rgb_to_name(rgb)
    except:
        print('No webcolors!')
        return None


def make_evoked_smooth_and_positive(evoked, positive=True, moving_average_win_size=100):
    evoked_smooth = None
    for cond_ind in range(evoked.shape[2]):
        for label_ind in range(evoked.shape[0]):
            x = evoked[label_ind, :, cond_ind]
            if positive:
                x *= np.sign(x[np.argmax(np.abs(x))])
            evoked[label_ind, :, cond_ind] = x
        if moving_average_win_size > 0:
            evoked_smooth_cond = moving_avg(evoked[:, :, cond_ind], moving_average_win_size)
            if evoked_smooth is None:
                evoked_smooth = np.zeros((evoked_smooth_cond.shape[0], evoked_smooth_cond.shape[1], evoked.shape[2]))
            evoked_smooth[:, :, cond_ind] = evoked_smooth_cond
    return evoked_smooth


def get_hemi_indifferent_roi(roi):
    return roi.replace('-rh', '').replace('-lh', '').replace('rh-', '').replace('lh-', '').\
        replace('.rh', '').replace('.lh', '').replace('rh.', '').replace('lh.', '').\
        replace('Right-', '').replace('Left-', '').replace('-Right', '').replace('-Left', '').\
        replace('Right.', '').replace('Left.', '').replace('.Right', '').replace('.Left', '').\
        replace('right-', '').replace('left-', '').replace('-right', '').replace('-left', '').\
        replace('right.', '').replace('left.', '').replace('.right', '').replace('.left', '')


def get_hemi_indifferent_rois(rois):
    return set(map(lambda roi:get_hemi_indifferent_roi(roi), rois))


def get_args_list(args, key):
    if ',' in args[key]:
        ret = args[key].split(',')
    elif len(args[key]) == 0:
        ret = []
    else:
        ret = [args[key]]
    return ret


def show_image(image_fname):
    image = mpimg.imread(image_fname)
    plt.axis("off")
    plt.imshow(image)
    plt.tight_layout()
    plt.show()
