import os
import sys
import shutil
import numpy as np
from collections import defaultdict, OrderedDict, Counter
import itertools
import time
import re
try:
    import matplotlib.pyplot as plt
    import matplotlib
    import matplotlib.image as mpimg
    import matplotlib.cm as cmx
except:
    print('No matplotlib')
import subprocess
import functools
from functools import partial, reduce
import warnings
import glob
import mne
import colorsys
# import math
import os.path as op
# import types
import traceback
import multiprocessing
import getpass

try:
    import scipy.io as sio
except:
    print('No scipy!')

try:
    from decorator import decorator
except:
    pass

from src.mmvt_addon import mmvt_utils as mu
# links to mmvt_utils
Bag = mu.Bag
make_dir = mu.make_dir
hemi_files_exists = mu.hemi_files_exists
natural_keys = mu.natural_keys
elec_group_number = mu.elec_group_number
elec_group = mu.elec_group
run_command_in_new_thread = mu.run_command_in_new_thread
is_linux = mu.is_linux
is_windows = mu.is_windows
is_mac = mu.is_mac
read_floats_rx = mu.read_floats_rx
read_numbers_rx = mu.read_numbers_rx
timeit = mu.timeit
profileit = mu.profileit
get_time = mu.get_time
get_data_max_min = mu.get_data_max_min
get_max_abs = mu.get_max_abs
calc_min_max = mu.calc_min_max
csv_file_reader = mu.csv_file_reader
time_to_go = mu.time_to_go
tryit = mu.tryit
to_str = mu.to_str
read_config_ini = mu.read_config_ini
make_link = mu.make_link
both_hemi_files_exist = mu.both_hemi_files_exist
other_hemi = mu.other_hemi
check_hemi = mu.check_hemi
file_modification_time = mu.file_modification_time
min_stc = mu.min_stc
max_stc = mu.max_stc
apply_trans = mu.apply_trans
remove_file = mu.remove_file
get_distinct_colors = mu.get_distinct_colors
is_float = mu.is_float
get_fname_folder = mu.get_fname_folder
change_fname_extension = mu.change_fname_extension
copy_file = mu.copy_file

from src.mmvt_addon.scripts import scripts_utils as su
get_link_dir = su.get_link_dir
get_real_atlas_name = su.get_real_atlas_name
select_one_file = su.select_one_file
waits_for_file = su.waits_for_file

from src.utils import args_utils as au
is_int = au.is_int


try:
    import cPickle as pickle
except:
    import pickle
import uuid

PLY_HEADER = 'ply\nformat ascii 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nelement face {}\nproperty list uchar int vertex_index\nend_header\n'
STAT_AVG, STAT_DIFF = range(2)
HEMIS = ['lh', 'rh']


def get_exisiting_dir(dirs):
    ex_dirs = [d for d in dirs if op.isdir(d)]
    if len(ex_dirs)==0:
        raise Exception('No exisiting dir!')
    else:
        return ex_dirs[0]


def get_exisiting_file(dirs):
    ex_files = [d for d in dirs if op.isfile(d)]
    if len(ex_files)==0:
        raise Exception('No exisiting file!')
    else:
        return ex_files[0]


def delete_folder_files(fol):
    if op.isdir(fol):
        shutil.rmtree(fol)
    os.makedirs(fol)


def get_scalar_map(x_min, x_max, color_map='jet'):
    cm = plt.get_cmap(color_map)
    cNorm = matplotlib.colors.Normalize(vmin=x_min, vmax=x_max)
    return cmx.ScalarMappable(norm=cNorm, cmap=cm)


def arr_to_colors(x, x_min=None, x_max=None, colors_map='jet', scalar_map=None, norm_percs=(1, 99)):
    if scalar_map is None:
        x_min, x_max = calc_min_max(x, x_min, x_max, norm_percs)
        scalar_map = get_scalar_map(x_min, x_max, colors_map)
    return scalar_map.to_rgba(x)


def mat_to_colors(x, x_min=None, x_max=None, colorsMap='jet', scalar_map=None, flip_cm=False):
    if flip_cm:
        x = -x
        x_min = np.min(x) if x_max is None else -x_max
        x_max = np.max(x) if x_min is None else -x_min

    x_min, x_max = calc_min_max(x, x_min, x_max)
    colors = arr_to_colors(x, x_min, x_max, colorsMap, scalar_map)
    if colors.ndim == 2:
        return colors[:, :3]
    elif colors.ndim == 3:
        return colors[:, :, :3]
    raise Exception('colors ndim not 2 or 3!')


def arr_to_colors_two_colors_maps(x, x_min=None, x_max=None, cm_big='YlOrRd', cm_small='PuBu', threshold=0, default_val=0,
                                  scalar_map_big=None, scalar_map_small=None, flip_cm_big=False, flip_cm_small=False,
                                  norm_percs=(3, 97), norm_by_percentile=True):
    colors = np.ones((len(x), 3)) * default_val
    norm_percs = norm_percs if norm_by_percentile else None
    x_min, x_max = calc_min_max(x, x_min, x_max, norm_percs)

    if np.sum(x >= threshold) > 0:
        if not flip_cm_big:
            big_colors = arr_to_colors(x[x >= threshold], threshold, x_max, cm_big, scalar_map_big)[:, :3]
        else:
            big_colors = arr_to_colors(-x[x >= threshold], -x_max, -threshold, cm_big, scalar_map_big)[:, :3]
        colors[x >= threshold, :] = big_colors
    if np.sum(x <= -threshold) > 0:
        if not flip_cm_small:
            small_colors = arr_to_colors(x[x <= -threshold], x_min, -threshold, cm_small, scalar_map_small)[:, :3]
        else:
            small_colors = arr_to_colors(-x[x <= -threshold], threshold, -x_min, cm_small, scalar_map_small)[:, :3]
        colors[x<=-threshold, :] = small_colors
    return colors


def calc_abs_minmax(x, norm_percs=None):
    x_min, x_max = calc_min_max(x, norm_percs=norm_percs)
    return max(map(abs, [x_min, x_max]))


def calc_minmax_abs_from_minmax(data_min, data_max):
    minmax = max(map(abs, [data_min, data_max]))
    return -minmax, minmax


def calc_minmax_from_arr(hemi_minmax):
    data_min, data_max = min([x[0] for x in hemi_minmax]), max([x[1] for x in hemi_minmax])
    if np.sign(data_max) != np.sign(data_min) and data_min != 0:
        data_min, data_max = calc_minmax_abs_from_minmax(data_min, data_max)
    return data_min, data_max


def mat_to_colors_two_colors_maps(x, x_min=None, x_max=None, cm_big='YlOrRd', cm_small='PuBu', threshold=0, default_val=0,
        scalar_map_big=None, scalar_map_small=None, flip_cm_big=False, flip_cm_small=False, min_is_abs_max=False,
        norm_percs = None):
    colors = np.ones((x.shape[0],x.shape[1], 3)) * default_val
    x_min, x_max = calc_min_max(x, x_min, x_max, norm_percs)
    if min_is_abs_max:
        x_max = max(map(abs, [x_min, x_max]))
        x_min = -x_max
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


def read_ply_file(ply_file, npz_fname=''):
    if file_type(ply_file) == '':
        ply_file = '{}.ply'.format(ply_file)
    npz_file = change_fname_extension(ply_file, 'npz')
    if file_type(ply_file) == 'ply' and not op.isfile(npz_file):
        # print('Reading {}'.format(ply_file))
        with open(ply_file, 'r') as f:
            lines = f.readlines()
            verts_num = int(lines[2].split(' ')[-1])
            faces_num = int(lines[6].split(' ')[-1])
            verts_lines = lines[9:9 + verts_num]
            faces_lines = lines[9 + verts_num:]
            verts = np.array([list(map(float, l.strip().split(' '))) for l in verts_lines])
            faces = np.array([list(map(int, l.strip().split(' '))) for l in faces_lines])[:,1:]
    elif file_type(ply_file) == 'npz' or op.isfile(npz_file):
        # print('Reading {}'.format(npz_file))
        d = np.load(npz_file)
        verts, faces = d['verts'], d['faces']
        faces = faces.astype(np.int)
    # elif npz_fname != '' and op.isfile(npz_fname):
    #     d = np.load(npz_fname)
    #     verts, faces = d['verts'], d['faces']
    else:
        raise Exception("Can't find ply/npz file!")
    return verts, faces


def load_surf(subject, mmvt_dir, subjects_dir):
    verts = {}
    for hemi in HEMIS:
        if op.isfile(op.join(mmvt_dir, subject, 'surf', '{}.pial.npz'.format(hemi))):
            hemi_verts, _ = read_pial(subject, mmvt_dir, hemi)
        elif op.isfile(op.join(subjects_dir, subject, 'surf', '{}.pial.ply'.format(hemi))):
            hemi_verts, _ = read_ply_file(
                op.join(subjects_dir, subject, 'surf', '{}.pial.ply'.format(hemi)))
        elif op.isfile(op.join(subjects_dir, subject, 'surf', '{}.pial'.format(hemi))):
            import nibabel as nib
            hemi_verts, _ = nib.freesurfer.read_geometry(op.join(subjects_dir, subject, 'surf', '{}.pial'.format(hemi)))
        else:
            print("Can't find {} pial ply/npz files!".format(hemi))
            return False
        verts[hemi] = hemi_verts
    return verts


# def read_pial_npz(subject, mmvt_dir, hemi):
#     d = np.load(op.join(mmvt_dir, subject, 'surf', '{}.pial.npz'.format(hemi)))
#     return d['verts'], d['faces']


@functools.lru_cache(maxsize=None)
def read_pial(subject, mmvt_dir, hemi, surface_type='pial'):
    verts, faces = read_ply_file(op.join(mmvt_dir, subject, 'surf', '{}.{}.ply'.format(hemi, surface_type)))
    return verts, faces


def write_ply_file(verts, faces, ply_file_name, write_also_npz=False):
    try:
        verts_num = verts.shape[0]
        faces_num = faces.shape[0]
        faces = faces.astype(np.int)
        faces_for_ply = np.hstack((np.ones((faces_num, 1)) * faces.shape[1], faces))
        with open(ply_file_name, 'w') as f:
            f.write(PLY_HEADER.format(verts_num, faces_num))
        with open(ply_file_name, 'ab') as f:
            np.savetxt(f, verts, fmt='%.5f', delimiter=' ')
            np.savetxt(f, faces_for_ply, fmt='%d', delimiter=' ')
        if write_also_npz:
            np.savez('{}.npz'.format(op.splitext(ply_file_name)[0]), verts=verts, faces=faces)
        return True
    except:
        print('Error in write_ply_file! ({})'.format(ply_file_name))
        print(traceback.format_exc())
        return False


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


def convert_mat_files_to_ply(mat_folder, overwrite=True):
    mat_files = glob.glob(op.join(mat_folder, '*.mat'))
    for mat_file in mat_files:
        ply_file = '{}.ply'.format(mat_file[:-4])
        if overwrite or not op.isfile(ply_file):
            d = Bag(**sio.loadmat(mat_file))
            write_ply_file(d.verts, d.faces - 1, ply_file, True)
            # srf2ply(srf_file, ply_file)


def get_ply_vertices_num(ply_file_template):
    if op.isfile(ply_file_template.format('rh')) and op.isfile(ply_file_template.format('lh')):
        rh_vertices, _ = read_ply_file(ply_file_template.format('rh'))
        lh_vertices, _ = read_ply_file(ply_file_template.format('lh'))
        return {'rh':rh_vertices.shape[0], 'lh':lh_vertices.shape[0]}
    else:
        print('No surface ply files!')
        return None


def calc_ply_faces_verts(verts, faces, out_file, overwrite=False, ply_name='', errors={}, verbose=False):
    if not overwrite and op.isfile(out_file):
        if verbose:
            print('{} already exist.'.format(out_file))
    else:
        _faces = faces.ravel()
        if verbose:
            print('{}: verts: {}, faces: {}, faces ravel: {}'.format(
                ply_name, verts.shape[0], faces.shape[0], len(_faces)))
        faces_arg_sort = np.argsort(_faces)
        faces_sort = np.sort(_faces)
        faces_count = Counter(faces_sort)
        max_len = max([v for v in faces_count.values()])
        print(ply_name, verts.shape[0], max_len)
        lookup = np.ones((verts.shape[0], max_len)) * -1
        diff = np.diff(faces_sort)
        n = 0
        for ind, (k, v) in enumerate(zip(faces_sort, faces_arg_sort)):
            lookup[k, n] = v
            n = 0 if ind < len(diff) and diff[ind] > 0 else n+1
        np.save(out_file, lookup.astype(np.int))
        if verbose:
            print('{} max lookup val: {}'.format(ply_name, int(np.max(lookup))))
        if len(_faces) != int(np.max(lookup)) + 1:
            errors[ply_name] = 'Wrong values in lookup table! ' + \
                'faces ravel: {}, max looup val: {}'.format(len(_faces), int(np.max(lookup)))
    return errors


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


def read_freesurfer_lookup_table(get_colors=False, return_dict=False, reverse_dict=False, lut_fname=''):
    if lut_fname == '':
        lut_name = 'FreeSurferColorLUT.txt'
        lut_fname = op.join(mmvt_fol(), lut_name)
    if not op.isfile(lut_fname):
        resources_lut_fname = op.join(get_resources_fol(), lut_name)
        if op.isfile(resources_lut_fname):
            shutil.copy(resources_lut_fname, lut_fname)
        else:
            freesurfer_lut_fname = op.join(freesurfer_fol(), lut_name)
            if op.isfile(freesurfer_lut_fname):
                shutil.copy(freesurfer_lut_fname, lut_fname)
            else:
                print("Can't find FreeSurfer Color LUT!")
                return None
    if get_colors:
        lut = np.genfromtxt(lut_fname, dtype=None, usecols=(0, 1, 2, 3, 4, 5), names=['id', 'name', 'r', 'g', 'b', 'a'])
    else:
        lut = np.genfromtxt(lut_fname, dtype=None, usecols=(0, 1), names=['id', 'name'])
    if return_dict:
        if reverse_dict:
            lut = {name.decode(sys.getfilesystemencoding(), 'ignore'):int(val) for val, name in lut}
        else:
            lut = {int(val):name.decode(sys.getfilesystemencoding(), 'ignore') for val, name in lut}

    return lut


def mmvt_fol():
    return get_link_dir(get_links_dir(), 'mmvt')


def freesurfer_fol():
    return get_link_dir(get_links_dir(), 'freesurfer', 'FREESURFER_HOME')


def get_environ_dir(var_name, default_val=''):
    ret_val = os.environ.get(var_name) if default_val == '' else default_val
    if not op.isdir(ret_val):
        raise Exception('get_environ_dir: No existing dir!')
    return ret_val


# def get_link_dir(links_dir, link_name, var_name='', default_val='', throw_exception=False):
#     link = op.join(links_dir, link_name)
#     # check if this is a windows folder shortcup
#     if op.isfile('{}.lnk'.format(link)):
#         from src.mmvt_addon.scripts import windows_utils as wu
#         sc = wu.MSShortcut('{}.lnk'.format(link))
#         return op.join(sc.localBasePath, sc.commonPathSuffix)
#         # return read_windows_dir_shortcut('{}.lnk'.format(val))
#     ret = op.realpath(link)
#     if not op.isdir(ret) and default_val != '':
#         ret = default_val
#     if not op.isdir(ret):
#         ret = os.environ.get(var_name, '')
#     if not op.isdir(ret):
#         ret = get_link_dir_from_csv(links_dir, link_name)
#         if ret == '':
#             if throw_exception:
#                 raise Exception('No {} dir!'.format(link_name))
#             else:
#                 print('No {} dir!'.format(link_name))
#     return ret


# def get_link_dir(links_dir, link_name, var_name='', default_val='', throw_exception=False):
#     val = op.join(links_dir, link_name)
#     # check if this is a windows folder shortcup
#     if op.isfile('{}.lnk'.format(val)):
#         from src.mmvt_addon.scripts import windows_utils as wu
#         sc = wu.MSShortcut('{}.lnk'.format(val))
#         return op.join(sc.localBasePath, sc.commonPathSuffix)
#         # return read_windows_dir_shortcut('{}.lnk'.format(val))
#     if not op.isdir(val) and default_val != '':
#         val = default_val
#     if not op.isdir(val):
#         val = os.environ.get(var_name, '')
#     if not op.isdir(val):
#         val = get_link_dir_from_csv(links_dir, link_name)
#         if val == '':
#             if throw_exception:
#                 raise Exception('No {} dir!'.format(link_name))
#             else:
#                 print('No {} dir!'.format(link_name))
#     return val


def get_links_dir(links_fol_name='links'):
    parent_fol = get_parent_fol(levels=3)
    links_dir = op.join(parent_fol, links_fol_name)
    return links_dir


def get_electrodes_labeling(subject, blender_root, atlas, bipolar=False, error_radius=3, elec_length=4, other_fname=''):
    if other_fname == '':
        # We remove the 'all_rois' and 'stretch' for the name!
        electrode_labeling_fname = op.join(blender_root, subject, 'electrodes',
            '{}_{}_electrodes_cigar_r_{}_l_{}{}.pkl'.format(subject, atlas, error_radius, elec_length,
            '_bipolar' if bipolar else ''))
    else:
        electrode_labeling_fname = other_fname
    if op.isfile(electrode_labeling_fname):
        labeling = load(electrode_labeling_fname)
        return labeling, electrode_labeling_fname
    else:
        print("Can't find the electrodes' labeling file in {}!".format(electrode_labeling_fname))
        return None, None

# def read_sub_cortical_lookup_table(lookup_table_file_name):
#     names = {}
#     with open(lookup_table_file_name, 'r') as f:
#         for line in f.readlines():
#             lines = line.strip().split('\t')
#             if len(lines) > 1:
#                 name, code = lines[0].strip(), int(lines[1])
#                 names[code] = name
#     return names


def get_numeric_index_to_label(label, lut=None):
    if lut is None:
        lut = read_freesurfer_lookup_table()
    lut_names = np.array([l.decode() for l in lut['name']])
    if type(label) == str:
        inds = np.where(lut_names == label)
        if len(inds[0]) == 0:
            return None, None
        else:
            seg_id = lut['id'][inds[0]][0]
            seg_name = label
    elif type(label) == int:
        seg_id = label
        seg_name = lut_names[lut['id'] == seg_id][0]
    if not isinstance(seg_name, str):
        seg_name = seg_name.astype(str)
    return seg_name, int(seg_id)


def lut_labels_to_indices(regions, lut):
    sub_corticals = []
    for reg in regions:
        name, id = get_numeric_index_to_label(reg, lut)
        sub_corticals.append(id)
    return sub_corticals


def how_many_curlies(str):
    return len(re.findall('\{*\}', str))


def run_script(cmd, verbose=False, cwd=None):
    try:
        if verbose:
            print('running: {}'.format(cmd))
        if is_windows():
            output = subprocess.call(cmd, cwd=cwd)
        else:
            # cmd = cmd.replace('\\\\', '')
            # output = subprocess.call(cmd)
            # output = subprocess.check_output(cmd, shell=True)
            output = subprocess.check_output('{} | tee /dev/stderr'.format(cmd), shell=True, cwd=cwd)
    except:
        print('Error in run_script!')
        print(traceback.format_exc())
        return ''

    if isinstance(output, str):
        output = output.decode(sys.getfilesystemencoding(), 'ignore')
    if verbose:
        print(output)
    return output


# def partial_run_script(vars, more_vars=None):
#     return partial(lambda cmd,v:run_script(cmd.format(**v)), v=vars)

def partial_run_script(vars, print_only=False, cwd=None):
    return partial(_run_script_wrapper, vars=vars, cwd=cwd, print_only=print_only)


def _run_script_wrapper(cmd, vars, cwd=None, print_only=False, **kwargs):
    for k,v in kwargs.items():
        vars[k] = v
    print(cmd.format(**vars))
    if not print_only:
        run_script(cmd.format(**vars), cwd=cwd)


def sub_cortical_voxels_generator(aseg, seg_labels, spacing=5, use_grid=True):
    # Read the segmentation data using nibabel
    aseg_data = aseg.get_data()

    # Read the freesurfer lookup table
    lut = read_freesurfer_lookup_table()

    # Generate a grid using spacing
    grid = None
    if use_grid:
        grid = generate_grid_using_spacing(spacing, aseg_data.shape)

    # Get the indices to the desired labels
    for label in seg_labels:
        seg_name, seg_id = get_numeric_index_to_label(label, lut)
        if seg_name is None:
            continue
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
    aseg_fname = op.join(subject_mri_dir, 'mri', 'aseg.mgz')
    aseg = nib.load(aseg_fname)
    aseg_hdr = aseg.get_header()
    return aseg_hdr


def namebase_sep(fname):
    name_with_ext = fname.split(op.sep)[-1]
    if not name_with_ext.endswith('nii.gz'):
        return '.'.join(name_with_ext.split('.')[:-1])
    else:
        return name_with_ext[:-len('nii.gz')]


def namebase(fname):
    if 'nii.gz' not in fname:
        return op.splitext(op.basename(fname))[0]
    else:
        nb = fname
        while '.' in nb:
            nb = op.splitext(op.basename(nb))[0]
        return nb


def file_type_sep(fname):
    if fname.endswith('nii.gz'):
        return 'nii.gz'
    else:
        return fname.split('.')[-1]


def file_type(fname):
    if 'nii.gz' in fname:
        return 'nii.gz'
    else:
        return op.splitext(op.basename(fname))[1][1:]
    # ret = '.'.join(fname.split(op.sep)[-1].split('.')[1:])
    # return ret


def is_file_type(fname, file_type):
    return fname[-len(file_type):] == file_type


def namebase_with_ext(fname):
    return fname.split(op.sep)[-1]


#todo: Move to labes utils
def read_labels_from_annot(subject, aparc_name, subjects_dir):
    labels = []
    annot_fname_temp = op.join(subjects_dir, subject, 'label', '{}.{}.annot'.format('{hemi}', aparc_name))
    for hemi in HEMIS:
        if op.isfile(annot_fname_temp.format(hemi=hemi)):
            labels_hemi = mne.read_labels_from_annot(subject, aparc_name)
            labels.extend(labels_hemi)
        else:
            print("Can't find the annotation file! {}".format(annot_fname_temp.format(hemi=hemi)))
            return []
    return labels


def rmtree(fol):
    if op.isdir(fol):
        shutil.rmtree(fol)

# def make_dir(fol):
#     if not op.isdir(fol):
#         os.makedirs(fol)
#     return fol


def get_subfolders(fol):
    return [op.join(fol,subfol) for subfol in os.listdir(fol) if op.isdir(op.join(fol,subfol))]


def get_spaced_colors(n):
    if n <= 7:
        colors = ['r', 'g', 'c', 'm', 'y', 'b', 'k'][:n]
    else:
        HSV_tuples = [(x*1.0/n, 0.5, 0.5) for x in range(n)]
        colors = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return colors


def downsample(x, R):
    if x.ndim == 1:
        return x.reshape(-1, R).mean(1)
    elif x.ndim == 2:
        return downsample_2d(x, R)
    else:
        raise Exception('Currently supports only matrices with up to 2 dims!')


def downsample_2d(x, R):
    return x.reshape(x.shape[0],-1,R).mean(2)


def downsample_3d(x, R):
    return x.reshape(x.shape[0],x.shape[1],-1, R).mean(3)


def read_sub_corticals_code_file(sub_corticals_codes_file, read_also_names=False):
    if op.isfile(sub_corticals_codes_file):
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


def dic2bunch(dic):
    from sklearn.datasets.base import Bunch
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


def build_remote_subject_dir(remote_subject_dir_template, subject):
    if remote_subject_dir_template != '':
        # remote_subject_dir_template = op.join(remote_subject_dir_template, subject)
        if '{subject}' in remote_subject_dir_template:
            if isinstance(remote_subject_dir_template, dict):
                if 'func' in remote_subject_dir_template:
                    template_val = remote_subject_dir_template['func'](subject)
                    remote_subject_dir = remote_subject_dir_template['template'].format(subject=template_val)
                else:
                    remote_subject_dir = remote_subject_dir_template['template'].format(subject=subject)
            else:
                remote_subject_dir = remote_subject_dir_template.format(subject=subject)
        else:
            remote_subject_dir = remote_subject_dir_template
    else:
        remote_subject_dir = ''
    return remote_subject_dir


def prepare_subject_folder(necessary_files, subject, remote_subject_dir, local_subjects_dir,
        sftp=False, sftp_username='', sftp_domain='', sftp_password='',
        overwrite_files=False, print_traceback=True, sftp_port=22, local_subject_dir=''):
    if local_subject_dir == '':
        local_subject_dir = op.join(local_subjects_dir, subject)
    mmvt_dir = get_link_dir(get_links_dir(), 'mmvt')
    if op.isdir(op.join(mmvt_dir, subject)):
        save(dict(remote_subject_dir=remote_subject_dir, sftp=sftp, sftp_username=sftp_username,
                  sftp_domain=sftp_domain, sftp_password=sftp_password, sftp_port=sftp_port),
             op.join(mmvt_dir, subject, 'remote_subject_info.pkl'))
    all_files_exists = False if overwrite_files else \
        check_if_all_necessary_files_exist(subject, necessary_files, local_subject_dir, trace=remote_subject_dir == '')
    if all_files_exists and not overwrite_files:
        return True
    elif remote_subject_dir == '':
        print('Not all the necessary files exist, and the remote_subject_dir was not set!')
        return False
    if sftp:
        sftp_copy_subject_files(subject, necessary_files, sftp_username, sftp_domain,
                                local_subjects_dir, remote_subject_dir, sftp_password,
                                overwrite_files, print_traceback, sftp_port)
    else:
        for fol, files in necessary_files.items():
            fol = fol.replace(':', op.sep)
            if not op.isdir(op.join(local_subject_dir, fol)):
                os.makedirs(op.join(local_subject_dir, fol))
            for file_name in files:
                try:
                    file_name = file_name.replace('{subject}', subject)
                    local_fname = op.join(local_subject_dir, fol, file_name)
                    remote_fname = op.join(remote_subject_dir, fol, file_name)
                    local_files = glob.glob(local_fname)
                    if len(local_files) == 0 or overwrite_files:
                        remote_files = glob.glob(remote_fname)
                        if len(remote_files) > 0:
                            remote_fname = remote_files[0]
                            remote_lower = namebase_with_ext(remote_fname).lower()
                            if subject.lower() in remote_lower and subject not in namebase(remote_fname):
                                ind = remote_lower.index(subject)
                                new_file_name = remote_lower[:ind] + subject + remote_lower[len(subject):]
                                local_fname = op.join(local_subject_dir, fol, new_file_name)
                            else:
                                local_fname = op.join(local_subject_dir, fol, namebase_with_ext(remote_fname))
                            print('coping {} to {}'.format(remote_fname, local_fname))
                            shutil.copyfile(remote_fname, local_fname)
                        else:
                            print("Remote file can't be found! {}".format(remote_fname))
                except:
                    if print_traceback:
                        print(traceback.format_exc())
    all_files_exists = check_if_all_necessary_files_exist(subject, necessary_files, local_subject_dir, True)
    return all_files_exists


def check_if_all_necessary_files_exist(subject, necessary_files, local_subject_dir, trace=True):
    all_files_exists = True
    for fol, files in necessary_files.items():
        fol = fol.replace(':', op.sep)
        # if fol == 'label':
        #     continue
        for file_name in files:
            file_name = file_name.replace('{subject}', subject)
            files = glob.glob(op.join(local_subject_dir, fol, file_name))
            if len(files) == 0:
                if trace:
                    print("{}: the file {} doesn't exist in the local subjects folder!!!".format(subject, file_name))
                all_files_exists = False
    return all_files_exists


def sftp_copy_subject_files(subject, necessary_files, username, domain, local_subjects_dir, remote_subject_dir,
                            password='', overwrite_files=False, print_traceback=True, port=22):
    import pysftp
    local_subject_dir = op.join(local_subjects_dir, subject)
    if password == '':
        password = ask_for_sftp_password(username)
    try:
        cnopts = pysftp.CnOpts()
        cnopts.hostkeys = None
        sftp_con = pysftp.Connection(domain, username=username, password=password, cnopts=cnopts, port=port)
    except:
        try:
            sftp_con = pysftp.Connection(domain, username=username, password=password, port=port)
        except:
            print("Can't connect via sftp!")
            if print_traceback:
                print(traceback.format_exc())
            return False
    with sftp_con as sftp:
        for fol, files in necessary_files.items():
            fol = fol.replace(':', op.sep)
            if not op.isdir(op.join(local_subject_dir, fol)):
                os.makedirs(op.join(local_subject_dir, fol))
            os.chdir(op.join(local_subject_dir, fol))
            for file_name in files:
                try:
                    file_name = file_name.replace('{subject}', subject)
                    local_fname = op.join(local_subject_dir, fol, file_name)
                    if not op.isfile(local_fname) or overwrite_files:
                        # with sftp.cd(op.join(remote_subject_dir, fol)):
                        try:
                            with sftp.cd(remote_subject_dir + '/' + fol):
                                print('sftp: getting {}'.format(file_name))
                                sftp.get(file_name)
                        except FileNotFoundError:
                            print('The file {} does not exist on the remote server!'.format(file_name))

                    if op.isfile(local_fname) and op.getsize(local_fname) == 0:
                        os.remove(local_fname)
                except:
                    if print_traceback:
                        print(traceback.format_exc())


def ask_for_sftp_password(username):
    return getpass.getpass('Please enter the sftp password for "{}": '.format(username))


def to_ras(points, round_coo=False):
    RAS_AFF = np.array([[-1, 0, 0, 128],
        [0, 0, -1, 128],
        [0, 1, 0, 128],
        [0, 0, 0, 1]])
    ras = [np.dot(RAS_AFF, np.append(p, 1))[:3] for p in points]
    if round_coo:
        ras = np.array([np.around(p) for p in ras], dtype=np.int16)
    return np.array(ras)


def check_for_necessary_files(necessary_files, root_fol):
    for fol, files in necessary_files.items():
        for file in files:
            full_path = op.join(root_fol, fol, file)
            if not op.isfile(full_path):
                raise Exception('{} does not exist!'.format(full_path))


def run_parallel(func, params, njobs=1):
    if njobs == 1:
        results = [func(p) for p in params]
    else:
        pool = multiprocessing.Pool(processes=njobs)
        results = pool.map(func, params)
        pool.close()
    return results


def get_current_fol():
    return op.dirname(op.realpath(__file__))


def get_parent_fol(curr_dir='', levels=1):
    if curr_dir == '':
        curr_dir = get_current_fol()
    parent_fol = op.split(curr_dir)[0]
    for _ in range(levels - 1):
        parent_fol = get_parent_fol(parent_fol)
    return parent_fol


def get_resources_fol():
    return op.join(get_parent_fol(levels=2), 'resources')


def get_figs_fol():
    return op.join(get_parent_fol(), 'figs')


def get_files_fol():
    return op.join(get_parent_fol(), 'pkls')


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


def plot_3d_PCA(X, names=None, colors=[], legend_labels=[], n_components=3):
    X_PCs = calc_PCA(X, n_components)
    plot_3d_scatter(X_PCs, names, colors=colors, legend_labels=legend_labels)


def calc_PCA(X, n_components=3):
    from sklearn import decomposition
    X = (X - np.mean(X, 0)) / np.std(X, 0) # You need to normalize your data first
    pca = decomposition.PCA(n_components=n_components)
    X = pca.fit(X).transform(X)
    print ('explained variance (first %d components): %.2f'%(n_components, sum(pca.explained_variance_ratio_)))
    return X


def gradient_scatter3d(X, colors_data, colorsMap='hot', do_show=True):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.cm as cmx
    cm = plt.get_cmap(colorsMap)
    cs = [colors_data[x, y, z] for x, y, z in zip(X[:, 0], X[:, 1], X[:, 2])]
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=scalarMap.to_rgba(cs))
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap)
    if do_show:
        plt.show()


def plot_3d_scatter(X, names=None, labels=None, classifier=None, labels_indices=[], colors=None, legend_labels=[],
                    title='', fname='', do_show=True):
    from mpl_toolkits.mplot3d import Axes3D, proj3d
    fig = plt.figure()
    ax = Axes3D(fig)
    if len(legend_labels) > 0:
        legend_labels = np.array(legend_labels)
        unique_labels = np.unique(legend_labels)
        for unique_label in unique_labels:
            inds = np.where(legend_labels == unique_label)[0]
            ax.scatter(X[inds, 0], X[inds, 1], X[inds, 2], c=[colors[ind] for ind in inds], label=unique_label)
    else:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors)

    if not names is None:
        if not labels is None:
            for label in labels:
                ind = names.index(label)
                add_annotation(ax, label, X[ind, 0], X[ind, 1], X[ind, 2])
        else:
            if len(labels_indices) > 0:
                for name, ind in zip(names, labels_indices):
                    add_annotation(ax, name, X[ind, 0], X[ind, 1], X[ind, 2])
            else:
                for x,y,z,name in zip(X[:, 0], X[:, 1], X[:, 2], names):
                    add_annotation(ax, name, x, y, z)

    if not classifier is None:
        make_ellipses(classifier, ax)

    if legend_labels is not None:
        plt.legend()

    if title != '':
        plt.title(title)

    if fname == '':
        if do_show:
            plt.show()
    else:
        plt.savefig(fname)
        plt.close()


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
        v, w = np.linalg.eigh(gmm.covariances_[n][:2, :2])
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
    return str(uuid.uuid4()).replace('-','')[:num]


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


def flat_list(lst):
    return list(itertools.chain.from_iterable(lst))


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


# def elec_group_number(elec_name, bipolar=False):
#     if isinstance(elec_name, bytes):
#         elec_name = elec_name.decode('utf-8')
#     if bipolar:
#         elec_name2, elec_name1 = elec_name.split('-')
#         group, num1 = elec_group_number(elec_name1, False)
#         _, num2 = elec_group_number(elec_name2, False)
#         return group, num1, num2
#     else:
#         elec_name = elec_name.strip()
#         num = int(re.sub('\D', ',', elec_name).split(',')[-1])
#         group = elec_name[:elec_name.rfind(str(num))]
#         return group, num


# def elec_group(elec_name, bipolar):
#     if bipolar:
#         group, _, _ = elec_group_number(elec_name, bipolar)
#     else:
#         group, _ = elec_group_number(elec_name, bipolar)
#     return group


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


# def time_to_go(now, run, runs_num, runs_num_to_print=10):
#     if run % runs_num_to_print == 0 and run != 0:
#         time_took = time.time() - now
#         more_time = time_took / run * (runs_num - run)
#         print('{}/{}, {:.2f}s, {:.2f}s to go!'.format(run, runs_num, time_took, more_time))


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
    return op.isfile(fpath) and os.access(fpath, os.X_OK)


def set_exe_permissions(fpath):
    os.chmod(fpath, 0o744)


def csv_from_excel(xlsx_fname, csv_fname, sheet_name=''):
    import xlrd
    import csv
    wb = xlrd.open_workbook(xlsx_fname)
    sheet_num = 0
    if len(wb.sheets()) > 1 and sheet_name == '':
        print('More than one sheet in the xlsx file:')
        for ind, sh in enumerate(wb.sheets()):
            print('{}) {}'.format(ind + 1, sh.name))
        sheet_num = input('Which one do you want to load (1, 2, ...)? ')
        while not is_int(sheet_num):
            print('Please enter a valid integer')
            sheet_num = input('Which one do you want to load (1, 2, ...)? ')
        sheet_num = int(sheet_num) - 1
    if sheet_name != '':
        sh = wb.sheet_by_name(sheet_name)
    else:
        sh = wb.sheets()[sheet_num]
    print('Converting sheet "{}" to csv'.format(sh.name))
    with open(csv_fname, 'w') as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        for rownum in range(sh.nrows):
            wr.writerow([val for val in sh.row_values(rownum)])
            # csv_file.write(b','.join([str(val).encode('utf_8') for val in sh.row_values(rownum)]) + b'\n')


def get_all_subjects(subjects_dir, prefix, exclude_substr):
    subjects = []
    folders = [namebase(fol) for fol in get_subfolders(subjects_dir)]
    for subject_fol in folders:
        if subject_fol[:len(prefix)].lower() == prefix and exclude_substr not in subject_fol:
            subjects.append(subject_fol)
    return subjects


# def read_labels(labels_fol, hemi='both'):
#     hemis = [hemi] if hemi != 'both' else HEMIS
#     labels = []
#     for hemi in hemis:
#         for label_file in glob.glob(op.join(labels_fol, '*{}.label'.format(hemi))):
#             print('read label from {}'.format(label_file))
#             label = mne.read_label(label_file)
#             labels.append(label)
#     return labels


# def read_labels_parallel(subject, subjects_dir, atlas, n_jobs):
#     labels_files = glob.glob(op.join(subjects_dir, subject, 'label', atlas, '*.label'))
#     files_chunks = chunks(labels_files, len(labels_files) / n_jobs)
#     results = run_parallel(_read_labels_parallel, files_chunks, n_jobs)
#     labels = []
#     for labels_chunk in results:
#         labels.extend(labels_chunk)
#     return labels


# def _read_labels_parallel(files_chunk):
#     labels = []
#     for label_fname in files_chunk:
#         label = mne.read_label(label_fname)
#         labels.append(label)
#     return labels


def merge_two_dics(dic1, dic2):
    # Only for python >= 3.5
    # return {**dic1, **dic2}
    ret = dic1.copy()
    ret.update(dic2)
    return ret


# def color_name_to_rgb(color_name):
#     try:
#         import webcolors
#         return webcolors.name_to_rgb(color_name)
#     except:
#         print('No webcolors!')
#         return None
#
#
# def color_name_to_rgb(rgb):
#     try:
#         import webcolors
#         return webcolors.rgb_to_name(rgb)
#     except:
#         print('No webcolors!')
#         return None


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
    if moving_average_win_size > 0:
        return evoked_smooth
    else:
        return evoked


def get_hemi_indifferent_roi(roi):
    return roi.replace('-rh', '').replace('-lh', '').replace('rh-', '').replace('lh-', '').\
        replace('.rh', '').replace('.lh', '').replace('rh.', '').replace('lh.', '').\
        replace('Right-', '').replace('Left-', '').replace('-Right', '').replace('-Left', '').\
        replace('Right.', '').replace('Left.', '').replace('.Right', '').replace('.Left', '').\
        replace('right-', '').replace('left-', '').replace('-right', '').replace('-left', '').\
        replace('right.', '').replace('left.', '').replace('.right', '').replace('.left', '')


def get_hemi_indifferent_rois(rois):
    return set(map(lambda roi:get_hemi_indifferent_roi(roi), rois))


def show_image(image_fname):
    image = mpimg.imread(image_fname)
    plt.axis("off")
    plt.imshow(image)
    plt.tight_layout()
    plt.show()


def get_n_jobs(n_jobs):
    cpu_num = multiprocessing.cpu_count()
    n_jobs = int(n_jobs)
    if n_jobs > cpu_num:
        n_jobs = cpu_num
    elif n_jobs < 0:
        n_jobs = cpu_num + n_jobs
    if n_jobs < 1:
        n_jobs == 1
    return n_jobs


def read_mat_file_into_bag(mat_fname):
    try:
        import scipy.io as sio
        x = sio.loadmat(mat_fname)
        return Bag(**x)
    except NotImplementedError:
        import tables
        from src.utils import tables_utils as tu
        x = tables.openFile(mat_fname)
        ret = Bag(**tu.read_tables_into_dict(x))
        x.close()
        return ret
    return None


def get_fol_if_exist(fols):
    for fol in fols:
        if op.isdir(fol):
            return fol
    return None


def get_file_if_exist(files):
    for fname in files:
        if op.isfile(fname):
            return fname
    return None


def rename_files(source_fnames, dest_fname):
    for source_fname in source_fnames:
        if op.isfile(source_fname):
            os.rename(source_fname, dest_fname)
            break


def vstack(arr1, arr2):
    arr1_np = np.array(arr1)
    arr2_np = np.array(arr2)
    if len(arr1) == 0 and len(arr2) == 0:
        return np.array([])
    elif len(arr1) == 0:
        return arr2_np
    elif len(arr2) == 0:
        return arr1_np
    else:
        return np.vstack((arr1_np, arr2_np))


def should_run(args, func_name):
    if 'exclude' not in args:
        args.exclude = []
    return ('all' in args.function or func_name in args.function) and func_name not in args.exclude


def trim_to_same_size(x1, x2):
    if len(x1) < len(x2):
        return x1, x2[:len(x1)]
    else:
        return x1[:len(x2)], x2


def sort_according_to_another_list(list_to_sort, list_to_sort_by):
    list_to_sort.sort(key=lambda x: list_to_sort_by.index(x.name))
    return list_to_sort


def get_sftp_password(subjects, subjects_dir, necessary_files, sftp_username, overwrite_fs_files=False):
    sftp_password = ''
    all_necessary_files_exist = False if overwrite_fs_files else np.all(
        [check_if_all_necessary_files_exist(subject, necessary_files, op.join(subjects_dir, subject), False)
         for subject in subjects])
    if not all_necessary_files_exist or overwrite_fs_files:
        sftp_password = ask_for_sftp_password(sftp_username)
    return sftp_password


def create_folder_link(real_fol, link_fol):
    if not is_link(link_fol):
        if is_windows():
            try:
                if not op.isdir(real_fol):
                    print('The target is not a directory!!')
                    return

                import winshell
                from win32com.client import Dispatch
                path = '{}.lnk'.format(link_fol)
                shell = Dispatch('WScript.Shell')
                shortcut = shell.CreateShortCut(path)
                shortcut.Targetpath = real_fol
                shortcut.save()
            except:
                print("Can't create a link to the folder {}!".format(real_fol))
        else:
            os.symlink(real_fol, link_fol)


def is_link(link_path):
    if is_windows():
        try:
            from src.mmvt_addon.scripts import windows_utils as wu
            sc = wu.MSShortcut('{}.lnk'.format(link_path))
            real_folder_path = op.join(sc.localBasePath, sc.commonPathSuffix)
            return op.isdir(real_folder_path)
        except:
            return False
    else:
        return op.islink(link_path)


def message_box(text, title=''):
    if is_windows():
        import ctypes
        return ctypes.windll.user32.MessageBoxW(0, text, title, 1)
    else:
        # print(text)
        from tkinter import Tk, Label
        root = Tk()
        w = Label(root, text=text)
        w.pack()
        root.mainloop()
        return 1


def choose_folder_gui():
    from tkinter.filedialog import askdirectory
    fol = askdirectory()
    if is_windows():
        fol = fol.replace('/', '\\')
    return fol


def list_flatten(l):
    return [item for sublist in l for item in sublist]


def all(arr):
    return list(set(arr))[0] == True


def ceil_floor(x):
    import math
    return math.ceil(x) if x > 0 else math.floor(x)


def add_str_to_file_name(fname, txt, suf=''):
    if suf == '':
        suf = file_type(fname)
    return op.join(get_parent_fol(fname), '{}{}.{}'.format(namebase(fname), txt, suf))


def locating_file(default_fname, glob_pattern, parent_fol, raise_exception=False):
    if op.isfile(default_fname):
        return default_fname, True
    if isinstance(glob_pattern, str):
        glob_pattern = [glob_pattern]
    glob_pattern_print = ','.join(glob_pattern)
    fname = op.join(parent_fol, default_fname)
    if '{cond}' in fname:
        exist = len(glob.glob(fname.replace('{cond}', '*'))) > 1
    else:
        exist = op.isfile(fname)
    if not exist:
        lists = [glob.glob(op.join(parent_fol, '**', gb), recursive=True) for gb in glob_pattern]
        files = list(itertools.chain.from_iterable(lists))
        exist = len(files) > 0
        if exist:
            if len(files) == 1:
                fname = files[0]
            else:
                for ind, fname in enumerate(files):
                    print('{}) {}'.format(ind+1, fname))
                ind = int(input('There are more than one {} files. Please choose the one you want to use: '.format(
                    glob_pattern_print)))
                if ind == 0 or ind > len(files):
                    return '', False
                fname_input = files[ind-1]
                if op.isfile(op.join(parent_fol, fname_input)):
                    fname = op.join(parent_fol, fname_input)
                else:
                    print("Couldn't find {}!".format(op.join(parent_fol, fname_input)))
    if not exist and raise_exception:
        raise Exception("locating_file: Couldn't find the file ({})".format(op.join(parent_fol, glob_pattern_print)))
    return fname, exist


def remove_link(source):
    try:
        os.unlink(source)
    except:
        pass


def read_list_from_file(fname, line_func=None, input_format='r'):
    import string
    printable = set(string.printable)
    arr = []
    with open(fname, input_format) as f:
        for line in f.readlines():
            line = line.strip()
            if not isinstance(line, str):
                line = line.decode(sys.getfilesystemencoding(), 'ignore')
                line = ''.join(list(filter(lambda x: x in printable, line)))
            if line.startswith('#'):
                continue
            if line != '':
                if line_func is not None:
                    line = line_func(line)
                arr.append(line)
    return arr


def write_list_to_file(list, fname):
    with open(fname, 'w') as f:
        for val in list:
            f.write('{}\n'.format(val))


def look_for_one_file(template, files_desc, pick_the_first_one=False, search_func=None):
    files = search_func(template) if search_func is not None else glob.glob(template)
    if len(files) == 0:
        print('No {} files were found in {}!'.format(files_desc, template))
        return None
    elif len(files) > 1:
        if pick_the_first_one:
            fname = files[0]
        else:
            fname = select_one_file(files, template, files_desc)
    else:
        fname = files[0]
    return fname


def get_logs_fol():
    logs_fol = op.join(get_parent_fol(__file__, 3), 'logs')
    make_dir(logs_fol)
    return logs_fol


def merge_text_files(input_files, output_fname):
    with open(output_fname, 'w') as outfile:
        for fname in input_files:
            with open(fname) as infile:
                outfile.write(infile.read())


def find_num_in_str(string):
    # return re.sub('\D', ',', string).replace(',', '')
    return ' '.join(re.sub('\D', ' ', string).split()).split()


def file_modification_time(fname):
    return time.strftime('%H:%M:%S %m/%d/%Y', time.gmtime(op.getmtime(fname)))


def file_is_newer(fname1, fname2):
    return time.gmtime(op.getmtime(fname1)) > time.gmtime(op.getmtime(fname2))


@decorator
def ignore_warnings(f, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        retval = f(*args, **kw)
    return retval


def check_for_freesurfer(func):
    def wrapper(*args, **kwargs):
        if os.environ.get('FREESURFER_HOME', '') == '':
            raise Exception('Source freesurfer and rerun')
        retval = func(*args, **kwargs)
        return retval
    return wrapper


def files_needed(necessary_files):
    def real_files_needed(func):
        def wrapper(*args, **kwargs):
            subjects_dir = get_link_dir(get_links_dir(), 'subjects', 'SUBJECTS_DIR')
            subject = kwargs.get('subject', args[0])
            remote_subject_dir = kwargs.get('remote_subject_dir', '')
            default_mmvt_args = Bag(
                sftp=False, sftp_username='', sftp_domain='', sftp_password='',
                overwrite_fs_files=False, print_traceback=False, sftp_port=22)
            mmvt_args = kwargs.get('mmvt_args', default_mmvt_args)
            if len(mmvt_args) == 0:
                mmvt_args = default_mmvt_args
            if remote_subject_dir == '':
                remote_subject_dir = mmvt_args.get('remote_subject_dir', '')
            ret = prepare_subject_folder(
                necessary_files, subject, remote_subject_dir, subjects_dir,
                mmvt_args.sftp, mmvt_args.sftp_username, mmvt_args.sftp_domain, mmvt_args.sftp_password,
                mmvt_args.overwrite_fs_files, mmvt_args.print_traceback, mmvt_args.sftp_port)
            if ret:
                if 'mmvt_args' in kwargs:
                    del kwargs['mmvt_args']
                retval = func(*args, **kwargs)
                return retval
            else:
                return False
        return wrapper
    return real_files_needed


def pca(x, comps_num=1):
    import sklearn.decomposition as deco

    remove_cols = np.where(np.all(x == np.mean(x, 0), 0))[0]
    x = np.delete(x, remove_cols, 1)
    x = (x - np.mean(x, 0)) / np.std(x, 0)
    pca = deco.PCA(comps_num)
    x = x.T
    x_r = pca.fit(x).transform(x)
    return x_r


def all_items_equall(arr):
    return all([x == arr[0] for x in arr])


def remove_mean_columnwise(x, lines=None):
    if lines is None:
        return  x - np.tile(np.mean(x, 0), (x.shape[0], 1))
    else:
        return x - np.tile(np.mean(x[lines], 0), (x.shape[0], 1))


def argmax2d(data):
    return np.unravel_index(np.argmax(data), data.shape)


def indices_of_elements(arr, values):
    return np.in1d(arr, values).nonzero()[0]


def is_locked(fname):
    if not op.isfile(fname):
        print('The file {} does not exist!'.format(fname))
        return None
    locked = True
    try:
        buffer_size = 8
        # Opening file in append mode and read the first 8 characters.
        file_object = open(fname, 'a', buffer_size)
        if file_object:
            locked = False
    except IOError:
        locked = True
    return locked


def non_nan_data(x):
    return x[np.where(~np.isnan(x))]


def get_common_letters(str_list):
    return ''.join([x[0] for x in zip(*str_list) if reduce(lambda a, b:(a == b) and a or None,x)])


def find_common_start(str_list):
    str_list = str_list[:]
    prev = None
    while True:
        common = get_common_letters(str_list)
        if common == prev:
            break
            str_list.append(common)
        prev = common
    return get_common_letters(str_list)


def print_last_error_line():
    try:
        print([l for l in traceback.format_exc().split('\n') if len(l) > 0][-1])
    except:
        pass

def wrapped_partial(func, *args, **kwargs):
    from functools import update_wrapper
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def calc_bands_power(x, dt, bands):
    from scipy.signal import welch
    f, psd = welch(x, fs=1. / dt)
    power = {band: np.mean(psd[np.where((f >= lf) & (f <= hf))]) for band, (lf, hf) in bands}
    return power


if __name__ == '__main__':
    print(file_modification_time('/home/npeled/code/procfast_indi'))