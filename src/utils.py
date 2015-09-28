import os
import shutil
import numpy as np
import scipy
import re
import matplotlib.pyplot as plt
import matplotlib
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

PLY_HEADER = 'ply\nformat ascii 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nelement face {}\nproperty list uchar int vertex_index\nend_header\n'


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

    if sum(x >= threshold) > 0:
        if not flip_cm_big:
            big_colors = arr_to_colors(x[x>=threshold], threshold, x_max, cm_big, scalar_map_big)[:, :3]
        else:
            big_colors = arr_to_colors(-x[x>=threshold], -x_max, -threshold, cm_big, scalar_map_big)[:, :3]
        colors[x>=threshold, :] = big_colors
    if sum(x <= -threshold) > 0:
        if not flip_cm_small:
            small_colors = arr_to_colors(x[x<=-threshold], x_min, -threshold, cm_small, scalar_map_small)[:, :3]
        else:
            small_colors = arr_to_colors(-x[x<=-threshold], threshold, -x_min, cm_small, scalar_map_small)[:, :3]
        colors[x<=-threshold, :] = small_colors
    return colors


def mat_to_colors_two_colors_maps(x, cm1='YlOrRd', cm2='PuBu', threshold=0, default_val=0):
    colors = np.ones((x.shape[0],x.shape[1], 3)) * default_val
    x_max, x_min = np.max(x), np.min(x)
    scalar_map_pos = get_scalar_map(threshold, x_max, cm1)
    scalar_map_neg = get_scalar_map(x_min, -threshold, cm2)
    for ind in range(x.shape[0]):
        colors[ind] = arr_to_colors_two_colors_maps(x[ind], x_min, x_max, cm1, cm2, threshold,
            default_val, scalar_map_pos, scalar_map_neg)
    return colors


def read_srf_file(srf_file):
    with open(srf_file, 'r') as f:
        lines = f.readlines()
        verts_num, faces_num = map(int, lines[1].strip().split(' '))
        sep = '  ' if len(lines[2].split('  ')) > 1 else ' '
        verts = np.array([map(float, l.strip().split(sep)) for l in lines[2:verts_num+2]])[:,:-1]
        faces = np.array([map(int, l.strip().split(' ')) for l in lines[verts_num+2:]])[:,:-1]
    return verts, faces, verts_num, faces_num


def read_ply_file(ply_file):
    with open(ply_file, 'r') as f:
        lines = f.readlines()
        verts_num = int(lines[2].split(' ')[-1])
        faces_num = int(lines[6].split(' ')[-1])
        verts_lines = lines[9:9 + verts_num]
        faces_lines = lines[9 + verts_num:]
        verts = np.array([map(float, l.strip().split(' ')) for l in verts_lines])
        faces = np.array([map(int, l.strip().split(' ')) for l in faces_lines])[:,1:]
    return verts, faces


def write_ply_file(verts, faces, ply_file_name):
    verts_num = verts.shape[0]
    faces_num = faces.shape[0]
    faces = np.hstack((np.ones((faces_num, 1)) * 3, faces))
    with open(ply_file_name, 'w') as f:
        f.write(PLY_HEADER.format(verts_num, faces_num))
        np.savetxt(f, verts, '%.5f', ' ')
        np.savetxt(f, faces, '%d', ' ')


def read_obj_file(obj_file):
    with open(obj_file, 'r') as f:
        lines = f.readlines()
        verts = np.array([[float(v) for v in l.strip().split(' ')[1:]] for l in lines if l[0]=='v'])
        faces = np.array([[int(v) for v in l.strip().split(' ')[1:]] for l in lines if l[0]=='f'])
    faces -= 1
    return verts, faces


def get_ply_vertices_num(ply_file_template):
    if os.path.isfile(ply_file_template.format('rh')) and \
        os.path.isfile(ply_file_template.format('lh')):
            rh_vertices, _ = read_ply_file(ply_file_template.format('rh'))
            lh_vertices, _ = read_ply_file(ply_file_template.format('lh'))
            return {'rh':rh_vertices.shape[0], 'lh':lh_vertices.shape[0]}
    else:
        print('No surface ply files!')
        return None


def check_hemi(hemi):
    if hemi in ['rh', 'lh']:
        hemi = [hemi]
    elif hemi=='both':
        hemi = ['rh', 'lh']
    else:
        raise ValueError('wrong hemi value!')
    return hemi


def get_data_max_min(data, norm_by_percentile, norm_percs):
    if norm_by_percentile:
        data_max = np.percentile(data, norm_percs[1])
        data_min = np.percentile(data, norm_percs[0])
    else:
        data_max = np.max(data)
        data_min = np.min(data)
    return data_max, data_min


def get_max_abs(data_max, data_min):
    return max(map(abs,[data_max, data_min]))


def normalize_data(data, norm_by_percentile, norm_percs):
    data_max, data_min = get_data_max_min(data, norm_by_percentile, norm_percs)
    max_abs = get_max_abs(data_max, data_min)
    norm_data = data / max_abs
    return norm_data


def read_freesurfer_lookup_table(freesurfer_home):
    lut_fname = os.path.join(freesurfer_home, 'FreeSurferColorLUT.txt')
    lut = np.genfromtxt(lut_fname, dtype=None, usecols=(0, 1),
                        names=['id', 'name'])
    return lut


# def read_sub_cortical_lookup_table(lookup_table_file_name):
#     names = {}
#     with open(lookup_table_file_name, 'r') as f:
#         for line in f.readlines():
#             lines = line.strip().split('\t')
#             if len(lines) > 1:
#                 name, code = lines[0].strip(), int(lines[1])
#                 names[code] = name
#     return names


def get_numeric_index_to_label(label, lut):
    if type(label) == str:
        seg_name = label
        seg_id = lut['id'][lut['name'] == seg_name][0]
    elif type(label) == int:
        seg_id = label
        seg_name = lut['name'][lut['id'] == seg_id][0]
    return seg_name, seg_id


def how_many_curlies(str):
    return len(re.findall('\{*\}', str))


def run_script(cmd):
    print('running: {}'.format(cmd))
    output = subprocess.check_output('{} | tee /dev/stderr'.format(cmd),
                                     shell=True)
    if (output != ''):
        print(output)


# def partial_run_script(vars, more_vars=None):
#     return partial(lambda cmd,v:run_script(cmd.format(**v)), v=vars)

def partial_run_script(vars, more_vars=None, print_only=False):
    return partial(_run_script_wrapper, vars=vars, print_only=print_only)


def _run_script_wrapper(cmd, vars, print_only=False, **kwargs):
    for k,v in kwargs.iteritems():
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


def namebase(file_name):
    return os.path.splitext(os.path.basename(file_name))[0]


def morph_labels_from_fsaverage(subject, subjects_dir='', aparc_name='aparc250', fs_labels_fol='', sub_labels_fol='', n_jobs=6):
    if subjects_dir=='':
        subjects_dir = os.environ['SUBJECTS_DIR']
    subject_dir = os.path.join(subjects_dir, subject)
    labels_fol = os.path.join(subjects_dir, 'fsaverage', 'label', aparc_name) if fs_labels_fol=='' else fs_labels_fol
    sub_labels_fol = os.path.join(subject_dir, 'label', aparc_name) if sub_labels_fol=='' else sub_labels_fol
    if not os.path.isdir(sub_labels_fol):
        os.makedirs(sub_labels_fol)
    for label_file in glob.glob(os.path.join(labels_fol, '*.label')):
        fs_label = mne.read_label(label_file)
        fs_label.values.fill(1.0)
        sub_label = fs_label.morph('fsaverage', subject, grade=None, n_jobs=n_jobs, subjects_dir=subjects_dir)
        sub_label.save(os.path.join(sub_labels_fol, '{}.label'.format(sub_label.name)))


def labels_to_annot(subject, subjects_dir='', aparc_name='aparc250', labels_fol='', overwrite=True):
    if subjects_dir=='':
        subjects_dir = os.environ['SUBJECTS_DIR']
    subject_dir = os.path.join(subjects_dir, subject)
    labels_fol = os.path.join(subject_dir, 'label', aparc_name) if labels_fol=='' else labels_fol
    labels = []
    for label_file in glob.glob(os.path.join(labels_fol, '*.label')):
        label = mne.read_label(label_file)
        print(label.name)
        labels.append(label)

    mne.write_labels_to_annot(subject=subject, labels=labels, parc=aparc_name, overwrite=overwrite,
                              subjects_dir=subjects_dir)


def get_hemis(hemi):
    return ['rh', 'lh'] if hemi == 'both' else [hemi]


def rmtree(fol):
    if os.path.isdir(fol):
        shutil.rmtree(fol)

def make_dir(fol):
    if not os.path.isdir(fol):
        os.makedirs(fol)

def get_subfolders(fol):
    return [os.path.join(fol,subfol) for subfol in os.listdir(fol) if os.path.isdir(os.path.join(fol,subfol))]


def get_spaced_colors(n):
    HSV_tuples = [(x*1.0/n, 0.5, 0.5) for x in range(n)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return RGB_tuples


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
            sub_corticals = codes
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


def get_activity_max_min(stc, norm_by_percentile=False, norm_percs=None, threshold=None):
    if type(stc) is types.DictType:
        if norm_by_percentile:
            data_max = max(np.percentile(stc['rh'], norm_percs[1]),
                           np.percentile(stc['lh'], norm_percs[1]))
            data_min = min(np.percentile(stc['rh'], norm_percs[0]),
                           np.percentile(stc['lh'], norm_percs[0]))
        else:
            data_max = max(np.max(stc['rh']), np.max(stc['lh']))
            data_min = max(np.min(stc['rh']), np.min(stc['lh']))
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


def read_labels(labels_fol, hemi='both', return_generator=False):
    hemis = [hemi] if hemi!='both' else ['rh', 'lh']
    labels = []
    for hemi in hemis:
        for label_file in glob.glob(os.path.join(labels_fol, '*{}.label'.format(hemi))):
            print('read label from {}'.format(label_file))
            label = mne.read_label(label_file)
            # if return_generator:
            #     yield label
            # else:
            labels.append(label)
    # if not return_generator:
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