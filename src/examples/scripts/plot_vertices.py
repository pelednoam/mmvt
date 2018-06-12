import os.path as op
import numpy as np
import bpy


def run(mmvt):
    input_name, default_cm = 'vertices_data.npz', 'jet'
    mu = mmvt.utils
    fol = mu.make_dir(op.join(mu.get_user_fol(), 'vertices_data'))
    input_fname = op.join(fol, input_name)
    if not op.isfile(input_fname):
        print("plot_vertices: Can't find {}!".format(input_fname))
        return
    vertices_data = mu.Bag(np.load(input_fname))
    subcorticals = get_subcorticals(mmvt, vertices_data)
    min_data, max_data = calc_data_min_max(mmvt, vertices_data, subcorticals)
    cb_title = str(vertices_data.get('cb_title', 'Vertices values'))
    colormap_name = str(vertices_data.get('colormap_name', default_cm))
    min_data = vertices_data.get('min_data', min_data)
    max_data = vertices_data.get('max_data', max_data)

    colors_ratio = set_colorbar(mmvt, min_data, max_data, cb_title, colormap_name)
    faces_verts = mmvt.coloring.get_faces_verts()
    hemi_verts_num = {hemi: faces_verts[hemi].shape[0] for hemi in mu.HEMIS}
    data = {hemi: np.zeros((hemi_verts_num[hemi])) for hemi in mu.HEMIS}
    for hemi in mu.HEMIS:
        if hemi not in vertices_data:
            print('plot_vertices: {} not in vertices_data!'.format(hemi))
            continue
        data[hemi][vertices_data[hemi][:, 0]] = vertices_data[hemi][:, 1]
        mmvt.coloring.color_hemi_data('inflated_{}'.format(hemi), data[hemi], min_data, colors_ratio, threshold=0.001)

    for subcortical, faces_verts, obj in subcorticals:
        verts_values = np.zeros((faces_verts.shape[0]))
        verts_values[vertices_data[subcortical][:, 0]] = vertices_data[subcortical][:, 1]
        mmvt.coloring.activity_map_obj_coloring(
            obj, verts_values, faces_verts, threshold=0.001, data_min=min_data, colors_ratio=colors_ratio)

    mmvt.appearance.show_activity()


def get_subcorticals(mmvt, vertices_data):
    mu = mmvt.utils
    subcorticals = []
    keys = [x for x in list(vertices_data.keys()) if x not in
            ['rh', 'lh', 'cb_title', 'colormap_name', 'min_data', 'max_data']]
    for subcortical in keys:
        obj = bpy.data.objects.get('{}_fmri_activity'.format(subcortical))
        if obj is None:
            print('plot_vertices: Can\'t find the object {}!'.format(subcortical))
            continue
        subcortical_faces_verts_fname = op.join(
            mu.get_user_fol(), 'subcortical', '{}_faces_verts.npy'.format(subcortical))
        if not op.isfile(subcortical_faces_verts_fname):
            print('plot_vertices: Can\'t find {}!'.format(subcortical_faces_verts_fname))
            continue
        subcortical_faces_verts = np.load(subcortical_faces_verts_fname)
        subcorticals.append((subcortical, subcortical_faces_verts, obj))
    return subcorticals


def calc_data_min_max(mmvt, vertices_data, subcorticals):
    max_data = 0
    min_data = np.inf
    subcorticals_objects_names = [k for k, _, _ in subcorticals]
    print('plot_vertices: subcorticals_objects_names: {}'.format(','.join(subcorticals_objects_names)))
    for region in mmvt.utils.HEMIS + subcorticals_objects_names:
        if region in vertices_data:
            max_data = max(max_data, np.nanmax(vertices_data[region][:, 1]))
            min_data = min(min_data, np.nanmin(vertices_data[region][:, 1]))
    return min_data, max_data


def set_colorbar(mmvt, data_min, data_max, cb_title='Vertices values', colormap_name=''):
    cb = mmvt.colorbar 
    if cb.colorbar_values_are_locked():
        data_max, data_min = cb.get_colorbar_max_min()
    else:
        cb.set_colorbar_max_min(data_max, data_min)
        cb.set_colorbar_title(cb_title)
        if colormap_name != '':
            cb.set_colormap(colormap_name)

    colors_ratio = 256 / (data_max - data_min)
    cb.set_colorbar_max_min(data_max, data_min)
    return colors_ratio