import os.path as op
import numpy as np


def run(mmvt, input_name='vertices_data.npz', default_cm='jet'):
    mu = mmvt.utils
    input_fname = op.join(mu.get_user_fol(), input_name)
    if not op.isfile(input_fname):
        print("plot_vertices: Can't find {}!".format(input_fname))
        return
    vertices_data = np.load(input_fname)
    min_data, max_data = calc_data_min_max(mmvt, vertices_data)
    cb_title = vertices_data.get('cb_title', 'Vertices values')
    colormap_name = vertices_data.get('colormap_name', default_cm)
    min_data = vertices_data.get('min_data', min_data)
    max_data = vertices_data.get('max_data', max_data)

    colors_ratio = set_colorbar(mmvt, min_data, max_data, cb_title, colormap_name)
    faces_verts = mmvt.coloring.get_faces_verts()
    hemi_verts_num = {hemi: faces_verts[hemi].shape[0] for hemi in mu.HEMIS}
    data = {hemi: np.zeros((hemi_verts_num[hemi], 4)) for hemi in mu.HEMIS}
    for hemi in mu.HEMIS:
        if hemi not in vertices_data:
            print('plot_vertices: {} not in vertices_data!'.format(hemi))
            continue
        hemi_vertices = vertices_data[hemi][:, 0]
        hemi_values = vertices_data[hemi][:, 1]
        data[hemi][hemi_vertices] = hemi_values
        mmvt.coloring.color_hemi_data('inflated_{}'.format(hemi), data[hemi], min_data, colors_ratio, threshold=0.5)

    subcorticals = [x for x in list(vertices_data.keys()) if x not in
                    ['rh', 'lh', 'cb_title', 'colormap_name', 'min_data', 'max_data']]
    for subcortical in subcorticals:
        lookup_file = op.join(mu.get_user_fol(), 'subcortical', '{}_faces_verts.npy'.format(subcortical))
        verts_file = op.join(mu.get_user_fol(), 'subcortical_fmri_activity', '{}.npy'.format(subcortical))
        if op.isfile(lookup_file) and op.isfile(verts_file):
            lookup = np.load(lookup_file)
            verts_values = np.zeros()
            activity_map_obj_coloring(cur_obj, verts_values, lookup, threshold, override_current_mat,
                                      use_abs=use_abs)

    mmvt.coloring.show_activity()


def calc_data_min_max(mmvt, vertices_data):
    max_data = 0
    min_data = np.inf
    for region in mmvt.utils.HEMIS + ['cerebellum']:
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