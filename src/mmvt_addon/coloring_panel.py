import mmvt_utils as mu
import colors_utils as cu
import numpy as np
import os.path as op
import time
import itertools
from collections import defaultdict, OrderedDict
import glob
import traceback
from functools import partial
import re

try:
    import bpy
    import bpy_extras
except:
    bpy = mu.empty_bpy
    bpy_extras = mu.empty_bpy_extras

try:
    import mne
    MNE_EXIST = True
except:
    MNE_EXIST = False

HEMIS = mu.HEMIS
# Should be moved to mmvt_addon
(WIC_MEG, WIC_MEG_LABELS, WIC_FMRI, WIC_FMRI_DYNAMICS, WIC_FMRI_LABELS, WIC_FMRI_CLUSTERS, WIC_EEG, WIC_MEG_SENSORS,
WIC_ELECTRODES, WIC_ELECTRODES_DISTS, WIC_ELECTRODES_SOURCES, WIC_ELECTRODES_STIM, WIC_MANUALLY, WIC_GROUPS, WIC_VOLUMES,
 WIC_CONN_LABELS_AVG, WIC_CONTOURS, WIC_LABELS_DATA) = range(18)


def _addon():
    return ColoringMakerPanel.addon


def get_activity_values(hemi):
    hemi = 'rh' if 'rh' in hemi else 'lh'
    return ColoringMakerPanel.activity_values[hemi]


def get_vertex_value(tkreg_ras=None, mni=None, voxel=None):
    val = None
    if tkreg_ras is not None:
        _addon().set_tkreg_ras(tkreg_ras)
    elif mni is not None:
        _addon().set_mni(mni)
    elif voxel is not None:
        _addon().set_voxel(voxel)
    if not bpy.context.scene.cursor_is_snapped:
        _addon().snap_cursor(True)
    vert_ind, mesh_name = _addon().get_closest_vertex_and_mesh_to_cursor()
    values = get_activity_values(mesh_name)
    if vert_ind < len(values):
        val = values[vert_ind]
    return val


def plot_fmri():
    activity_map_coloring('FMRI')


def get_eeg_sensors_data():
    return ColoringMakerPanel.eeg_sensors_data, ColoringMakerPanel.eeg_sensors_meta


def get_meg_sensors_data():
    return ColoringMakerPanel.meg_sensors_data, ColoringMakerPanel.meg_sensors_meta


def plot_meg(t=-1, save_image=False, view_selected=False):
    if t != -1:
        bpy.context.scene.frame_current = t
    activity_map_coloring('MEG')
    if save_image:
        return _addon().save_image('meg', view_selected, t)
    else:
        return True


# @mu.dump_args
def plot_stc(stc, t=-1, threshold=None, cb_percentiles=None, save_image=False,
             view_selected=False, subject='', save_prev_colors=False, cm=None,
             n_jobs=-1):
    import mne
    subject = mu.get_user() if subject == '' else subject
    n_jobs = mu.get_n_jobs(n_jobs)

    def create_stc_t(stc, t):
        if len(stc.times) == 1:
            return stc
        C = max([stc.rh_data.shape[0], stc.lh_data.shape[0]])
        stc_lh_data = stc.lh_data[:, t:t + 1] if stc.lh_data.shape[0] > 0 else np.zeros((C, 1))
        stc_rh_data = stc.rh_data[:, t:t + 1] if stc.rh_data.shape[0] > 0 else np.zeros((C, 1))
        data = np.concatenate([stc_lh_data, stc_rh_data])
        vertno = max([len(stc.lh_vertno), len(stc.rh_vertno)])
        lh_vertno = stc.lh_vertno if len(stc.lh_vertno) > 0 else np.arange(0, vertno)
        rh_vertno = stc.rh_vertno if len(stc.rh_vertno) > 0 else np.arange(0, vertno) + max(lh_vertno) + 1
        vertices = [lh_vertno, rh_vertno]
        stc_t = mne.SourceEstimate(data, vertices, stc.tmin + t * stc.tstep, stc.tstep, subject=subject)
        return stc_t

    # subjects_dir = mu.get_link_dir(mu.get_links_dir(), 'subjects')
    subjects_dir = mu.get_parent_fol(mu.get_user_fol())
    if subjects_dir:
        print('subjects_dir: {}'.format(subjects_dir))
    if t == -1:
        t = get_current_time()
    if isinstance(stc, str):
        stc_name = stc
        if not op.isfile(stc):
            stc = op.join(mu.get_user_fol(), 'meg', stc)
        if not op.isfile(stc):
            stc = '{}-rh.stc'.format(stc)
        if not op.isfile(stc):
            print("Can't find the stc file! ({})".format(stc_name))
            return '', None
        ColoringMakerPanel.stc = stc = mne.read_source_estimate(stc)
        stc.name = stc_name
        # data_min, data_max, _ = calc_stc_minmax(cb_percentiles, stc_name)
        # if not _addon().colorbar_values_are_locked():
        #     _addon().set_colorbar_max_min(data_max, data_min, force_update=True)
    else:
        if stc is None:
            stc = mne.read_source_estimate(get_stc_full_fname())
        stc.name = bpy.context.scene.meg_files
    if cb_percentiles is not None:
        set_meg_minmax_prec(*cb_percentiles, stc.name)
    if threshold is not None:
        set_lower_threshold(threshold)
    else:
        threshold = get_lower_threshold()
    stc_t = create_stc_t(stc, t)
    if len(bpy.data.objects.get('inflated_rh').data.vertices) == len(stc_t.rh_vertno) and \
            len(bpy.data.objects.get('inflated_lh').data.vertices) == len(stc_t.lh_vertno):
        stc_t_smooth = stc_t
    else:
        vertices_to = mne.grade_to_vertices(subject, None, subjects_dir=subjects_dir)
        n_jobs = 1 if mu.IS_WINDOWS else n_jobs
        stc_t_smooth = mne.morph_data(
            subject, subject, stc_t, n_jobs=n_jobs, grade=vertices_to, subjects_dir=subjects_dir)

    if _addon().colorbar_values_are_locked():
        data_max, data_min = _addon().get_colorbar_max_min()
    else:
        data_min, data_max = ColoringMakerPanel.meg_data_min, ColoringMakerPanel.meg_data_max
        if cm is not None:
            _addon().set_colormap(cm)
            _addon().change_colorbar_default_cm(('hot', 'hot'))
        else:
            _addon().set_colorbar_default_cm()
        # if data_min >= 0 and data_max >= 0:
        #     _addon().set_colormap('YlOrRd')
        _addon().set_colorbar_max_min(data_max, data_min)
        _addon().set_colorbar_prec(2)
        _addon().set_colorbar_title('MEG')
    if threshold > data_max:
        print('threshold > data_max!')
        threshold = bpy.context.scene.coloring_lower_threshold = 0
    colors_ratio = 256 / (data_max - data_min)
    # set_default_colormap(data_min, data_max)
    fname = plot_stc_t(stc_t_smooth.rh_data, stc_t_smooth.lh_data, t, data_min, colors_ratio,
                       threshold, save_image, view_selected, save_prev_colors=save_prev_colors)
    return fname, stc_t_smooth


# def set_default_colormap(data_min, data_max):
#     # todo: should read default values from ini file
#     if not (data_min == 0 and data_max == 0) and not _addon().colorbar_values_are_locked():
#         if data_min == 0 or np.sign(data_min) == np.sign(data_max):
#             _addon().set_colormap('YlOrRd')
#         else:
#             _addon().set_colormap('BuPu-YlOrRd')


def plot_stc_t(rh_data, lh_data, t, data_min=None, colors_ratio=None, threshold=0, save_image=False,
               view_selected=False, save_prev_colors=False):
    if data_min is None or colors_ratio is None:
        data_min = min([np.min(rh_data), np.min(lh_data)])
        data_max = max([np.max(rh_data), np.max(lh_data)])
        colors_ratio = 256 / (data_max - data_min)
    else:
        data_max = (256 + colors_ratio * data_min) / colors_ratio
    print('plot stc ({}-{})'.format(data_min, data_max))
    for hemi in mu.HEMIS:
        data = rh_data if hemi == 'rh' else lh_data
        color_hemi_data(hemi, data, data_min, colors_ratio, threshold, save_prev_colors=save_prev_colors)
    if save_image:
        return _addon().save_image('stc', view_selected, t)
    # elif render_image:
    #     return _addon().render_image('stc', view_selected, t)
    else:
        return True


def set_lower_threshold(val):
    if not isinstance(val, float):
        val = float(val)
    bpy.context.scene.coloring_lower_threshold = val


def get_lower_threshold():
    return bpy.context.scene.coloring_lower_threshold


def set_use_abs_threshold(val):
    bpy.context.scene.coloring_use_abs = val


def can_color_obj(obj):
    cur_mat = obj.active_material
    return 'RGB' in cur_mat.node_tree.nodes


def object_coloring(obj, rgb):
    # print('ploting {} with {}'.format(obj.name, rgb))
    if not obj:
        print('object_coloring: obj is None!')
        return False
    bpy.context.scene.objects.active = obj
    # todo: do we need to select the object here? In diff mode it's a problem
    # obj.select = True
    cur_mat = obj.active_material
    new_color = (rgb[0], rgb[1], rgb[2], 1)
    try:
        cur_mat.diffuse_color = new_color[:3]
    except:
        print('object_coloring: No diffuse_color')
        return False
    if can_color_obj(obj):
        cur_mat.node_tree.nodes["RGB"].outputs[0].default_value = new_color
    # else:
    #     print("Can't color {}".format(obj.name))
    #     return False
    # new_color = get_obj_color(obj)
    # print('{} new color: {}'.format(obj.name, new_color))
    if _addon().is_solid():
        cur_mat.use_nodes = False
    else:
        cur_mat.use_nodes = True
    # print(cur_mat.diffuse_color)
    return True


def get_obj_color(obj):
    cur_mat = obj.active_material
    try:
        if can_color_obj(obj):
            cur_color = tuple(cur_mat.node_tree.nodes["RGB"].outputs[0].default_value)
        else:
            cur_color = (1, 1, 1)
    except:
        cur_color = (1, 1, 1)
        print("Can't get the color!")
    return cur_color


def clear_subcortical_fmri_activity():
    for sub_obj in bpy.data.objects['Subcortical_fmri_activity_map'].children:
        clear_object_vertex_colors(sub_obj)
    for sub_obj in bpy.data.objects['Subcortical_meg_activity_map'].children:
        object_coloring(sub_obj, (1, 1, 1))
    for sub_obj in bpy.data.objects['Subcortical_structures'].children:
        object_coloring(sub_obj, (1, 1, 1))


def clear_cortex(hemis=HEMIS):
    for hemi in hemis:
        for cur_obj in [bpy.data.objects[hemi], mu.get_hemi_obj(hemi)]:
            clear_object_vertex_colors(cur_obj)
        # if bpy.context.scene.coloring_both_pial_and_inflated:
        #     for cur_obj in [bpy.data.objects[hemi], mu.get_hemi_obj(hemi)]:
        #         clear_object_vertex_colors(cur_obj)
        # else:
        #     # if _addon().is_pial():
        #     #     cur_obj = bpy.data.objects[hemi]
        #     # elif _addon().is_inflated():
        #     cur_obj = mu.get_hemi_obj(hemi)
        #     clear_object_vertex_colors(cur_obj)
        for label_obj in bpy.data.objects['Cortex-{}'.format(hemi)].children:
            object_coloring(label_obj, (1, 1, 1))


#todo: call this code from the coloring
def clear_object_vertex_colors(cur_obj):
    mesh = cur_obj.data
    scn = bpy.context.scene
    scn.objects.active = cur_obj
    cur_obj.select = True
    # bpy.ops.mesh.vertex_color_remove()
    # vcol_layer = mesh.vertex_colors.new()
    # rerecreate_coloring_layers(cur_obj, mesh, 'Col')
    recreate_coloring_layers(mesh)


# def rerecreate_coloring_layers(cur_obj, mesh, active_layer='Col'):
#     if len(mesh.vertex_colors) > 1 and 'inflated' in cur_obj.name:
#         # mesh.vertex_colors.active_index = 1
#         mesh.vertex_colors.active_index = mesh.vertex_colors.keys().index('Col')
#     if not (len(mesh.vertex_colors) == 1 and 'inflated' in cur_obj.name):
#         bpy.ops.mesh.vertex_color_remove()
#     if mesh.vertex_colors.get('Col') is None:
#         vcol_layer = mesh.vertex_colors.new('Col')
#     if mesh.vertex_colors.get('contours') is not None:
#         mesh.vertex_colors.active_index = mesh.vertex_colors.keys().index('contours')
#         bpy.ops.mesh.vertex_color_remove()
#         mesh.vertex_colors.new('contours')
#     if len(mesh.vertex_colors) > 1 and 'inflated' in cur_obj.name:
#         # mesh.vertex_colors.active_index = 1
#         mesh.vertex_colors.active_index = mesh.vertex_colors.keys().index('Col')
#         mesh.vertex_colors['Col'].active_render = True
#

# todo: do something with the threshold parameter
# @mu.timeit
def color_objects_homogeneously(data, names, conditions, data_min, colors_ratio, threshold=0, postfix_str=''):
    if data is None:
        print('color_objects_homogeneously: No data to color!')
        return
    if names is None:
        data, names, conditions = data['data'], data['names'], data['conditions']
    # default_color = (1, 1, 1)
    cur_frame = bpy.context.scene.frame_current
    for obj_name, values in zip(names, data):
        if not isinstance(obj_name, str):
            obj_name = obj_name.astype(str)
        if values.ndim == 0:
            values = [values]
        t_ind = min(len(values) - 1, cur_frame)
        if values[t_ind].ndim == 0:
            value = values[t_ind]
        # elif bpy.context.scene.selection_type == 'spec_cond':
        #     cond_inds = np.where(conditions == bpy.context.scene.conditions_selection)[0]
        #     if len(cond_inds) == 0:
        #         print("!!! Can't find the current condition in the data['conditions'] !!!")
        #         return
        #     else:
        #         cond_ind = cond_inds[0]
        #         object_colors = object_colors[:, cond_ind]
        #         value = values[t_ind, cond_ind]
        else:
            value = np.diff(values[t_ind])[0] if values.shape[1] > 1 else np.squeeze(values[t_ind])
        # todo: there is a difference between value and real_value, what should we do?
        # real_value = mu.get_fcurve_current_frame_val('Deep_electrodes', obj_name, cur_frame)
        # new_color = object_colors[cur_frame] if abs(value) > threshold else default_color
        new_color = calc_colors([value], data_min, colors_ratio)[0]
        # todo: check if the stat should be avg or diff
        obj = bpy.data.objects.get(obj_name+postfix_str)
        # if obj and not obj.hide:
        #     print('trying to color {} with {}'.format(obj_name+postfix_str, new_color))
        # obj.active_material = bpy.data.materials['selected_label_Mat_subcortical']
        ret = object_coloring(obj, new_color)
        # if not ret:
        #     print('color_objects_homogeneously: Error in coloring the object {}!'.format(obj_name))
            # print(obj_name, value, new_color)
        # else:
        #     print('color_objects_homogeneously: {} was not loaded!'.format(obj_name))

    # _addon().show_rois()
    # print('Finished coloring!!')


def init_activity_map_coloring(map_type, subcorticals=True):
    # _addon().set_appearance_show_activity_layer(bpy.context.scene, True)
    # _addon().set_filter_view_type(bpy.context.scene, 'RENDERS')
    _addon().show_activity()
    # _addon().change_to_rendered_brain()

    if not bpy.context.scene.objects_show_hide_sub_cortical:
        if subcorticals:
            _addon().show_hide_hierarchy(map_type != 'FMRI', 'Subcortical_fmri_activity_map')
            _addon().show_hide_hierarchy(map_type != 'MEG', 'Subcortical_meg_activity_map')
        else:
            _addon().show_hide_sub_corticals(not subcorticals)
    # change_view3d()


def load_faces_verts():
    faces_verts = {}
    current_root_path = mu.get_user_fol()
    if op.isfile(op.join(current_root_path, 'faces_verts_lh.npy') and \
                         op.join(current_root_path, 'faces_verts_rh.npy')):
        faces_verts['lh'] = np.load(op.join(current_root_path, 'faces_verts_lh.npy'))
        faces_verts['rh'] = np.load(op.join(current_root_path, 'faces_verts_rh.npy'))
    return faces_verts


def load_subs_faces_verts():
    faces_verts = {}
    verts = {}
    current_root_path = mu.get_user_fol()
    subcoticals = glob.glob(op.join(current_root_path, 'subcortical', '*_faces_verts.npy'))
    for subcortical_file in subcoticals:
        subcortical = mu.namebase(subcortical_file)[:-len('_faces_verts')]
        lookup_file = op.join(current_root_path, 'subcortical', '{}_faces_verts.npy'.format(subcortical))
        verts_file = op.join(current_root_path, 'subcortical', '{}.npz'.format(subcortical))
        if op.isfile(lookup_file) and op.isfile(verts_file):
            faces_verts[subcortical] = np.load(lookup_file)
            verts[subcortical] = np.load(verts_file)['verts']
    return faces_verts, verts


def load_meg_subcortical_activity():
    meg_sub_activity = None
    subcortical_activity_file = op.join(mu.get_user_fol(), 'subcortical_meg_activity.npz')
    if op.isfile(subcortical_activity_file) and bpy.context.scene.coloring_meg_subcorticals:
        meg_sub_activity = np.load(subcortical_activity_file)
    return meg_sub_activity


# @mu.dump_args
def activity_map_coloring(map_type, clusters=False, threshold=None):
    init_activity_map_coloring(map_type)
    if threshold is None:
        threshold = bpy.context.scene.coloring_lower_threshold
    meg_sub_activity = None
    if map_type == 'MEG':
        if bpy.context.scene.coloring_meg_subcorticals:
            meg_sub_activity = load_meg_subcortical_activity()
        ColoringMakerPanel.what_is_colored.add(WIC_MEG)
        mu.remove_items_from_set(ColoringMakerPanel.what_is_colored, [WIC_FMRI, WIC_FMRI_CLUSTERS, WIC_FMRI_DYNAMICS])
        plot_subcorticals = bpy.context.scene.coloring_meg_subcorticals
    elif map_type == 'FMRI':
        if not clusters:
            ColoringMakerPanel.what_is_colored.add(WIC_FMRI)
        else:
            ColoringMakerPanel.what_is_colored.add(WIC_FMRI_CLUSTERS)
        mu.remove_items_from_set(ColoringMakerPanel.what_is_colored, [WIC_MEG, WIC_MEG_LABELS, WIC_FMRI_DYNAMICS])
        plot_subcorticals = False
    elif map_type == 'FMRI_DYNAMICS':
        ColoringMakerPanel.what_is_colored.add(WIC_FMRI_DYNAMICS)
        mu.remove_items_from_set(ColoringMakerPanel.what_is_colored, [WIC_MEG, WIC_MEG_LABELS, WIC_FMRI, WIC_FMRI_CLUSTERS])
        # todo: support in subcorticals
        plot_subcorticals = False
    return plot_activity(map_type, ColoringMakerPanel.faces_verts, threshold, meg_sub_activity, clusters=clusters,
                  plot_subcorticals=plot_subcorticals)
    # setup_environment_settings()


def meg_labels_coloring(override_current_mat=True):
    ColoringMakerPanel.what_is_colored.add(WIC_MEG_LABELS)
    init_activity_map_coloring('MEG')
    threshold = bpy.context.scene.coloring_lower_threshold
    hemispheres = [hemi for hemi in HEMIS if not mu.get_hemi_obj(hemi).hide]
    user_fol = mu.get_user_fol()
    atlas, em = bpy.context.scene.atlas, bpy.context.scene.meg_labels_extract_method
    labels_data_minimax = np.load(op.join(user_fol, 'meg', 'labels_data_{}_{}_minmax.npz'.format(atlas, em)))
    meg_labels_min, meg_labels_max = labels_data_minimax['labels_diff_minmax'] \
        if bpy.context.scene.meg_labels_coloring_type == 'diff' else labels_data_minimax['labels_minmax']
    data_minmax = max(map(abs, [meg_labels_max, meg_labels_min]))
    meg_labels_min, meg_labels_max = -data_minmax, data_minmax
    for hemi in hemispheres:
        labels_data = np.load(op.join(user_fol, 'meg', 'labels_data_{}_{}_{}.npz'.format(atlas, em, hemi)))
        labels_coloring_hemi(
            labels_data, ColoringMakerPanel.faces_verts, hemi, threshold, bpy.context.scene.meg_labels_coloring_type,
            override_current_mat, meg_labels_min, meg_labels_max)


def color_connections_labels_avg(override_current_mat=True):
    ColoringMakerPanel.what_is_colored.add(WIC_CONN_LABELS_AVG)
    init_activity_map_coloring('MEG')
    threshold = bpy.context.scene.coloring_lower_threshold
    hemispheres = [hemi for hemi in HEMIS if not mu.get_hemi_obj(hemi).hide]
    user_fol = mu.get_user_fol()
    # files_names = [mu.namebase(fname).replace('_', ' ').replace('{} '.format(atlas), '') for fname in
    #                conn_labels_avg_files]
    atlas = bpy.context.scene.atlas
    # labels_data_minimax = np.load(op.join(user_fol, 'meg', 'meg_labels_{}_{}_minmax.npz'.format(atlas, em)))
    # meg_labels_min, meg_labels_max = labels_data_minimax['labels_diff_minmax'] \
    #     if bpy.context.scene.meg_labels_coloring_type == 'diff' else labels_data_minimax['labels_minmax']
    # data_minmax = max(map(abs, [meg_labels_max, meg_labels_min]))
    # meg_labels_min, meg_labels_max = -data_minmax, data_minmax

    file_name = bpy.context.scene.conn_labels_avg_files.replace(' ', '_').replace('_labels_avg.npz', '')
    file_name = '{}_{}_labels_avg.npz'.format(file_name, atlas)
    data = np.load(op.join(user_fol, 'connectivity', file_name))
    for hemi in hemispheres:
        labels_coloring_hemi(
            labels_data, ColoringMakerPanel.faces_verts, hemi, threshold, bpy.context.scene.meg_labels_coloring_type,
            override_current_mat, meg_labels_min, meg_labels_max)


def color_connectivity_degree():
    corr = ColoringMakerPanel.static_conn
    if corr.ndim != 2:
        print('ColoringMakerPanel.static_conn has {} dim instead of 2!'.format(corr.ndim))
        return
    threshold = bpy.context.scene.connectivity_degree_threshold
    labels = ColoringMakerPanel.connectivity_labels
    if bpy.context.scene.connectivity_degree_threshold_use_abs:
        degree_mat = np.sum(abs(corr) >= threshold, 0)
    else:
        degree_mat = np.sum(corr >= threshold, 0)
    data_max = max(degree_mat)
    colors_ratio = 256 / (data_max)
    _addon().set_colorbar_max_min(data_max, 0)
    _addon().set_colormap('YlOrRd')
    _addon().set_colorbar_title('fMRI connectivity degree')
    # _addon().fix_labels_material(labels)

    color_objects_homogeneously(degree_mat, labels, ['rest'], 0, colors_ratio)
    _addon().show_rois()
    if bpy.context.scene.connectivity_degree_save_image:
        _addon().save_image()


def static_conn_files_update(self, context):
    ColoringMakerPanel.static_conn = np.load(op.join(mu.get_user_fol(), 'connectivity', '{}.npy'.format(
        bpy.context.scene.static_conn_files)))


def update_connectivity_degree_threshold(self, context):
    color_connectivity_degree()


def fmri_labels_coloring(override_current_mat=True, use_abs=None):
    if use_abs is None:
        use_abs = bpy.context.scene.coloring_use_abs
    ColoringMakerPanel.what_is_colored.add(WIC_FMRI_LABELS)
    if not bpy.context.scene.color_rois_homogeneously:
        init_activity_map_coloring('FMRI')
    threshold = bpy.context.scene.coloring_lower_threshold
    hemispheres = [hemi for hemi in HEMIS if not mu.get_hemi_obj(hemi).hide]
    user_fol = mu.get_user_fol()
    atlas, em = bpy.context.scene.atlas, bpy.context.scene.fmri_labels_extract_method
    if _addon().colorbar_values_are_locked():
        labels_min, labels_max = _addon().get_colorbar_max_min()
    else:
        labels_min, labels_max  = np.load(
            op.join(user_fol, 'fmri', 'labels_data_{}_{}_minmax.pkl'.format(atlas, em)))
        if labels_min > 0:
            labels_min = 0
        _addon().set_colorbar_max_min(labels_max, labels_min)
        _addon().set_colorbar_title('fMRI')
    colors_ratio = 256 / (labels_max - labels_min)
    # set_default_colormap(labels_min, labels_max)
    for hemi in hemispheres:
        if mu.get_hemi_obj(hemi).hide:
            continue
        labels_data = np.load(op.join(user_fol, 'fmri', 'labels_data_{}_{}_{}.npz'.format(atlas, em, hemi)))
        # fix_labels_material(labels_data['names'])
        # todo check this also in other labels coloring
        if bpy.context.scene.color_rois_homogeneously:
            color_objects_homogeneously(
                labels_data['data'], labels_data['names'], ['rest'], labels_min, colors_ratio, threshold)
            _addon().show_rois()
        else:
            labels_coloring_hemi(
                labels_data, ColoringMakerPanel.faces_verts, hemi, threshold, 'avg', override_current_mat,
                labels_min, labels_max)

    if ColoringMakerPanel.fmri_subcorticals_mean_exist:
        em = bpy.context.scene.fmri_labels_extract_method
        subs_data = np.load(op.join(user_fol, 'fmri', 'subcorticals_{}.npz'.format(em)))
        # fix_labels_material(subs_data['names'])
        if bpy.context.scene.color_rois_homogeneously:
            color_objects_homogeneously(
                subs_data['data'], subs_data['names'], ['rest'], labels_min, colors_ratio, threshold)
            _addon().show_rois()
        else:
            t = bpy.context.scene.frame_current
            for sub_name, sub_data in zip(subs_data['names'], subs_data['data']):
                verts = ColoringMakerPanel.subs_verts[sub_name]
                data = np.tile(sub_data[t], (len(verts), 1))
                cur_obj = bpy.data.objects.get('{}_fmri_activity'.format(sub_name))
                activity_map_obj_coloring(
                    cur_obj, data, ColoringMakerPanel.subs_faces_verts[sub_name], 0, True,
                    labels_min, colors_ratio, use_abs)


def init_labels_colorbar(data, cb_title='', labels_max=None, labels_min=None, cmap=None):
    if _addon().colorbar_values_are_locked():
        labels_max, labels_min = _addon().get_colorbar_max_min()
    else:
        if labels_max is None:
            labels_max = np.max(data)
        if labels_min is None:
            labels_min = np.min(data)
        _addon().set_colorbar_max_min(labels_max, labels_min)
        if cmap is not None:
            _addon().set_colormap(cmap)
        else:
            _addon().set_colorbar_default_cm()
        if cb_title != '':
            _addon().set_colorbar_title(cb_title)
    return labels_max, labels_min


def color_labels_data(labels, data, atlas, cb_title='', labels_max=None, labels_min=None, cmap=None):
    data = data.ravel()
    if len(labels) != len(data):
        print('color_labels_data: len(labels) ({}) != len(data) ({})!'.format(len(labels), len(data)))
        return
    if len(labels) == 0:
        print('color_labels_data: len(labels) == 0!')
        return
    labels_max, labels_min = init_labels_colorbar(data, cb_title, labels_max, labels_min, cmap)
    colors_ratio = 256 / (labels_max - labels_min)

    threshold = bpy.context.scene.coloring_lower_threshold
    if threshold > labels_max:
        print('threshold > labels_max ({}), setting the threshold to labels_min ({})'.format(labels_max, labels_min))
        threshold = bpy.context.scene.coloring_lower_threshold = labels_min
    for hemi in mu.HEMIS:
        if mu.get_hemi_obj(hemi).hide:
            continue
        if bpy.context.scene.color_rois_homogeneously:
            color_objects_homogeneously(data, labels, [], labels_min, colors_ratio)
            _addon().show_rois()
        else:
            labels_data = dict(data=data, names=labels)
            labels_coloring_hemi(
                labels_data, ColoringMakerPanel.faces_verts, hemi, threshold, '', True,
                labels_min, labels_max, cmap=cmap, atlas=atlas)
    ColoringMakerPanel.what_is_colored.add(WIC_LABELS_DATA)


def load_labels_vertices(atlas):
    labels_vertices_fname = op.join(mu.get_user_fol(), 'labels_vertices_{}.pkl'.format(atlas))
    if op.isfile(labels_vertices_fname):
        labels_names, labels_vertices = mu.load(labels_vertices_fname)
        ColoringMakerPanel.labels_vertices[atlas] = \
            dict(labels_names=labels_names,labels_vertices=labels_vertices)


def labels_coloring_hemi(labels_data, faces_verts, hemi, threshold=0, labels_coloring_type='diff',
                         override_current_mat=True, colors_min=None, colors_max=None, use_abs=None, cmap=None,
                         atlas=None):
    if use_abs is None:
        use_abs = bpy.context.scene.coloring_use_abs
    if atlas is None:
        atlas = bpy.context.scene.atlas
    if atlas not in ColoringMakerPanel.labels_vertices:
        load_labels_vertices(atlas)
        if atlas not in ColoringMakerPanel.labels_vertices:
            print('Creating a label vertices file for atlas {}'.format(atlas))
            print('Please try again after the file will be created.')
            mu.run_mmvt_func('src.preproc.anatomy', 'save_labels_vertices', flags='-a {}'.format(atlas))
            return
    now = time.time()
    colors_ratio = None
    labels_names = ColoringMakerPanel.labels_vertices[atlas]['labels_names']
    labels_vertices = ColoringMakerPanel.labels_vertices[atlas]['labels_vertices']
    vertices_num = max(itertools.chain.from_iterable(labels_vertices[hemi])) + 1
    no_t = labels_data['data'][0].ndim == 0
    t = bpy.context.scene.frame_current
    if not colors_min is None and not colors_max is None:
        colors_data = np.zeros((vertices_num))
    else:
        colors_data = np.ones((vertices_num, 4))
        colors_data[:, 0] = 0
    colors = labels_data['colors'] if 'colors' in labels_data else [None] * len(labels_data['names'])
    if not colors_min is None and not colors_max is None:
        if _addon().colorbar_values_are_locked():
            colors_max, colors_min = _addon().get_colorbar_max_min()
            colors_ratio = 256 / (colors_max - colors_min)
        else:
            colors_ratio = 256 / (colors_max - colors_min)
            _addon().set_colorbar_max_min(colors_max, colors_min)
            if cmap is not None:
                _addon().set_colormap(cmap)
            else:
                _addon().set_colorbar_default_cm()
            # elif colors_min == 0 or np.sign(colors_min) == np.sign(colors_max):
            #     _addon().set_colormap('YlOrRd')
            # else:
            #     _addon().set_colormap('BuPu-YlOrRd')
    for label_data, label_colors, label_name in zip(labels_data['data'], colors, labels_data['names']):
        label_name = mu.to_str(label_name)
        if label_data.ndim == 0:
            label_data = np.array([label_data])
        if not colors_min is None and not colors_max is None:
            if label_data.ndim > 1:
                if labels_coloring_type == 'diff':
                    if label_data.squeeze().ndim == 1:
                        labels_data = label_data.squeeze()
                    else:
                        label_data = np.squeeze(np.diff(label_data))
                else:
                    cond_ind = np.where(labels_data['conditions'] == labels_coloring_type)[0]
                    label_data = np.squeeze(label_data[:, cond_ind])
            label_colors = calc_colors(label_data, colors_min, colors_ratio)
        else:
            label_colors = np.array(label_colors)
            if 'unknown' in label_name:
                continue
            # if label_colors.ndim == 3:
        #         cond_inds = np.where(labels_data['conditions'] == bpy.context.scene.conditions_selection)[0]
        #         if len(cond_inds) == 0:
        #             print("!!! Can't find the current condition in the data['conditions'] !!!")
        #             return
        #         label_colors = label_colors[:, cond_inds[0], :]
        #         label_data = label_data[:, cond_inds[0]]
        if label_name not in labels_names[hemi]:
            continue
        label_index = labels_names[hemi].index(label_name)
        label_vertices = np.array(labels_vertices[hemi][label_index])
        if len(label_vertices) > 0:
            if no_t:
                label_data_t, label_colors_t = label_data, label_colors
            else:
                label_data_t, label_colors_t = (label_data[t], label_colors[t]) if 0 < t < len(label_data) else (0, 0)
            # print('coloring {} with {}'.format(label_name, label_colors_t))
            if not colors_min is None and not colors_max is None:
                colors_data[label_vertices] = label_data_t
            else:
                label_colors_data = np.hstack((label_data_t, label_colors_t))
                label_colors_data = np.tile(label_colors_data, (len(label_vertices), 1))
                colors_data[label_vertices, :] = label_colors_data
    # if bpy.context.scene.coloring_both_pial_and_inflated:
    #     for cur_obj in [bpy.data.objects[hemi], mu.get_hemi_obj(hemi)]:
    #         activity_map_obj_coloring(
    #             cur_obj, colors_data, faces_verts[hemi], threshold, override_current_mat, colors_min, colors_ratio)
    # else:
    #     if _addon().is_pial():
    #         cur_obj = bpy.data.objects[hemi]
    #     elif _addon().is_inflated():
    #         cur_obj = mu.get_hemi_obj(hemi)
    #     activity_map_obj_coloring(
    #         cur_obj, colors_data, faces_verts[hemi], threshold, override_current_mat, colors_min, colors_ratio)
    cur_obj = mu.get_hemi_obj(hemi)
    activity_map_obj_coloring(
        cur_obj, colors_data, faces_verts[hemi], threshold, override_current_mat, colors_min, colors_ratio, use_abs)
    print('Finish labels_coloring_hemi, hemi {}, {:.2f}s'.format(hemi, time.time() - now))


def color_hemi_data(hemi, data, data_min=None, colors_ratio=None, threshold=0, override_current_mat=True,
                    save_prev_colors=False, coloring_layer='Col', use_abs=None, check_valid_verts=True):
    if use_abs is None:
        use_abs = bpy.context.scene.coloring_use_abs
    if hemi in mu.HEMIS:
        pial_hemi = hemi
        hemi = 'inflated_{}'.format(hemi)
    elif hemi.startswith('inflated'):
        pial_hemi = hemi[len('inflated_'):]
    if bpy.data.objects[hemi].hide:
        return
    faces_verts = ColoringMakerPanel.faces_verts[pial_hemi]
    cur_obj = bpy.data.objects[hemi]
    activity_map_obj_coloring(cur_obj, data, faces_verts, threshold, override_current_mat, data_min,
                              colors_ratio, use_abs, save_prev_colors=save_prev_colors, coloring_layer=coloring_layer,
                              check_valid_verts=check_valid_verts)


@mu.timeit
def plot_activity(map_type, faces_verts, threshold, meg_sub_activity=None,
        plot_subcorticals=True, override_current_mat=True, clusters=False):

    current_root_path = mu.get_user_fol()# bpy.path.abspath(bpy.context.scene.conf_path)
    not_hiden_hemis = [hemi for hemi in HEMIS if not mu.get_hemi_obj(hemi).hide]
    t = bpy.context.scene.frame_current
    frame_str = str(t)
    data_min, data_max = 0, 0
    f = None
    for hemi in not_hiden_hemis:
        colors_ratio, data_min = None, None
        if map_type in ['MEG', 'FMRI_DYNAMICS']:
            if map_type == 'MEG':
                activity_type = bpy.context.scene.meg_files
                activity_type = '' if activity_type == 'conditions diff' else '{}_'.format(activity_type)
                fname = op.join(current_root_path, 'activity_map_{}{}'.format(
                    activity_type, hemi), 't' + frame_str + '.npy')
                colors_ratio = ColoringMakerPanel.meg_activity_colors_ratio
                data_min, data_max = ColoringMakerPanel.meg_activity_data_minmax
                cb_title = 'MEG'
            elif map_type == 'FMRI_DYNAMICS':
                fname =  op.join(current_root_path, 'fmri', 'activity_map_' + hemi, 't' + frame_str + '.npy')
                colors_ratio = ColoringMakerPanel.fmri_activity_colors_ratio
                data_min, data_max = ColoringMakerPanel.fmri_activity_data_minmax
                cb_title = 'fMRI'
            if op.isfile(fname):
                f = np.load(fname)
                if _addon().colorbar_values_are_locked():
                    data_max, data_min = _addon().get_colorbar_max_min()
                    colors_ratio = 256 / (data_max - data_min)
                else:
                    _addon().set_colorbar_max_min(data_max, data_min)
            else:
                print("Can't load {}".format(fname))
                return False
        elif map_type == 'FMRI':
            if not ColoringMakerPanel.fmri_activity_data_minmax is None:
                if _addon().colorbar_values_are_locked():
                    data_max, data_min = _addon().get_colorbar_max_min()
                    colors_ratio = 256 / (data_max - data_min)
                else:
                    colors_ratio = ColoringMakerPanel.fmri_activity_colors_ratio
                    data_min, data_max = ColoringMakerPanel.fmri_activity_data_minmax
                    _addon().set_colorbar_max_min(data_max, data_min)
                    _addon().set_colorbar_title('fMRI')
                _addon().set_colorbar_max_min(data_max, data_min)
            if clusters:
                f = [c for h, c in ColoringMakerPanel.fMRI_clusters.items() if h == hemi]
            else:
                f = ColoringMakerPanel.fMRI[hemi]

        if threshold > data_max:
            print('Threshold > data_max, changing it to data_min')
            threshold = data_min

        if f.ndim == 2:
            if t >= f.shape[1]:
                bpy.context.scene.frame_current = t = f.shape[1] - 1
                bpy.data.scenes['Scene'].frame_preview_start = 0
                bpy.data.scenes['Scene'].frame_preview_end = t
            f = f[:, t]

        # # todo: should read default values from ini file
        # if not (data_min == 0 and data_max == 0) and not _addon().colorbar_values_are_locked():
        #     if data_min == 0:
        #         _addon().set_colormap('YlOrRd')
        #     else:
        #         _addon().set_colormap('BuPu-YlOrRd')

        color_hemi_data(hemi, f, data_min, colors_ratio, threshold, override_current_mat)
        # if bpy.context.scene.coloring_both_pial_and_inflated:
        #     for cur_obj in [bpy.data.objects[hemi], mu.get_hemi_obj(hemi)]:
        #         activity_map_obj_coloring(cur_obj, f, faces_verts[hemi], threshold, override_current_mat, data_min,
        #                                   colors_ratio)
        # else:
        #     if _addon().is_pial():
        #         cur_obj = bpy.data.objects[hemi]
        #     elif _addon().is_inflated():
        #         cur_obj = mu.get_hemi_obj(hemi)
        #     activity_map_obj_coloring(cur_obj, f, faces_verts[hemi], threshold, override_current_mat, data_min, colors_ratio)

    if plot_subcorticals and not bpy.context.scene.objects_show_hide_sub_cortical and not meg_sub_activity is None:
        if map_type == 'MEG' and not bpy.data.objects['Subcortical_meg_activity_map'].hide:
                if f is None:
                    if _addon().colorbar_values_are_locked():
                        data_max, data_min = _addon().get_colorbar_max_min()
                        colors_ratio = 256 / (data_max - data_min)
                    else:
                        data_max, data_min = meg_sub_activity['data_minmax'], -meg_sub_activity['data_minmax']
                        colors_ratio = 256 / (2 * data_max)
                        _addon().set_colorbar_max_min(data_max, data_min)
                        _addon().set_colorbar_title('Subcortical MEG')
                color_objects_homogeneously(
                    meg_sub_activity['data'], meg_sub_activity['names'], meg_sub_activity['conditions'], data_min,
                    colors_ratio, threshold, '_meg_activity')
                _addon().show_rois()
        elif map_type == 'FMRI' and not bpy.data.objects['Subcortical_fmri_activity_map'].hide:
            fmri_subcortex_activity_color(threshold, override_current_mat)

    return True
    # return loop_indices
    # Noam: not sure this is necessary
    #deselect_all()
    #bpy.data.objects['Brain'].select = True


def fmri_subcortex_activity_color(threshold, override_current_mat=True, use_abs=None):
    if use_abs is None:
        use_abs = bpy.context.scene.coloring_use_abs
    current_root_path = mu.get_user_fol() # bpy.path.abspath(bpy.context.scene.conf_path)
    subcoticals = glob.glob(op.join(current_root_path, 'subcortical_fmri_activity', '*.npy'))
    for subcortical_file in subcoticals:
        subcortical = op.splitext(op.basename(subcortical_file))[0]
        cur_obj = bpy.data.objects.get('{}_fmri_activity'.format(subcortical))
        if cur_obj is None:
            print("Can't find the object {}!".format(subcortical))
        else:
            lookup_file = op.join(current_root_path, 'subcortical', '{}_faces_verts.npy'.format(subcortical))
            verts_file = op.join(current_root_path, 'subcortical_fmri_activity', '{}.npy'.format(subcortical))
            if op.isfile(lookup_file) and op.isfile(verts_file):
                lookup = np.load(lookup_file)
                verts_values = np.load(verts_file)
                activity_map_obj_coloring(cur_obj, verts_values, lookup, threshold, override_current_mat,
                                          use_abs=use_abs)


def create_inflated_curv_coloring():

    def color_obj_curvs(cur_obj, curv, lookup):
        mesh = cur_obj.data
        if not 'curve' in mesh.vertex_colors.keys():
            scn = bpy.context.scene
            verts_colors = np.zeros((curv.shape[0], 3))
            verts_colors[np.where(curv == 0)] = [1, 1, 1]
            verts_colors[np.where(curv == 1)] = [0.55, 0.55, 0.55]
            scn.objects.active = cur_obj
            cur_obj.select = True
            bpy.ops.mesh.vertex_color_remove()
            vcol_layer = mesh.vertex_colors.new('curve')
            for vert in range(curv.shape[0]):
                x = lookup[vert]
                for loop_ind in x[x>-1]:
                    d = vcol_layer.data[loop_ind]
                    d.color = verts_colors[vert]

    try:
        # todo: check not to overwrite
        print('Creating the inflated curvatures coloring')
        for hemi in mu.HEMIS:
            cur_obj = mu.get_hemi_obj(hemi)
            curv_fname = op.join(mu.get_user_fol(), 'surf', '{}.curv.npy'.format(hemi))
            faces_verts_fname = op.join(mu.get_user_fol(), 'faces_verts_{}.npy'.format(hemi))
            if not op.isfile(curv_fname) or not op.isfile(faces_verts_fname):
                print("Can't plot the {} curves!".format(hemi))
                continue
            curv = np.load(curv_fname)
            lookup = np.load(op.join(mu.get_user_fol(), 'faces_verts_{}.npy'.format(hemi)))
            color_obj_curvs(cur_obj, curv, lookup)
        for hemi in mu.HEMIS:
            curvs_fol = op.join(mu.get_user_fol(), 'surf', '{}_{}_curves'.format(bpy.context.scene.atlas, hemi))
            lookup_fol = op.join(mu.get_user_fol(), 'labels', '{}.pial.{}'.format(bpy.context.scene.atlas, hemi))
            for cur_obj in bpy.data.objects['Cortex-{}'.format(hemi)].children:
                try:
                    label = cur_obj.name
                    inflated_cur_obj = bpy.data.objects['inflated_{}'.format(label)]
                    curv_file = op.join(curvs_fol, '{}_curv.npy'.format(label))
                    if op.isfile(curv_file):
                        curv = np.load(curv_file)
                    else:
                        print('Can\'t find the file {}'.format(curv_file))

                    faces_verts_file = op.join(lookup_fol, '{}_faces_verts.npy'.format(label))
                    if op.isfile(faces_verts_file):
                        lookup = np.load(faces_verts_file)
                    else:
                        print('Can\'t find the file {}'.format(faces_verts_file))
                    color_obj_curvs(inflated_cur_obj, curv, lookup)
                except:
                    print("Can't create {}'s curves!".format(cur_obj.name))
                    print(traceback.format_exc())
    except:
        print('Error in create_inflated_curv_coloring!')
        print(traceback.format_exc())


def get_obj_color_data(obj, color, val=1):
    if isinstance(obj, str):
        obj = bpy.data.objects.get(obj, None)
    if obj is None:
        print("set_color_for_obj: Can't find {}".format(obj))
        return
    data = np.ones((len(obj.data.vertices), 4)) * val
    if isinstance(color, str):
        color = cu.name_to_rgb(color)
    data[:, 1:] = color
    return data


def calc_colors(vert_values, data_min, colors_ratio, cm=None):
    if cm is None:
        cm = _addon().get_cm()
    if cm is None:
        return np.zeros((len(vert_values), 3))
    return mu.calc_colors_from_cm(vert_values, data_min, colors_ratio, cm)


def find_valid_verts(values, threshold, use_abs, bigger_or_equall):
    if use_abs is None:
        use_abs = bpy.context.scene.coloring_use_abs
    if use_abs:
        if bigger_or_equall:
            valid_verts = np.where(np.abs(values) >= threshold)[0]
        else:
            valid_verts = np.where(np.abs(values) > threshold)[0]
    else:
        if bigger_or_equall:
            valid_verts = np.where(values >= threshold)[0]
        else:
            valid_verts = np.where(values > threshold)[0]
    return valid_verts


def set_activity_values(cur_obj, values):
    if mu.obj_is_cortex(cur_obj):
        hemi = 'rh' if 'rh' in cur_obj.name else 'lh'
        ColoringMakerPanel.activity_values[hemi] = values
        if bpy.context.scene.cursor_is_snapped:
            vert_ind, mesh_name = _addon().get_closest_vertex_and_mesh_to_cursor()
            closest_hemi = 'rh' if 'rh' in mesh_name else 'lh'
            if closest_hemi == hemi:
                if  vert_ind < len(values):
                    _addon().set_vertex_data(values[vert_ind])
                else:
                    print('vert_ind {} > len(values) ({}) in mesh {}'.format(vert_ind, len(values), mesh_name))


# @mu.timeit
def activity_map_obj_coloring(cur_obj, vert_values, lookup=None, threshold=0, override_current_mat=True, data_min=None,
                              colors_ratio=None, use_abs=None, bigger_or_equall=False, save_prev_colors=False,
                              coloring_layer='Col', check_valid_verts=True):
    if isinstance(cur_obj, str):
        cur_obj = bpy.data.objects[cur_obj]
    if lookup is None:
        lookup_fnames = glob.glob(
            op.join(mu.get_user_fol(), '**', '{}_faces_verts.npy'.format(cur_obj.name)), recursive=True)
        if len(lookup_fnames) == 0:
            print("activity_map_obj_coloring: Can't find the lookup file for {}".format(cur_obj))
            return False
        lookup = np.load(lookup_fnames[0])

    mesh = cur_obj.data
    scn = bpy.context.scene

    if use_abs is None:
        use_abs = bpy.context.scene.coloring_use_abs
    _addon().show_activity()
    values = vert_values[:, 0] if vert_values.ndim > 1 else vert_values
    if coloring_layer == 'Col':
        set_activity_values(cur_obj, values)
    valid_verts = find_valid_verts(values, threshold, use_abs, bigger_or_equall)
    if len(valid_verts) == 0 and check_valid_verts:
        # print('No vertices values are above the threhold ({} to {})'.format(np.min(values), np.max(values)))
        return
    colors_picked_from_cm = False
    # cm = _addon().get_cm()
    if vert_values.ndim > 1 and vert_values.squeeze().ndim == 1:
        vert_values = vert_values.squeeze()
    if vert_values.ndim == 1 and data_min is not None:
        verts_colors = calc_colors(vert_values, data_min, colors_ratio)
        colors_picked_from_cm = True
    #check if our mesh already has Vertex Colors, and if not add some... (first we need to make sure it's the active object)
    scn.objects.active = cur_obj
    cur_obj.select = True
    if override_current_mat:
        recreate_coloring_layers(mesh, coloring_layer)
        # vcol_layer = mesh.vertex_colors["Col"]

    if len(mesh.vertex_colors) > 1 and 'inflated' in cur_obj.name:
        # mesh.vertex_colors.active_index = 1
        mesh.vertex_colors.active_index = mesh.vertex_colors.keys().index(coloring_layer)
        mesh.vertex_colors[coloring_layer].active_render = True
    # else:
    #     vcol_layer = mesh.vertex_colors.active
    # print('cur_obj: {}, max vert in lookup: {}, vcol_layer len: {}'.format(cur_obj.name, np.max(lookup), len(vcol_layer.data)))
    vcol_layer = mesh.vertex_colors[coloring_layer]
    if save_prev_colors:
        ColoringMakerPanel.prev_colors[cur_obj.name] = {'lookup':lookup, 'vcol_layer':vcol_layer, 'colors':{}}
    if colors_picked_from_cm:
        verts_lookup_loop_coloring(
            valid_verts, lookup, vcol_layer, lambda vert:verts_colors[vert], cur_obj.name, save_prev_colors)
    else:
        verts_lookup_loop_coloring(
            valid_verts, lookup, vcol_layer, lambda vert:vert_values[vert, 1:], cur_obj.name, save_prev_colors)


def verts_lookup_loop_coloring(valid_verts, lookup, vcol_layer, colors_func, cur_obj_name, save_prev_colors=False):
    if save_prev_colors:
        ColoringMakerPanel.prev_colors[cur_obj_name]['colors'] = defaultdict(dict)
    for vert in valid_verts:
        x = lookup[vert]
        for loop_ind in x[x > -1]:
            d = vcol_layer.data[loop_ind]
            if save_prev_colors:
                ColoringMakerPanel.prev_colors[cur_obj_name]['colors'][vert][loop_ind] = d.color.copy()
            d.color = colors_func(vert)


def recreate_coloring_layers(mesh, coloring_layer='Col'):
    # if len(mesh.vertex_colors) > 1 and 'inflated' in cur_obj.name:
    #     # mesh.vertex_colors.active_index = 1
    #     mesh.vertex_colors.active_index = mesh.vertex_colors.keys().index('Col')
    # if not (len(mesh.vertex_colors) == 1 and 'inflated' in cur_obj.name):
    #     c = mu.get_graph_context()
    #     bpy.ops.mesh.vertex_color_remove(c)
    #     # except:
    #     #     print("Can't remove vertex color!")
    # if mesh.vertex_colors.get('Col') is None:
    #     vcol_layer = mesh.vertex_colors.new('Col')

    coloring_layers = ['Col', 'contours'] if coloring_layer == 'Col' else ['contours']
    for cl in coloring_layers:
        if mesh.vertex_colors.get(cl) is not None:
            mesh.vertex_colors.active_index = mesh.vertex_colors.keys().index(cl)
            mesh.vertex_colors.remove(mesh.vertex_colors.get(cl))
            # bpy.ops.mesh.vertex_color_remove()
    for cl in coloring_layers:
        if mesh.vertex_colors.get(cl) is None:
            mesh.vertex_colors.new(cl)
    # if mesh.vertex_colors.get('Col') is None:
    #     vcol_layer = mesh.vertex_colors.new('Col')
    # if mesh.vertex_colors.get('Col') is None:
    #     mesh.vertex_colors.new('contours')
    mesh.vertex_colors.active_index = mesh.vertex_colors.keys().index('Col')
    mesh.vertex_colors['Col'].active_render = True
    # return vcol_layer


def get_activity_colors(vert_values, threshold, data_min, colors_ratio, bigger_or_equall=False):
    values = vert_values[:, 0] if vert_values.ndim > 1 else vert_values
    if bigger_or_equall:
        valid_verts = np.where(np.abs(values) >= threshold)[0]
    else:
        valid_verts = np.where(np.abs(values) > threshold)[0]
    if vert_values.ndim > 1 and vert_values.squeeze().ndim == 1:
        vert_values = vert_values.squeeze()
    verts_colors = calc_colors(vert_values[valid_verts], data_min, colors_ratio)
    return verts_colors, valid_verts


def get_prev_colors():
    return ColoringMakerPanel.prev_colors


def color_prev_colors(verts, obj_name, coloring_layer='Col'):
    if obj_name not in ColoringMakerPanel.prev_colors:
        return False
    cur_obj = bpy.data.objects[obj_name]
    mesh = cur_obj.data
    scn = bpy.context.scene
    scn.objects.active = cur_obj
    cur_obj.select = True
    lookup = ColoringMakerPanel.prev_colors[obj_name]['lookup']
    hemi = 'rh' if 'rh' in obj_name else 'lh'

    if len(mesh.vertex_colors) > 1 and 'inflated' in cur_obj.name:
        mesh.vertex_colors.active_index = mesh.vertex_colors.keys().index(coloring_layer)
        mesh.vertex_colors[coloring_layer].active_render = True
    vcol_layer = mesh.vertex_colors[coloring_layer]
    for vert in verts:
        x = lookup[vert]
        for loop_ind in x[x > -1]:
            d = vcol_layer.data[loop_ind]
            if vert in ColoringMakerPanel.prev_colors[obj_name]['colors']:
                prev_color = ColoringMakerPanel.prev_colors[obj_name]['colors'][vert][loop_ind]
                if d.color != prev_color:
                    d.color = prev_color
            else:
                if ColoringMakerPanel.curvs is not None:
                    default_color = [1, 1, 1] if ColoringMakerPanel.curvs[hemi][vert] == 0 else [0.55, 0.55, 0.55]
                else:
                    default_color = [1, 1, 1]
                d.color = default_color
    return True


def color_groups_manually():
    ColoringMakerPanel.what_is_colored.add(WIC_GROUPS)
    init_activity_map_coloring('FMRI')
    labels = ColoringMakerPanel.labels_groups[bpy.context.scene.labels_groups]
    objects_names, colors, data = defaultdict(list), defaultdict(list), defaultdict(list)
    for label in labels:
        obj_type = mu.check_obj_type(label['name'])
        if obj_type is not None:
            objects_names[obj_type].append(label['name'])
            colors[obj_type].append(np.array(label['color']) / 255.0)
            data[obj_type].append(1.)
    clear_subcortical_fmri_activity()
    color_objects(objects_names, colors, data)


def color_manually():
    ColoringMakerPanel.what_is_colored.add(WIC_MANUALLY)
    init_activity_map_coloring('FMRI')
    subject_fol = mu.get_user_fol()
    objects_names, colors, data = defaultdict(list), defaultdict(list), defaultdict(list)
    values = []
    coloring_objects, coloring_labels = False, False
    # labels_delim, labels_pos, _, _ = mu.get_hemi_delim_and_pos(bpy.data.objects['Cortex-lh'].children[0].name)
    coloring_name = '{}.csv'.format(bpy.context.scene.coloring_files)
    no_colors_lines_num = sum(
        [1 for line in mu.csv_file_reader(op.join(subject_fol, 'coloring', coloring_name))if len(line) == 1])
    rand_colors = itertools.cycle(mu.get_distinct_colors(no_colors_lines_num))
    subs_names = [o.name for o in bpy.data.objects['Subcortical_structures'].children]
    cortex_labels_names = [o.name for o in bpy.data.objects['Cortex-lh'].children + bpy.data.objects['Cortex-rh'].children]
    other_atals_labels = defaultdict(list)
    if coloring_name.startswith('_labels_'):
        atlas = coloring_name.split('_')[2]
        obj_type = mu.OBJ_TYPE_LABEL
        coloring_labels = True
    else:
        obj_type = ''
        coloring_objects = True
        atlas = ''
    # atlas_labels_rh, atlas_labels_lh = {}, {}
    for line in mu.csv_file_reader(op.join(subject_fol, 'coloring', coloring_name)):
        if len(line) == 0 or line[0][0] == '#':
            continue
        object_added = False
        if line[0].startswith('atlas'):
            atlas = line[0].split('=')[-1].strip()
            # atlas_labels_rh[atlas] = mu.read_labels_from_annots(atlas, hemi='rh')
            # atlas_labels_lh[atlas] = mu.read_labels_from_annots(atlas, hemi='lh')
            # if len(atlas_labels_rh[atlas]) + len(atlas_labels_lh[atlas]) == 0:
            #     print("Couldn't find any labels for atlas {}!".format(atlas))
            #     return
            continue
        if len(line) >= 1:
            obj_name = line[0]
        if len(line) == 1 or line[1] == '':
            color_name = ''
        elif isinstance(line[1], str):
            color_name = line[1]
        if len(line) >= 4 and all([mu.is_float(x) for x in line[1:4]]):
            color_name = line[1:4]
        if isinstance(color_name, str) and color_name != '':
            color_rgb = cu.name_to_rgb(color_name)
        # Check if the color is already in RBG
        elif len(color_name) == 3:
            if max(list(map(float, color_name))) > 1:
                color_name = [float(c) / 256 for c in color_name]
            color_rgb = color_name
        elif len(color_name) == 0:
            color_rgb = next(rand_colors)
        else:
            print('Unrecognize color! ({})'.format(color_name))
            continue
        color_rgb = list(map(float, color_rgb))
        if len(line) == 5:
            values.append(float(line[4]))
        # if isinstance(color_name, list) and len(color_name) == 1:
        #     color_name = color_name[0]
        if not coloring_labels:
            obj_type = mu.check_obj_type(obj_name)
        # todo: support coloring of corticals objects and atlas labels
        if obj_type is None or atlas != '' and obj_type in (mu.OBJ_TYPE_CORTEX_LH, mu.OBJ_TYPE_CORTEX_RH):
            # Check if the obj_name is an sub-cortical
            obj_name = '_'.join(re.split('\W+', obj_name))
            for sub_name in subs_names:
                if obj_name.lower() in '_'.join(re.split('\W+', sub_name)).lower():
                    obj_type = mu.OBJ_TYPE_SUBCORTEX
                    objects_names[obj_type].append(sub_name)
                    colors[obj_type].append(color_rgb)
                    data[obj_type].append(1.)
                    object_added = True
            if object_added:
                continue
            # Check if this is a label from another atlas
            if len(line) >= 3 and isinstance(line[2], str) or atlas != '':
                line_atlas = line[2] if len(line) >= 3 and isinstance(line[2], str) else atlas
                other_atals_labels, object_added = find_atlas_labels(
                    obj_name, line_atlas, color_rgb, other_atals_labels)
                # labels_fol = mu.get_atlas_labels_fol(atlas)
                # if labels_fol != '':
                #     for atlas_label_name in atlas_labels:
                #         if obj_name.lower() in '_'.join(re.split('\W+', atlas_label_name)).lower():
                #             other_atals_labels[atlas].append((atlas_label_name, color_rgb))
                #             object_added = True
                    # if not atlas_label_found:
                    #     other_atals_labels[atlas].append((obj_name, color_rgb))
                    #     object_added = True
            if object_added:
                continue

            other_atals_labels, object_added = find_atlas_labels(
                obj_name, bpy.context.scene.atlas, color_rgb, other_atals_labels, cortex_labels_names)
            if object_added:
                continue

            # todo: merge these two cases
            for cortex_labels_name in cortex_labels_names:
                if obj_name.lower() in '_'.join(re.split('\W+', cortex_labels_name)).lower():
                    hemi = mu.get_hemi_from_fname(cortex_labels_name)
                    obj_type = mu.OBJ_TYPE_CORTEX_LH if hemi == 'lh' else mu.OBJ_TYPE_CORTEX_RH
                    objects_names[obj_type].append(cortex_labels_name)
                    colors[obj_type].append(color_rgb)
                    data[obj_type].append(1.)
                    object_added = True
            if object_added:
                continue
            if obj_type is None and atlas == '':
                # check if the obj_name is an existing label with a different delim/pos
                delim, pos, label, hemi = mu.get_hemi_delim_and_pos(obj_name)
                if delim != '' and pos != '' and hemi != '' and label != obj_name:
                    labels_delim, labels_pos, labels_label, _ = mu.get_hemi_delim_and_pos(
                        bpy.data.objects['Cortex-lh'].children[0].name)
                    label_obj_name = mu.build_label_name(labels_delim, labels_pos, label, hemi)
                    obj_type = mu.check_obj_type(label_obj_name)

        # else:
        if obj_type is not None and not object_added:
            objects_names[obj_type].append(obj_name)
            colors[obj_type].append(color_rgb)
            data[obj_type].append(1.)

    if coloring_objects:
        color_objects(objects_names, colors, data)
    elif coloring_labels:
        plot_labels(objects_names[mu.OBJ_TYPE_LABEL], colors[mu.OBJ_TYPE_LABEL], atlas)
    ColoringMakerPanel.labels_plotted = []
    for atlas, labels_tup in other_atals_labels.items():
        plot_labels([t[0] for t in labels_tup], [t[1] for t in labels_tup], atlas,
                    do_plot=False) # atlas_labels_rh[atlas], atlas_labels_lh[atlas],
    if len(other_atals_labels) > 0:
        _plot_labels()
    if len(values) > 0:
        _addon().set_colorbar_max_min(np.max(values), np.min(values))
    _addon().set_colorbar_title(bpy.context.scene.coloring_files.replace('_', ' '))

    if op.isfile(op.join(subject_fol, 'coloring', '{}_legend.jpg'.format(bpy.context.scene.coloring_files))):
        cmd = '{} -m src.preproc.electrodes_preproc -s {} -a {} -f show_labeling_coloring --ignore_missing 1'.format(
            bpy.context.scene.python_cmd, mu.get_user(), bpy.context.scene.atlas)
        print('Running {}'.format(cmd))
        mu.run_command_in_new_thread(cmd, False)


def find_atlas_labels(obj_name, atlas, color_rgb, other_atals_labels, atlas_labels=None):
    labels_fol = mu.get_atlas_labels_fol(atlas)
    object_added = False
    if labels_fol != '':
        if atlas_labels is None:
            atlas_labels = [l.name for l in mu.read_labels_from_annots(atlas)]
        delim, pos, label, hemi = mu.get_hemi_delim_and_pos(obj_name)
        labels_delim, labels_pos, labels_label, _ = mu.get_hemi_delim_and_pos(atlas_labels[0])
        label_obj_name = mu.build_label_name(labels_delim, labels_pos, label, hemi)
        label_obj_name = '_'.join(re.split('\W+', label_obj_name)).lower()
        for atlas_label_name in atlas_labels:
            if label_obj_name.lower() in '_'.join(re.split('\W+', atlas_label_name)).lower():
                other_atals_labels[atlas].append((atlas_label_name, color_rgb))
                object_added = True
    return other_atals_labels, object_added


def color_objects(objects_names, colors, data):
    for hemi in HEMIS:
        # This code should be merged with the plot_labels function
        obj_type = mu.OBJ_TYPE_CORTEX_LH if hemi=='lh' else mu.OBJ_TYPE_CORTEX_RH
        if obj_type not in objects_names or len(objects_names[obj_type]) == 0:
            continue
        labels_data = dict(data=np.array(data[obj_type]), colors=colors[obj_type], names=objects_names[obj_type])
        # print('color hemi {}: {}'.format(hemi, labels_names))
        labels_coloring_hemi(labels_data, ColoringMakerPanel.faces_verts, hemi, labels_coloring_type='avg')
    clear_subcortical_regions()
    if mu.OBJ_TYPE_SUBCORTEX in objects_names:
        for region, color in zip(objects_names[mu.OBJ_TYPE_SUBCORTEX], colors[mu.OBJ_TYPE_SUBCORTEX]):
            print('color {}: {}'.format(region, color))
            color_subcortical_region(region, color)
    if mu.OBJ_TYPE_ELECTRODE in objects_names:
        for electrode, color in zip(objects_names[mu.OBJ_TYPE_ELECTRODE], colors[mu.OBJ_TYPE_ELECTRODE]):
            obj = bpy.data.objects.get(electrode)
            if obj and not obj.hide:
                object_coloring(obj, color)
    if mu.OBJ_TYPE_CEREBELLUM in objects_names:
        for cer, color in zip(objects_names[mu.OBJ_TYPE_CEREBELLUM], colors[mu.OBJ_TYPE_CEREBELLUM]):
            obj = bpy.data.objects.get(cer)
            if obj and not obj.hide:
                object_coloring(obj, color)
    bpy.context.scene.subcortical_layer = 'fmri'
    _addon().show_activity()


def color_volumetric():
    ColoringMakerPanel.what_is_colored.add(WIC_VOLUMES)
    pass


def color_subcortical_region(region_name, color, use_abs=None):
    if use_abs is None:
        use_abs = bpy.context.scene.coloring_use_abs
    #todo: read the ColoringPanel.subs_verts_faces like in fmri labels coloring
    cur_obj = bpy.data.objects.get(region_name + '_fmri_activity', None)
    obj_ana_fname = op.join(mu.get_user_fol(), 'subcortical', '{}.npz'.format(region_name))
    obj_lookup_fname = op.join(mu.get_user_fol(), 'subcortical', '{}_faces_verts.npy'.format(region_name))
    if not cur_obj is None and op.isfile(obj_lookup_fname):
        # todo: read only the verts number
        if not op.isfile(obj_ana_fname):
            verts, faces = mu.read_ply_file(op.join(mu.get_user_fol(), 'subcortical', '{}.ply'.format(region_name)))
            np.savez(obj_ana_fname, verts=verts, faces=faces)
        else:
            d = np.load(obj_ana_fname)
            verts =  d['verts']
        lookup = np.load(obj_lookup_fname)
        region_colors_data = np.hstack((np.array([1.]), color))
        region_colors_data = np.tile(region_colors_data, (len(verts), 1))
        activity_map_obj_coloring(cur_obj, region_colors_data, lookup, 0, True, use_abs=use_abs)
    else:
        if cur_obj and not 'white' in cur_obj.name.lower():
            print("Don't have the necessary files for coloring {}!".format(region_name))


def clear_subcortical_regions():
    clear_colors_from_parent_childrens('Subcortical_meg_activity_map')
    clear_subcortical_fmri_activity()


def clear_colors_from_parent_childrens(parent_object):
    parent_obj = bpy.data.objects.get(parent_object)
    if parent_obj is not None:
        for obj in parent_obj.children:
            if 'RGB' in obj.active_material.node_tree.nodes:
                obj.active_material.node_tree.nodes['RGB'].outputs['Color'].default_value = (1, 1, 1, 1)
            obj.active_material.diffuse_color = (1, 1, 1)


def default_coloring(loop_indices):
    for hemi, indices in loop_indices.items():
        cur_obj = bpy.data.objects[hemi]
        mesh = cur_obj.data
        vcol_layer = mesh.vertex_colors.active
        for loop_ind in indices:
            vcol_layer.data[loop_ind].color = [1, 1, 1]


def meg_files_update(self, context):
    _meg_files_update(context)


def _meg_files_update(context):
    user_fol = mu.get_user_fol()
    ColoringMakerPanel.activity_map_chosen = False
    ColoringMakerPanel.stc_file_chosen = False
    activity_types = [mu.namebase(f)[len('activity_map_'):-2] for f in glob.glob(op.join(user_fol, 'activity_map_*rh'))]
    for activity_type in activity_types:
        if activity_type != '':
            activity_type = activity_types[:-1]
        if bpy.context.scene.meg_files == activity_type or activity_type == '' and \
                        bpy.context.scene.meg_files == 'conditions diff':
            ColoringMakerPanel.activity_map_chosen = True
            meg_data_maxmin_fname = op.join(mu.get_user_fol(), 'meg_activity_map_{}minmax.pkl'.format(activity_type))
            data_min, data_max = mu.load(meg_data_maxmin_fname)
            ColoringMakerPanel.meg_activity_colors_ratio = 256 / (data_max - data_min)
            ColoringMakerPanel.meg_activity_data_minmax = (data_min, data_max)
            if not _addon().colorbar_values_are_locked():
                _addon().set_colorbar_max_min(data_max, data_min, True)
            break
    else:
        full_stc_fname = get_stc_full_fname()
        if full_stc_fname == '':
            print("Can't find the stc file in {}".format(op.join(user_fol, 'meg', '*.stc')))
        else:
            # ColoringMakerPanel.stc = mne.read_source_estimate(full_stc_fname)
            ColoringMakerPanel.stc = None
            # if not _addon().colorbar_values_are_locked():
            if not _addon().colorbar_values_are_locked():
                data_min, data_max, data_len = calc_stc_minmax()
                _addon().set_colorbar_max_min(data_max, data_min, force_update=True)
                _addon().set_colorbar_title('MEG')
            T = data_len - 1
            bpy.data.scenes['Scene'].frame_preview_start = 0
            bpy.data.scenes['Scene'].frame_preview_end = T
            if context.scene.frame_current > T:
                context.scene.frame_current = T
            # try:
            #     bpy.context.scene.meg_max_t = max([np.argmax(np.max(stc.rh_data, 0)), np.argmax(np.max(stc.lh_data, 0))])
            # except:
            #     bpy.context.scene.meg_max_t = 0


def get_stc_full_fname():
    full_stc_fname = ''
    user_fol = mu.get_user_fol()
    template = op.join(user_fol, 'meg', '*.stc')
    stcs_files = glob.glob(template)
    # todo: Do something better...
    stcs_files += glob.glob(op.join(user_fol, 'eeg', '*.stc'))
    for stc_file in stcs_files:
        _, _, label, hemi = mu.get_hemi_delim_and_pos(mu.namebase(stc_file))
        if label == bpy.context.scene.meg_files:
            full_stc_fname = stc_file
            ColoringMakerPanel.stc_file_chosen = True
            break
    return full_stc_fname


def meg_minmax_prec_update(selc, context):
    if not ColoringMakerPanel.run_meg_minmax_prec_update:
        return
    ColoringMakerPanel.run_meg_minmax_prec_update = False
    if bpy.context.scene.meg_min_prec >= bpy.context.scene.meg_max_prec:
        bpy.context.scene.meg_min_prec = bpy.context.scene.meg_max_prec - 1
    if bpy.context.scene.meg_max_prec <= bpy.context.scene.meg_min_prec:
        bpy.context.scene.meg_max_prec = bpy.context.scene.meg_min_prec + 1
    ColoringMakerPanel.run_meg_minmax_prec_update = True
    calc_stc_minmax()


def set_meg_minmax_prec(min_prec, max_prec, stc_name):
    if ColoringMakerPanel.stc_exist:
        ColoringMakerPanel.run_meg_minmax_prec_update = False
        bpy.context.scene.meg_min_prec = min_prec
        bpy.context.scene.meg_max_prec = max_prec
        ColoringMakerPanel.run_meg_minmax_prec_update = True
        calc_stc_minmax((min_prec, max_prec), stc_name)


def calc_stc_minmax(meg_min_max_prec=None, stc_name=''):
    if meg_min_max_prec is not None:
        meg_min_prec, meg_max_prec = meg_min_max_prec
    else:
        meg_min_prec = bpy.context.scene.meg_min_prec
        meg_max_prec = bpy.context.scene.meg_max_prec
    if stc_name == '':
        stc_name = bpy.context.scene.meg_files.replace(' ', '_')
    # print('calc_stc_minmax: {}'.format(bpy.context.scene.meg_files))
    maxmin_fname = op.join(mu.get_user_fol(), 'meg', '{}_{}_{}_minmax.pkl'.format(stc_name, meg_min_prec, meg_max_prec))
    if op.isfile(maxmin_fname):
        ColoringMakerPanel.meg_data_min, ColoringMakerPanel.meg_data_max, stc_data_len = mu.load(maxmin_fname)
        return ColoringMakerPanel.meg_data_min, ColoringMakerPanel.meg_data_max, stc_data_len

    if ColoringMakerPanel.stc is None:
        ColoringMakerPanel.stc = mne.read_source_estimate(get_stc_full_fname())
    data_min = mu.min_stc(ColoringMakerPanel.stc, meg_min_prec)
    data_max = mu.max_stc(ColoringMakerPanel.stc, meg_max_prec)
    if data_min == data_max == 0:
        data_min = mu.min_stc(ColoringMakerPanel.stc)
        data_max = mu.max_stc(ColoringMakerPanel.stc)
    # data_minmax = mu.get_max_abs(data_max, data_min)
    # factor = -int(mu.ceil_floor(np.log10(data_minmax)))
    if np.max(ColoringMakerPanel.stc._data) < 1e-5:
        ColoringMakerPanel.stc._data *= np.power(10, 9)
        data_min *= np.power(10, 9)
        data_max *= np.power(10, 9)
    if np.sign(data_max) != np.sign(data_min) and data_min != 0:
        data_minmax = max(map(abs, [data_min, data_max]))
        ColoringMakerPanel.meg_data_min, ColoringMakerPanel.meg_data_max = -data_minmax, data_minmax
    else:
        if data_min >= 0 and data_max >= 0:
            ColoringMakerPanel.meg_data_min, ColoringMakerPanel.meg_data_max = 0, data_max
        elif data_min <= 0 and data_max <= 0:
            ColoringMakerPanel.meg_data_min, ColoringMakerPanel.meg_data_max = data_min, 0
        else:
            ColoringMakerPanel.meg_data_min, ColoringMakerPanel.meg_data_max = data_min, data_max
    if not _addon().colorbar_values_are_locked():
        _addon().set_colorbar_max_min(data_max, data_min, force_update=True)
    stc_data_len = ColoringMakerPanel.stc.data.shape[1]

    mu.save((ColoringMakerPanel.meg_data_min, ColoringMakerPanel.meg_data_max, stc_data_len), maxmin_fname)
    return ColoringMakerPanel.meg_data_min, ColoringMakerPanel.meg_data_max, stc_data_len


def get_meg_data_minmax():
    return (ColoringMakerPanel.meg_data_min, ColoringMakerPanel.meg_data_max)


@mu.tryit()
def plot_fmri_file(fmri_template_fname):
    fmri_template_fname_no_hemi = mu.remove_hemi_template(mu.namebase(fmri_template_fname))
    init_fmri_files(current_fmri_file=fmri_template_fname_no_hemi)
    activity_map_coloring('FMRI')


def _fmri_files_update(fmri_file_name):
    #todo: there are two frmi files list (the other one in fMRI panel)
    user_fol = mu.get_user_fol()
    # fname_template = op.join(user_fol, 'fmri', 'fmri_{}_{}.npy'.format(fmri_file_name, '{hemi}'))
    fname_template = op.join(user_fol, 'fmri', 'fmri_*{}*{}.npy'.format('{hemi}', fmri_file_name))
    if not mu.both_hemi_files_exist(fname_template):
        fname_template = op.join(user_fol, 'fmri', 'fmri_*{}*{}.npy'.format(fmri_file_name, '{hemi}'))
    if not mu.both_hemi_files_exist(fname_template):
        print('fmri_files_update: {} does not exist!'.format(fname_template))
        ColoringMakerPanel.fMRI['rh'] = None
        ColoringMakerPanel.fMRI['lh'] = None
        return False
    for hemi in mu.HEMIS:
        fname = glob.glob(fname_template.format(hemi=hemi))[0]
        ColoringMakerPanel.fMRI[hemi] = np.load(fname)
    # fmri_data_maxmin_fname = op.join(mu.get_user_fol(), 'fmri', 'fmri_activity_map_minmax_{}.pkl'.format(
    #     bpy.context.scene.fmri_files))
    # if not op.isfile(fmri_data_maxmin_fname):
    fmri_data_maxmin_fname = op.join(mu.get_user_fol(), 'fmri', '{}_minmax.pkl'.format(fmri_file_name))
    #     bpy.context.scene.fmri_files))
    if not op.isfile(fmri_data_maxmin_fname):
        calc_fmri_min_max(fmri_data_maxmin_fname, fname_template)
    data_min, data_max = mu.load(fmri_data_maxmin_fname)
    ColoringMakerPanel.fmri_activity_colors_ratio = 256 / (data_max - data_min)
    ColoringMakerPanel.fmri_activity_data_minmax = (data_min, data_max)
    if not _addon().colorbar_values_are_locked():
        _addon().set_colorbar_max_min(data_max, data_min)
        _addon().set_colorbar_title('fMRI')
    return True


def fmri_files_update(self, context):
    _fmri_files_update(bpy.context.scene.fmri_files)


def calc_fmri_min_max(fmri_data_maxmin_fname, fname_template):
    # Remove the RGB columns
    for hemi in mu.HEMIS:
        fname = glob.glob(fname_template.format(hemi=hemi))[0] if '*' in fname_template else \
            fname_template.format(hemi=hemi)
        x = np.load(fname)
        if x.ndim > 1 and x.shape[1] == 4:
            x = x[:, 0]
            np.save(fname, x)
        ColoringMakerPanel.fMRI[hemi] = x
    data = np.hstack((ColoringMakerPanel.fMRI[hemi] for hemi in mu.HEMIS))
    data_min, data_max = np.nanmin(data), np.nanmax(data)
    print('fmri_files_update: min: {}, max: {}'.format(data_min, data_max))
    data_minmax = mu.get_max_abs(data_max, data_min)
    if np.sign(data_max) != np.sign(data_min) and data_min != 0:
        data_max, data_min = data_minmax, -data_minmax
    mu.save((data_min, data_max), fmri_data_maxmin_fname)


def electrodes_sources_files_update(self, context):
    ColoringMakerPanel.electrodes_sources_labels_data, ColoringMakerPanel.electrodes_sources_subcortical_data = \
        get_elecctrodes_sources()


def get_elecctrodes_sources():
    labels_fname = op.join(mu.get_user_fol(), 'electrodes', '{}-{}.npz'.format(
        bpy.context.scene.electrodes_sources_files, '{hemi}'))
    subcorticals_fname = labels_fname.replace('labels', 'subcortical').replace('-{hemi}', '')
    electrodes_sources_labels_data = \
        {hemi:np.load(labels_fname.format(hemi=hemi)) for hemi in mu.HEMIS}
    electrodes_sources_subcortical_data = np.load(subcorticals_fname)
    return electrodes_sources_labels_data, electrodes_sources_subcortical_data


def color_electrodes_sources():
    ColoringMakerPanel.what_is_colored.add(WIC_ELECTRODES_SOURCES)
    labels_data = ColoringMakerPanel.electrodes_sources_labels_data
    rh_data, lh_data = labels_data['rh']['data'], labels_data['lh']['data']
    subcorticals = ColoringMakerPanel.electrodes_sources_subcortical_data
    sub_data = subcorticals['data']

    _max = partial(np.percentile, q=97)
    _min = partial(np.percentile, q=3)
    if _addon().colorbar_values_are_locked():
        data_max, data_min = _addon().get_colorbar_max_min()
        colors_ratio = 256 / (data_max - data_min)
    else:
        data_min = min([_min(sub_data), _min(rh_data), _min(lh_data)])
        data_max = max([_max(sub_data), _max(rh_data), _max(lh_data)])
        data_minmax = max(map(abs, [data_min, data_max]))
        data_min, data_max = -data_minmax, data_minmax
        colors_ratio = 256 / (data_max - data_min)
        _addon().set_colorbar_max_min(data_max, data_min)
        _addon().set_colorbar_title('Electordes-Cortex')
    curr_sub_data = sub_data[:, bpy.context.scene.frame_current, 0]
    subs_colors = calc_colors(curr_sub_data, data_min, colors_ratio)
    # cond_inds = np.where(subcortical_data['conditions'] == bpy.context.scene.conditions_selection)[0]
    # if len(cond_inds) == 0:
    #     print("!!! Can't find the current condition in the data['conditions'] !!!")
    #     return {"FINISHED"}
    clear_subcortical_regions()
    for region, color in zip(subcorticals['names'], subs_colors):
        # print('electrodes source: color {} with {}'.format(region, color))
        color_subcortical_region(region, color)
    for hemi in mu.HEMIS:
        if mu.get_hemi_obj(hemi).hide:
            continue
        labels_coloring_hemi(labels_data[hemi], ColoringMakerPanel.faces_verts, hemi, 0,
                             colors_min=data_min, colors_max=data_max)


def color_eeg_helmet(use_abs=None):
    if use_abs is None:
        use_abs = bpy.context.scene.coloring_use_abs
    fol = mu.get_user_fol()
    data_fname = op.join(fol, 'eeg', 'eeg_data.npy')
    data = np.load(data_fname)
    data = np.diff(data).squeeze()
    lookup = np.load(op.join(fol, 'eeg', 'eeg_faces_verts.npy'))
    threshold = 0
    if _addon().colorbar_values_are_locked():
        data_max, data_min = _addon().get_colorbar_max_min()
    else:
        data_min, data_max = np.percentile(data, 3), np.percentile(data, 97)
        _addon().set_colorbar_max_min(data_max, data_min, True)
        _addon().set_colorbar_title('EEG')
    colors_ratio = 256 / (data_max - data_min)
    data_t = data[:, bpy.context.scene.frame_current]

    cur_obj = bpy.data.objects['eeg_helmet']
    activity_map_obj_coloring(cur_obj, data_t, lookup, threshold, True, data_min=data_min,
                              colors_ratio=colors_ratio, bigger_or_equall=False, use_abs=use_abs)


def color_meg_sensors():
    _addon().show_hide_meg_sensors()
    ColoringMakerPanel.what_is_colored.add(WIC_MEG_SENSORS)
    threshold = bpy.context.scene.coloring_lower_threshold
    data, meta = get_meg_sensors_data()
    if _addon().colorbar_values_are_locked():
        data_max, data_min = _addon().get_colorbar_max_min()
    else:
        data_min, data_max = ColoringMakerPanel.meg_sensors_data_minmax
        _addon().set_colorbar_max_min(data_max, data_min)
    colors_ratio = 256 / (data_max - data_min)
    if bpy.context.scene.meg_sensors_conditions != 'diff':
        cond_ind = np.where(meta['conditions'] == bpy.context.scene.meg_sensors_conditions)[0][0]
        data = data[:, :, cond_ind]
        _addon().set_colorbar_title('MEG sensors {} condition'.format(meta['conditions'][cond_ind]))
    else:
        _addon().set_colorbar_title('MEG sensors conditions difference')
    color_objects_homogeneously(data, meta['names'], meta['conditions'], data_min, colors_ratio, threshold)


def color_eeg_sensors():
    _addon().show_hide_eeg()
    ColoringMakerPanel.what_is_colored.add(WIC_EEG)
    threshold = bpy.context.scene.coloring_lower_threshold
    data, meta = get_eeg_sensors_data()
    if _addon().colorbar_values_are_locked():
        data_max, data_min = _addon().get_colorbar_max_min()
    else:
        data_min, data_max = ColoringMakerPanel.eeg_sensors_data_minmax
        _addon().set_colorbar_max_min(data_max, data_min)
    colors_ratio = 256 / (data_max - data_min)
    if bpy.context.scene.eeg_sensors_conditions != 'diff':
        cond_ind = np.where(meta['conditions'] == bpy.context.scene.eeg_sensors_conditions)[0][0]
        data = data[:, :, cond_ind]
        if not _addon().colorbar_values_are_locked():
            _addon().set_colorbar_title('EEG sensors {} condition'.format(meta['conditions'][cond_ind]))
    else:
        if not _addon().colorbar_values_are_locked():
            _addon().set_colorbar_title('EEG sensors conditions difference')
    color_objects_homogeneously(data, meta['names'], meta['conditions'], data_min, colors_ratio, threshold)


def color_electrodes_dists():
    bpy.context.scene.show_hide_electrodes = True
    _addon().show_hide_electrodes(True)
    ColoringMakerPanel.what_is_colored.add(WIC_ELECTRODES_DISTS)
    dists = _addon().load_electrodes_dists()
    _, names, conditions = _addon().load_electrodes_data()
    selected_objects = bpy.context.selected_objects
    if not (len(selected_objects) == 1 and selected_objects[0].name in names):
        return

    selected_elec = selected_objects[0].name
    selected_elec_ind = np.where(names == selected_elec)[0]
    data = dists[selected_elec_ind].squeeze()
    data_max = np.max(dists)
    colors_ratio = 256 / data_max
    if not _addon().colorbar_values_are_locked():
        _addon().set_colorbar_max_min(data_max, 0)
        _addon().set_colorbar_title('Electordes conditions difference')
        _addon().set_colormap('hot')
    color_objects_homogeneously(data, names, conditions, 0, colors_ratio, 0)
    _addon().show_electrodes()


def color_electrodes():
    # mu.set_show_textured_solid(False)
    bpy.context.scene.show_hide_electrodes = True
    _addon().show_hide_electrodes(True)
    ColoringMakerPanel.what_is_colored.add(WIC_ELECTRODES)
    threshold = bpy.context.scene.coloring_lower_threshold
    data, names, conditions = _addon().load_electrodes_data()
    if not _addon().colorbar_values_are_locked():
        norm_percs = (3, 97)  # todo: add to gui
        data_max, data_min = mu.get_data_max_min(data, True, norm_percs=norm_percs, data_per_hemi=False, symmetric=True)
        colors_ratio = 256 / (data_max - data_min)
        _addon().set_colorbar_max_min(data_max, data_min)
    else:
        data_max, data_min = _addon().get_colorbar_max_min()
        colors_ratio = 256 / (data_max - data_min)
    if bpy.context.scene.electrodes_conditions != 'diff':
        cond_ind = np.where(conditions == bpy.context.scene.electrodes_conditions)[0][0]
        data = data[:, :, cond_ind]
        if not _addon().colorbar_values_are_locked():
            _addon().set_colorbar_title('Electrodes {} condition'.format(conditions[cond_ind]))
    else:
        if not _addon().colorbar_values_are_locked():
            _addon().set_colorbar_title('Electrodes conditions difference')
    color_objects_homogeneously(data, names, conditions, data_min, colors_ratio, threshold)
    _addon().show_electrodes()
    # for obj in bpy.data.objects['Deep_electrodes'].children:
    #     bpy.ops.object.editmode_toggle()
    #     bpy.ops.object.editmode_toggle()

    # mu.update()
    # mu.set_show_textured_solid(True)
    # _addon().change_to_rendered_brain()


    # deselect_all()
    # mu.select_hierarchy('Deep_electrodes', False)


def what_is_colored():
    return ColoringMakerPanel.what_is_colored


def color_electrodes_stim():
    ColoringMakerPanel.what_is_colored.add(WIC_ELECTRODES_STIM)
    threshold = bpy.context.scene.coloring_lower_threshold
    stim_fname = 'stim_electrodes_{}.npz'.format(bpy.context.scene.stim_files.replace(' ', '_'))
    stim_data_fname = op.join(mu.get_user_fol(), 'electrodes', stim_fname)
    data = np.load(stim_data_fname)
    color_objects_homogeneously(data, threshold=threshold)
    _addon().show_electrodes()
    _addon().change_to_rendered_brain()


def color_connections(threshold=None):
    clear_connections()
    _addon().plot_connections(_addon().connections_data(), bpy.context.scene.frame_current, threshold)


def clear_and_recolor():
    color_meg = partial(activity_map_coloring, map_type='MEG')
    color_fmri = partial(activity_map_coloring, map_type='FMRI')
    color_fmri_clusters = partial(activity_map_coloring, map_type='FMRI', clusters=True)

    wic_funcs = {
        WIC_MEG:color_meg,
        WIC_MEG_LABELS:meg_labels_coloring,
        WIC_FMRI:color_fmri,
        WIC_FMRI_CLUSTERS:color_fmri_clusters,
        WIC_ELECTRODES:color_electrodes,
        WIC_ELECTRODES_SOURCES:color_electrodes_sources,
        WIC_ELECTRODES_STIM:color_electrodes_stim,
        WIC_MANUALLY:color_manually,
        WIC_GROUPS:color_groups_manually,
        WIC_VOLUMES:color_volumetric}

    what_is_colored = ColoringMakerPanel.what_is_colored
    clear_colors()
    for wic in what_is_colored:
        wic_funcs[wic]()


def get_condditions_from_labels_fcurves():
    conditions = []
    parent_obj = bpy.data.objects.get('Cortex-lh')
    if parent_obj and len(parent_obj.children) > 0:
        label_obj = parent_obj.children[0]
        fcurves_names = mu.get_fcurves_names(label_obj)
        conditions = [fc_name.split('_')[-1] for fc_name in fcurves_names]
    return conditions


def plot_labels_folder():
    labels_files = glob.glob(op.join(bpy.path.abspath(bpy.context.scene.labels_folder), '*.label'))
    if len(labels_files) == 0:
        print('No labels were found in {}!'.format(op.realpath(bpy.context.scene.labels_folder)))
    else:
        colors = cu.get_distinct_colors_hs(len(labels_files))
        # todo: same color for each hemi
        for label_fname, color in zip(labels_files, colors):
            plot_label(label_fname)


class ColorEEGHelmet(bpy.types.Operator):
    bl_idname = "mmvt.eeg_helmet"
    bl_label = "mmvt eeg helmet"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        color_eeg_helmet()
        return {"FINISHED"}


class ColorConnections(bpy.types.Operator):
    bl_idname = "mmvt.connections_color"
    bl_label = "mmvt connections color"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        color_connections()
        return {"FINISHED"}


class ColorStaticConnectionsDegree(bpy.types.Operator):
    bl_idname = "mmvt.connectivity_degree"
    bl_label = "mmvt connectivity_degree"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        color_connectivity_degree()
        return {"FINISHED"}


class ColorConnectionsLabelsAvg(bpy.types.Operator):
    bl_idname = "mmvt.connections_labels_avg"
    bl_label = "mmvt connections labels avg"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        color_connections_labels_avg()
        return {"FINISHED"}


class ColorMEGSensors(bpy.types.Operator):
    bl_idname = "mmvt.color_meg_sensors"
    bl_label = "mmvt color_meg_sensors"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        color_meg_sensors()
        return {"FINISHED"}


class ColorEEGSensors(bpy.types.Operator):
    bl_idname = "mmvt.color_eeg_sensors"
    bl_label = "mmvt color_eeg_sensors"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        color_eeg_sensors()
        return {"FINISHED"}


class ColorElectrodes(bpy.types.Operator):
    bl_idname = "mmvt.electrodes_color"
    bl_label = "mmvt electrodes color"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        color_electrodes()
        return {"FINISHED"}


class ColorElectrodesDists(bpy.types.Operator):
    bl_idname = "mmvt.electrodes_dists_color"
    bl_label = "mmvt electrodes dists color"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        color_electrodes_dists()
        return {"FINISHED"}


class ColorElectrodesLabels(bpy.types.Operator):
    bl_idname = "mmvt.electrodes_color_labels"
    bl_label = "mmvt electrodes color labels"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        color_electrodes_sources()
        return {"FINISHED"}


class ColorElectrodesStim(bpy.types.Operator):
    bl_idname = "mmvt.electrodes_color_stim"
    bl_label = "mmvt electrodes color stim"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        color_electrodes_stim()
        return {"FINISHED"}


class ColorManually(bpy.types.Operator):
    bl_idname = "mmvt.man_color"
    bl_label = "mmvt man color"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        color_manually()
        return {"FINISHED"}


class ColorVol(bpy.types.Operator):
    bl_idname = "mmvt.vol_color"
    bl_label = "mmvt vol color"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        color_volumetric()
        return {"FINISHED"}


class ColorGroupsManually(bpy.types.Operator):
    bl_idname = "mmvt.man_groups_color"
    bl_label = "mmvt man groups color"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        color_groups_manually()
        return {"FINISHED"}


class ColorfMRIDynamics(bpy.types.Operator):
    bl_idname = "mmvt.fmri_dynamics_color"
    bl_label = "mmvt fmri dynamics color"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        activity_map_coloring('FMRI_DYNAMICS')
        return {"FINISHED"}


class ColorMeg(bpy.types.Operator):
    bl_idname = "mmvt.meg_color"
    bl_label = "mmvt meg color"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        if ColoringMakerPanel.stc_file_chosen:
            plot_stc(ColoringMakerPanel.stc, bpy.context.scene.frame_current,
                     threshold=bpy.context.scene.coloring_lower_threshold, save_image=False)
        elif ColoringMakerPanel.activity_map_chosen:
            activity_map_coloring('MEG')
        else:
            print('ColorMeg: Both stc_file_chosen and activity_map_chosen are False!')
        return {"FINISHED"}


class ColorMegMax(bpy.types.Operator):
    bl_idname = "mmvt.meg_max_color"
    bl_label = "mmvt meg max color"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        if ColoringMakerPanel.stc_file_chosen:
            if ColoringMakerPanel.stc is None:
                ColoringMakerPanel.stc = mne.read_source_estimate(get_stc_full_fname())
            max_vert, bpy.context.scene.frame_current = ColoringMakerPanel.stc.get_peak(
                time_as_index=True, vert_as_index=True, mode=bpy.context.scene.meg_peak_mode)
            print(max_vert, bpy.context.scene.frame_current)
            plot_stc(ColoringMakerPanel.stc, bpy.context.scene.frame_current,
                     threshold=bpy.context.scene.coloring_lower_threshold, save_image=False)
        elif ColoringMakerPanel.activity_map_chosen:
            #todo: implement
            pass
            # activity_map_coloring('MEG')
        else:
            print('ColorMegMax: Both stc_file_chosen and activity_map_chosen are False!')
        return {"FINISHED"}

class ColorMegLabels(bpy.types.Operator):
    bl_idname = "mmvt.meg_labels_color"
    bl_label = "mmvt meg labels color"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        meg_labels_coloring()
        return {"FINISHED"}


class ColorfMRI(bpy.types.Operator):
    bl_idname = "mmvt.fmri_color"
    bl_label = "mmvt fmri color"
    bl_description = 'Plot the current fMRI activity map.\nScript: mmvt.plot_fmri()'
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        plot_fmri()
        return {"FINISHED"}


class ColorfMRILabels(bpy.types.Operator):
    bl_idname = "mmvt.fmri_labels_color"
    bl_label = "mmvt fmri labels_color"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        fmri_labels_coloring()
        return {"FINISHED"}


class ColorClustersFmri(bpy.types.Operator):
    bl_idname = "mmvt.fmri_clusters_color"
    bl_label = "mmvt fmri clusters color"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        activity_map_coloring('FMRI', clusters=True)
        return {"FINISHED"}


class ClearColors(bpy.types.Operator):
    bl_idname = "mmvt.colors_clear"
    bl_label = "mmvt colors clear"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        clear_colors()
        return {"FINISHED"}


class ChooseLabelFile(bpy.types.Operator, bpy_extras.io_utils.ExportHelper):
    bl_idname = "mmvt.plot_label_file"
    bl_label = "Plot label file"

    filename_ext = '.label'
    filter_glob = bpy.props.StringProperty(default='*.label', options={'HIDDEN'}, maxlen=255)

    def execute(self, context):
        label_fname = self.filepath
        plot_label(label_fname)
        return {'FINISHED'}


class PlotLabelsFolder(bpy.types.Operator):
    bl_idname = "mmvt.plot_labels_folder"
    bl_label = "mmvt plot labels folder"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        plot_labels_folder()
        return {'FINISHED'}


def plot_label(label, color=''):
    if isinstance(label, str):
        label = mu.read_label_file(label)
    else:
        try:
            _, _, _ = label.name, label.vertices, label.pos
        except:
            raise Exception('plot_label: label can be label fname or label object!')

    if bpy.context.scene.plot_label_contour:
        _addon().color_contours(specific_labels=[label.name])
    else:
        color = list(bpy.context.scene.labels_color) if color == '' else color
        ColoringMakerPanel.labels_plotted.append((label, color))
        _plot_labels()


def _plot_labels(labels_plotted_tuple=None, faces_verts=None):
    if labels_plotted_tuple is None:
        labels_plotted_tuple = ColoringMakerPanel.labels_plotted
    if faces_verts is None:
        faces_verts = ColoringMakerPanel.faces_verts
    hemi_verts_num = {hemi: faces_verts[hemi].shape[0] for hemi in mu.HEMIS}
    data = {hemi: np.zeros((hemi_verts_num[hemi], 4)) for hemi in mu.HEMIS}
    for label, color in labels_plotted_tuple:
        label.vertices = label.vertices[label.vertices < hemi_verts_num[label.hemi]]
        data[label.hemi][label.vertices] = [1, *color]
    for hemi in mu.HEMIS:
        color_hemi_data('inflated_{}'.format(hemi), data[hemi], threshold=0.5)
    _addon().show_activity()


def plot_labels(labels_names, colors, atlas, atlas_labels_rh=[], atlas_labels_lh=[], do_plot=True):
    if len(atlas_labels_rh) == 0 and len(atlas_labels_lh) == 0: # or atlas == bpy.context.scene.atlas:
        atlas_labels_rh = mu.read_labels_from_annots(atlas, hemi='rh')
        atlas_labels_lh = mu.read_labels_from_annots(atlas, hemi='lh')
    atlas_labels = atlas_labels_rh + atlas_labels_lh
    if len(atlas_labels) == 0:
        print("Couldn't find the atlas! ({})".format(atlas))
        return
    annot_verts_ok = check_annot_verts(atlas_labels_lh, atlas_labels_rh, atlas)
    if not annot_verts_ok:
        return
    org_delim, org_pos, label, label_hemi = mu.get_hemi_delim_and_pos(atlas_labels[0].name)
    labels_names_fix = []
    for label_name in labels_names:
        delim, pos, label, label_hemi = mu.get_hemi_delim_and_pos(label_name)
        label = mu.get_label_hemi_invariant_name(label_name)
        label_fix = mu.build_label_name(org_delim, org_pos, label, label_hemi)
        labels_names_fix.append(label_fix)
    labels = [l for l in atlas_labels if l.name in labels_names_fix]
    if len(labels) < len(labels_names):
        dump_fname = op.join(mu.get_user_fol(), 'logs', '{}_labels.txt'.format(atlas))
        print("Can't find all the labels ({}) in the {} atlas!".format(labels_names, atlas))
        print("Take a look here for the {} labels names: {}".format(atlas, dump_fname))
        with open(dump_fname, 'w') as output_file:
            output_file.write("Can't find all the labels ({}) in the {} atlas!\n".format(labels_names, atlas))
            output_file.write("Take a look here for the {} labels names:\n".format(atlas))
            for label in atlas_labels:
                output_file.write('{}\n'.format(label.name))
        import webbrowser
        webbrowser.open_new(dump_fname)
        return
    labels.sort(key=lambda x: labels_names_fix.index(x.name))
    # todo: check if bpy.context.scene.color_rois_homogeneously
    for label, color in zip(labels, colors):
        print('color {}: {}'.format(label, color))
        # plot_label(label, color)
        ColoringMakerPanel.labels_plotted.append((label, color))
    if do_plot:
        _plot_labels()


def check_annot_verts(atlas_labels_lh, atlas_labels_rh, atlas):
    annot_verts_num_lh = max([max(l.vertices) for l in atlas_labels_lh])
    annot_verts_num_rh = max([max(l.vertices) for l in atlas_labels_rh])
    hemi_verts_num_lh = len(bpy.data.objects['lh'].data.vertices)
    hemi_verts_num_rh = len(bpy.data.objects['rh'].data.vertices)
    if annot_verts_num_lh >= hemi_verts_num_lh or annot_verts_num_rh >= hemi_verts_num_rh:
        print('{} has wrong number of vertices!'.format(atlas))
        print('rh: annot {}, hemi {}'.format(annot_verts_num_lh, hemi_verts_num_lh))
        print('lh: annot {}, hemi {}'.format(annot_verts_num_rh, hemi_verts_num_rh))
        return False
    else:
        return True


def clear_colors():
    clear_cortex()
    clear_subcortical_fmri_activity()
    for root in ['Subcortical_meg_activity_map', 'Deep_electrodes', 'EEG_sensors', 'MEG_sensors']:
        clear_colors_from_parent_childrens(root)
    clear_connections()
    ColoringMakerPanel.what_is_colored = set()
    ColoringMakerPanel.labels_plotted = []


def clear_connections():
    vertices_obj = bpy.data.objects.get('connections_vertices')
    if vertices_obj:
        if any([obj.hide for obj in vertices_obj.children]):
            _addon().plot_connections(_addon().connections_data(), bpy.context.scene.frame_current, 0)
            _addon().filter_nodes(False)
            _addon().filter_nodes(True)


def get_fMRI_activity(hemi, clusters=False):
    if clusters:
        f = [c for c in ColoringMakerPanel.fMRI_clusters if c['hemi'] == hemi]
    else:
        f = ColoringMakerPanel.fMRI[hemi]
    return f
    # return ColoringMakerPanel.fMRI_clusters[hemi] if clusters else ColoringMakerPanel.fMRI[hemi]


def get_faces_verts():
    return ColoringMakerPanel.faces_verts


def read_groups_labels(colors):
    groups_fname = op.join(mu.get_parent_fol(mu.get_user_fol()), '{}_groups.csv'.format(bpy.context.scene.atlas))
    if not op.isfile(groups_fname):
        return {}
    groups = defaultdict(list) # OrderedDict() # defaultdict(list)
    color_ind = 0
    for line in mu.csv_file_reader(groups_fname):
        group_name = line[0]
        group_color = line[1]
        labels = line[2:]
        if group_name[0] == '#':
            continue
        groups[group_name] = []
        for label in labels:
            # group_color = cu.name_to_rgb(colors[color_ind])
            groups[group_name].append(dict(name=label, color=cu.name_to_rgb(group_color)))
            color_ind += 1
    order_groups = OrderedDict()
    groups_names = sorted(list(groups.keys()))
    for group_name in groups_names:
        order_groups[group_name] = groups[group_name]
    return order_groups


# def set_current_time_update(self=None, context=None):
#     bpy.data.scenes['Scene'].frame_current = context.scene.set_current_time


def set_current_time(t):
    if not isinstance(t, int):
        t = int(t)
    bpy.context.scene.frame_current = t


def get_current_time():
    return bpy.context.scene.frame_current


def draw(self, context):
    layout = self.layout
    user_fol = mu.get_user_fol()
    atlas = bpy.context.scene.atlas
    faces_verts_exist = mu.hemi_files_exists(op.join(user_fol, 'faces_verts_{hemi}.npy'))
    fmri_files = glob.glob(op.join(user_fol, 'fmri', 'fmri_*lh*.npy'))  # mu.hemi_files_exists(op.join(user_fol, 'fmri_{hemi}.npy'))
    # fmri_clusters_files_exist = mu.hemi_files_exists(op.join(user_fol, 'fmri', 'fmri_clusters_{hemi}.npy'))
    # meg_ext_meth = bpy.context.scene.meg_labels_extract_method
    # meg_labels_data_exist = mu.hemi_files_exists(op.join(user_fol, 'meg', 'labels_data*_{}_{}_{}.npz'.format(
    #     atlas, meg_ext_meth, '{hemi}')))
    # meg_labels_data_minmax_exist = op.isfile(
    #     op.join(user_fol, 'meg', 'meg_labels_{}_{}_minmax.npz'.format(atlas, meg_ext_meth)))
    manually_color_files_exist = len(glob.glob(op.join(user_fol, 'coloring', '*.csv'))) > 0
    static_connectivity_exist = len(glob.glob(op.join(user_fol, 'connectivity', '*.npy'))) > 0
    if _addon() is None:
        connections_files_exit = False
    else:
        connections_files_exit = _addon().connections_exist() and not _addon().connections_data() is None
    layout.prop(context.scene, 'coloring_lower_threshold', text="Threshold")
    # row = layout.row(align=True)
    # row.prop(context.scene, 'coloring_add_upper_threhold', text="Add upper threhold")
    # if context.scene.coloring_add_upper_threhold:
    #     layout.prop(context.scene, 'coloring_upper_threshold', text="Upper threshold")
    layout.prop(context.scene, "frame_current", text="Set time")
    layout.prop(context.scene, 'coloring_use_abs', text="Use abs")
    layout.prop(context.scene, 'color_rois_homogeneously', text="Color ROIs homogeneously")

    if faces_verts_exist:
        meg_current_activity_data_exist = mu.hemi_files_exists(
            op.join(user_fol, 'activity_map_{hemi}', 't{}.npy'.format(bpy.context.scene.frame_current)))
        if ColoringMakerPanel.meg_activity_data_exist and meg_current_activity_data_exist or \
                ColoringMakerPanel.stc_file_exist:
            col = layout.box().column()
            # mu.add_box_line(col, '', 'MEG', 0.4)
            col.prop(context.scene, 'meg_files', '')
            # col.label(text='T max: {}'.format(bpy.context.scene.meg_max_t))
            col.operator(ColorMeg.bl_idname, text="Plot MEG ", icon='POTATO')
            row = col.row(align=True)
            row.operator(ColorMegMax.bl_idname, text="Plot MEG peak", icon='POTATO')
            row.prop(context.scene, 'meg_peak_mode', '')
            row = col.row(align=True)
            row.prop(context.scene, 'meg_min_prec', 'min percentile')
            row.prop(context.scene, 'meg_max_prec', 'max percentile')
            if op.isfile(op.join(mu.get_user_fol(), 'subcortical_meg_activity.npz')):
                col.prop(context.scene, 'coloring_meg_subcorticals', text="Plot also subcorticals")
        if ColoringMakerPanel.meg_labels_data_exist and ColoringMakerPanel.meg_labels_data_minmax_exist:
            col = layout.box().column()
            # col.label('MEG labels')
            col.prop(context.scene, 'meg_labels_coloring_type', '')
            col.operator(ColorMegLabels.bl_idname, text="Plot MEG Labels ", icon='POTATO')
        if len(fmri_files) > 0 or ColoringMakerPanel.fmri_activity_map_exist or ColoringMakerPanel.fmri_labels_exist:
            col = layout.box().column()
            # col.label('fMRI')
            if len(fmri_files) > 0:
                col.prop(context.scene, "fmri_files", text="")
                col.operator(ColorfMRI.bl_idname, text="Plot fMRI file", icon='POTATO')
            if ColoringMakerPanel.fmri_activity_map_exist:
                col.operator(ColorfMRIDynamics.bl_idname, text="Plot fMRI Dynamics", icon='POTATO')
            if ColoringMakerPanel.fmri_labels_exist:
                col.operator(ColorfMRILabels.bl_idname, text="Plot fMRI Labels", icon='POTATO')

    if ColoringMakerPanel.meg_sensors_exist:
        col = layout.box().column()
        col.prop(context.scene, "meg_sensors_conditions", text="")
        col.operator(ColorMEGSensors.bl_idname, text="Plot MEG sensors", icon='POTATO')

    if ColoringMakerPanel.eeg_exist:
        col = layout.box().column()
        col.prop(context.scene, "eeg_sensors_conditions", text="")
        col.operator(ColorEEGSensors.bl_idname, text="Plot EEG sensors", icon='POTATO')
        # if not bpy.data.objects.get('eeg_helmet', None) is None:
        #     layout.operator(ColorEEGHelmet.bl_idname, text="Plot EEG Helmet", icon='POTATO')

    if ColoringMakerPanel.electrodes_files_exist:
        col = layout.box().column()
        col.prop(context.scene, "electrodes_conditions", text="")
        col.operator(ColorElectrodes.bl_idname, text="Plot Electrodes", icon='POTATO')
        # if ColoringMakerPanel.electrodes_dists_exist:
        #     col.operator(ColorElectrodesDists.bl_idname, text="Plot Electrodes Dists", icon='POTATO')
        if ColoringMakerPanel.electrodes_labels_files_exist:
            col.prop(context.scene, "electrodes_sources_files", text="")
            col.operator(ColorElectrodesLabels.bl_idname, text="Plot Electrodes Sources", icon='POTATO')
        if ColoringMakerPanel.electrodes_stim_files_exist:
            col.operator(ColorElectrodesStim.bl_idname, text="Plot Electrodes Stimulation", icon='POTATO')
    if connections_files_exit:
        col = layout.box().column()
        col.operator(ColorConnections.bl_idname, text="Plot Connections", icon='POTATO')
        col.prop(context.scene, 'hide_connection_under_threshold', text='Hide connections under threshold')
        if ColoringMakerPanel.conn_labels_avg_files_exit:
            col.prop(context.scene, 'conn_labels_avg_files', text='')
            col.operator(ColorConnectionsLabelsAvg.bl_idname, text="Plot Connections Labels Avg", icon='POTATO')
    if static_connectivity_exist:
        col = layout.box().column()
        col.prop(context.scene, 'static_conn_files', text='')
        col.prop(context.scene, 'connectivity_degree_threshold', text="Threshold")
        col.prop(context.scene, 'connectivity_degree_threshold_use_abs', text="Use connectivity absolute value")
        col.prop(context.scene, 'connectivity_degree_save_image', text="Save an image each update")
        col.operator(ColorStaticConnectionsDegree.bl_idname, text="Plot Connectivity Degree", icon='POTATO')

    if faces_verts_exist:
        if manually_color_files_exist:
            col = layout.box().column()
            # col.label('Manual coloring files')
            col.prop(context.scene, "coloring_files", text="")
            col.operator(ColorManually.bl_idname, text="Color Manually", icon='POTATO')
            layout.prop(context.scene, 'plot_label_contour', text='Plot labels as contour')
        if len(ColoringMakerPanel.labels_plotted) > 0:
            layout.prop(context.scene, 'show_labels_plotted', text='Show labels list')
            if bpy.context.scene.show_labels_plotted:
                layout.label(text='Labels plotted:')
                box = layout.box()
                col = box.column()
                for label, color in ColoringMakerPanel.labels_plotted:
                    mu.add_box_line(col, label.name, percentage=1)

    # layout.label(text="Choose labels' folder")
    # row = layout.row(align=True)
    # row.prop(context.scene, 'labels_folder', text='')
    # if bpy.context.scene.labels_folder != '':
    #     row.operator(PlotLabelsFolder.bl_idname, text='', icon='GAME')
    row = layout.row(align=True)
    row.operator(ChooseLabelFile.bl_idname, text="plot a label", icon='GAME').filepath = op.join(
        mu.get_user_fol(), '*.label')
    row.prop(context.scene, 'labels_color', text='')
    layout.operator(ClearColors.bl_idname, text="Clear", icon='PANEL_CLOSE')


bpy.types.Scene.hide_connection_under_threshold = bpy.props.BoolProperty(
    default=True, description="Hide connections under threshold")
bpy.types.Scene.meg_activitiy_type = bpy.props.EnumProperty(
    items=[('diff', 'Conditions difference', '', 0)], description="MEG activity type")
bpy.types.Scene.meg_peak_mode = bpy.props.EnumProperty(
    items=[('abs', 'Absolute values', '', 0), ('pos', 'Only positive', '', 1), ('neg', 'only negative', '', 2)])
bpy.types.Scene.meg_min_prec = bpy.props.FloatProperty(min=0, default=0, max=100, update=meg_minmax_prec_update)
bpy.types.Scene.meg_max_prec = bpy.props.FloatProperty(min=0, default=0, max=100, update=meg_minmax_prec_update)
bpy.types.Scene.meg_files = bpy.props.EnumProperty(items=[], description="MEG files")
bpy.types.Scene.meg_labels_coloring_type = bpy.props.EnumProperty(items=[], description="MEG labels coloring type")
bpy.types.Scene.coloring_fmri = bpy.props.BoolProperty(default=True, description="Plot FMRI")
bpy.types.Scene.coloring_electrodes = bpy.props.BoolProperty(default=False, description="Plot Deep electrodes")
bpy.types.Scene.coloring_lower_threshold = bpy.props.FloatProperty(default=0.5, min=0, description="")
bpy.types.Scene.coloring_use_abs = bpy.props.BoolProperty(default=True)
bpy.types.Scene.fmri_files = bpy.props.EnumProperty(items=[('', '', '', 0)], description="fMRI files")
bpy.types.Scene.stc_files = bpy.props.EnumProperty(items=[('', '', '', 0)], description="STC files")
bpy.types.Scene.meg_sensors_conditions= bpy.props.EnumProperty(items=[])
bpy.types.Scene.eeg_sensors_conditions= bpy.props.EnumProperty(items=[])
bpy.types.Scene.electrodes_conditions= bpy.props.EnumProperty(items=[])
bpy.types.Scene.meg_max_t = bpy.props.IntProperty(default=0, min=0, description="MEG max t")
bpy.types.Scene.electrodes_sources_files = bpy.props.EnumProperty(items=[], description="electrodes sources files")
bpy.types.Scene.coloring_files = bpy.props.EnumProperty(items=[], description="Coloring files")
bpy.types.Scene.vol_coloring_files = bpy.props.EnumProperty(items=[], description="Coloring volumetric files")
# bpy.types.Scene.coloring_both_pial_and_inflated = bpy.props.BoolProperty(default=False, description="")
bpy.types.Scene.color_rois_homogeneously = bpy.props.BoolProperty(default=False, description="")
bpy.types.Scene.coloring_meg_subcorticals = bpy.props.BoolProperty(default=False, description="")
bpy.types.Scene.conn_labels_avg_files = bpy.props.EnumProperty(items=[], description="Connectivity labels avg")
bpy.types.Scene.labels_color = bpy.props.FloatVectorProperty(
    name="labels_color", subtype='COLOR', default=(0, 0.5, 0), min=0.0, max=1.0, description="color picker")
bpy.types.Scene.static_conn_files = bpy.props.EnumProperty(items=[])
bpy.types.Scene.connectivity_degree_threshold = bpy.props.FloatProperty(
    default=0.7, min=0, max=1, description="", update=update_connectivity_degree_threshold)
bpy.types.Scene.connectivity_degree_threshold_use_abs = bpy.props.BoolProperty(default=False, description="")
bpy.types.Scene.connectivity_degree_save_image = bpy.props.BoolProperty(default=False, description="")
bpy.types.Scene.plot_label_contour = bpy.props.BoolProperty(default=False, description="")
bpy.types.Scene.show_labels_plotted = bpy.props.BoolProperty(default=True, description="Show labels list")
bpy.types.Scene.labels_folder = bpy.props.StringProperty(subtype='DIR_PATH')
# bpy.types.Scene.set_current_time = bpy.props.IntProperty(name="Current time:", min=0,
#                                                          max=bpy.data.scenes['Scene'].frame_preview_end,
#                                                          update=set_current_time_update)

class ColoringMakerPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "Activity Maps"
    addon = None
    init = False
    fMRI = {}
    fMRI = {}
    fMRI_clusters = {}
    labels_vertices = {}
    electrodes_sources_labels_data = None
    electrodes_sources_subcortical_data = None
    what_is_colored = set()
    fmri_activity_data_minmax, fmri_activity_colors_ratio = None, None
    meg_activity_data_minmax, meg_activity_colors_ratio = None, None
    meg_data_min, meg_data_max = 0, 0
    eeg_sensors_data_minmax, eeg_colors_ratio = None, None
    meg_sensors_data_minmax, meg_sensors_colors_ratio = None, None
    labels_plotted = []
    static_conn = None
    connectivity_labels = []
    activity_values = {hemi:[] for hemi in mu.HEMIS}
    prev_colors = {}
    stc = None
    stc_file_chosen = False
    activity_map_chosen = False
    stc_file_exist = False
    meg_activity_data_exist = False
    fmri_labels_exist = False
    fmri_subcorticals_mean_exist = False
    fmri_activity_map_exist = False
    eeg_exist = False
    meg_sensors_exist = False
    electrodes_files_exist = False
    electrodes_dists_exist = False
    electrodes_stim_files_exist = False
    electrodes_labels_files_exist = False
    conn_labels_avg_files_exit = False
    meg_labels_data_exist = False
    meg_labels_data_minmax_exist = False
    eeg_sensors_data, eeg_sensors_meta, eeg_sensors_data_minmax = None, None, None
    meg_sensors_data, meg_sensors_meta, meg_sensors_data_minmax = None, None, None
    no_plotting = False
    stc = None
    stc_exist = False
    curvs = {hemi:None for hemi in mu.HEMIS}
    activity_types = []
    run_meg_minmax_prec_update = True
    # activity_map_coloring = activity_map_coloring

    def draw(self, context):
        draw(self, context)


def panel_initialized():
    return ColoringMakerPanel.init


def get_no_plotting():
    return ColoringMakerPanel.no_plotting


def set_no_plotting(val):
    ColoringMakerPanel.no_plotting = val


# @mu.profileit('cumtime', op.join(mu.get_user_fol()))
def init(addon):
    ColoringMakerPanel.addon = addon
    ColoringMakerPanel.faces_verts = None
    ColoringMakerPanel.subs_faces_verts, ColoringMakerPanel.subs_verts = None, None

    register()
    init_meg_activity_map()
    init_fmri_activity_map()
    init_meg_labels_coloring_type()
    init_electrodes()
    init_fmri_files()
    init_fmri_labels()
    init_electrodes_sources()
    init_coloring_files()
    init_labels_groups()
    init_labels_vertices()
    init_eeg_sensors()
    init_meg_sensors()
    init_connectivity_labels_avg()
    init_static_conn()

    ColoringMakerPanel.faces_verts = load_faces_verts()
    bpy.context.scene.coloring_meg_subcorticals = False
    bpy.context.scene.meg_peak_mode = 'abs'
    bpy.context.scene.coloring_use_abs = True
    bpy.context.scene.color_rois_homogeneously = False
    set_meg_minmax_prec(1, 99, bpy.context.scene.meg_files)
    ColoringMakerPanel.init = True
    for hemi in ['lh', 'rh']:
        mesh = mu.get_hemi_obj(hemi).data
        if mesh.vertex_colors.get('Col') is None:
            mesh.vertex_colors.new('Col')
        if mesh.vertex_colors.get('contours') is None:
            mesh.vertex_colors.new('contours')
        curv_fname = op.join(mu.get_user_fol(), 'surf', '{}.curv.npy'.format(hemi))
        if op.isfile(curv_fname):
            ColoringMakerPanel.curvs[hemi] = np.load(curv_fname)
    if any([ColoringMakerPanel.curvs[hemi] is None for hemi in mu.HEMIS]):
        print('No curv.npy files! To create them run')
        print('python -m src.preproc.anatomy -s {} -a {} -f save_hemis_curv'.format(
            bpy.context.scene.atlas, mu.get_user()))


def init_labels_vertices():
    user_fol = mu.get_user_fol()
    labels_vertices_files = glob.glob(op.join(user_fol, 'labels_vertices_*.pkl'))
    for labels_vertices_fname in labels_vertices_files:
        atlas = mu.namebase(labels_vertices_fname)[len('labels_vertices_'):]
        if op.isfile(labels_vertices_fname):
            labels_names, labels_vertices = mu.load(labels_vertices_fname)
            ColoringMakerPanel.labels_vertices[atlas] = dict(labels_names=labels_names, labels_vertices=labels_vertices)


def init_meg_activity_map():
    user_fol = mu.get_user_fol()
    list_items = []
    activity_types = [mu.namebase(f)[len('activity_map_'):-2] for f in glob.glob(op.join(user_fol, 'activity_map_*rh'))]
    for activity_type in activity_types:
        if activity_type != '':
            activity_type = activity_type[:-1]
        meg_files_exist = len(glob.glob(op.join(user_fol, 'activity_map_{}rh'.format(activity_type), 't*.npy'))) > 0 and \
                          len(glob.glob(op.join(user_fol, 'activity_map_{}lh'.format(activity_type), 't*.npy'))) > 0
        meg_data_maxmin_fname = op.join(mu.get_user_fol(), 'meg_activity_map_{}minmax.pkl'.format(activity_type))
        if meg_files_exist and op.isfile(meg_data_maxmin_fname):
            data_min, data_max = mu.load(meg_data_maxmin_fname)
            ColoringMakerPanel.meg_activity_colors_ratio = 256 / (data_max - data_min)
            ColoringMakerPanel.meg_activity_data_minmax = (data_min, data_max)
            print('data meg: {}-{}'.format(data_min, data_max))
            if not _addon().colorbar_values_are_locked():
                _addon().set_colorbar_max_min(data_max, data_min, True)
                _addon().set_colorbar_title('MEG')
            ColoringMakerPanel.meg_activity_data_exist = True
            if activity_type == '':
                activity_type = 'conditions diff'
            list_items.append((activity_type, activity_type, '', len(list_items)))
    if MNE_EXIST:
        list_items = create_stc_files_list(list_items)
    else:
        print('No MNE installed in Blender. Run python -m src.setup -f install_blender_reqs')
    if len(list_items) > 0:
        bpy.types.Scene.meg_files = bpy.props.EnumProperty(
            items=list_items, description="MEG files", update=meg_files_update)
        bpy.context.scene.meg_files = list_items[0][0]
    if ColoringMakerPanel.meg_activity_data_exist or ColoringMakerPanel.stc_file_exist:
        ColoringMakerPanel.activity_types.append('meg')


def create_stc_files_list(list_items=[]):
    user_fol = mu.get_user_fol()
    stcs_files = glob.glob(op.join(user_fol, 'meg', '*.stc'))
    # todo: seperate between MEG and EEG stc files
    stcs_files += glob.glob(op.join(user_fol, 'eeg', '*.stc'))
    if len(stcs_files) > 0:
        ColoringMakerPanel.stc_exist = True
        stc_files_dic = defaultdict(list)
        for stc_file in stcs_files:
            _, _, label, hemi = mu.get_hemi_delim_and_pos(mu.namebase(stc_file))
            stc_files_dic[label].append(hemi)
        stc_names = [label for label, hemis in stc_files_dic.items() if len(hemis) == 2]
        if len(stc_names) > 0:
            ColoringMakerPanel.stc_file_exist = True
            list_items.extend([(c, c, '', ind + len(list_items)) for ind, c in enumerate(stc_names)])
        return list_items
    else:
        ColoringMakerPanel.stc_exist = False
        return []


def init_fmri_activity_map():
    user_fol = mu.get_user_fol()
    fmri_files_exist = mu.hemi_files_exists(op.join(user_fol, 'fmri', 'activity_map_{hemi}', 't0.npy'))
    fmri_data_maxmin_fname = op.join(user_fol, 'fmri', 'activity_map_minmax.npy')
    if fmri_files_exist and op.isfile(fmri_data_maxmin_fname):
        ColoringMakerPanel.fmri_activity_map_exist = True
        data_min, data_max = np.load(fmri_data_maxmin_fname)
        ColoringMakerPanel.fmri_activity_colors_ratio = 256 / (data_max - data_min)
        ColoringMakerPanel.fmri_activity_data_minmax = (data_min, data_max)
        if not _addon().colorbar_values_are_locked():
            _addon().set_colorbar_max_min(data_max, data_min, True)
            _addon().set_colorbar_title('fMRI')
        ColoringMakerPanel.activity_types.append('fmri')


def init_electrodes():
    user_fol = mu.get_user_fol()
    bip = 'bipolar_' if bpy.context.scene.bipolar else ''
    ColoringMakerPanel.electrodes_files_exist = \
        op.isfile(op.join(user_fol, 'electrodes', 'electrodes_{}data.npz'.format(bip))) or \
        op.isfile(op.join(user_fol, 'electrodes', 'electrodes_{}data.npy'.format(bip)))
    ColoringMakerPanel.electrodes_dists_exist = op.isfile(op.join(mu.get_user_fol(), 'electrodes', 'electrodes_dists.npy'))
    ColoringMakerPanel.electrodes_stim_files_exist = len(glob.glob(op.join(
        mu.get_user_fol(), 'electrodes', 'stim_electrodes_*.npz'))) > 0
    ColoringMakerPanel.electrodes_labels_files_exist = len(glob.glob(op.join(
        mu.get_user_fol(), 'electrodes', '*_labels_*.npz'))) > 0 and \
        len(glob.glob(op.join(mu.get_user_fol(), 'electrodes', '*_subcortical_*.npz'))) > 0
    if ColoringMakerPanel.electrodes_files_exist:
        ColoringMakerPanel.activity_types.append('elecs')
    if ColoringMakerPanel.electrodes_stim_files_exist:
        ColoringMakerPanel.activity_types.append('stim')
    if ColoringMakerPanel.electrodes_files_exist:
        _, _, conditions = _addon().load_electrodes_data()
        items = [(c, c, '', ind + 1) for ind, c in enumerate(conditions)]
        items.append(('diff', 'Conditions difference', '', len(conditions) +1))
        bpy.types.Scene.electrodes_conditions = bpy.props.EnumProperty(items=items, description="Electrodes")
        bpy.context.scene.electrodes_conditions = 'diff'



def init_meg_sensors():
    user_fol = mu.get_user_fol()
    meg_data_files = glob.glob(op.join(mu.get_user_fol(), 'meg', '*sensors_evoked_data.npy'))
    if len(meg_data_files) == 0:
        return
    # todo: should be according to the data panel
    meg_data_files = mu.namebase(meg_data_files[0])
    meg_sensors_data_fname = op.join(user_fol, 'meg', '{}.npy'.format(meg_data_files))
    meg_sensors_meta_data_fname = op.join(user_fol, 'meg', '{}_meta.npz'.format(meg_data_files))
    meg_sensors_data_minmax_fname = op.join(user_fol, 'meg', '{}_minmax.npy'.format(meg_data_files[:-5]))
    if all([op.isfile(f) for f in [
            meg_sensors_data_fname, meg_sensors_meta_data_fname, meg_sensors_data_minmax_fname]]) and \
            bpy.data.objects.get('MEG_sensors') is not None:
        ColoringMakerPanel.meg_sensors_exist = True
        ColoringMakerPanel.meg_sensors_data, ColoringMakerPanel.meg_sensors_meta, = \
            _addon().load_meg_sensors_data(meg_sensors_data_fname, meg_sensors_meta_data_fname)
        data_min, data_max = np.load(meg_sensors_data_minmax_fname)
        ColoringMakerPanel.meg_sensors_colors_ratio = 256 / (data_max - data_min)
        ColoringMakerPanel.meg_sensors_data_minmax = (data_min, data_max)
        items = [(c, c, '', ind + 1) for ind, c in enumerate(ColoringMakerPanel.meg_sensors_meta['conditions'])]
        items.append(('diff', 'Conditions difference', '', len(ColoringMakerPanel.meg_sensors_meta['conditions']) +1))
        bpy.types.Scene.meg_sensors_conditions = bpy.props.EnumProperty(items=items, description="MEG sensors")
        bpy.context.scene.meg_sensors_conditions = 'diff'
        ColoringMakerPanel.activity_types.append('meg_sensors')


def init_eeg_sensors():
    user_fol = mu.get_user_fol()
    eeg_data_files = glob.glob(op.join(mu.get_user_fol(), 'eeg', '*sensors_evoked_data.npy'))
    if len(eeg_data_files) == 0:
        return
    # todo: should be according to the data panel
    eeg_data_files = mu.namebase(eeg_data_files[0])
    eeg_sensors_data_fname = op.join(user_fol, 'eeg', '{}.npy'.format(eeg_data_files))
    eeg_sensors_meta_data_fname = op.join(user_fol, 'eeg', '{}_meta.npz'.format(eeg_data_files))
    eeg_sensors_data_minmax_fname = op.join(user_fol, 'eeg', '{}_minmax.npy'.format(eeg_data_files[:-5]))
    if all([op.isfile(f) for f in
            [eeg_sensors_data_fname, eeg_sensors_meta_data_fname, eeg_sensors_data_minmax_fname]]) and \
            bpy.data.objects.get('EEG_sensors', None) is  not None:
        ColoringMakerPanel.eeg_exist = True
        ColoringMakerPanel.eeg_sensors_data, ColoringMakerPanel.eeg_sensors_meta, = \
            _addon().load_eeg_sensors_data(eeg_sensors_data_fname, eeg_sensors_meta_data_fname)
        data_min, data_max = np.load(eeg_sensors_data_minmax_fname)
        ColoringMakerPanel.eeg_colors_ratio = 256 / (data_max - data_min)
        ColoringMakerPanel.eeg_sensors_data_minmax = (data_min, data_max)
        items = [(c, c, '', ind + 1) for ind, c in enumerate(ColoringMakerPanel.eeg_sensors_meta['conditions'])]
        items.append(('diff', 'Conditions difference', '', len(ColoringMakerPanel.eeg_sensors_meta['conditions']) +1))
        bpy.types.Scene.eeg_sensors_conditions = bpy.props.EnumProperty(items=items, description="EEG sensors")
        bpy.context.scene.eeg_sensors_conditions = 'diff'
        ColoringMakerPanel.activity_types.append('eeg_sensors')


def init_meg_labels_coloring_type():
    user_fol = mu.get_user_fol()
    atlas = bpy.context.scene.atlas
    conditions = get_condditions_from_labels_fcurves()
    # em = bpy.context.scene.meg_labels_extract_method
    files = glob.glob(op.join(user_fol, 'meg', 'labels_data_{}_*_rh.npz'.format(atlas)))
    if len(files) > 0:
        em = '_'.join(mu.namebase(files[0]).split('_')[3:-1])
        ColoringMakerPanel.meg_labels_data_exist = mu.hemi_files_exists(
            op.join(user_fol, 'meg', 'labels_data_{}_{}_{}.npz'.format(atlas, em, '{hemi}')))
        ColoringMakerPanel.meg_labels_data_minmax_exist = op.isfile(
            op.join(user_fol, 'meg', 'labels_data_{}_{}_minmax.npz'.format(atlas, em)))
        if len(conditions) > 0 and ColoringMakerPanel.meg_labels_data_exist and \
                ColoringMakerPanel.meg_labels_data_minmax_exist:
            items = [('diff', 'Conditions differece', '', 0)]
            items.extend([(cond, cond, '', ind + 1) for ind, cond in enumerate(conditions)])
            bpy.types.Scene.meg_labels_coloring_type = bpy.props.EnumProperty(
                items=items, description="meg_labels_coloring_type")
            bpy.context.scene.meg_labels_coloring_type = 'diff'
            ColoringMakerPanel.activity_types.append('meg_labels')


def init_fmri_labels():
    user_fol = mu.get_user_fol()
    atlas = bpy.context.scene.atlas
    em = bpy.context.scene.fmri_labels_extract_method
    fmri_labels_data_exist = mu.hemi_files_exists(
        op.join(user_fol, 'fmri', 'labels_data_{}_{}_{}.npz'.format(atlas, em, '{hemi}')))
    fmri_labels_data_minmax_exist = op.isfile(
        op.join(user_fol, 'fmri', 'labels_data_{}_{}_minmax.pkl'.format(atlas, em)))
    ColoringMakerPanel.fmri_labels_exist = fmri_labels_data_exist and fmri_labels_data_minmax_exist
    ColoringMakerPanel.fmri_subcorticals_mean_exist = op.isfile(
        op.join(user_fol, 'fmri', 'subcorticals_{}.npz'.format(em)))
    if ColoringMakerPanel.subs_faces_verts is None or ColoringMakerPanel.subs_verts is None:
        ColoringMakerPanel.subs_faces_verts, ColoringMakerPanel.subs_verts = load_subs_faces_verts()
    if ColoringMakerPanel.fmri_labels_exist:
        ColoringMakerPanel.activity_types.append('fmri_labels')


def init_fmri_files(current_fmri_file=''):
    user_fol = mu.get_user_fol()
    fmri_files = glob.glob(op.join(user_fol, 'fmri', 'fmri_*lh*.npy'))
    if len(fmri_files) > 0:
        files_names = [mu.get_label_hemi_invariant_name(mu.namebase(fname)[5:]) for fname in fmri_files]
        clusters_items = [(c, c, '', ind) for ind, c in enumerate(files_names)]
        bpy.types.Scene.fmri_files = bpy.props.EnumProperty(
            items=clusters_items, description="fMRI files", update=fmri_files_update)
        if current_fmri_file == '':
            bpy.context.scene.fmri_files = files_names[0]
        else:
            bpy.context.scene.fmri_files = current_fmri_file


def init_static_conn():
    user_fol = mu.get_user_fol()
    conn_labels_names_fname = op.join(mu.get_user_fol(), 'connectivity', 'labels_names.npy')
    if op.isfile(conn_labels_names_fname):
        ColoringMakerPanel.connectivity_labels = np.load(conn_labels_names_fname)
    static_conn_files = glob.glob(op.join(user_fol, 'connectivity', '*mean*.npy')) + \
                        glob.glob(op.join(user_fol, 'connectivity', '*statics*.npy'))
    if len(static_conn_files) > 0:
        files_names = [mu.namebase(fname) for fname in static_conn_files]
        files_names = [name for name in files_names if 'backup' not in name]
        items = [(c, c, '', ind) for ind, c in enumerate(files_names)]
        bpy.types.Scene.static_conn_files = bpy.props.EnumProperty(
            items=items, description="static conn files", update=static_conn_files_update)
        bpy.context.scene.static_conn_files = files_names[0]


def init_electrodes_sources():
    user_fol = mu.get_user_fol()
    electrodes_source_files = glob.glob(op.join(user_fol, 'electrodes', '*_labels_*-rh.npz'))
    if len(electrodes_source_files) > 0:
        files_names = [mu.namebase(fname)[:-len('-rh')] for fname in electrodes_source_files]
        items = [(c, c, '', ind) for ind, c in enumerate(files_names)]
        bpy.types.Scene.electrodes_sources_files = bpy.props.EnumProperty(
            items=items, description="electrodes sources", update=electrodes_sources_files_update)
        bpy.context.scene.electrodes_sources_files = files_names[0]


def init_coloring_files():
    user_fol = mu.get_user_fol()
    mu.make_dir(op.join(user_fol, 'coloring'))
    manually_color_files = glob.glob(op.join(user_fol, 'coloring', '*.csv'))
    if len(manually_color_files) > 0:
        files_names = [mu.namebase(fname) for fname in manually_color_files]
        coloring_items = [(c, c, '', ind) for ind, c in enumerate(files_names)]
        bpy.types.Scene.coloring_files = bpy.props.EnumProperty(items=coloring_items, description="Coloring files")
        bpy.context.scene.coloring_files = files_names[0]


def init_colume_coloring_files():
    user_fol = mu.get_user_fol()
    vol_color_files = glob.glob(op.join(user_fol, 'coloring', 'volumetric', '*.csv'))
    if len(vol_color_files) > 0:
        files_names = [mu.namebase(fname) for fname in vol_color_files]
        coloring_items = [(c, c, '', ind) for ind, c in enumerate(files_names)]
        bpy.types.Scene.vol_coloring_files = bpy.props.EnumProperty(
            items=coloring_items, description="Volumetric Coloring files")
        bpy.context.scene.vol_coloring_files = files_names[0]


def init_labels_groups():
    from random import shuffle
    ColoringMakerPanel.colors = list(set(list(cu.NAMES_TO_HEX.keys())) - set(['black']))
    shuffle(ColoringMakerPanel.colors)
    ColoringMakerPanel.labels_groups = read_groups_labels(ColoringMakerPanel.colors)
    if len(ColoringMakerPanel.labels_groups) > 0:
        groups_items = [(gr, gr, '', ind) for ind, gr in enumerate(list(ColoringMakerPanel.labels_groups.keys()))]
        bpy.types.Scene.labels_groups = bpy.props.EnumProperty(
            items=groups_items, description="Groups")


def init_connectivity_labels_avg():
    user_fol = mu.get_user_fol()
    conn_labels_avg_files = glob.glob(op.join(user_fol, 'connectivity', '*_labels_avg.npz'))
    atlas = bpy.context.scene.atlas
    if len(conn_labels_avg_files) > 0:
        files_names = [mu.namebase(fname).replace('_', ' ').replace('{} '.format(atlas), '') for fname in conn_labels_avg_files]
        items = [(c, c, '', ind) for ind, c in enumerate(files_names)]
        bpy.types.Scene.conn_labels_avg_files = bpy.props.EnumProperty(
            items=items, description="Connectivity labels avg")
        bpy.context.scene.conn_labels_avg_files = files_names[0]
        ColoringMakerPanel.conn_labels_avg_files_exit = True
        ColoringMakerPanel.activity_types.append('labels_connectivity')


def register():
    try:
        bpy.utils.register_class(ColorElectrodes)
        bpy.utils.register_class(ColorElectrodesDists)
        bpy.utils.register_class(ColorElectrodesStim)
        bpy.utils.register_class(ColorElectrodesLabels)
        bpy.utils.register_class(ColorManually)
        bpy.utils.register_class(ColorVol)
        bpy.utils.register_class(ColorGroupsManually)
        bpy.utils.register_class(ColorMeg)
        bpy.utils.register_class(ColorMegMax)
        bpy.utils.register_class(ColorMegLabels)
        bpy.utils.register_class(ColorfMRI)
        bpy.utils.register_class(ColorfMRILabels)
        bpy.utils.register_class(ColorfMRIDynamics)
        bpy.utils.register_class(ColorClustersFmri)
        bpy.utils.register_class(ColorMEGSensors)
        bpy.utils.register_class(ColorEEGSensors)
        bpy.utils.register_class(ColorEEGHelmet)
        bpy.utils.register_class(ColorConnections)
        bpy.utils.register_class(ColorConnectionsLabelsAvg)
        bpy.utils.register_class(ClearColors)
        bpy.utils.register_class(ColoringMakerPanel)
        bpy.utils.register_class(ChooseLabelFile)
        bpy.utils.register_class(PlotLabelsFolder)
        bpy.utils.register_class(ColorStaticConnectionsDegree)
        # print('Freeview Panel was registered!')
    except:
        print("Can't register Freeview Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(ColorElectrodes)
        bpy.utils.unregister_class(ColorElectrodesDists)
        bpy.utils.unregister_class(ColorElectrodesStim)
        bpy.utils.unregister_class(ColorElectrodesLabels)
        bpy.utils.unregister_class(ColorManually)
        bpy.utils.unregister_class(ColorVol)
        bpy.utils.unregister_class(ColorGroupsManually)
        bpy.utils.unregister_class(ColorMeg)
        bpy.utils.unregister_class(ColorMegMax)
        bpy.utils.unregister_class(ColorMegLabels)
        bpy.utils.unregister_class(ColorfMRI)
        bpy.utils.unregister_class(ColorfMRILabels)
        bpy.utils.unregister_class(ColorfMRIDynamics)
        bpy.utils.unregister_class(ColorClustersFmri)
        bpy.utils.unregister_class(ColorMEGSensors)
        bpy.utils.unregister_class(ColorEEGSensors)
        bpy.utils.unregister_class(ColorEEGHelmet)
        bpy.utils.unregister_class(ColorConnections)
        bpy.utils.unregister_class(ColorConnectionsLabelsAvg)
        bpy.utils.unregister_class(ClearColors)
        bpy.utils.unregister_class(ColoringMakerPanel)
        bpy.utils.unregister_class(ChooseLabelFile)
        bpy.utils.unregister_class(PlotLabelsFolder)
        bpy.utils.unregister_class(ColorStaticConnectionsDegree)
    except:
        pass
        # print("Can't unregister Freeview Panel!")


if __name__ == '__main__':
    import mne
    stc = mne.read_source_estimate('/homes/5/npeled/space1/mmvt/fsaverage5/meg/fsaverage5_msit_nTSSS_interference_diff_1-15-dSPM-lh.stc')
    plot_stc(stc, 100, threshold=1, save_image=False, view_selected=False, subject='fsaverage5', n_jobs=4)
