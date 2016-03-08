import bpy
import mmvt_utils as mu
import colors_utils as cu
import numpy as np
import os.path as op
import time
import itertools
from collections import defaultdict
import glob

HEMIS = mu.HEMIS


def object_coloring(obj, rgb):
    bpy.context.scene.objects.active = obj
    # todo: do we need to select the object here? In diff mode it's a problem
    # obj.select = True
    cur_mat = obj.active_material
    new_color = (rgb[0], rgb[1], rgb[2], 1)
    cur_mat.node_tree.nodes["RGB"].outputs[0].default_value = new_color


def clear_subcortical_fmri_activity():
    for cur_obj in bpy.data.objects['Subcortical_fmri_activity_map'].children:
        clear_object_vertex_colors(cur_obj)


def clear_cortex(hemis=HEMIS):
    for hemisphere in hemis:
        cur_obj = bpy.data.objects[hemisphere]
        clear_object_vertex_colors(cur_obj)


def clear_object_vertex_colors(cur_obj):
    mesh = cur_obj.data
    scn = bpy.context.scene
    scn.objects.active = cur_obj
    cur_obj.select = True
    bpy.ops.mesh.vertex_color_remove()
    vcol_layer = mesh.vertex_colors.new()


# todo: do something with the threshold parameter
def color_object_homogeneously(data, postfix_str='', threshold=0):
    if data is None:
        print('color_object_homogeneously: No data to color!')
        return

    default_color = (1, 1, 1)
    cur_frame = bpy.context.scene.frame_current
    for obj_name, object_colors, values in zip(data['names'], data['colors'], data['data']):
        obj_name = obj_name.astype(str)
        value = np.diff(values[cur_frame])[0]
        # todo: there is a difference between value and real_value, what should we do?
        # real_value = mu.get_fcurve_current_frame_val('Deep_electrodes', obj_name, cur_frame)
        new_color = object_colors[cur_frame] if abs(value) > threshold else default_color
        # todo: check if the stat should be avg or diff
        obj = bpy.data.objects.get(obj_name+postfix_str)
        if obj and not obj.hide:
            # print('trying to color {} with {}'.format(obj_name+postfix_str, new_color))
            object_coloring(obj, new_color)
            print(obj_name, value, new_color)
        # else:
        #     print('color_object_homogeneously: {} was not loaded!'.format(obj_name))

    print('Finished coloring!!')


def init_activity_map_coloring(map_type):
    ColoringMakerPanel.addon.set_appearance_show_activity_layer(bpy.context.scene, True)
    ColoringMakerPanel.addon.set_filter_view_type(bpy.context.scene, 'RENDERS')
    ColoringMakerPanel.addon.show_hide_hierarchy(map_type != 'FMRI', 'Subcortical_fmri_activity_map')
    ColoringMakerPanel.addon.show_hide_hierarchy(map_type != 'MEG', 'Subcortical_meg_activity_map')
    # change_view3d()
    faces_verts = load_faces_verts()
    return faces_verts


def load_faces_verts():
    faces_verts = {}
    current_root_path = mu.get_user_fol()
    faces_verts['lh'] = np.load(op.join(current_root_path, 'faces_verts_lh.npy'))
    faces_verts['rh'] = np.load(op.join(current_root_path, 'faces_verts_rh.npy'))
    return faces_verts


def load_meg_subcortical_activity():
    meg_sub_activity = None
    current_root_path = mu.get_user_fol() # bpy.path.abspath(bpy.context.scene.conf_path)
    subcortical_activity_file = op.join(current_root_path,'subcortical_meg_activity.npz')
    if op.isfile(subcortical_activity_file):
        meg_sub_activity = np.load(subcortical_activity_file)
    return meg_sub_activity


def activity_map_coloring(map_type):
    faces_verts = init_activity_map_coloring(map_type)
    threshold = bpy.context.scene.coloring_threshold
    meg_sub_activity = None
    if map_type == 'MEG':
        meg_sub_activity = load_meg_subcortical_activity()
    plot_activity(map_type, faces_verts, threshold, meg_sub_activity)
    # setup_environment_settings()


def meg_labels_coloring(self, context, override_current_mat=True):
    faces_verts = init_activity_map_coloring('MEG')
    threshold = bpy.context.scene.coloring_threshold
    hemispheres = [hemi for hemi in HEMIS if not bpy.data.objects[hemi].hide]
    user_fol = mu.get_user_fol()
    for hemi_ind, hemi in enumerate(hemispheres):
        labels_names, labels_vertices = mu.load(op.join(user_fol, 'labels_vertices_{}.pkl'.format(bpy.context.scene.atlas)))
        labels_data = np.load(op.join(user_fol, 'meg_labels_coloring_{}.npz'.format(hemi)))
        meg_labels_coloring_hemi(labels_names, labels_vertices, labels_data, faces_verts, hemi, threshold, override_current_mat)


def meg_labels_coloring_hemi(labels_names, labels_vertices, labels_data, faces_verts, hemi, threshold, override_current_mat=True):
    now = time.time()
    vertices_num = max(itertools.chain.from_iterable(labels_vertices[hemi])) + 1
    colors_data = np.ones((vertices_num, 4))
    colors_data[:, 0] = 0
    no_t = labels_data['data'][0].ndim == 0
    t = bpy.context.scene.frame_current
    for label_data, label_colors, label_name in zip(labels_data['data'], labels_data['colors'], labels_data['names']):
        if 'unknown' in label_name:
            continue
        label_index = labels_names[hemi].index(label_name)
        label_vertices = np.array(labels_vertices[hemi][label_index])
        if len(label_vertices) > 0:
            label_data_t, label_colors_t = (label_data, label_colors) if no_t else (label_data[t], label_colors[t])
            # print('coloring {} with {}'.format(label_name, label_colors_t))
            label_colors_data = np.hstack((label_data_t, label_colors_t))
            label_colors_data = np.tile(label_colors_data, (len(label_vertices), 1))
            colors_data[label_vertices, :] = label_colors_data
    cur_obj = bpy.data.objects[hemi]
    activity_map_obj_coloring(cur_obj, colors_data, faces_verts[hemi], threshold, override_current_mat)
    print('Finish meg_labels_coloring_hemi, hemi {}, {:.2f}s'.format(hemi, time.time()-now))


def plot_activity(map_type, faces_verts, threshold, meg_sub_activity=None,
        plot_subcorticals=True, override_current_mat=True):
    current_root_path = mu.get_user_fol() # bpy.path.abspath(bpy.context.scene.conf_path)
    hemispheres = [hemi for hemi in HEMIS if not bpy.data.objects[hemi].hide]
    frame_str = str(bpy.context.scene.frame_current)

    # loop_indices = {}
    for hemi in hemispheres:
        if map_type == 'MEG':
            f = np.load(op.join(current_root_path, 'activity_map_' + hemi, 't' + frame_str + '.npy'))
        elif map_type == 'FMRI':
            f = np.load(op.join(current_root_path, 'fmri_' + hemi + '.npy'))
        cur_obj = bpy.data.objects[hemi]
        # loop_indices[hemi] =
        activity_map_obj_coloring(cur_obj, f, faces_verts[hemi], threshold, override_current_mat)

    if plot_subcorticals:
        if map_type == 'MEG':
            if not bpy.data.objects['Subcortical_meg_activity_map'].hide:
                color_object_homogeneously(meg_sub_activity, '_meg_activity', threshold)
        if map_type == 'FMRI':
            fmri_subcortex_activity_color(threshold, override_current_mat)

    # return loop_indices
    # Noam: not sure this is necessary
    #deselect_all()
    #bpy.data.objects['Brain'].select = True


def fmri_subcortex_activity_color(threshold, override_current_mat=True):
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
                activity_map_obj_coloring(cur_obj, verts_values, lookup, threshold, override_current_mat)


def activity_map_obj_coloring(cur_obj, vert_values, lookup, threshold, override_current_mat):
    mesh = cur_obj.data
    scn = bpy.context.scene

    valid_verts = np.where(np.abs(vert_values[:,0])>threshold)[0]
    #check if our mesh already has Vertex Colors, and if not add some... (first we need to make sure it's the active object)
    scn.objects.active = cur_obj
    cur_obj.select = True
    if override_current_mat:
        bpy.ops.mesh.vertex_color_remove()
    vcol_layer = mesh.vertex_colors.new()
    # else:
    #     vcol_layer = mesh.vertex_colors.active
        # loop_indices = set()
    print('cur_obj: {}, max vert in lookup: {}, vcol_layer len: {}'.format(cur_obj.name, np.max(lookup), len(vcol_layer.data)))
    for vert in valid_verts:
        x = lookup[vert]
        for loop_ind in x[x>-1]:
            d = vcol_layer.data[loop_ind]
            d.color = vert_values[vert, 1:]


def color_manually():
    ColoringMakerPanel.addon.show_hide_hierarchy(do_hide=False, obj='Subcortical_meg_activity_map')
    ColoringMakerPanel.addon.show_hide_hierarchy(do_hide=True, obj='Subcortical_fmri_activity_map')
    subject_fol = mu.get_user_fol()
    labels_names, labels_vertices = mu.load(
        op.join(subject_fol, 'labels_vertices_{}.pkl'.format(bpy.context.scene.atlas)))
    faces_verts = load_faces_verts()
    objects_names, colors, data = defaultdict(list), defaultdict(list), defaultdict(list)
    for line in mu.csv_file_reader(op.join(subject_fol, 'coloring.csv')):
        obj_name, color_name = line
        obj_type = mu.check_obj_type(obj_name)
        color_rgb = cu.name_to_rgb(color_name)
        if obj_type is not None:
            objects_names[obj_type].append(obj_name)
            colors[obj_type].append(color_rgb)
            data[obj_type].append(1.)
    # coloring
    for hemi in HEMIS:
        obj_type = mu.OBJ_TYPE_CORTEX_LH if hemi=='lh' else mu.OBJ_TYPE_CORTEX_RH
        labels_data = dict(data=np.array(data[obj_type]), colors=colors[obj_type], names=objects_names[obj_type])
        meg_labels_coloring_hemi(labels_names, labels_vertices, labels_data, faces_verts, hemi, 0)
    for region, color in zip(objects_names[mu.OBJ_TYPE_SUBCORTEX], colors[mu.OBJ_TYPE_SUBCORTEX]):
        color_subcortical_region(region, color)


def color_subcortical_region(region_name, rgb):
    obj = bpy.data.objects.get(region_name + '_meg_activity', None)
    if not obj is None:
        object_coloring(obj, rgb)


def clear_subcortical_regions():
    clear_colors_from_parent_childrens('Subcortical_meg_activity_map')


def clear_colors_from_parent_childrens(parent_object):
    parent_obj = bpy.data.objects.get(parent_object)
    if parent_obj is not None:
        for obj in parent_obj.children:
            if 'RGB' in obj.active_material.node_tree.nodes:
                obj.active_material.node_tree.nodes['RGB'].outputs['Color'].default_value=(1, 1, 1, 1)


def default_coloring(loop_indices):
    for hemi, indices in loop_indices.items():
        cur_obj = bpy.data.objects[hemi]
        mesh = cur_obj.data
        vcol_layer = mesh.vertex_colors.active
        for loop_ind in indices:
            vcol_layer.data[loop_ind].color = [1, 1, 1]


class ColorElectrodes(bpy.types.Operator):
    bl_idname = "ohad.electrodes_color"
    bl_label = "ohad electrodes color"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        threshold = bpy.context.scene.coloring_threshold
        data = np.load(op.join(mu.get_user_fol(),'electrodes_data_{}.npz'.format(
            'avg' if bpy.context.scene.selection_type == 'conds' else 'diff')))
        color_object_homogeneously(data, threshold=threshold)
        # deselect_all()
        # mu.select_hierarchy('Deep_electrodes', False)
        ColoringMakerPanel.addon.set_appearance_show_electrodes_layer(bpy.context.scene, True)
        ColoringMakerPanel.addon.change_to_rendered_brain()
        return {"FINISHED"}


class ColorManually(bpy.types.Operator):
    bl_idname = "ohad.man_color"
    bl_label = "ohad man color"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        color_manually()
        return {"FINISHED"}


class ColorMeg(bpy.types.Operator):
    bl_idname = "ohad.meg_color"
    bl_label = "ohad meg color"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        activity_map_coloring('MEG')
        return {"FINISHED"}


class ColorMegLabels(bpy.types.Operator):
    bl_idname = "ohad.meg_labels_color"
    bl_label = "ohad meg labels color"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        # todo: should send also aparc_name
        meg_labels_coloring(self, context)
        return {"FINISHED"}


class ColorFmri(bpy.types.Operator):
    bl_idname = "ohad.fmri_color"
    bl_label = "ohad fmri color"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        activity_map_coloring('FMRI')
        return {"FINISHED"}


class ClearColors(bpy.types.Operator):
    bl_idname = "ohad.colors_clear"
    bl_label = "ohad colors clear"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        clear_cortex()
        clear_subcortical_fmri_activity()
        for root in ['Subcortical_meg_activity_map', 'Deep_electrodes']:
            clear_colors_from_parent_childrens(root)
        return {"FINISHED"}


bpy.types.Scene.coloring_fmri = bpy.props.BoolProperty(default=True, description="Plot FMRI")
bpy.types.Scene.coloring_electrodes = bpy.props.BoolProperty(default=False, description="Plot Deep electrodes")
bpy.types.Scene.coloring_threshold = bpy.props.FloatProperty(default=0.5, min=0, description="")


class ColoringMakerPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
    bl_label = "Activity Maps"
    addon = None

    def draw(self, context):
        layout = self.layout
        user_fol = mu.get_user_fol()
        aparc_name = bpy.context.scene.atlas
        faces_verts_exist = mu.hemi_files_exists(op.join(user_fol, 'faces_verts_{hemi}.npy'))
        fmri_files_exist = mu.hemi_files_exists(op.join(user_fol, 'fmri_{hemi}.npy'))
        meg_files_exist = mu.hemi_files_exists(op.join(user_fol, 'activity_map_{hemi}', 't0.npy'))
        meg_labels_files_exist = op.isfile(op.join(user_fol, 'labels_vertices_{}.pkl'.format(aparc_name))) and \
            mu.hemi_files_exists(op.join(user_fol, 'meg_labels_coloring_{hemi}.npz'))
        electrodes_files_exist = op.isfile(op.join(mu.get_user_fol(),'electrodes_data_{}.npz'.format(
            'avg' if bpy.context.scene.selection_type == 'conds' else 'diff')))
        manually_color_file_exist = op.isfile(op.join(user_fol, 'coloring.csv'))
        layout.prop(context.scene, 'coloring_threshold', text="Threshold")
        layout.operator(ColorMeg.bl_idname, text="Plot MEG ", icon='POTATO')
        if faces_verts_exist:
            if meg_files_exist:
                # layout.operator(ColorMeg.bl_idname, text="Plot MEG ", icon='POTATO')
                pass
            if meg_labels_files_exist:
                layout.operator(ColorMegLabels.bl_idname, text="Plot MEG Labels ", icon='POTATO')
            if fmri_files_exist:
                layout.operator(ColorFmri.bl_idname, text="Plot FMRI ", icon='POTATO')
            if manually_color_file_exist:
                layout.operator(ColorManually.bl_idname, text="Color Manually", icon='POTATO')
        if electrodes_files_exist:
            layout.operator(ColorElectrodes.bl_idname, text="Plot Electrodes ", icon='POTATO')
        layout.operator(ClearColors.bl_idname, text="Clear", icon='PANEL_CLOSE')


def init(addon):
    ColoringMakerPanel.addon = addon
    register()


def register():
    try:
        unregister()
        bpy.utils.register_class(ColorElectrodes)
        bpy.utils.register_class(ColorManually)
        bpy.utils.register_class(ColorMeg)
        bpy.utils.register_class(ColorMegLabels)
        bpy.utils.register_class(ColorFmri)
        bpy.utils.register_class(ClearColors)
        bpy.utils.register_class(ColoringMakerPanel)
        print('Freeview Panel was registered!')
    except:
        print("Can't register Freeview Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(ColorElectrodes)
        bpy.utils.unregister_class(ColorManually)
        bpy.utils.unregister_class(ColorMeg)
        bpy.utils.unregister_class(ColorMegLabels)
        bpy.utils.unregister_class(ColorFmri)
        bpy.utils.unregister_class(ClearColors)
        bpy.utils.unregister_class(ColoringMakerPanel)
    except:
        print("Can't unregister Freeview Panel!")
