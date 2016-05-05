import bpy
import mmvt_utils as mu
import colors_utils as cu
import numpy as np
import os.path as op
import time
import itertools
from collections import defaultdict, OrderedDict
import glob

HEMIS = mu.HEMIS


def can_color_obj(obj):
    cur_mat = obj.active_material
    return 'RGB' in cur_mat.node_tree.nodes


def object_coloring(obj, rgb):
    bpy.context.scene.objects.active = obj
    # todo: do we need to select the object here? In diff mode it's a problem
    # obj.select = True
    cur_mat = obj.active_material
    new_color = (rgb[0], rgb[1], rgb[2], 1)
    if can_color_obj(obj):
        cur_mat.node_tree.nodes["RGB"].outputs[0].default_value = new_color
        # new_color = get_obj_color(obj)
        # print('{} new color: {}'.format(obj.name, new_color))
    else:
        print("Can't color {}".format(obj.name))


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
        # if obj and not obj.hide:
            # print('trying to color {} with {}'.format(obj_name+postfix_str, new_color))
        object_coloring(obj, new_color)
            # print(obj_name, value, new_color)
        # else:
        #     print('color_object_homogeneously: {} was not loaded!'.format(obj_name))

    print('Finished coloring!!')


def init_activity_map_coloring(map_type, subcorticals=True):
    # ColoringMakerPanel.addon.set_appearance_show_activity_layer(bpy.context.scene, True)
    # ColoringMakerPanel.addon.set_filter_view_type(bpy.context.scene, 'RENDERS')
    ColoringMakerPanel.addon.show_activity()
    ColoringMakerPanel.addon.change_to_rendered_brain()

    if subcorticals:
        ColoringMakerPanel.addon.show_hide_hierarchy(map_type != 'FMRI', 'Subcortical_fmri_activity_map')
        ColoringMakerPanel.addon.show_hide_hierarchy(map_type != 'MEG', 'Subcortical_meg_activity_map')
    else:
        hide_subcorticals = not subcorticals
        ColoringMakerPanel.addon.show_hide_sub_corticals(hide_subcorticals)
    # change_view3d()


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


def activity_map_coloring(map_type, clusters=False, threshold=None):
    init_activity_map_coloring(map_type)
    if threshold is None:
        threshold = bpy.context.scene.coloring_threshold
    meg_sub_activity = None
    if map_type == 'MEG':
        meg_sub_activity = load_meg_subcortical_activity()
    plot_activity(map_type, ColoringMakerPanel.faces_verts, threshold, meg_sub_activity, clusters=clusters)
    # setup_environment_settings()


def meg_labels_coloring(self, context, override_current_mat=True):
    init_activity_map_coloring('MEG')
    threshold = bpy.context.scene.coloring_threshold
    hemispheres = [hemi for hemi in HEMIS if not bpy.data.objects[hemi].hide]
    user_fol = mu.get_user_fol()
    for hemi_ind, hemi in enumerate(hemispheres):
        labels_data = np.load(op.join(user_fol, 'meg_labels_coloring_{}.npz'.format(hemi)))
        meg_labels_coloring_hemi(labels_data, ColoringMakerPanel.faces_verts,
                                 hemi, threshold, override_current_mat)


def meg_labels_coloring_hemi(labels_data, faces_verts, hemi, threshold, override_current_mat=True):
    now = time.time()
    labels_names = ColoringMakerPanel.labels_vertices['labels_names']
    labels_vertices = ColoringMakerPanel.labels_vertices['labels_vertices']
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
        plot_subcorticals=True, override_current_mat=True, clusters=False):
    current_root_path = mu.get_user_fol() # bpy.path.abspath(bpy.context.scene.conf_path)
    hemispheres = [hemi for hemi in HEMIS if not bpy.data.objects[hemi].hide]
    frame_str = str(bpy.context.scene.frame_current)

    # loop_indices = {}
    for hemi in hemispheres:
        if map_type == 'MEG':
            f = np.load(op.join(current_root_path, 'activity_map_' + hemi, 't' + frame_str + '.npy'))
        elif map_type == 'FMRI':
            # fname = op.join(current_root_path, 'fmri_{}{}.npy'.format('clusters_' if clusters else '', hemi))
            # f = np.load(fname)
            if clusters:
                f = [c for h, c in ColoringMakerPanel.fMRI_clusters.items() if h == hemi]
            else:
                f = ColoringMakerPanel.fMRI[hemi]
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
    # print('cur_obj: {}, max vert in lookup: {}, vcol_layer len: {}'.format(cur_obj.name, np.max(lookup), len(vcol_layer.data)))
    for vert in valid_verts:
        x = lookup[vert]
        for loop_ind in x[x>-1]:
            d = vcol_layer.data[loop_ind]
            d.color = vert_values[vert, 1:]


def color_groups_manually():
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
    init_activity_map_coloring('FMRI')
    subject_fol = mu.get_user_fol()
    objects_names, colors, data = defaultdict(list), defaultdict(list), defaultdict(list)
    for line in mu.csv_file_reader(op.join(subject_fol, 'coloring', '{}.csv'.format(bpy.context.scene.coloring_files))):
        obj_name, color_name = line[0], line[1:]
        if obj_name[0] == '#':
            continue
        obj_type = mu.check_obj_type(obj_name)
        if isinstance(color_name, str) and color_name.startswith('mark'):
            import filter_panel
            filter_panel.filter_roi_func(obj_name, mark=color_name)
        else:
            if isinstance(color_name, str):
                color_rgb = cu.name_to_rgb(color_name)
            # Check if the color is already in RBG
            elif len(color_name) == 3:
                color_rgb = color_name
            else:
                print('Unrecognize color! ({})'.format(color_name))
                continue
            color_rgb = list(map(float, color_rgb))
            if obj_type is not None:
                objects_names[obj_type].append(obj_name)
                colors[obj_type].append(color_rgb)
                data[obj_type].append(1.)

    color_objects(objects_names, colors, data)
    if op.isfile(op.join(subject_fol, 'coloring', '{}_legend.jpg'.format(bpy.context.scene.coloring_files))):
        cmd = '{} -m src.preproc.electrodes_preproc -s {} -a {} -f show_labeling_coloring'.format(
            bpy.context.scene.python_cmd, mu.get_user(), bpy.context.scene.atlas)
        print('Running {}'.format(cmd))
        mu.run_command_in_new_thread(cmd, False)


def color_objects(objects_names, colors, data):
    for hemi in HEMIS:
        obj_type = mu.OBJ_TYPE_CORTEX_LH if hemi=='lh' else mu.OBJ_TYPE_CORTEX_RH
        if len(objects_names[obj_type]) == 0:
            continue
        labels_data = dict(data=np.array(data[obj_type]), colors=colors[obj_type], names=objects_names[obj_type])
        # print('color hemi {}: {}'.format(hemi, labels_names))
        meg_labels_coloring_hemi(labels_data, ColoringMakerPanel.faces_verts, hemi, 0)
    for region, color in zip(objects_names[mu.OBJ_TYPE_SUBCORTEX], colors[mu.OBJ_TYPE_SUBCORTEX]):
        print('color {}: {}'.format(region, color))
        color_subcortical_region(region, color)
    for electrode, color in zip(objects_names[mu.OBJ_TYPE_ELECTRODE], colors[mu.OBJ_TYPE_ELECTRODE]):
        obj = bpy.data.objects.get(electrode)
        if obj and not obj.hide:
            object_coloring(obj, color)
    bpy.context.scene.subcortical_layer = 'fmri'
    ColoringMakerPanel.addon.show_activity()


def color_subcortical_region(region_name, color):
    # obj = bpy.data.objects.get(region_name + '_meg_activity', None)
    # if not obj is None:
    #     object_coloring(obj,     color)
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
        activity_map_obj_coloring(cur_obj, region_colors_data, lookup, 0, True)
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
                obj.active_material.node_tree.nodes['RGB'].outputs['Color'].default_value=(1, 1, 1, 1)


def default_coloring(loop_indices):
    for hemi, indices in loop_indices.items():
        cur_obj = bpy.data.objects[hemi]
        mesh = cur_obj.data
        vcol_layer = mesh.vertex_colors.active
        for loop_ind in indices:
            vcol_layer.data[loop_ind].color = [1, 1, 1]


def fmri_files_update(self, context):
    #todo: there are two frmi files list (the other one in fMRI panel)
    user_fol = mu.get_user_fol()
    # fmri_files = glob.glob(op.join(user_fol, 'fmri', '*_lh.npy'))
    for hemi in mu.HEMIS:
        fname = op.join(user_fol, 'fmri', 'fmri_{}_{}.npy'.format(bpy.context.scene.fmri_files, hemi))
        ColoringMakerPanel.fMRI[hemi] = np.load(fname)


class ColorElectrodes(bpy.types.Operator):
    bl_idname = "ohad.electrodes_color"
    bl_label = "ohad electrodes color"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        threshold = bpy.context.scene.coloring_threshold
        data = np.load(op.join(mu.get_user_fol(), 'electrodes', 'electrodes_data_{}.npz'.format(
            'avg' if bpy.context.scene.selection_type == 'conds' else 'diff')))
        color_object_homogeneously(data, threshold=threshold)
        # deselect_all()
        # mu.select_hierarchy('Deep_electrodes', False)
        ColoringMakerPanel.addon.show_electrodes()
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


class ColorGroupsManually(bpy.types.Operator):
    bl_idname = "ohad.man_groups_color"
    bl_label = "ohad man groups color"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        color_groups_manually()
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


class ColorClustersFmri(bpy.types.Operator):
    bl_idname = "ohad.fmri_clusters_color"
    bl_label = "ohad fmri clusters color"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        activity_map_coloring('FMRI', clusters=True)
        return {"FINISHED"}


class ClearColors(bpy.types.Operator):
    bl_idname = "ohad.colors_clear"
    bl_label = "ohad colors clear"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        clear_colors()
        return {"FINISHED"}


def clear_colors():
    clear_cortex()
    clear_subcortical_fmri_activity()
    for root in ['Subcortical_meg_activity_map', 'Deep_electrodes']:
        clear_colors_from_parent_childrens(root)


bpy.types.Scene.coloring_fmri = bpy.props.BoolProperty(default=True, description="Plot FMRI")
bpy.types.Scene.coloring_electrodes = bpy.props.BoolProperty(default=False, description="Plot Deep electrodes")
bpy.types.Scene.coloring_threshold = bpy.props.FloatProperty(default=0.5, min=0, description="")
bpy.types.Scene.fmri_files = bpy.props.EnumProperty(items=[], description="fMRI files", update=fmri_files_update)
bpy.types.Scene.coloring_files = bpy.props.EnumProperty(items=[], description="Coloring files")


class ColoringMakerPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
    bl_label = "Activity Maps"
    addon = None
    fMRI = {}
    fMRI_clusters = {}
    labels_vertices = {}

    def draw(self, context):
        layout = self.layout
        user_fol = mu.get_user_fol()
        aparc_name = bpy.context.scene.atlas
        faces_verts_exist = mu.hemi_files_exists(op.join(user_fol, 'faces_verts_{hemi}.npy'))
        fmri_files = glob.glob(op.join(user_fol, 'fmri', '*_lh.npy'))  # mu.hemi_files_exists(op.join(user_fol, 'fmri_{hemi}.npy'))
        # fmri_clusters_files_exist = mu.hemi_files_exists(op.join(user_fol, 'fmri', 'fmri_clusters_{hemi}.npy'))
        meg_files_exist = mu.hemi_files_exists(op.join(user_fol, 'activity_map_{hemi}', 't0.npy'))
        meg_labels_files_exist = op.isfile(op.join(user_fol, 'labels_vertices_{}.pkl'.format(aparc_name))) and \
            mu.hemi_files_exists(op.join(user_fol, 'meg_labels_coloring_{hemi}.npz'))
        electrodes_files_exist = op.isfile(op.join(mu.get_user_fol(), 'electrodes', 'electrodes_data_{}.npz'.format(
            'avg' if bpy.context.scene.selection_type == 'conds' else 'diff')))
        manually_color_files_exist = len(glob.glob(op.join(user_fol, 'coloring', '*.csv'))) > 0
        manually_groups_file_exist = op.isfile(op.join(mu.get_parent_fol(user_fol), '{}_groups.csv'.format(bpy.context.scene.atlas)))
        layout.prop(context.scene, 'coloring_threshold', text="Threshold")
        if faces_verts_exist:
            if meg_files_exist:
                layout.operator(ColorMeg.bl_idname, text="Plot MEG ", icon='POTATO')
            if meg_labels_files_exist:
                layout.operator(ColorMegLabels.bl_idname, text="Plot MEG Labels ", icon='POTATO')
            if len(fmri_files) > 0:
                layout.prop(context.scene, "fmri_files", text="")
                layout.operator(ColorFmri.bl_idname, text="Plot fMRI ", icon='POTATO')
            # if fmri_clusters_files_exist:
            #     layout.operator(ColorClustersFmri.bl_idname, text="Plot Clusters fMRI ", icon='POTATO')
            if manually_color_files_exist:
                layout.prop(context.scene, "coloring_files", text="")
                layout.operator(ColorManually.bl_idname, text="Color Manually", icon='POTATO')
            if manually_groups_file_exist:
                layout.prop(context.scene, 'labels_groups', text="")
                layout.operator(ColorGroupsManually.bl_idname, text="Color Groups", icon='POTATO')
        if electrodes_files_exist:
            layout.operator(ColorElectrodes.bl_idname, text="Plot Electrodes ", icon='POTATO')
        layout.operator(ClearColors.bl_idname, text="Clear", icon='PANEL_CLOSE')


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


def init(addon):
    from random import shuffle
    ColoringMakerPanel.addon = addon
    # Load fMRI data
    user_fol = mu.get_user_fol()
    labels_names, labels_vertices = mu.load(
        op.join(user_fol, 'labels_vertices_{}.pkl'.format(bpy.context.scene.atlas)))
    ColoringMakerPanel.labels_vertices = dict(labels_names=labels_names, labels_vertices=labels_vertices)
    fmri_files = glob.glob(op.join(user_fol, 'fmri', 'fmri_*_lh.npy')) # mu.hemi_files_exists(op.join(user_fol, 'fmri_{hemi}.npy'))
    # fmri_clusters_files_exist = mu.hemi_files_exists(op.join(user_fol, 'fmri', 'fmri_clusters_{hemi}.npy'))
    if len(fmri_files) > 0:
        # if len(fmri_files) > 1:
        files_names = [mu.namebase(fname)[5:-3] for fname in fmri_files]
        clusters_items = [(c, c, '', ind) for ind, c in enumerate(files_names)]
        bpy.types.Scene.fmri_files = bpy.props.EnumProperty(
            items=clusters_items, description="fMRI files", update=fmri_files_update)
        bpy.context.scene.fmri_files = files_names[0]
        for hemi in mu.HEMIS:
            ColoringMakerPanel.fMRI[hemi] = np.load('{}_{}.npy'.format(fmri_files[0][:-7], hemi))
        # if fmri_clusters_files_exist:
        #     for hemi in mu.HEMIS:
        #         ColoringMakerPanel.fMRI_clusters[hemi] = np.load(
        #             op.join(user_fol, 'fmri', 'fmri_clusters_{}.npy'.format(hemi)))
    mu.make_dir(op.join(user_fol, 'coloring'))
    manually_color_files = glob.glob(op.join(user_fol, 'coloring', '*.csv'))
    if len(manually_color_files) > 0:
        files_names = [mu.namebase(fname) for fname in manually_color_files]
        coloring_items = [(c, c, '', ind) for ind, c in enumerate(files_names)]
        bpy.types.Scene.coloring_files = bpy.props.EnumProperty(items=coloring_items, description="Coloring files")
        bpy.context.scene.coloring_files = files_names[0]

    ColoringMakerPanel.colors = list(set(list(cu.NAMES_TO_HEX.keys())) - set(['black']))
    shuffle(ColoringMakerPanel.colors)
    ColoringMakerPanel.labels_groups = read_groups_labels(ColoringMakerPanel.colors)
    if len(ColoringMakerPanel.labels_groups) > 0:
        groups_items = [(gr, gr, '', ind) for ind, gr in enumerate(list(ColoringMakerPanel.labels_groups.keys()))]
        bpy.types.Scene.labels_groups = bpy.props.EnumProperty(
            items=groups_items, description="Groups")

    ColoringMakerPanel.faces_verts = load_faces_verts()
    register()


def register():
    try:
        unregister()
        bpy.utils.register_class(ColorElectrodes)
        bpy.utils.register_class(ColorManually)
        bpy.utils.register_class(ColorGroupsManually)
        bpy.utils.register_class(ColorMeg)
        bpy.utils.register_class(ColorMegLabels)
        bpy.utils.register_class(ColorFmri)
        bpy.utils.register_class(ColorClustersFmri)
        bpy.utils.register_class(ClearColors)
        bpy.utils.register_class(ColoringMakerPanel)
        # print('Freeview Panel was registered!')
    except:
        print("Can't register Freeview Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(ColorElectrodes)
        bpy.utils.unregister_class(ColorManually)
        bpy.utils.unregister_class(ColorGroupsManually)
        bpy.utils.unregister_class(ColorMeg)
        bpy.utils.unregister_class(ColorMegLabels)
        bpy.utils.unregister_class(ColorFmri)
        bpy.utils.unregister_class(ColorClustersFmri)
        bpy.utils.unregister_class(ClearColors)
        bpy.utils.unregister_class(ColoringMakerPanel)
    except:
        pass
        # print("Can't unregister Freeview Panel!")
