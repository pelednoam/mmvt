import bpy
import numpy as np
import os.path as op
import math
from collections import defaultdict, OrderedDict
import mmvt_utils as mu
import os

PARENT_OBJ = 'connections'
HEMIS_WITHIN, HEMIS_BETWEEN = range(2)


#todo: read labels from matlab file
# d: labels, locations, hemis, con_colors (L, W, 2, 3), con_values (L, W, 2), indices, con_names, conditions
def create_keyframes(context, d, condition, threshold):
    layers_rods = [False] * 20
    rods_layer = 3
    layers_rods[rods_layer] = True
    mu.create_empty_if_doesnt_exists(PARENT_OBJ, rods_layer, layers_rods)
    cond_id = [i for i, cond in enumerate(d.conditions) if cond == condition][0]
    windows_num = d.con_colors.shape[1]
    norm_fac = ConnectionsPanel.addon.T / windows_num
    mask = np.max(d.con_values[:, :, 0], axis=1) > threshold
    indices = np.where(mask)[0]
    parent_obj = bpy.data.objects[PARENT_OBJ]
    mu.delete_hierarchy(PARENT_OBJ)

    radius = .05
    for ind, conn_name, (i, j) in zip(indices, d.con_names[mask], d.con_indices[mask]):
        print('keyframing {}'.format(conn_name))
        p1, p2 = d.locations[i, :] * 0.1, d.locations[j, :] * 0.1
        mu.cylinder_between(p1, p2, radius)
        con_color = np.hstack((d.con_colors[ind, 0, cond_id, :], [0.]))
        mu.create_material('{}_mat'.format(conn_name), con_color, 1)
        bpy.context.active_object.name = conn_name
        bpy.context.active_object.parent = parent_obj
        # if not bpy.data.objects.get(conn_name):
        #     continue
        mu.insert_keyframe_to_custom_prop(parent_obj, conn_name, 0, 1)
        mu.insert_keyframe_to_custom_prop(parent_obj, conn_name, d.con_values[ind, -1, cond_id], ConnectionsPanel.addon.T + 1)
        mu.insert_keyframe_to_custom_prop(parent_obj, conn_name, 0, ConnectionsPanel.addon.T + 2)
        for t in range(windows_num):
            timepoint = t * norm_fac + 2
            mu.insert_keyframe_to_custom_prop(parent_obj, conn_name, d.con_values[ind, t, cond_id], timepoint)

    for fcurve in parent_obj.animation_data.action.fcurves:
        fcurve.modifiers.new(type='LIMITS')

    print('finish keyframing!')


# d: labels, locations, hemis, con_colors (L, W, 2, 3), con_values (L, W, 2), indices, con_names, conditions, con_types
def filter_graph(context, d, condition, threshold, connections_type):
    mu.show_hide_hierarchy(False, PARENT_OBJ)
    masked_con_names = calc_masked_con_names(d, threshold, connections_type)
    parent_obj = bpy.data.objects[PARENT_OBJ]
    for fcurve in parent_obj.animation_data.action.fcurves:
        con_name = fcurve.data_path[2:-2]
        cur_obj = bpy.data.objects[con_name]
        fcurve.hide = con_name not in masked_con_names
        fcurve.select = not fcurve.hide
        cur_obj.hide = con_name not in masked_con_names
        cur_obj.select = not cur_obj.hide


def calc_masked_con_names(d, threshold, connections_type):
    threshold_mask = np.max(d.con_values[:, :, 0], axis=1) > threshold
    if connections_type == 'between':
        con_names_hemis = set(d.con_names[d.con_types == HEMIS_BETWEEN])
    elif connections_type == 'within':
        con_names_hemis = set(d.con_names[d.con_types == HEMIS_WITHIN])
    else:
        con_names_hemis = set(d.con_names)
    return set(d.con_names[threshold_mask]) & con_names_hemis


# d: labels, locations, hemis, con_colors (L, W, 2, 3), con_values (L, W, 2), indices, con_names, conditions, con_types
def plot_connections(context, d, time, connections_type, condition, threshold, abs_threshold=True):
    windows_num = d.con_colors.shape[1]
    cond_id = [i for i, cond in enumerate(d.conditions) if cond == condition][0]
    t = int(time / ConnectionsPanel.addon.T * windows_num)
    # for conn_name, conn_colors in colors.items():
    for ind, con_name in enumerate(d.con_names):
        cur_obj = bpy.data.objects.get(con_name)
        if cur_obj and not cur_obj.hide:
            con_color = np.hstack((d.con_colors[ind, t, cond_id, :], [0.]))
            bpy.context.scene.objects.active = cur_obj
            mu.create_material('{}_mat'.format(con_name), con_color, 1, False)
    print(con_color, d.con_values[ind, t, cond_id])


def filter(target):
    values = ConnectionsPanel.d['selected_values']
    names = ConnectionsPanel.d['selected_connections']
    closest_value = values[np.argmin(np.abs(np.array(values)-target))]
    indices = np.where(values == closest_value)[0]
    exceptions = []
    for index in indices:
        obj_name = str(names[index])
        bpy.data.objects[obj_name].select = False
        exceptions.append(obj_name)
    mu.delete_hierarchy(PARENT_OBJ, exceptions=exceptions)
    parent_obj = bpy.data.objects[PARENT_OBJ]
    for index in indices:
        bpy.data.objects[str(names[index])].select = True
        mu.add_keyframe(parent_obj, str(names[index]), float(values[index]), ConnectionsPanel.addon.T)
    selected_colors = np.array(ConnectionsPanel.d['selected_colors'])[indices]
    for fcurve, color in zip(parent_obj.animation_data.action.fcurves, selected_colors):
        fcurve.modifiers.new(type='LIMITS')
        fcurve.color_mode = 'CUSTOM'
        fcurve.color = tuple(color)


def show_hide_connections(context, do_show, d, condition, threshold, connections_type, time):
    mu.show_hide_hierarchy(do_show, PARENT_OBJ)
    bpy.data.objects[PARENT_OBJ].select = do_show
    cond_id = [i for i, cond in enumerate(d.conditions) if cond == condition][0]
    windows_num = d.con_colors.shape[1]
    t = int(time / ConnectionsPanel.addon.T * windows_num)
    for ind, con_name in enumerate(d.con_names):
        do_show_con = do_show # and con_name in masked_con_names
        cur_obj = bpy.data.objects.get(con_name)
        if cur_obj:
            if do_show_con:
                con_color = np.hstack((d.con_colors[ind, t, cond_id, :], [0.]))
            else:
                con_color = [1., 1., 1., 1.]
            bpy.context.scene.objects.active = cur_obj
            transparency = 1 if do_show_con else 0
            mu.create_material('{}_mat'.format(con_name), con_color, transparency, False)

    parent_obj = bpy.data.objects[PARENT_OBJ]
    for fcurve in parent_obj.animation_data.action.fcurves:
        fcurve.hide = not do_show
        fcurve.select = do_show
    if do_show:
        filter_graph(context, d, condition, threshold, connections_type)
        mu.view_all_in_graph_editor(context)


def filter_electrodes_via_connections(context, do_filter):
    for elc_name in ConnectionsPanel.addon.play_panel.PlayPanel.electrodes_names:
        cur_obj = bpy.data.objects.get(elc_name)
        if cur_obj:
            cur_obj.hide = do_filter
            cur_obj.select = not do_filter
            for fcurve in cur_obj.animation_data.action.fcurves:
                fcurve.hide = do_filter
                fcurve.select = not do_filter

    if do_filter:
        for ind, con_name in enumerate(ConnectionsPanel.d.con_names):
            cur_obj = bpy.data.objects.get(con_name)
            if cur_obj and not cur_obj.hide:
                electrodes = con_name.split('-')
                for elc in electrodes:
                    cur_obj = bpy.data.objects.get(elc)
                    if cur_obj:
                        cur_obj.hide = False
                        cur_obj.select = True
                        for fcurve in cur_obj.animation_data.action.fcurves:
                            fcurve.hide = False
                            fcurve.select = True

    mu.view_all_in_graph_editor(context)

class CreateConnections(bpy.types.Operator):
    bl_idname = "ohad.create_connections"
    bl_label = "ohad create connections"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        connections_type = bpy.context.scene.connections_type
        threshold = bpy.context.scene.connections_threshold
        abs_threshold = False #bpy.context.scene.abs_threshold
        condition = bpy.context.scene.conditions
        print(connections_type, condition, threshold, abs_threshold)
        create_keyframes(context, ConnectionsPanel.d, condition, threshold)
        return {"FINISHED"}


class FilterGraph(bpy.types.Operator):
    bl_idname = "ohad.filter_graph"
    bl_label = "ohad filter graph"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        connections_type = bpy.context.scene.connections_type
        threshold = bpy.context.scene.connections_threshold
        abs_threshold = False #bpy.context.scene.abs_threshold
        condition = bpy.context.scene.conditions
        print(connections_type, condition, threshold, abs_threshold)
        filter_graph(context, ConnectionsPanel.d, condition, threshold, connections_type)
        return {"FINISHED"}


class PlotConnections(bpy.types.Operator):
    bl_idname = "ohad.plot_connections"
    bl_label = "ohad plot connections"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        connections_type = bpy.context.scene.connections_type
        threshold = bpy.context.scene.connections_threshold
        abs_threshold = bpy.context.scene.abs_threshold
        condition = bpy.context.scene.conditions
        time = bpy.context.scene.frame_current
        print(connections_type, condition, threshold, abs_threshold)
        # mu.delete_hierarchy(PARENT_OBJ)
        plot_connections(context, ConnectionsPanel.d, time, connections_type, condition, threshold, abs_threshold)
        return {"FINISHED"}


class ShowHideConnections(bpy.types.Operator):
    bl_idname = "ohad.show_hide_connections"
    bl_label = "ohad show_hide_connections"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        d = ConnectionsPanel.d
        condition = bpy.context.scene.conditions
        connections_type = bpy.context.scene.connections_type
        threshold = bpy.context.scene.connections_threshold
        time = bpy.context.scene.frame_current
        show_hide_connections(context, ConnectionsPanel.show_connections, d, condition, threshold, connections_type, time)
        ConnectionsPanel.show_connections = not ConnectionsPanel.show_connections
        return {"FINISHED"}


class FilterElectrodes(bpy.types.Operator):
    bl_idname = "ohad.filter_electrodes"
    bl_label = "ohad filter electrodes"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        filter_electrodes_via_connections(context, ConnectionsPanel.do_filter)
        ConnectionsPanel.do_filter = not ConnectionsPanel.do_filter
        return {"FINISHED"}


class ClearConnections(bpy.types.Operator):
    bl_idname = "ohad.clear_connections"
    bl_label = "ohad clear connections"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        mu.delete_hierarchy(PARENT_OBJ)
        return {"FINISHED"}


def connections_draw(self, context):
    layout = self.layout
    layout.prop(context.scene, "connections_origin", text="")
    layout.operator("ohad.create_connections", text="Create connections ", icon='RNA_ADD')
    layout.prop(context.scene, 'connections_threshold', text="Threshold")
    # layout.prop(context.scene, 'abs_threshold')
    layout.prop(context.scene, "connections_type", text="")
    layout.prop(context.scene, "conditions", text="")
    layout.operator("ohad.filter_graph", text="Filter graph ", icon='BORDERMOVE')
    layout.operator("ohad.plot_connections", text="Plot connections ", icon='POTATO')
    if ConnectionsPanel.show_connections:
        layout.operator("ohad.show_hide_connections", text="Show connections ", icon='RESTRICT_VIEW_OFF')
    else:
        layout.operator("ohad.show_hide_connections", text="Hide connections ", icon='RESTRICT_VIEW_OFF')

    filter_obj_name = 'electrodes' if bpy.context.scene.connections_origin == 'electrodes' else 'MEG labels'
    filter_text = '{} {}'.format('Filter' if ConnectionsPanel.do_filter else 'Remove filter from', filter_obj_name)
    layout.operator("ohad.filter_electrodes", text=filter_text, icon='BORDERMOVE')
    layout.operator("ohad.clear_connections", text="Clear", icon='PANEL_CLOSE')


bpy.types.Scene.connections_origin = bpy.props.EnumProperty(
        items=[("electrodes", "Between electrodes", "", 1), ("meg", "Between MEG labels", "", 2)],
        description="Conditions origin", update=connections_draw)
bpy.types.Scene.connections_threshold = bpy.props.FloatProperty(default=5, min=0, description="")
bpy.types.Scene.abs_threshold = bpy.props.BoolProperty(name='abs threshold',
    description="check if abs(val) > threshold")
bpy.types.Scene.connections_type = bpy.props.EnumProperty(
    items=[("all", "All connections", "", 1), ("between", "Only between hemispheres", "", 2),
           ("within", "Only within hemispheres", "", 3)],
    description="Conetions type")
bpy.types.Scene.conditions = bpy.props.EnumProperty(items=[], description="Conditions")


class ConnectionsPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
    bl_label = "Connections"
    addon = None
    d = None
    show_connections = True
    do_filter = True

    def draw(self, context):
        connections_draw(self, context)


def init(addon):
    # colors = mu.arr_to_colors(10)
    # print(colors)
    connections_file = op.join(mu.get_user_fol(), 'electrodes_coh.npz')
    if not os.path.isfile(connections_file):
        print('No connections file! {}'.format(connections_file))
        # self.report({'ERROR'}, "My message")
    else:
        print('loading {}'.format(connections_file))
        d = mu.Bag(np.load(connections_file))
        # d.data = d.data.astype(np.double)
        d.labels = [l.astype(str) for l in d.labels]
        d.hemis = [l.astype(str) for l in d.hemis]
        d.con_names = np.array([l.astype(str) for l in d.con_names], dtype=np.str)
        d.conditions = [l.astype(str) for l in d.conditions]
        confitions_items = [(cond, cond, '', cond_ind) for cond_ind, cond in enumerate(d.conditions)]

        # diff_cond = '{}-{}'.format(d.conditions[0], d.conditions[1])
        # confitions_items.append((diff_cond, diff_cond, '', len(d.conditions)))
        bpy.types.Scene.conditions = bpy.props.EnumProperty(items=confitions_items, description="Conditions")
        register()
        ConnectionsPanel.addon = addon
        ConnectionsPanel.d = d
        print('connection panel initialization completed successfully!')


def register():
    try:
        unregister()
        bpy.utils.register_class(ConnectionsPanel)
        bpy.utils.register_class(CreateConnections)
        bpy.utils.register_class(PlotConnections)
        bpy.utils.register_class(ShowHideConnections)
        bpy.utils.register_class(ClearConnections)
        bpy.utils.register_class(FilterGraph)
        bpy.utils.register_class(FilterElectrodes)
        print('ConnectionsPanel was registered!')
    except:
        print("Can't register ConnectionsPanel!")


def unregister():
    try:
        bpy.utils.unregister_class(ConnectionsPanel)
        bpy.utils.unregister_class(CreateConnections)
        bpy.utils.unregister_class(PlotConnections)
        bpy.utils.unregister_class(ShowHideConnections)
        bpy.utils.unregister_class(ClearConnections)
        bpy.utils.unregister_class(FilterGraph)
        bpy.utils.unregister_class(FilterElectrodes)
    except:
        print("Can't unregister ConnectionsPanel!")