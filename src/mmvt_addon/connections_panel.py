import bpy
import numpy as np
import os.path as op
import time
import mmvt_utils as mu
import colors_utils as cu
import os

# try:
#     import matplotlib as mpl
#     mpl.use('Agg')
#     import matplotlib.pyplot as plt
# except ImportError:
#     print('No matplotlib!')

PARENT_OBJ = 'connections'
HEMIS_WITHIN, HEMIS_BETWEEN = range(2)
STAT_AVG, STAT_DIFF = range(2)

CONNECTIONS_LAYER, BRAIN_EMPTY_LAYER = 3,5


# d(Bag): labels, locations, hemis, con_colors (L, W, 2, 3), con_values (L, W, 2), indices, con_names, conditions
def create_keyframes(d, threshold, threshold_type, radius=.1, stat=STAT_DIFF):
    layers_rods = [False] * 20
    rods_layer = CONNECTIONS_LAYER
    layers_rods[rods_layer] = True
    mu.delete_hierarchy(PARENT_OBJ)
    mu.create_empty_if_doesnt_exists(PARENT_OBJ, BRAIN_EMPTY_LAYER, None, 'Functional maps')
    # cond_id = [i for i, cond in enumerate(d.conditions) if cond == condition][0]
    if d.con_colors.ndim == 3:
        windows_num = d.con_colors.shape[1]
    else:
        windows_num = 1
    T = ConnectionsPanel.addon.get_max_time_steps()
    if T == 0:
        T = windows_num
    norm_fac = T / windows_num
    # todo: Check if we really want to let the user create connection according to the first option
    # if bpy.context.scene.selection_type == 'conds':
    #     # Takes all the connections that at least one condition pass the threshold
    #     # Ex: np.array([True, False, False]) | np.array([False, True, False]) = array([ True,  True, False], dtype=bool)
    #     mask1 = np.max(d.con_values[:, :, 0], axis=1) > threshold
    #     mask2 = np.max(d.con_values[:, :, 1], axis=1) > threshold
    #     mask = mask1 | mask2
    # else:
    stat_data = calc_stat_data(d.con_values, stat)
    mask = calc_mask(stat_data, threshold, threshold_type, windows_num)
    indices = np.where(mask)[0]
    parent_obj = bpy.data.objects[PARENT_OBJ]
    parent_obj.animation_data_clear()
    N = len(indices)
    print('{} connections are above the threshold'.format(N))
    create_conncection_for_both_conditions(d, layers_rods, indices, mask, windows_num, norm_fac, T, radius)
    print('Create connections for the conditions {}'.format('difference' if stat == STAT_DIFF else 'mean'))
    create_keyframes_for_parent_obj(d, indices, mask, windows_num, norm_fac, T, stat)
    print('finish keyframing!')


def calc_mask(stat_data, threshold, threshold_type, windows_num):
    if threshold_type == 'percentile':
        threshold = np.percentile(np.abs(stat_data), threshold)
    threshold_type = bpy.context.scene.above_below_threshold
    if stat_data.ndim == 1:
        mask = abs(stat_data) >= threshold if threshold_type == 'above' else abs(stat_data) <= threshold
    else:
        mask = np.max(abs(stat_data), axis=1) >= threshold if threshold_type == 'above' else \
            np.max(abs(stat_data), axis=1) <= threshold
    return mask


def create_conncection_for_both_conditions(d, layers_rods, indices, mask, windows_num, norm_fac, T, radius):
    N = len(indices)
    parent_obj = bpy.data.objects[PARENT_OBJ]
    print('Create connections for both conditions')
    now = time.time()
    for run, (ind, conn_name, (i, j)) in enumerate(zip(indices, d.con_names[mask], d.con_indices[mask])):
        mu.time_to_go(now, run, N, runs_num_to_print=10)
        if d.con_colors.ndim == 3:
            con_color = np.hstack((d.con_colors[ind, 0, :], [0.]))
        else:
            con_color = np.hstack((d.con_colors[ind, :], [0.]))
        p1, p2 = d.locations[i, :] * 0.1, d.locations[j, :] * 0.1
        mu.cylinder_between(p1, p2, radius, layers_rods)
        # mu.create_material('{}_mat'.format(conn_name), (0, 0, 1, 1), 1)
        mu.create_material('{}_mat'.format(conn_name), con_color, 1)
        cur_obj = bpy.context.active_object
        cur_obj.name = conn_name
        cur_obj.parent = parent_obj
        # cur_obj.animation_data_clear()
        if windows_num == 1:
            continue
        for cond_id, cond in enumerate(d.conditions):
            # insert_frame_keyframes(cur_obj, '{}-{}'.format(conn_name, cond), d.con_values[ind, -1, cond_id], T)
            for t in range(windows_num):
                extra_time_points = 0 if norm_fac ==1 else 2
                timepoint = t * norm_fac + extra_time_points
                mu.insert_keyframe_to_custom_prop(cur_obj, '{}-{}'.format(conn_name, cond),
                                                  d.con_values[ind, t, cond_id], timepoint)
        finalize_fcurves(cur_obj)


def create_keyframes_for_parent_obj(d, indices, mask, windows_num, norm_fac, T, stat=STAT_DIFF):
    # Create keyframes for the parent obj (conditions diff)
    if windows_num == 1:
        return
    if stat not in [STAT_DIFF, STAT_AVG]:
        print("Wrong type of stat!")
        return
    parent_obj = bpy.data.objects[PARENT_OBJ]
    stat_data = calc_stat_data(d.con_values, stat)
    N = len(indices)
    now = time.time()
    for run, (ind, conn_name) in enumerate(zip(indices, d.con_names[mask])):
        mu.time_to_go(now, run, N, runs_num_to_print=100)
        # insert_frame_keyframes(parent_obj, conn_name, stat_data[ind, -1], T)
        for t in range(windows_num):
            extra_time_points = 0 if norm_fac ==1 else 2
            timepoint = t * norm_fac + extra_time_points
            mu.insert_keyframe_to_custom_prop(parent_obj, conn_name, stat_data[ind, t], timepoint)
    finalize_fcurves(parent_obj)
    finalize_objects_creations()


def calc_stat_data(data, stat):
    axis = data.ndim - 1
    if data.shape[axis] == 1:
        stat = STAT_AVG
    if stat == STAT_AVG:
        stat_data = np.squeeze(np.mean(data, axis=axis))
    elif stat == STAT_DIFF:
        stat_data = np.squeeze(np.diff(data, axis=axis))
    else:
        raise Exception('Wrong stat value!')
    return stat_data


def insert_frame_keyframes(parent_obj, conn_name, last_data, T):
    mu.insert_keyframe_to_custom_prop(parent_obj, conn_name, 0, 1)
    mu.insert_keyframe_to_custom_prop(parent_obj, conn_name, 0, T + 1)  # last_data, T + 1)
    mu.insert_keyframe_to_custom_prop(parent_obj, conn_name, 0, T + 2)


def finalize_fcurves(parent_obj, interpolation=''):
    for fcurve in parent_obj.animation_data.action.fcurves:
        fcurve.modifiers.new(type='LIMITS')
        if interpolation == '':
            interpolation = 'BEZIER' if len(fcurve.keyframe_points) > 10 else 'LINEAR'
        for kf in fcurve.keyframe_points:
            kf.interpolation = interpolation
    mu.change_fcurves_colors([parent_obj])


def finalize_objects_creations():
    try:
        bpy.ops.graph.previewrange_set()
    except:
        pass
    for obj in bpy.data.objects:
        obj.select = False
    if bpy.data.objects.get(' '):
        bpy.context.scene.objects.active = bpy.data.objects[' ']


def unfilter_graph(context, d, condition, connections_type):
    filter_graph(context, d, condition, 0, 'value', connections_type)


# d: labels, locations, hemis, con_colors (L, W, 2, 3), con_values (L, W, 2), indices, con_names, conditions, con_types
def filter_graph(context, d, condition, threshold, threshold_type, connections_type, stat=STAT_DIFF):
    mu.show_hide_hierarchy(False, PARENT_OBJ)
    masked_con_names = calc_masked_con_names(d, threshold, threshold_type, connections_type, condition, stat)
    parent_obj = bpy.data.objects[PARENT_OBJ]
    for con_name in d.con_names:
        cur_obj = bpy.data.objects.get(con_name)
        if cur_obj:
            cur_obj.hide = con_name not in masked_con_names
            cur_obj.hide_render = con_name not in masked_con_names
            if bpy.context.scene.selection_type == 'conds':
                cur_obj.select = not cur_obj.hide
    if parent_obj.animation_data is None:
        return
    now = time.time()
    fcurves_num = len(parent_obj.animation_data.action.fcurves)
    for fcurve_index, fcurve in enumerate(parent_obj.animation_data.action.fcurves):
        # mu.time_to_go(now, fcurve_index, fcurves_num, runs_num_to_print=10)
        con_name = mu.fcurve_name(fcurve)
        # cur_obj = bpy.data.objects[con_name]
        # cur_obj.hide = con_name not in masked_con_names
        # cur_obj.hide_render = con_name not in masked_con_names
        if bpy.context.scene.selection_type != 'conds':
            fcurve.hide = con_name not in masked_con_names
            fcurve.select = not fcurve.hide

    parent_obj.select = True
    mu.view_all_in_graph_editor(context)


def calc_masked_con_names(d, threshold, threshold_type, connections_type, condition, stat):
    # For now, we filter only according to both conditions, not each one seperatly
    if bpy.context.scene.selection_type == 'conds' and d.con_values.ndim == 3:
        if threshold_type == 'percentile':
            threshold = np.percentile(np.abs(np.max(d.con_values, axis=1)), threshold)
        above_below_threshold = bpy.context.scene.above_below_threshold
        mask1 = np.max(d.con_values[:, :, 0], axis=1) >= threshold if above_below_threshold == 'above' else \
            np.max(d.con_values[:, :, 0], axis=1) <= threshold
        mask2 = np.max(d.con_values[:, :, 1], axis=1) >= threshold if above_below_threshold == 'above' else \
            np.max(d.con_values[:, :, 1], axis=1) <= threshold
        threshold_mask = mask1 | mask2
    else:
        stat_data = calc_stat_data(d.con_values, stat)
        windows_num = d.con_values.shape[1]
        threshold_mask = calc_mask(stat_data, threshold, threshold_type, windows_num)
        # threshold_mask = np.max(stat_data, axis=1) > threshold
    if connections_type == 'between':
        con_names_hemis = set(d.con_names[d.con_types == HEMIS_BETWEEN])
    elif connections_type == 'within':
        con_names_hemis = set(d.con_names[d.con_types == HEMIS_WITHIN])
    else:
        con_names_hemis = set(d.con_names)
    return set(d.con_names[threshold_mask]) & con_names_hemis


# d: labels, locations, hemis, con_colors (L, W, 3), con_values (L, W, 2), indices, con_names, conditions, con_types
def plot_connections(self, context, d, plot_time, connections_type, condition, threshold, abs_threshold=True):
    windows_num = d.con_colors.shape[1]
    # xs, ys = get_fcurve_values('RPT3-RAT5')
    # cond_id = [i for i, cond in enumerate(d.conditions) if cond == condition][0]
    t = int(plot_time / ConnectionsPanel.addon.get_max_time_steps() * windows_num)
    print('plotting connections for t:{}'.format(t))
    if t >= d.con_colors.shape[1]:
        mu.message(self, 'time out of bounds! {}'.format(plot_time))
    else:
        # for conn_name, conn_colors in colors.items():
        # vals = []
        selected_objects, selected_indices = get_all_selected_connections(d)
        for ind, con_name in zip(selected_indices, selected_objects):
            # print(con_name, d.con_values[ind, t, 0], d.con_values[ind, t, 1], np.diff(d.con_values[ind, t, :]))
            cur_obj = bpy.data.objects.get(con_name)
            con_color = np.hstack((d.con_colors[ind, t, :], [0.]))
            bpy.context.scene.objects.active = cur_obj
            mu.create_material('{}_mat'.format(con_name), con_color, 1, False)
            # vals.append(np.diff(d.con_values[ind, t])[0])
        bpy.data.objects[PARENT_OBJ].select = True
        ConnectionsPanel.addon.show_connections()
        # print(max(vals), min(vals))
        # print(con_color, d.con_values[ind, t, cond_id])


def get_all_selected_connections(d):
    objs, inds = [], []
    if bpy.context.scene.selection_type == 'conds':
        for ind, con_name in enumerate(d.con_names):
            cur_obj = bpy.data.objects.get(con_name)
            if cur_obj and not cur_obj.hide:
                objs.append(cur_obj.name)
                inds.append(ind)
    else:
        parent_obj = bpy.data.objects[PARENT_OBJ]
        for fcurve in parent_obj.animation_data.action.fcurves:
            con_name = mu.fcurve_name(fcurve)
            if fcurve.select and not fcurve.hide:
                objs.append(con_name)
                ind = np.where(d.con_names == con_name)[0][0]
                inds.append(ind)
    return objs, inds


# Called from FilterPanel, FindCurveClosestToCursor
def find_connections_closest_to_target_value(closet_object_name, closest_curve_name, target):
    parent_obj = bpy.data.objects[PARENT_OBJ]
    if bpy.context.scene.selection_type == 'conds':
        for cur_obj in parent_obj.children:
            if not cur_obj.animation_data:
                continue
            for fcurve in cur_obj.animation_data.action.fcurves:
                if cur_obj.name == closet_object_name:
                    fcurve_name = mu.fcurve_name(fcurve)
                    fcurve.select = fcurve_name == closest_curve_name
                    fcurve.hide = fcurve_name != closest_curve_name
                else:
                    fcurve.select = False
                    fcurve.hide = True
    else:  # diff
        # todo: implement this part
        for fcurve in parent_obj.animation_data.action.fcurves:
            conn_name = mu.fcurve_name(conn_name)


def filter_electrodes_via_connections(context, do_filter):
    display_conds = bpy.context.scene.selection_type == 'conds'
    for elc_name in ConnectionsPanel.addon.play_panel.PlayPanel.electrodes_names:
        cur_obj = bpy.data.objects.get(elc_name)
        if cur_obj:
            cur_obj.hide = do_filter
            cur_obj.hide_render = do_filter
            cur_obj.select = not do_filter and display_conds
            for fcurve in cur_obj.animation_data.action.fcurves:
                fcurve.hide = do_filter
                fcurve.select = not do_filter

    elecs_parent_obj = bpy.data.objects['Deep_electrodes']
    elecs_parent_obj.select = not display_conds
    for fcurve in elecs_parent_obj.animation_data.action.fcurves:
        fcurve.hide = do_filter
        fcurve.select = not do_filter

    if do_filter:
        selected_electrodes = set()
        for ind, con_name in enumerate(ConnectionsPanel.d.con_names):
            cur_obj = bpy.data.objects.get(con_name)
            if not cur_obj or cur_obj.hide:
                continue
            electrodes = con_name.split('-')
            for elc in electrodes:
                cur_elc = bpy.data.objects.get(elc)
                if not cur_elc or elc in selected_electrodes:
                    continue
                selected_electrodes.add(elc)
                # if bpy.context.scene.selection_type == 'conds':
                cur_elc.hide = False
                cur_elc.hide_render = False
                cur_elc.select = display_conds
                for fcurve in cur_elc.animation_data.action.fcurves:
                    fcurve.hide = False
                    fcurve.select = True
        for fcurve in elecs_parent_obj.animation_data.action.fcurves:
            elc_name = mu.fcurve_name(fcurve)
            if elc_name in selected_electrodes:
                fcurve.hide = False
                fcurve.select = True

    mu.view_all_in_graph_editor(context)


def capture_graph_data(per_condition):
    parent_obj = bpy.data.objects[PARENT_OBJ]
    time_range = range(0, ConnectionsPanel.addon.get_max_time_steps(), bpy.context.scene.play_dt)
    if per_condition:
        #todo: implement
        pass
    else:
        data, colors = mu.evaluate_fcurves(parent_obj, time_range)
    return data, colors


# def capture_graph(context, image_fol):
#     data, colors = {}, {}
#     data['Coherence'], colors['Coherence'] = capture_graph_data()
#     data['Electrodes'], colors['Electrodes'] = ConnectionsPanel.addon.play_panel.get_electrodes_data()
#     # data.update(elcs_data)
#     # colors.update(elcs_colors)
#     ConnectionsPanel.addon.play_panel.plot_graph(context, data, colors, image_fol)
#     ConnectionsPanel.addon.play_panel.save_graph_data(data, colors, image_fol)

def load_connections_file():
    print('loading {}'.format(bpy.context.scene.connections_file))
    d = mu.Bag(np.load(bpy.context.scene.connections_file))
    d.labels = [l.astype(str) for l in d.labels]
    d.hemis = [l.astype(str) for l in d.hemis]
    d.con_names = np.array([l.astype(str) for l in d.con_names], dtype=np.str)
    d.conditions = [l.astype(str) for l in d.conditions]
    ConnectionsPanel.d = d


def create_connections():
    load_connections_file()
    conditions_items = [(cond, cond, '', cond_ind) for cond_ind, cond in enumerate(d.conditions)]
    # diff_cond = '{}-{}'.format(d.conditions[0], d.conditions[1])
    # conditions_items.append((diff_cond, diff_cond, '', len(d.conditions)))
    bpy.types.Scene.conditions = bpy.props.EnumProperty(items=conditions_items, description="Conditions")

    threshold = bpy.context.scene.connections_threshold
    threshold_type = bpy.context.scene.connections_threshold_type
    create_keyframes(ConnectionsPanel.d, threshold, threshold_type, stat=STAT_DIFF)


class CreateConnections(bpy.types.Operator):
    bl_idname = "ohad.create_connections"
    bl_label = "ohad create connections"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        create_connections()
        return {"FINISHED"}


class FilterGraph(bpy.types.Operator):
    bl_idname = "ohad.filter_graph"
    bl_label = "ohad filter graph"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        if not bpy.data.objects.get(PARENT_OBJ):
            self.report({'ERROR'}, 'No parent node was found, you first need to create the connections.')
        else:
            connections_type = bpy.context.scene.connections_type
            threshold = bpy.context.scene.connections_threshold
            threshold_type = bpy.context.scene.connections_threshold_type
            # abs_threshold = False  # bpy.context.scene.abs_threshold
            condition = bpy.context.scene.conditions
            # print(connections_type, condition, threshold, abs_threshold)
            if bpy.context.scene.connections_filter:
                unfilter_graph(context, ConnectionsPanel.d, condition, connections_type)
            else:
                filter_graph(context, ConnectionsPanel.d, condition, threshold, threshold_type, connections_type)
            bpy.context.scene.connections_filter = not bpy.context.scene.connections_filter
        return {"FINISHED"}


# todo: Should move to coloring_panel
class PlotConnections(bpy.types.Operator):
    bl_idname = "ohad.plot_connections"
    bl_label = "ohad plot connections"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        if not bpy.data.objects.get(PARENT_OBJ):
            self.report({'ERROR'}, 'No parent node was found, you first need to create the connections.')
        else:
            connections_type = bpy.context.scene.connections_type
            threshold = bpy.context.scene.connections_threshold
            abs_threshold = bpy.context.scene.abs_threshold
            condition = bpy.context.scene.conditions
            plot_time = bpy.context.scene.frame_current
            print(connections_type, condition, threshold, abs_threshold)
            # mu.delete_hierarchy(PARENT_OBJ)
            plot_connections(self, context, ConnectionsPanel.d, plot_time, connections_type, condition, threshold,
                             abs_threshold)
        return {"FINISHED"}


# class ShowHideConnections(bpy.types.Operator):
#     bl_idname = "ohad.show_hide_connections"
#     bl_label = "ohad show_hide_connections"
#     bl_options = {"UNDO"}
#
#     @staticmethod
#     def invoke(self, context, event=None):
#         if not bpy.data.objects.get(PARENT_OBJ):
#             self.report({'ERROR'}, 'No parent node was found, you first need to create the connections.')
#         else:
#             d = ConnectionsPanel.d
#             condition = bpy.context.scene.conditions
#             connections_type = bpy.context.scene.connections_type
#             threshold = bpy.context.scene.connections_threshold
#             time = bpy.context.scene.frame_current
#             show_hide_connections(context, ConnectionsPanel.show_connections, d, condition,
#                                   threshold, connections_type, time)
#             ConnectionsPanel.show_connections = not ConnectionsPanel.show_connections
#         return {"FINISHED"}


class FilterElectrodes(bpy.types.Operator):
    bl_idname = "ohad.filter_electrodes"
    bl_label = "ohad filter electrodes"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        if not bpy.data.objects.get(PARENT_OBJ):
            self.report({'ERROR'}, 'No parent node was found, you first need to create the connections.')
        else:
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


# class ExportGraph(bpy.types.Operator):
#     bl_idname = "ohad.export_graph"
#     bl_label = "ohad export_graph"
#     bl_options = {"UNDO"}
#     uuid = mu.rand_letters(5)
#
#     @staticmethod
#     def invoke(self, context, event=None):
#         image_fol = op.join(mu.get_user_fol(), 'images', ExportGraph.uuid)
#         capture_graph(context, image_fol)
#         return {"FINISHED"}


def connections_draw(self, context):
    layout = self.layout
    layout.prop(context.scene, "connections_origin", text="")
    layout.operator(CreateConnections.bl_idname, text="Create connections ", icon='RNA_ADD')
    layout.prop(context.scene, 'connections_threshold', text="Threshold")
    layout.prop(context.scene, 'above_below_threshold', text='')
    # layout.prop(context.scene, 'abs_threshold')
    layout.prop(context.scene, "connections_type", text="")
    # layout.prop(context.scene, "conditions", text="")
    layout.label(text='Filter type:')
    layout.prop(context.scene, 'connections_threshold_type', text='threshold type', expand=True)
    filter_text = 'Remove filter' if bpy.context.scene.connections_filter else 'Filter connections'
    layout.operator(FilterGraph.bl_idname, text=filter_text, icon='BORDERMOVE')
    layout.operator(PlotConnections.bl_idname, text="Plot connections ", icon='POTATO')
    # if ConnectionsPanel.show_connections:
    #     layout.operator("ohad.show_hide_connections", text="Show connections ", icon='RESTRICT_VIEW_OFF')
    # else:
    #     layout.operator("ohad.show_hide_connections", text="Hide connections ", icon='RESTRICT_VIEW_OFF')

    if bpy.context.scene.connections_origin == 'electrodes':
        filter_text = '{} electrodes'.format('Filter' if ConnectionsPanel.do_filter else 'Remove filter from')
        layout.operator(FilterElectrodes.bl_idname, text=filter_text, icon='BORDERMOVE')
    # layout.operator("ohad.export_graph", text="Export graph", icon='SNAP_NORMAL')
    # layout.operator("ohad.clear_connections", text="Clear", icon='PANEL_CLOSE')


bpy.types.Scene.connections_origin = bpy.props.EnumProperty(items=[],
        description="Conditions origin", update=connections_draw)
bpy.types.Scene.connections_threshold = bpy.props.FloatProperty(default=5, min=0, description="")
bpy.types.Scene.abs_threshold = bpy.props.BoolProperty(name='abs threshold',
                                                       description="check if abs(val) > threshold")
bpy.types.Scene.connections_type = bpy.props.EnumProperty(
        items=[("all", "All connections", "", 1), ("between", "Only between hemispheres", "", 2),
               ("within", "Only within hemispheres", "", 3)], description="Connections type")
bpy.types.Scene.above_below_threshold = bpy.props.EnumProperty(
        items=[("above", "Above threshold", "", 1), ("below", "Below threshold", "", 2)], description="Threshold type")
bpy.types.Scene.conditions = bpy.props.EnumProperty(items=[], description="Conditions")
bpy.types.Scene.connections_file = bpy.props.StringProperty(default='', description="connection file")
bpy.types.Scene.connections_threshold_type = bpy.props.EnumProperty(
    items=[("value", "value", "", 1), ("percentile", "percentile", "", 2)], #, ("top_k", "top k", "", 3)],
    description="Threshold type")
bpy.types.Scene.connections_filter = bpy.props.BoolProperty(name='connections_filter')


class ConnectionsPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
    bl_label = "Connections"
    addon = None
    d = mu.Bag({})
    show_connections = True
    do_filter = True

    def draw(self, context):
        connections_draw(self, context)


def connections_origin_update(self, context):
    _connections_origin_update()


def _connections_origin_update():
    if bpy.context.scene.connections_origin == 'rois':
        bpy.context.scene.connections_file = op.join(mu.get_user_fol(), 'rois_con.npz')
    elif bpy.context.scene.connections_origin == 'electrodes':
        bpy.context.scene.connections_file = op.join(mu.get_user_fol(), 'electrodes', 'electrodes_con.npz')
    else:
        print('Wrong connection type!!!')
    if ConnectionsPanel.d == {}:
        load_connections_file()


def set_connection_type(connection_type):
    bpy.context.scene.connections_origin = connection_type


def set_connections_threshold(threshold):
    bpy.context.scene.connections_threshold = threshold


def init(addon):
    electrodes_connections_file = op.join(mu.get_user_fol(), 'electrodes', 'electrodes_con.npz')
    # meg_bev_connections_file = op.join(mu.get_user_fol(), 'meg_coh_bev.npz')
    rois_connections_file = op.join(mu.get_user_fol(), 'rois_con.npz')

    items = []
    if op.isfile(electrodes_connections_file):
        items.append(("electrodes", "Between electrodes", "", 1))
    elif op.isfile(rois_connections_file):
        items.append(("rois", "Between MEG labels", "", 2))
    else:
        print('No connections file!')
    bpy.types.Scene.connections_origin = bpy.props.EnumProperty(
        items=items, description="Conditions origin", update=connections_origin_update)

    if len(items) != '':
        ConnectionsPanel.addon = addon
        bpy.context.scene.connections_threshold = 0
        _connections_origin_update()
        register()
        # print('connection panel initialization completed successfully!')


def register():
    try:
        unregister()
        bpy.utils.register_class(ConnectionsPanel)
        bpy.utils.register_class(CreateConnections)
        bpy.utils.register_class(PlotConnections)
        bpy.utils.register_class(FilterGraph)
        bpy.utils.register_class(FilterElectrodes)
        # print('ConnectionsPanel was registered!')
    except:
        print("Can't register ConnectionsPanel!")


def unregister():
    try:
        bpy.utils.unregister_class(ConnectionsPanel)
        bpy.utils.unregister_class(CreateConnections)
        bpy.utils.unregister_class(PlotConnections)
        bpy.utils.unregister_class(FilterGraph)
        bpy.utils.unregister_class(FilterElectrodes)
    except:
        pass
        # print("Can't unregister ConnectionsPanel!")
