import bpy
import numpy as np
import os.path as op
import math
import mmvt_utils as mu
import os

PARENT_OBJ = 'connections'

def create_connections(context, d, connections_type, condition, threshold, abs_threshold=True):
    # 'coh_colors', 'locations', 'labels', 'conditions', 'data', 'hemis'
    cond_id = [cond_ind for cond_ind, cond in enumerate(d.conditions) if cond == condition][0]
    # data = d.data[:, :, cond_id]
    radius = .05
    rodsLayer = 3

    LayersRods=[False] * 20
    LayersRods[rodsLayer]=True

    mu.create_empty_if_doesnt_exists(PARENT_OBJ,rodsLayer,LayersRods)
    parent_obj = bpy.data.objects[PARENT_OBJ]

    conn_names = set()
    d['flags'], d['selected_values'], d['selected_connections'], d['selected_colors'] = [], [], [], []
    for conn_ind, conn in enumerate(d.coh_colors[:, :, cond_id]):
        i, j, val, conn_color = int(conn[0])-1, int(conn[1])-1,  conn[2], conn[3:]
        conn_name = '{}-{}'.format(d.labels[i], d.labels[j])
        if conn_name in conn_names:
            continue
        conn_names.add(conn_name)
        # threshold_flag = np.isclose([abs(val)], [0])[0] if cond_id == 2 else abs(val) < threshold
        threshold_flag = abs(val) <= threshold if abs_threshold else val <= threshold
        if (connections_type == 'between' and d.hemis[i] == d.hemis[j]) or \
            (connections_type == 'within' and d.hemis[i] != d.hemis[j]) or threshold_flag:
            d['flags'].append(False)
            continue
        d['flags'].append(True)
        d['selected_values'].append(val)
        d['selected_connections'].append(conn_name)
        d['selected_colors'].append(conn_color)
        conn_color = np.hstack((conn_color, [0.]))
        p1, p2 = d.locations[i, :] * 0.1, d.locations[j, :] * 0.1
        mu.cylinder_between(p1, p2, radius)
        mat_name = 'conn{}_mat'.format(conn_ind)
        mu.create_material(mat_name, conn_color, 1)
        # print(val, conn_color)

        bpy.context.active_object.name = conn_name
        bpy.context.active_object.parent = parent_obj
        mu.add_keyframe(parent_obj, conn_name, val, ConnectionsPanel.addon.T)
        # mark_objs([labels[i], labels[j]])
    if sum(d['flags']) > 0:
        for fcurve, color in zip(parent_obj.animation_data.action.fcurves, d['selected_colors']):
            fcurve.modifiers.new(type='LIMITS')
            fcurve.color_mode = 'CUSTOM'
            fcurve.color = tuple(color)
            # fcurve.select = True
            # fcurve.lock = False
            # fcurve.update()
            # print(fcurve.color)
    bpy.context.scene.update()
    bpy.data.objects[PARENT_OBJ].select = True
    return d


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
        print(connections_type, condition, threshold, abs_threshold)
        mu.delete_hierarchy(PARENT_OBJ)
        create_connections(context, ConnectionsPanel.d, connections_type, condition, threshold, abs_threshold)
        return {"FINISHED"}


class ClearConnections(bpy.types.Operator):
    bl_idname = "ohad.clear_connections"
    bl_label = "ohad clear connections"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        mu.delete_hierarchy(PARENT_OBJ)
        return {"FINISHED"}


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

    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene, 'connections_threshold', text="Threshold")
        layout.prop(context.scene, 'abs_threshold')
        layout.prop(context.scene, "connections_type", text="")
        layout.prop(context.scene, "conditions", text="")
        layout.operator("ohad.plot_connections", text="Plot connections ", icon='POTATO')
        layout.operator("ohad.clear_connections", text="Clear", icon='PANEL_CLOSE')


def init(addon):
    # colors = mu.arr_to_colors(10)
    # print(colors)
    connections_file = op.join(mu.get_user_fol(), 'coh.npz')
    if not os.path.isfile(connections_file):
        print('No connections file! {}'.format(connections_file))
    else:
        d = mu.Bag(np.load(connections_file))
        d.data = d.data.astype(np.double)
        d.labels = [str(l) for l in d.labels]
        d.hemis = [str(l) for l in d.hemis]
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
        bpy.utils.register_class(PlotConnections)
        bpy.utils.register_class(ClearConnections)
        print('ConnectionsPanel was registered!')
    except:
        print("Can't register ConnectionsPanel!")


def unregister():
    try:
        bpy.utils.unregister_class(ConnectionsPanel)
        bpy.utils.unregister_class(PlotConnections)
        bpy.utils.unregister_class(ClearConnections)
    except:
        print("Can't unregister ConnectionsPanel!")