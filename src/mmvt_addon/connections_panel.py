import bpy
import numpy as np
import os.path as op
import math
import mmvt_utils as mu

PARENT_OBJ = 'connections'

def create_connections(d, connections_type, condition, threshold):
    # 'coh_colors', 'locations', 'labels', 'conditions', 'data', 'hemis'
    d.labels = [str(l) for l in d.labels]
    d.hemis = [str(l) for l in d.hemis]
    cond_id = [cond_ind for cond_ind, cond in enumerate(d.conditions) if cond == condition],
    radius = .05
    rodsLayer = 3

    LayersRods=[False] * 20
    LayersRods[rodsLayer]=True

    mu.create_empty_if_doesnt_exists(PARENT_OBJ,rodsLayer,LayersRods)
    parent_obj = bpy.data.objects[PARENT_OBJ]

    conn_names = set()
    d['flags'], d['selected_values'], d['selected_connections'] = [], [], []
    for conn_ind, conn in enumerate(d.coh_colors):
        i, j, conn_color = int(conn[0])-1, int(conn[1])-1,  conn[2:]
        conn_name = '{}-{}'.format(d.labels[i], d.labels[j])
        if conn_name in conn_names:
            continue
        if (connections_type == 'between' and d.hemis[i] == d.hemis[j]) or \
            (connections_type == 'within' and d.hemis[i] != d.hemis[j]) or \
            d.data[i, j, cond_id] < threshold:
            d['flags'].append(False)
            continue
        d['flags'].append(True)
        d['selected_values'].append(d.data[i, j, cond_id])
        d['selected_connections'].append(conn_name)
        conn_color = np.hstack((conn_color, [0.]))
        p1, p2 = d.locations[i, :] * 0.1, d.locations[j, :] * 0.1
        mu.cylinder_between(p1, p2, radius)
        mat_name = 'conn{}_mat'.format(conn_ind)
        mu.create_material(mat_name, conn_color, 1)

        bpy.context.active_object.name = conn_name
        bpy.context.active_object.parent = parent_obj
        mu.add_keyframe(parent_obj, conn_name, float(d.data[i, j, cond_id]), ConnectionsPanel.addon.T)
        # mark_objs([labels[i], labels[j]])
    for fcurve in parent_obj.animation_data.action.fcurves:
        fcurve.modifiers.new(type='LIMITS')
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
    for index in indices:
        bpy.data.objects[str(names[index])].select = True
        mu.add_keyframe(bpy.data.objects[PARENT_OBJ], str(names[index]), float(values[index]), ConnectionsPanel.addon.T)


class PlotConnections(bpy.types.Operator):
    bl_idname = "ohad.plot_connections"
    bl_label = "ohad plot connections"
    bl_options = {"UNDO"}
    d = None

    @staticmethod
    def invoke(self, context, event=None):
        connections_type = bpy.context.scene.connections_type
        threshold = bpy.context.scene.connections_threshold
        condition = bpy.context.scene.conditions
        print(connections_type, condition, threshold)
        mu.delete_hierarchy(PARENT_OBJ)
        create_connections(ConnectionsPanel.d, connections_type, condition, threshold)
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
bpy.types.Scene.connections_type = bpy.props.EnumProperty(
    items=[("all", "All connections", "", 1), ("between", "Only between hemispheres", "", 2),
           ("within", "Only within hemispheres", "", 3)],
    description="Conetions type")
bpy.types.Scene.conditions = None


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
        layout.prop(context.scene, "connections_type", text="")
        layout.prop(context.scene, "conditions", text="")
        layout.operator("ohad.plot_connections", text="Plot connections ", icon='POTATO')
        layout.operator("ohad.clear_connections", text="Clear", icon='PANEL_CLOSE')


def init(addon):
    ROOT = '/homes/5/npeled/space3/glassy_brain'
    d = mu.Bag(np.load(op.join(ROOT, 'coh.npz')))
    bpy.types.Scene.conditions = bpy.props.EnumProperty(
            items=[(cond, cond, '', cond_ind) for cond_ind, cond in enumerate(d.conditions)],
            description="Conditions")
    register()
    ConnectionsPanel.addon = addon
    ConnectionsPanel.d = d

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