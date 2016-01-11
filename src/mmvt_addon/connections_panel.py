import bpy
import numpy as np
import os.path as op
import math

bpy.types.Scene.connections_threshold = bpy.props.FloatProperty(default=5, min=0, description="")


def create_connections():
    ROOT = '/home/noam/Documents/glassy_brain'
    d = np.load(op.join(ROOT, 'coh_data.npz'))
    coh_colors, locations, labels, data = d['coh_colors'], d['locations'], d['labels'], d['data']

    radius = .05
    rodsLayer = 3

    LayersRods=[False] * 20
    LayersRods[rodsLayer]=True

    create_empty_if_doesnt_exists('coherence_connections',rodsLayer,LayersRods)
    parent_obj = bpy.data.objects['coherence_connections']

    conn_names = set()
    for conn_ind, conn in enumerate(coh_colors):
        i, j, conn_color = conn[0]-1, conn[1]-1, conn[2:]
        conn_name = '{}-{}'.format(labels[i], labels[j])
        if conn_name in conn_names:
            continue
        conn_color = np.hstack((conn_color, [0.]))
        p1, p2 = locations[i, :] * 0.1, locations[j, :] * 0.1
        cylinder_between(p1, p2, radius)
        mat_name = 'conn{}_mat'.format(conn_ind)
        create_material(mat_name, conn_color, 1)

        bpy.context.active_object.name = conn_name
        bpy.context.active_object.parent = parent_obj
        # mark_objs([labels[i], labels[j]])
        insert_keyframe_to_custom_prop(parent_obj, bpy.context.active_object.name, data[i, j], 1)
        insert_keyframe_to_custom_prop(parent_obj, bpy.context.active_object.name, data[i, j], 2502)
    for fcurve in parent_obj.animation_data.action.fcurves:
        fcurve.modifiers.new(type='LIMITS')
    select_hierarchy('coherence_connections', True, False)


def insert_keyframe_to_custom_prop(obj, prop_name, value, keyframe):
    bpy.context.scene.objects.active = obj
    obj.select = True
    obj[prop_name] = value
    obj.keyframe_insert(data_path='[' + '"' + prop_name + '"' + ']', frame=keyframe)


def mark_objs(objs_names):
    for obj_name in objs_names:
        try:
            bpy.data.objects[obj_name].active_material = bpy.data.materials['selected_label_Mat']
        except:
            print("Can't find {}".format(obj_name))


def cylinder_between(p1, p2, r):
    # From http://blender.stackexchange.com/questions/5898/how-can-i-create-a-cylinder-linking-two-points-with-python
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    dist = math.sqrt(dx**2 + dy**2 + dz**2)

    bpy.ops.mesh.primitive_cylinder_add(
      radius = r,
      depth = dist,
      location = (dx/2 + x1, dy/2 + y1, dz/2 + z1)
    )

    phi = math.atan2(dy, dx)
    theta = math.acos(dz/dist)
    bpy.context.object.rotation_euler[1] = theta
    bpy.context.object.rotation_euler[2] = phi


def create_empty_if_doesnt_exists(name,brain_layer,layers_array,root_fol='Brain'):
    if bpy.data.objects.get(name) is None:
        layers_array[brain_layer] = True
        bpy.ops.object.empty_add(type='PLAIN_AXES', radius=1, view_align=False, location=(0, 0, 0), layers=layers_array)
        bpy.data.objects['Empty'].name = name
        if name != root_fol:
            bpy.data.objects[name].parent = bpy.data.objects[root_fol]


def select_hierarchy(obj, val=True, select_parent=True):
    if bpy.data.objects.get(obj) is not None:
        bpy.data.objects[obj].select = select_parent
        for child in bpy.data.objects[obj].children:
            child.select = val


def create_material(name, diffuseColors, transparency):
    #curMat = bpy.data.materials['OrigPatchesMat'].copy()
    curMat = bpy.data.materials['OrigPatchMatTwoCols'].copy()
    curMat.name = name
    bpy.context.active_object.active_material = curMat
    curMat.node_tree.nodes['MyColor'].inputs[0].default_value = diffuseColors
    curMat.node_tree.nodes['MyColor1'].inputs[0].default_value = diffuseColors
    curMat.node_tree.nodes['MyTransparency'].inputs['Fac'].default_value = transparency


class PlotConnections(bpy.types.Operator):
    bl_idname = "ohad.plot_connections"
    bl_label = "ohad plot connections"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        print('invoke!')
        create_connections()
        return {"FINISHED"}


class ClearConnections(bpy.types.Operator):
    bl_idname = "ohad.clear_connections"
    bl_label = "ohad clear connections"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        bpy.ops.object.select_all(action='DESELECT')
        select_hierarchy('coherence_connections', True)
        bpy.ops.object.delete()
        select_hierarchy('coherence_connections', False, False)
        return {"FINISHED"}


class ConnectionsPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
    bl_label = "Connections"

    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene, 'connections_threshold', text="Threshold")
        layout.operator("ohad.plot_connections", text="Plot connections ", icon='POTATO')
        layout.operator("ohad.clear_connections", text="Clear", icon='PANEL_CLOSE')


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