import bpy
import traceback
import math
import numpy as np
import os
import os.path as op
import uuid
from collections import OrderedDict
import time

try:
    import cPickle as pickle
except:
    import pickle


def namebase(file_name):
    return op.splitext(op.basename(file_name))[0]


def save(obj, fname):
    with open(fname, 'wb') as fp:
        # protocol=2 so we'll be able to load in python 2.7
        pickle.dump(obj, fp)


def load(fname):
    with open(fname, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


class Bag( dict ):
    """ a dict with d.key short for d["key"]
        d = Bag( k=v ... / **dict / dict.items() / [(k,v) ...] )  just like dict
    """
        # aka Dotdict

    def __init__(self, *args, **kwargs):
        dict.__init__( self, *args, **kwargs )
        self.__dict__ = self

    def __getnewargs__(self):  # for cPickle.dump( d, file, protocol=-1)
        return tuple(self)


def add_keyframe(parent_obj, conn_name, value, T):
    try:
        insert_keyframe_to_custom_prop(parent_obj, conn_name, 0, 0)
        insert_keyframe_to_custom_prop(parent_obj, conn_name, value, 1)
        insert_keyframe_to_custom_prop(parent_obj, conn_name, value, T)
        insert_keyframe_to_custom_prop(parent_obj, conn_name, 0, T + 1)
        # print('insert keyframe with value of {}'.format(value))
    except:
        print("Can't add a keyframe! {}, {}, {}".format(parent_obj, conn_name, value))
        print(traceback.format_exc())


def insert_keyframe_to_custom_prop(obj, prop_name, value, keyframe):
    bpy.context.scene.objects.active = obj
    obj.select = True
    obj[prop_name] = value
    obj.keyframe_insert(data_path='[' + '"' + prop_name + '"' + ']', frame=keyframe)


def mark_objects(objs_names):
    for obj_name in objs_names:
        if bpy.data.objects.get(obj_name):
            bpy.data.objects[obj_name].active_material = bpy.data.materials['selected_label_Mat']


def cylinder_between(p1, p2, r, layers_array):
    # From http://blender.stackexchange.com/questions/5898/how-can-i-create-a-cylinder-linking-two-points-with-python
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    dist = math.sqrt(dx**2 + dy**2 + dz**2)

    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=dist, location=(dx/2 + x1, dy/2 + y1, dz/2 + z1))#, layers=layers_array)

    phi = math.atan2(dy, dx)
    theta = math.acos(dz/dist)
    bpy.context.object.rotation_euler[1] = theta
    bpy.context.object.rotation_euler[2] = phi
    bpy.ops.object.move_to_layer(layers=layers_array)


def create_empty_if_doesnt_exists(name, brain_layer, layers_array=None,root_fol='Brain'):
    if layers_array is None:
        # layers_array = bpy.context.scene.layers
        layers_array = [False] * 20
        layers_array[brain_layer] = True
    if bpy.data.objects.get(name) is None:
        # layers_array[brain_layer] = True
        bpy.ops.object.empty_add(type='PLAIN_AXES', radius=1, view_align=False, location=(0, 0, 0), layers=layers_array)
        bpy.ops.object.move_to_layer(layers=layers_array)
        bpy.data.objects['Empty'].name = name
        if name != root_fol:
            bpy.data.objects[name].parent = bpy.data.objects[root_fol]


def select_hierarchy(obj, val=True, select_parent=True):
    if bpy.data.objects.get(obj) is not None:
        bpy.data.objects[obj].select = select_parent
        for child in bpy.data.objects[obj].children:
            child.select = val


def create_material(name, diffuseColors, transparency, copy_material=True):
    curMat = bpy.context.active_object.active_material
    if copy_material or 'MyColor' not in curMat.node_tree.nodes:
        #curMat = bpy.data.materials['OrigPatchesMat'].copy()
        curMat = bpy.data.materials['OrigPatchMatTwoCols'].copy()
        curMat.name = name
        bpy.context.active_object.active_material = curMat
    curMat.node_tree.nodes['MyColor'].inputs[0].default_value = diffuseColors
    curMat.node_tree.nodes['MyColor1'].inputs[0].default_value = diffuseColors
    curMat.node_tree.nodes['MyTransparency'].inputs['Fac'].default_value = transparency
    bpy.context.active_object.active_material.diffuse_color = diffuseColors[:3]


def delete_hierarchy(parent_obj_name, exceptions=(), delete_only_animation=False):
    bpy.ops.object.select_all(action='DESELECT')
    obj = bpy.data.objects.get(parent_obj_name)
    if obj is None:
        return
    obj.animation_data_clear()
    # Go over all the objects in the hierarchy like @zeffi suggested:
    names = set()
    def get_child_names(obj):
        for child in obj.children:
            names.add(child.name)
            if child.children:
                get_child_names(child)

    get_child_names(obj)
    names = names - set(exceptions)
    # Remove the animation from the all the child objects
    for child_name in names:
        bpy.data.objects[child_name].animation_data_clear()

    bpy.context.scene.objects.active = obj
    if not delete_only_animation:
        objects = bpy.data.objects
        [setattr(objects[n], 'select', True) for n in names]
        result = bpy.ops.object.delete()
        if result == {'FINISHED'}:
            print ("Successfully deleted object")
        else:
            print ("Could not delete object")


def get_user_fol():
    user = namebase(bpy.data.filepath).split('_')[0]
    root_fol = bpy.path.abspath('//')
    return op.join(root_fol, user)


def view_all_in_graph_editor(context):
    graph_area = [context.screen.areas[k] for k in range(len(context.screen.areas)) if
                  context.screen.areas[k].type == 'GRAPH_EDITOR'][0]
    graph_window_region = [graph_area.regions[k] for k in range(len(graph_area.regions)) if
                           graph_area.regions[k].type == 'WINDOW'][0]

    c = context.copy()  # copy the context
    c['area'] = graph_area
    c['region'] = graph_window_region
    bpy.ops.graph.view_all(c)


def show_hide_hierarchy(val, obj, also_parent=False):
    if bpy.data.objects.get(obj) is not None:
        if also_parent:
            bpy.data.objects[obj].hide_render = not val
        for child in bpy.data.objects[obj].children:
            child.hide = not val
            child.hide_render = not val
            child.select = val


def rand_letters(num):
    return str(uuid.uuid4())[:num]


def evaluate_fcurves(parent_obj, time_range):
    data = OrderedDict()
    colors = OrderedDict()
    for fcurve in parent_obj.animation_data.action.fcurves:
        if fcurve.hide:
            continue
        name = fcurve.data_path.split('"')[1]
        print('{} extrapolation'.format(name))
        for kf in fcurve.keyframe_points:
            kf.interpolation = 'BEZIER'
        data[name] = []
        for t in time_range:
            d = fcurve.evaluate(t)
            data[name].append(d)
        colors[name] = tuple(fcurve.color)
    return data, colors


def get_fcurve_current_frame_val(parent_obj_name, obj_name, cur_frame):
    for fcurve in bpy.data.objects[parent_obj_name].animation_data.action.fcurves:
        name = fcurve_name(fcurve)
        if name == obj_name:
            return fcurve.evaluate(cur_frame)


def fcurve_name(fcurve):
    return fcurve.data_path.split('"')[1]


def show_only_selected_fcurves(context):
    space = context.space_data
    dopesheet = space.dopesheet
    dopesheet.show_only_selected = True


def get_fcurve_values(parent_name, fcurve_name):
    xs, ys = [], []
    parent_obj = bpy.data.objects[parent_name]
    for fcurve in parent_obj.animation_data.action.fcurves:
        if fcurve_name(fcurve) == fcurve_name:
            for kp in fcurve.keyframe_points:
                xs.append(kp.co[0])
                ys.append(kp.co[1])
    return xs, ys


def time_to_go(now, run, runs_num, runs_num_to_print=10):
    if run % runs_num_to_print == 0 and run != 0:
        time_took = time.time() - now
        more_time = time_took / run * (runs_num - run)
        print('{}/{}, {:.2f}s, {:.2f}s to go!'.format(run, runs_num, time_took, more_time))


def show_hide_obj_and_fcurves(objs, val):
    for obj in objs:
        obj.select = val
        if obj.animation_data:
            for fcurve in obj.animation_data.action.fcurves:
                if val:
                    fcurve.hide = not val
                    fcurve.hide = not val
                fcurve.select = val
        else:
            print('No animation in {}'.format(obj.name))


def message(self, message):
    # todo: Find how to send messaages without the self
    if self:
        self.report({'ERROR'}, message)
    else:
        print(message)


def show_only_group_objects(context, objects, group_name):
    space = context.space_data
    dopesheet = space.dopesheet
    selected_group = bpy.data.groups.get(group_name, bpy.data.groups.new(group_name))
    for obj in objects:
        selected_group.objects.link(obj)
    dopesheet.filter_group = selected_group
    dopesheet.show_only_group_objects = True


def create_spline(points, layers_array, bevel_depth=0.045, resolution_u=5):
    # points = [ [1,1,1], [-1,1,1], [-1,-1,1], [1,-1,-1] ]
    curvedata = bpy.data.curves.new(name="Curve", type='CURVE')
    curvedata.dimensions = '3D'
    curvedata.fill_mode = 'FULL'
    curvedata.bevel_depth = bevel_depth
    ob = bpy.data.objects.new("CurveObj", curvedata)
    bpy.context.scene.objects.link(ob)

    spline = curvedata.splines.new('BEZIER')
    spline.bezier_points.add(len(points)-1)
    for num in range(len(spline.bezier_points)):
        spline.bezier_points[num].co = points[num]
        spline.bezier_points[num].handle_right_type = 'AUTO'
        spline.bezier_points[num].handle_left_type = 'AUTO'
    spline.resolution_u = resolution_u
    #spline.order_u = 6
    #spline.use_bezier_u = True
    #spline.radius_interpolation = 'BSPLINE'
    #print(spline.type)
    #spline.use_smooth = True
    bpy.ops.object.move_to_layer(layers=layers_array)
    return ob


def get_subfolders(fol):
    return [os.path.join(fol,subfol) for subfol in os.listdir(fol) if os.path.isdir(os.path.join(fol,subfol))]
