import bpy
import traceback
import math
import numpy as np
import os.path as op
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


def delete_hierarchy(parent_obj_name, exceptions=()):
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
    objects = bpy.data.objects
    [setattr(objects[n], 'select', True) for n in names]
    # Remove the animation from the all the child objects
    for child_name in names:
        bpy.data.objects[child_name].animation_data_clear()

    bpy.context.scene.objects.active = obj
    result = bpy.ops.object.delete()
    if result == {'FINISHED'}:
        print ("Successfully deleted object")
    else:
        print ("Could not delete object")


def get_user_fol():
    user = namebase(bpy.data.filepath).split('_')[0]
    root_fol = bpy.path.abspath('//')
    return op.join(root_fol, user)


# def get_scalar_map(x_min, x_max, color_map='jet'):
#     import matplotlib.colors
#     import matplotlib.cm
#     root = bpy.path.abspath('//')
#     cm = load(op.join(root, 'mmvt_code', 'color_map_{}.pkl'.format(color_map)))
#     cNorm = matplotlib.colors.Normalize(vmin=x_min, vmax=x_max)
#     return matplotlib.cm .ScalarMappable(norm=cNorm, cmap=cm)
#
#
# def arr_to_colors(x, x_min=None, x_max=None, colors_map='jet', scalar_map=None):
#     if scalar_map is None:
#         x_min, x_max = check_min_max(x, x_min, x_max)
#         scalar_map = get_scalar_map(x_min, x_max, colors_map)
#     return scalar_map.to_rgba(x)
#
#
# def check_min_max(x, x_min, x_max):
#     if x_min is None:
#         x_min = np.min(x)
#     if x_max is None:
#         x_max = np.max(x)
#     return x_min, x_max
#



# EXTERNAL_PYTHON_PATH = '/homes/5/npeled/space3/anaconda3/lib/python3.5/'
# EXTERNAL_PYTHON_PACKAGES_PATH = '/homes/5/npeled/space3/anaconda3/lib/python3.5/site-packages'


# def insert_into_path(rel_paths, root):
#     import sys
#     import os
#     for rel_path in rel_paths:
#         wpath = os.path.join(root, rel_path)
#         try:
#             sys.path.remove(wpath)
#         except:
#             pass
#         if wpath not in sys.path:
#             sys.path.insert(1, wpath)
#             print('insert into path: {}'.format(wpath))


# def insert_external_path():
#     import sys
#     for p in sys.path:
#         if 'python3.4' in p:
#             sys.path.remove(p)
#
#     insert_into_path(['', ], EXTERNAL_PYTHON_PACKAGES_PATH)
#     insert_into_path(['', ], EXTERNAL_PYTHON_PATH)
#     insert_into_path(['setuptools-18.5-py3.5.egg', 'numpy', 'numpy/core'], EXTERNAL_PYTHON_PACKAGES_PATH)
#     import imp
#     print('imp: ', imp.__file__)
#
#
# def lib_check():
#     # for p in sys.path:
#     #     if 'python3.4' in p: # /site-packages/numpy
#     #         sys.path.remove(p)
#     # sys.path = ['/homes/5/npeled/space3/anaconda3/lib/python3.5', '/homes/5/npeled/space3/anaconda3/lib/python3.5/site-packages',
#     #             '/homes/5/npeled/space3/anaconda3/lib/python3.5/site-packages/setuptools-18.5-py3.5.egg',
#     #             '/homes/5/npeled/space3/anaconda3/lib/python3.5/site-packages/numpy/core',
#     #             '/homes/5/npeled/space3/anaconda3/lib/python3.5/site-packages/numpy/core/multiarray.cpython-35m-x86_64-linux-gnu.so'] + sys.path
#     # sys.path.insert(1, '/homes/5/npeled/space3/anaconda3/lib/python3.5',)
#     # insert_into_path(['', 'setuptools-18.5-py3.5.egg', 'numpy', 'numpy/core', 'numpy/core/multiarray.cpython-35m-x86_64-linux-gnu.so'])
#     import matplotlib
#     # imp.reload(matplotlib)
#     print(matplotlib.__version__, matplotlib.__file__)
#     # import numpy
#     # print(numpy.__version__, numpy.__file__)
#     import numpy.core
#     # imp.reload(numpy.core)
#     print(numpy.core.__version__, numpy.core.__file__)
#     # import numpy.core.multiarray
#     # imp.reload(numpy.core.multiarray)
#     # imp.load_dynamic(numpy.core.multiarray, '/homes/5/npeled/space3/anaconda3/lib/python3.5/site-packages/numpy/core/multiarray.cpython-35m-x86_64-linux-gnu.so')
#     print(numpy.core.multiarray.__version__, numpy.core.multiarray.__file__)
#     import matplotlib.pyplot
#     # import mmvt_utils
#     # imp.reload(mmvt_utils)
#     # mmvt_utils.arr_to_colors(range(10), colors_map='jet')
#     import imp
#     imp.load_dynamic('numpy.core.multiarray', '/homes/5/npeled/space3/anaconda3/lib/python3.5/site-packages/numpy/core/multiarray.cpython-35m-x86_64-linux-gnu.so')
#     imp.load_dynamic('_csv', '/homes/5/npeled/space3/anaconda3/lib/python3.5/lib-dynload/_csv.cpython-35m-x86_64-linux-gnu.so')
