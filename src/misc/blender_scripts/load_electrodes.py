import bpy
import os.path as op
import glob
import numpy as np

# LAYERS
(TARGET_LAYER, LIGHTS_LAYER, EMPTY_LAYER, BRAIN_EMPTY_LAYER, ROIS_LAYER, ACTIVITY_LAYER, INFLATED_ROIS_LAYER,
 INFLATED_ACTIVITY_LAYER, ELECTRODES_LAYER, CONNECTIONS_LAYER, EEG_LAYER, MEG_LAYER) = range(12)


def namebase(file_name):
    return op.splitext(op.basename(file_name))[0]


def change_layer(layer):
    bpy.context.scene.layers = [ind == layer for ind in range(len(bpy.context.scene.layers))]


def create_empty_if_doesnt_exists(name, brain_layer, layers_array=None, root_fol='Brain'):
    # if not bpy.data.objects.get(root_fol):
    #     print('root fol, {}, does not exist'.format(root_fol))
    #     return
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
    return bpy.data.objects[name]


def create_sphere(loc, rad, my_layers, name):
    bpy.ops.mesh.primitive_uv_sphere_add(
        ring_count=30, size=rad, view_align=False, enter_editmode=False, location=loc, layers=my_layers)
    bpy.ops.object.shade_smooth()
    bpy.context.active_object.name = name


def create_and_set_material(obj):
    # curMat = bpy.data.materials['OrigPatchesMat'].copy()
    if obj.active_material is None or obj.active_material.name != obj.name + '_Mat':
        if obj.name + '_Mat' in bpy.data.materials:
            cur_mat = bpy.data.materials[obj.name + '_Mat']
        else:
            cur_mat = bpy.data.materials['Deep_electrode_mat'].copy()
            cur_mat.name = obj.name + '_Mat'
        # Wasn't it originally (0, 0, 1, 1)?
        cur_mat.node_tree.nodes["RGB"].outputs[0].default_value = (1, 1, 1, 1) # (0, 0, 1, 1) # (0, 1, 0, 1)
        obj.active_material = cur_mat


input_file = '/home/npeled/Angelique/Niles/electrodes.npz'
parnet_name = 'Deep_electrodes'
electrode_size = 0.15
bipolar = False
f = np.load(input_file)

layers_array = bpy.context.scene.layers
create_empty_if_doesnt_exists(parnet_name, BRAIN_EMPTY_LAYER, layers_array, parnet_name)

layers_array = [False] * 20
layers_array[ELECTRODES_LAYER] = True

for (x, y, z), name in zip(f['pos'], f['names']):
    elc_name = name.astype(str)
    if not bpy.data.objects.get(elc_name) is None:
        elc_obj = bpy.data.objects[elc_name]
        elc_obj.location = [x * 0.1, y * 0.1, z * 0.1]
    else:
        print('creating {}: {}'.format(elc_name, (x, y, z)))
        create_sphere((x * 0.1, y * 0.1, z * 0.1), electrode_size, layers_array, elc_name)
        cur_obj = bpy.data.objects[elc_name]
        cur_obj.select = True
        cur_obj.parent = bpy.data.objects[parnet_name]
        create_and_set_material(cur_obj)
