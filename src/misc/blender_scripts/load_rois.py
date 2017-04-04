import bpy
import os.path as op
import glob

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


root = '/home/npeled/Angelique/Niles/plys'
anatomy = {'Cortex-lh': op.join(root, 'labels'),
           'Subcortical_structures': op.join(root, 'subs'),
           'lh': op.join(root, 'whole_brain')}
brain_layer = BRAIN_EMPTY_LAYER
bpy.context.scene.layers = [ind == brain_layer for ind in range(len(bpy.context.scene.layers))]
layers_array = bpy.context.scene.layers
for name in ['Brain', 'Cortex-lh', 'Subcortical_structures']:
    create_empty_if_doesnt_exists(name, brain_layer, layers_array, 'Brain')
for name in ['Functional maps', 'lh']:
    create_empty_if_doesnt_exists(name, brain_layer, layers_array, 'Functional maps')
bpy.context.scene.layers = [ind == ROIS_LAYER for ind in range(len(bpy.context.scene.layers))]

current_mat = bpy.data.materials['unselected_label_Mat_cortex']
for parent_fol, base_path in anatomy.items():
    for ply_fname in glob.glob(op.join(base_path, '*.ply')):
        print('loading {}'.format(ply_fname))
        new_obj_name = namebase(ply_fname)
        surf_name = 'pial'
        change_layer(ROIS_LAYER)
        if not bpy.data.objects.get(new_obj_name) is None:
            # print('{} was already imported'.format(new_obj_name))
            continue
        bpy.ops.object.select_all(action='DESELECT')
        # print(ply_fname)
        bpy.ops.import_mesh.ply(filepath=ply_fname)
        cur_obj = bpy.context.selected_objects[0]
        cur_obj.select = True
        bpy.ops.object.shade_smooth()
        cur_obj.parent = bpy.data.objects[parent_fol]
        cur_obj.scale = [0.1] * 3
        cur_obj.active_material = current_mat
        cur_obj.hide = False
        cur_obj.name = new_obj_name
