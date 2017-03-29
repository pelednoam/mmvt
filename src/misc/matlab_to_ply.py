import glob
import os.path as op
import numpy as np

from src.utils import utils

def mat_to_ply():
    fol = '/homes/5/npeled/space1/Angelique/Lionel Recon/'
    files = glob.glob(op.join(fol, 'Iso*.mat'))
    avg_fname = op.join(fol, 'blender', 'mean_vertices.npy')
    if not op.isfile(avg_fname):
        vertices_avg = []
        for f in files:
            print(f)
            m = utils.read_mat_file_into_bag(f)
            vertices_avg.append(np.mean(m.vertices.T / 10, 0))
            m.clear()
        vertices_avg = np.mean(vertices_avg)
        np.save(avg_fname, vertices_avg)
    else:
        vertices_avg = np.load(avg_fname)

    for f in files:
        print(f)
        m = utils.read_mat_file_into_bag(f)
        ply_fname = utils.change_fname_extension(f, 'ply')
        utils.write_ply_file(m.vertices.T / 10.0 - vertices_avg , m.faces.T, ply_fname, True)
        m.clear()


def obj_to_ply():
    fol = '/homes/5/npeled/space1/Angelique/Lionel Recon/blender/objs'
    files = glob.glob(op.join(fol, '*.obj'))
    for f in files:
        print(f)
        verts, faces = utils.read_obj_file(f)
        ply_fname = utils.change_fname_extension(f, 'ply')
        utils.write_ply_file(verts, faces, ply_fname, True)



def import_plys():
    import bpy
    import os.path as op
    import glob

    (TARGET_LAYER, LIGHTS_LAYER, EMPTY_LAYER, BRAIN_EMPTY_LAYER, ROIS_LAYER, ACTIVITY_LAYER, INFLATED_ROIS_LAYER,
     INFLATED_ACTIVITY_LAYER, ELECTRODES_LAYER, CONNECTIONS_LAYER, EEG_LAYER, MEG_LAYER) = range(12)

    def namebase(fname):
        return op.splitext(op.basename(fname))[0]

    def create_empty_if_doesnt_exists(name, brain_layer=None, layers_array=None, parent_obj_name='Brain'):
        if brain_layer is None:
            brain_layer = BRAIN_EMPTY_LAYER
        if layers_array is None:
            layers_array = bpy.context.scene.layers
        if bpy.data.objects.get(name) is None:
            layers_array[brain_layer] = True
            bpy.ops.object.empty_add(type='PLAIN_AXES', radius=1, view_align=False, location=(0, 0, 0), layers=layers_array)
            bpy.data.objects['Empty'].name = name
            if name != parent_obj_name:
                bpy.data.objects[name].parent = bpy.data.objects[parent_obj_name]
        return bpy.data.objects[name]


    emptys_names = ['Brain', 'Cortex']
    for name in emptys_names:
        create_empty_if_doesnt_exists(name)
    current_mat = bpy.data.materials['unselected_label_Mat_cortex']
    plys_fol = '/homes/5/npeled/space1/Angelique/Lionel Recon/blender/plys'
    anatomy_name = 'Cortex'
    for ply_fname in glob.glob(op.join(plys_fol, 'Iso*.ply')):
        new_obj_name = namebase(ply_fname)
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.import_mesh.ply(filepath=ply_fname)
        cur_obj = bpy.context.selected_objects[0]
        cur_obj.select = True
        bpy.ops.object.shade_smooth()
        cur_obj.parent = bpy.data.objects[anatomy_name]
        cur_obj.scale = [0.1] * 3
        cur_obj.active_material = current_mat
        cur_obj.hide = False
        cur_obj.name = new_obj_name


if __name__ == '__main__':
    obj_to_ply()