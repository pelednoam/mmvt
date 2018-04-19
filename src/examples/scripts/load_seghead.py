import os.path as op
import nibabel.freesurfer as nib_fs
import numpy as np
import bpy


PLY_HEADER = 'ply\nformat ascii 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nelement face {}\nproperty list uchar int vertex_index\nend_header\n'


def run(mmvt):
    surf_fname = op.join(mmvt.utils.get_subjects_dir(), mmvt.utils.get_user(), 'surf', 'lh.seghead')
    if not op.isfile(surf_fname):
        print("Can't find the seghead surface! ({})".format(surf_fname))
        return
    ply_fname = op.join(mmvt.utils.get_user_fol(), 'surf', 'seghead.ply')
    if not op.isfile(ply_fname):
        verts, faces = nib_fs.read_geometry(surf_fname)
        write_ply_file(verts, faces, ply_fname)
    mmvt.utils.change_layer(mmvt.INFLATED_ACTIVITY_LAYER)
    bpy.ops.import_mesh.ply(filepath=ply_fname)
    o = bpy.context.selected_objects[0]
    o.name = 'seghead'
    o.data.update()
    o.select = True
    bpy.ops.object.shade_smooth()
    o.scale = [0.1] * 3
    o.hide = False
    o.active_material = bpy.data.materials['Activity_map_mat']
    o.parent = bpy.data.objects["Functional maps"]
    o.hide_select = True
    o.data.vertex_colors.new()


def write_ply_file(verts, faces, ply_file_name):
    verts_num = verts.shape[0]
    faces_num = faces.shape[0]
    faces = faces.astype(np.int)
    faces_for_ply = np.hstack((np.ones((faces_num, 1)) * faces.shape[1], faces))
    with open(ply_file_name, 'w') as f:
        f.write(PLY_HEADER.format(verts_num, faces_num))
    with open(ply_file_name, 'ab') as f:
        np.savetxt(f, verts, fmt='%.5f', delimiter=' ')
        np.savetxt(f, faces_for_ply, fmt='%d', delimiter=' ')
