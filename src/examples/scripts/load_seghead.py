import os.path as op
import nibabel.freesurfer as nib_fs


def run(mmvt):
    surf_fname = op.join(mmvt.utils.get_subjects_dir(), mmvt.utils.get_user(), 'surf', 'lh.seghead')
    if not op.isfile(surf_fname):
        print("Can't find the seghead surface! ({})".format(surf_fname))
        return
    ply_fname = op.join(mmvt.utils.get_user_fol(), 'surf', 'seghead.ply')
    if not op.isfile(ply_fname):
        verts, faces = nib_fs.read_geometry(surf_fname)
        mmvt.utils.write_ply_file(verts, faces, ply_fname)
    mmvt.data.load_ply(ply_fname, 'seghead', new_material_name='seghead_mat')
    mmvt.appearance.set_transparency('seghead_mat', 1)
