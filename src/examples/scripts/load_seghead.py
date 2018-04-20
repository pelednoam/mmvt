import os.path as op
import nibabel.freesurfer as nib_fs


def run(mmvt):
    surf_fname = op.join(mmvt.utils.get_subjects_dir(), mmvt.utils.get_user(), 'surf', 'lh.seghead')
    if not op.isfile(surf_fname):
        print("Trying to create the seghead surface ({})".format(surf_fname))
        mmvt.utils.run_mmvt_func('src.utils.freesurfer_utils', 'create_seghead', add_subject=True)
        return
    ply_fname = op.join(mmvt.utils.get_user_fol(), 'surf', 'seghead.ply')
    if not op.isfile(ply_fname):
        verts, faces = nib_fs.read_geometry(surf_fname)
        mmvt.utils.write_ply_file(verts, faces, ply_fname)
    mmvt.data.load_ply(ply_fname, 'seghead', new_material_name='seghead_mat')
    mmvt.appearance.set_transparency('seghead_mat', 1)
    mmvt.appearance.set_layers_depth_trans('seghead_mat', 10)

