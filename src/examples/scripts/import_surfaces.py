import os.path as op
import glob
import nibabel.freesurfer as nib_fs


def run(mmvt):
    surfaces_fnames = glob.glob(op.join(mmvt.utils.get_user_fol(), 'surf', 'import', '*'))
    for surf_fname in surfaces_fnames:
        ply_fname = op.join(mmvt.utils.get_user_fol(), 'surf', '{}.ply'.format(mmvt.utils.namebase(surf_fname)))
        if not op.isfile(ply_fname):
            verts, faces = nib_fs.read_geometry(surf_fname)
            mmvt.utils.write_ply_file(verts, faces, ply_fname)
        mmvt.data.load_ply(ply_fname, mmvt.utils.namebase(surf_fname))