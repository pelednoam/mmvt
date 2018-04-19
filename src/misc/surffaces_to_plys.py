import os.path as op
import nibabel.freesurfer as nib_fs
from src.utils import utils
from src.utils import preproc_utils as pu

SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()

def convert(subject, surfaces_fnames):
    for surf_fname in surfaces_fnames:
        ply_fname = op.join(SUBJECTS_DIR, subject, 'surf', '{}.ply'.format(utils.namebase(surf_fname)))
        if op.isfile(ply_fname):
            continue
        verts, faces = nib_fs.read_geometry(surf_fname)
        utils.write_ply_file(verts, faces, ply_fname)