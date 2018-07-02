import os.path as op
import time
import nibabel as nib
import numpy as np
import scipy.ndimage
from skimage import measure

from src.utils import utils
from src.utils import preproc_utils as pu
from src.preproc import anatomy as anat

SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()


def create_surf(subject, subcortical_code, subcortical_name):
    aseg = nib.load(op.join(SUBJECTS_DIR, subject, 'mri', 'aseg.mgz')).get_data()
    t1_header = nib.load(op.join(SUBJECTS_DIR, subject, 'mri', 'T1.mgz')).header
    aseg[aseg != subcortical_code] = 255
    aseg[aseg == subcortical_code] = 10
    aseg = np.array(aseg, dtype=np.float)
    aseg_smooth = scipy.ndimage.gaussian_filter(aseg, sigma=1)
    verts_vox, faces, _, _ = measure.marching_cubes(aseg_smooth, 100)
    # Doesn't seem to fix the normals directions that should be out...
    faces = measure.correct_mesh_orientation(aseg_smooth, verts_vox, faces)
    verts = utils.apply_trans(t1_header.get_vox2ras_tkr(), verts_vox)
    fol = utils.make_dir(op.join(MMVT_DIR, subject, 'subcortical_test'))
    ply_file_name = op.join(fol, '{}.ply'.format(subcortical_name))
    utils.write_ply_file(verts, faces, ply_file_name, False)


if __name__ == '__main__':
    subs = anat.load_subcortical_lookup_table()
    now, N = time.time(), len(subs)
    for ind, (k, name) in enumerate(subs.items()):
        utils.time_to_go(now, ind, N, runs_num_to_print=1)
        create_surf('mg116', k, name)