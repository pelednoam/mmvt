import os.path as op
import numpy as np
from src.utils import utils
from src.utils import freesurfer_utils as fu
from src.preproc import meg

LINKS_DIR = utils.get_links_dir()
FMRI_DIR = utils.get_link_dir(utils.get_links_dir(), 'fMRI')
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')


def morph_fmri(morph_from, morph_to, nii_template):
    utils.make_dir(op.join(MMVT_DIR, morph_to, 'fmri'))
    for hemi in utils.HEMIS:
        fu.surf2surf(
            morph_from, morph_to, hemi, op.join(MMVT_DIR, morph_from, 'fmri', nii_template.format(hemi=hemi)),
            op.join(MMVT_DIR, morph_to, 'fmri', nii_template.format(hemi=hemi)))


def morph_stc(subject, events, morph_to_subject, inverse_method='dSPM', grade=5, smoothing_iterations=None,
              overwrite=False, n_jobs=6):
    meg.morph_stc(subject, events, morph_to_subject, inverse_method, grade, smoothing_iterations,
              overwrite, n_jobs)


def change_stc_according_to_source():
    pass


def morph_sensors_to_template(morph_from, morph_to, modality='meg'):
    sensors = np.load(op.join(MMVT_DIR, morph_from, modality, f'{modality}_sensors_positions.npz'))
    coords = sensors['pos']
    mophed_coords = fu.transform_subject_to_subject_coordinates(morph_from, morph_to, coords, SUBJECTS_DIR)
    # Not sure why, but there is a shift: (-0.18, -2.95, 1.53)
    utils.make_dir(op.join(MMVT_DIR, morph_to, modality))
    np.savez(op.join(MMVT_DIR, morph_to, modality, f'{modality}_sensors_positions.npz'), names=sensors['names'], pos=mophed_coords)
    print('The morphed sensors were saved in {}'.format(op.join(MMVT_DIR, morph_to, modality, f'{modality}_sensors_positions.npz')))


if __name__ == '__main__':
    morph_from = 'colin27'
    morph_to = 'matt_hibert'
    # morph_fmri(morph_from, morph_to, 'non-interference-v-interference_{hemi}.mgz')
    morph_sensors_to_template(morph_from, morph_to, 'meg')