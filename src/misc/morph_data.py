import os.path as op
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


if __name__ == '__main__':
    morph_from = 'mg78'
    morph_to = 'colin27'
    morph_fmri(morph_from, morph_to, 'non-interference-v-interference_{hemi}.mgz')