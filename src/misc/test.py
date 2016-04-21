from src import freesurfer_utils as fu
from src import utils

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')


def prepare_darpa_csv(subject, coords):
    fu.transform_subject_to_mni_coordinates(subject, coords, SUBJECTS_DIR)


if __name__ == '__main__':
    prepare_darpa_csv()