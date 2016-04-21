from src import freesurfer_utils as fu
from src import utils
from src.preproc import electrodes_preproc as elec_pre

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')


def prepare_darpa_csv(subject, bipolar=False):
    elecs_names, elecs_coords = elec_pre.read_electrodes_file(subject, bipolar)
    elecs_coords_mni = fu.transform_subject_to_mni_coordinates(subject, elecs_coords, SUBJECTS_DIR)
    elec_pre.save_electrodes_file(subject, bipolar, elecs_names, elecs_coords_mni, '_mni')


if __name__ == '__main__':
    prepare_darpa_csv('mg96', [])