from src.utils import utils

links_dir = utils.get_links_dir()
subjects_dir = utils.get_link_dir(links_dir, 'subjects', 'SUBJECTS_DIR')
freesurfer_home = utils.get_link_dir(links_dir, 'freesurfer', 'FREESURFER_HOME')
mmvt_dir = utils.get_link_dir(links_dir, 'mmvt')


def main(args):
    pass