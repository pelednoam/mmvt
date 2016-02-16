import nibabel as nib
import os.path as op
from src import utils

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')


def load_dracula_merged(subject):
    dti_fname = op.join(SUBJECTS_DIR, subject, 'dti', 'merged_avg33_mni_bbr.mgz')
    dti_file = nib.load(dti_fname)
    dti_data = dti_file.get_data()
    dti_header = dti_file.get_header()
    print('sdf')


if __name__ == '__main__':
    subject = 'hc008'
    load_dracula_merged(subject)
