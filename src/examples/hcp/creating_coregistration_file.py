import os.path as op
import hcp

from src.utils import utils

links_dir = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(links_dir, 'subjects', 'SUBJECTS_DIR')
MMVT_DIR = utils.get_link_dir(links_dir, 'mmvt')
HCP_DIR = utils.get_link_dir(links_dir, 'hcp')


def make_mne_anatomy(subject):
    hcp.make_mne_anatomy(
         subject, subjects_dir=SUBJECTS_DIR,
         hcp_path=HCP_DIR, recordings_path=op.join(HCP_DIR, 'hcp-meg'))
    return op.isfile(op.join(op.join(HCP_DIR, 'hcp-meg', subject, '{}-head_mri-trans.fif'.format(subject))))


if __name__ == '__main__':
    subject = '100307'  # our test subject
    ret = make_mne_anatomy(subject)
    print('make_mne_anatomy: {}'.format(ret))