import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from src.utils import utils
from src.utils import preproc_utils as pu
from src.preproc import meg
from src.preproc import connectivity as con

links_dir = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(links_dir, 'subjects', 'SUBJECTS_DIR')
MMVT_DIR = utils.get_link_dir(links_dir, 'mmvt')
HCP_DIR = utils.get_link_dir(links_dir, 'hcp')
MEG_DIR = utils.get_link_dir(links_dir, 'meg')
recordings_path = op.join(HCP_DIR, 'hcp-meg')


def analyze_fMRI_rest(subject):
    h = nib.load(op.join(HCP_DIR, subject, 'MNINonLinear', 'Results', 'rfMRI_REST1_LR',
                         'rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii'))
    x = h.get_data().squeeze()
    print(x.shape)
    h2 =
    print('asdfa')
    pass


if __name__ == '__main__':
    subject = '100307'
    analyze_fMRI_rest(subject)
    print('Finish!')