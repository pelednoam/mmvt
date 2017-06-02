import os.path as op
import glob
from src.utils import utils
from src.utils import preproc_utils as pu

SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()

for folder in glob.glob(op.join(MMVT_DIR, 'nmr*')):
    sub = utils.namebase(folder)
    pu.backup_folder(sub, 'fmri')
    pu.backup_folder(sub, 'connectivity')
