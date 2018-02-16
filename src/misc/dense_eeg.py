import sys
import os.path as op
import glob
import shutil
from functools import partial

from src.preproc import eeg
from src.preproc import meg
from src.utils import utils

def main(eeg_fol, subject):
    raw_files = glob.glob(op.join(eeg_fol, '*_raw.fif'))
    for raw_file in raw_files:
        if '-s' not in sys.argv:
            sys.argv.append('-s')
            sys.argv.append(subject)
        args = eeg.read_cmd_args()
        if not op.isfile(op.join(EEG_DIR, utils.namebase_with_ext(raw_file))):
            shutil.copy(raw_file, EEG_DIR)
        eeg.init(subject, args)
        eeg.read_eeg_sensors_layout(subject, subject, args)


if __name__ == '__main__':
    eeg_fol = '/cluster/neuromind/npeled/MGH_Highdensity_source/EEG'
    subject = 'mgh128'
    EEG_DIR = utils.make_dir(op.join(eeg.SUBJECTS_EEG_DIR, subject))
    main(eeg_fol, subject)