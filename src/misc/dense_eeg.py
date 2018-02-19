import sys
import os.path as op
import glob
import shutil
from functools import partial

from src.preproc import eeg
from src.preproc import meg
from src.utils import utils

LINKS_DIR = utils.get_links_dir()
EEG_DIR = utils.get_link_dir(LINKS_DIR, 'eeg')


def main(eeg_fol, subject):
    raw_files = glob.glob(op.join(eeg_fol, '*_raw.fif'))
    for raw_file in raw_files:
        # if '-s' not in sys.argv:
        #     sys.argv.append('-s')
        #     sys.argv.append(subject)
        args = eeg.read_cmd_args(subject=subject)
        if not op.isfile(op.join(EEG_DIR, utils.namebase_with_ext(raw_file))):
            shutil.copy(raw_file, EEG_DIR)
        args.raw_template = raw_file
        cond = raw_file.split('_')[1]
        args.conditions = [cond]
        args.files_includes_cond = True
        eeg.init(subject, args)
        # args.fname_format = '{subject}_{cond}-{ana_type}.{file_type}''
        eeg.read_eeg_sensors_layout(subject, subject, args)
        eeg.calc_evokes(subject, {cond:0}, args)


if __name__ == '__main__':
    # eeg_fol = '/cluster/neuromind/npeled/MGH_Highdensity_source/EEG'
    subject = 'mgh128'
    eeg_fol = op.join(EEG_DIR, subject)
    EEG_DIR = utils.make_dir(op.join(eeg.SUBJECTS_EEG_DIR, subject))
    main(eeg_fol, subject)