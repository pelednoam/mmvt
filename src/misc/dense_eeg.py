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
        eve_fname = f'{raw_file[:-4]}-annot.fif'
        if not op.isfile(eve_fname):
            print(f'No annot file for {raw_file}!')
            continue
        args.task = raw_file.split('_')[1]
        fol = utils.make_dir(op.join(EEG_DIR, subject, args.task))
        if not op.isfile(op.join(fol, utils.namebase_with_ext(raw_file))):
            shutil.copy(raw_file, fol)
        if not op.isfile(op.join(fol, utils.namebase_with_ext(eve_fname))):
            shutil.copy(eve_fname, fol)
        raw_file = op.join(fol, utils.namebase_with_ext(raw_file))
        eve_fname = op.join(fol, utils.namebase_with_ext(eve_fname))
        args.raw_template = args.raw_fname = raw_file # todo: why both?
        args.eve_template = eve_fname
        args.conditions = [1001, 1002]
        # args.files_includes_cond = False
        args.sub_dirs_for_tasks = True
        eeg.init(subject, args)
        args.t_min, args.t_max = -0.2, 0.2
        args.l_freq, args.h_freq = 1, 40

        eeg.read_eeg_sensors_layout(subject, subject, args)
        eeg.calc_evokes(subject, {'s1':1001, 's2':1002}, args)


if __name__ == '__main__':
    # eeg_fol = '/cluster/neuromind/npeled/MGH_Highdensity_source/EEG'
    subject = 'mgh128'
    eeg_fol = op.join(EEG_DIR, subject)
    main(eeg_fol, subject)