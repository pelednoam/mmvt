import os.path as op
import os
import sys
import mne
from mne.datasets import sample

from src.utils import utils

MMVT_DIR = op.join(utils.get_links_dir(), 'mmvt')

code_root_fol = os.environ['MMVT_CODE']
if code_root_fol not in sys.path:
    sys.path.append(code_root_fol)

from src.mmvt_addon.scripts import run_mmvt
mmvt = run_mmvt.run('sample', run_in_background=False, debug=True)

data_path = sample.data_path()

raw_fname_sample = data_path + '/MEG/sample/sample_audvis_raw.fif'
fwd_fname_sample = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'

subjects_dir_sample = data_path + '/subjects'

# Read the forward solutions with surface orientation
fwd_sample = mne.read_forward_solution(fwd_fname_sample)
mne.convert_forward_solution(fwd_sample, surf_ori=True, copy=False)
eeg_map = mne.sensitivity_map(fwd_sample, ch_type='eeg', mode='fixed')
mmvt.coloring.plot_stc(eeg_map, threshold=0)
# eeg_map.save(op.join(MMVT_DIR, 'sample', 'meg', 'eeg_sensitivity_map'))
# eeg_map.plot(time_label='Gradiometer sensitivity', subjects_dir=subjects_dir_sample)
