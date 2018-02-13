import os
import os.path as op
from pizco import Proxy
import mne

os.chdir(os.environ['MMVT_CODE'])
from src.mmvt_addon.scripts import run_mmvt


subject = 'sample'
atlas = 'dkt'
run_mmvt.run(subject, atlas, run_in_background=False)
mmvt = Proxy('tcp://127.0.0.1:8000')
mne_sample_data_fol = mne.datasets.sample.data_path()
stc_fname = op.join(mne_sample_data_fol, 'MEG', 'sample', 'sample_audvis-meg-rh.stc')
mmvt.plot_stc(stc_fname, 10, threshold=0)
