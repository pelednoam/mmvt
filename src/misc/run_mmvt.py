import os
import os.path as op
from pizco import Proxy

os.chdir(os.environ['MMVT_CODE'])
from src.mmvt_addon.scripts import run_mmvt
from src.utils import utils

subject, atlas = 'DC', 'laus250'
run_mmvt.run(subject, atlas, run_in_background=False, debug=True)
mmvt = Proxy('tcp://127.0.0.1:8001')
meg_dir = utils.get_link_dir(utils.get_links_dir(), 'meg')
stc_fname = op.join(meg_dir, subject, 'left-MNE-1-15-lh.stc')
mmvt.plot_stc(stc_fname, 10, threshold=0)
