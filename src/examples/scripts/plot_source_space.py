import os.path as op
import numpy as np
import glob
import mne


def run(mmvt):
    stcs = glob.glob(op.join(op.join(mmvt.utils.get_user_fol(), 'meg', '*.stc')))
    print('plot_source_space: reading {}'.format(stcs[0]))
    stc = mne.read_source_estimate(stcs[0])
    print('rh: {}, lh: {}'.format(len(stc.rh_vertno), len(stc.lh_vertno)))
    for hemi in mmvt.utils.HEMIS:
        vertices = stc.lh_vertno if hemi=='lh' else stc.rh_vertno
        data = np.zeros((max(vertices) + 1, 4))
        data[vertices, 0] = 1
        data[vertices, 1:] = [0, 1, 0]
        mmvt.coloring.color_hemi_data(hemi, data)

