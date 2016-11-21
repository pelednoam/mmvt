import os
import os.path as op
import numpy as np
import mne.io
import glob

from src.utils import utils
from src.preproc import meg as meg

LINKS_DIR = utils.get_links_dir()
SUBJECTS_MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
SUBJECTS_EEG_DIR = utils.get_link_dir(LINKS_DIR, 'eeg')
if SUBJECTS_EEG_DIR == '':
    print('No EEG folder, using MEG folder')
    SUBJECTS_EEG_DIR = SUBJECTS_MEG_DIR
if SUBJECTS_EEG_DIR == '':
    raise Exception('No EEG folder (not MEG)!')
SUBJECT_EEG_DIR = ''
SUBJECTS_MRI_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
FREESURFER_HOME = utils.get_link_dir(LINKS_DIR, 'freesurfer', 'FREESURFER_HOME')
print('FREE_SURFER_HOME: {}'.format(FREESURFER_HOME))
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')
os.environ['SUBJECTS_DIR'] = SUBJECTS_MRI_DIR


def read_eeg_sensors_layout(mri_subject):
    if not op.isfile(meg.INFO):
        raw = mne.io.read_raw_fif(meg.RAW)
        info = raw.info
        utils.save(info, meg.INFO)
    else:
        info = utils.load(meg.INFO)
    eeg_picks = mne.io.pick.pick_types(info, meg=False, eeg=True)
    eeg_pos = np.array([info['chs'][k]['loc'][:3] for k in eeg_picks])
    eeg_names = np.array([info['ch_names'][k] for k in eeg_picks])
    if 'Event' in eeg_names:
        event_ind = np.where(eeg_names == 'Event')[0]
        eeg_names = np.delete(eeg_names, event_ind)
        eeg_pos = np.delete(eeg_pos, event_ind)
    fol = op.join(MMVT_DIR, mri_subject, 'eeg')
    utils.make_dir(fol)
    output_fname = op.join(fol, 'eeg_positions.npz')
    if len(eeg_pos) > 0:
        trans_files = glob.glob(op.join(SUBJECTS_MRI_DIR, '*COR*.fif'))
        if len(trans_files) == 1:
            trans = mne.transforms.read_trans(trans_files[0])
            head_mri_t = mne.transforms._ensure_trans(trans, 'head', 'mri')
            eeg_pos = mne.transforms.apply_trans(head_mri_t, eeg_pos)
            eeg_pos *= 1000
            np.savez(output_fname, pos=eeg_pos, names=eeg_names)
            return True
    return False


def save_evoked_to_blender(mri_subject, events, args, evoked=None):
    fol = op.join(MMVT_DIR, mri_subject, 'eeg')
    utils.make_dir(fol)
    if '{cond}' in meg.EVO:
        for event_ind, event_id in enumerate(events.keys()):
            if evoked is None:
                evo = mne.read_evokeds(meg.get_cond_fname(meg.EVO, event_id))
            else:
                evo = evoked[event_id]
            if event_ind == 0:
                ch_names = np.array(evo[0].ch_names)
                dt = np.diff(evo[0].times[:2])[0]
                data = np.zeros((evo[0].data.shape[0], evo[0].data.shape[1], 2))
            data[:, :, event_ind] = evo[0].data
    else:
        if evoked is None:
            evoked = mne.read_evokeds(meg.EVO)
        data = evoked[0].data
        data = data[..., np.newaxis]
        ch_names = np.array(evoked[0].ch_names)
        dt = np.diff(evoked[0].times[:2])[0]
    if 'Event' in ch_names:
        event_ind = np.where(ch_names == 'Event')[0]
        ch_names = np.delete(ch_names, event_ind)
        data = np.delete(data, event_ind, 0)
    if args.normalize_evoked:
        data_max, data_min = utils.get_data_max_min(data, args.norm_by_percentile, args.norm_percs)
        max_abs = utils.get_max_abs(data_max, data_min)
        data = data / max_abs
    np.save(op.join(fol, 'eeg_data.npy'), data)
    np.savez(op.join(fol, 'eeg_data_meta.npz'), names=ch_names, conditions=list(events.keys()), dt=dt)
    return True


def create_eeg_mesh(subject):
    from scipy.spatial import Delaunay
    input_file = op.join(MMVT_DIR, subject, 'eeg', 'eeg_positions.npz')
    f = np.load(input_file)
    verts = [(x, y, z) for x, y, z in f['pos']]
    tri = Delaunay(verts)
    areas = [poly_area(f['pos'][poly]) for poly in tri.convex_hull]
    inds = [k for k, s in enumerate(areas) if s > np.percentile(areas, 97)]
    tris = np.delete(tri.convex_hull, inds, 0)
    np.savez(input_file, pos=f['pos'], names=f['names'], tri=tris)


#unit normal vector of plane defined by points a, b, and c
def unit_normal(a, b, c):
    x = np.linalg.det([[1,a[1],a[2]],
         [1,b[1],b[2]],
         [1,c[1],c[2]]])
    y = np.linalg.det([[a[0],1,a[2]],
         [b[0],1,b[2]],
         [c[0],1,c[2]]])
    z = np.linalg.det([[a[0],a[1],1],
         [b[0],b[1],1],
         [c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    return (x/magnitude, y/magnitude, z/magnitude)

#area of polygon poly
def poly_area(poly):
    if len(poly) < 3: # not a plane - no area
        return 0
    total = [0, 0, 0]
    N = len(poly)
    for i in range(N):
        vi1 = poly[i]
        vi2 = poly[(i+1) % N]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = np.dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)


def main(subject, mri_subject, inverse_method, args):
    evoked, epochs = None, None
    fname_format, fname_format_cond, conditions = meg.init_main(subject, mri_subject, args)
    meg.init_globals_args(subject, mri_subject, fname_format, fname_format_cond, SUBJECTS_EEG_DIR, SUBJECTS_MRI_DIR,
                     MMVT_DIR, args)
    meg.SUBJECTS_MEG_DIR = SUBJECTS_EEG_DIR
    meg.FWD = meg.FWD_EEG
    meg.INV = meg.INV_EEG
    stat = meg.STAT_AVG if len(conditions) == 1 else meg.STAT_DIFF
    flags = {}

    if utils.should_run(args, 'read_eeg_sensors_layout'):
        flags['read_eeg_sensors_layout'] = read_eeg_sensors_layout(mri_subject)

    flags = meg.calc_evoked_wrapper(subject, conditions, args, flags)

    if utils.should_run(args, 'create_eeg_mesh'):
        create_eeg_mesh(subject)

    if utils.should_run(args, 'save_evoked_to_blender'):
        flags['save_evoked_to_blender'] = save_evoked_to_blender(mri_subject, conditions, args, evoked)
    if not op.isfile(meg.COR):
        eeg_cor = op.join(meg.SUBJECT_MEG_FOLDER, '{}-cor-trans.fif'.format(subject))
        if not op.isfile(eeg_cor):
            raise Exception("Can't find head-MRI transformation matrix. Should be in {} or in {}".format(meg.COR, eeg_cor))
        meg.COR = eeg_cor
    flags = meg.calc_fwd_inv_wrapper(subject, mri_subject, conditions, args, flags)
    flags = meg.calc_stc_per_condition_wrapper(subject, conditions, inverse_method, args, flags)
    return flags


def run_on_subjects(args):
    from src.preproc.eeg import main as eeg_main
    meg.run_on_subjects(args, eeg_main)


def read_cmd_args(argv=None):
    args = meg.read_cmd_args(argv)
    args.pick_meg = False
    args.pick_eeg = True
    args.reject = False
    args.fwd_usingMEG = False
    args.fwd_usingEEG = True
    return args


if __name__ == '__main__':
    args = read_cmd_args()
    run_on_subjects(args)
    print('finish!')
