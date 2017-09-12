import os.path as op
import numpy as np
import mne.io
import traceback

from src.utils import utils
from src.utils import preproc_utils as pu
from src.preproc import meg as meg

SUBJECTS_MRI_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()

LINKS_DIR = utils.get_links_dir()
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
SUBJECTS_EEG_DIR = utils.get_link_dir(LINKS_DIR, 'eeg')
if SUBJECTS_EEG_DIR == '':
    print('No EEG folder, using MEG folder')
    SUBJECTS_EEG_DIR = MEG_DIR
if SUBJECTS_EEG_DIR == '':
    raise Exception('No EEG folder (not MEG)!')
SUBJECT_EEG_DIR = ''


def read_eeg_sensors_layout(subject, mri_subject, args):
    return meg.read_sensors_layout(subject, mri_subject, args, pick_meg=False, pick_eeg=True)


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
    data_max, data_min = utils.get_data_max_min(data, args.norm_by_percentile, args.norm_percs)
    max_abs = utils.get_max_abs(data_max, data_min)
    if args.normalize_evoked:
        data = data / max_abs
    np.save(op.join(fol, 'eeg_data.npy'), data)
    np.savez(op.join(fol, 'eeg_data_meta.npz'), names=ch_names, conditions=list(events.keys()), dt=dt,
             minmax=(-max_abs, max_abs))
    return True


def calc_minmax(mri_subject, args):
    #todo: merge into save_evoked_to_blender
    fol = op.join(MMVT_DIR, mri_subject, 'eeg')
    data = np.load(op.join(fol, 'eeg_data.npy'))
    data_max, data_min = utils.get_data_max_min(np.diff(data), args.norm_by_percentile, args.norm_percs)
    max_abs = utils.get_max_abs(data_max, data_min)
    np.save(op.join(fol, 'eeg_data_minmax.npy'), [-max_abs, max_abs])
    return op.isfile(op.join(fol, 'eeg_data_minmax.npy'))


def create_eeg_mesh(subject, excludes=[], overwrite_faces_verts=True):
    try:
        from scipy.spatial import Delaunay
        from src.utils import trig_utils
        input_file = op.join(MMVT_DIR, subject, 'eeg', 'eeg_positions.npz')
        mesh_ply_fname = op.join(MMVT_DIR, subject, 'eeg', 'eeg_helmet.ply')
        faces_verts_out_fname = op.join(MMVT_DIR, subject, 'eeg', 'eeg_faces_verts.npy')
        f = np.load(input_file)
        verts = f['pos']
        excluded_inds = [np.where(f['names'] == e)[0] for e in excludes]
        # verts = np.delete(verts, excluded_inds, 0)
        verts_tup = [(x, y, z) for x, y, z in verts]
        tris = Delaunay(verts_tup)
        faces = tris.convex_hull
        areas = [trig_utils.poly_area(verts[poly]) for poly in tris.convex_hull]
        inds = [k for k, s in enumerate(areas) if s > np.percentile(areas, 97)]
        faces = np.delete(faces, inds, 0)
        utils.write_ply_file(verts, faces, mesh_ply_fname, True)
        utils.calc_ply_faces_verts(verts, faces, faces_verts_out_fname, overwrite_faces_verts,
                                   utils.namebase(faces_verts_out_fname))
        np.savez(input_file, pos=f['pos'], names=f['names'], tri=faces, excludes=excludes)
    except:
        print('Error in create_eeg_mesh!')
        print(traceback.format_exc())
        return False
    return True


def main(tup, remote_subject_dir, args, flags):
    (subject, mri_subject), inverse_method = tup
    evoked, epochs = None, None
    fname_format, fname_format_cond, conditions = meg.init_main(subject, mri_subject, remote_subject_dir, args)
    meg.init_globals_args(subject, mri_subject, fname_format, fname_format_cond, SUBJECTS_EEG_DIR, SUBJECTS_MRI_DIR,
                     MMVT_DIR, args)
    meg.MEG_DIR = SUBJECTS_EEG_DIR
    meg.FWD = meg.FWD_EEG
    meg.INV = meg.INV_EEG
    stat = meg.STAT_AVG if len(conditions) == 1 else meg.STAT_DIFF

    if utils.should_run(args, 'read_eeg_sensors_layout'):
        flags['read_eeg_sensors_layout'] = read_eeg_sensors_layout(subject, mri_subject, args)

    flags, evoked, epochs = meg.calc_evokes_wrapper(subject, conditions, args, flags, mri_subject=mri_subject)

    if utils.should_run(args, 'create_eeg_mesh'):
        flags['create_eeg_mesh'] = create_eeg_mesh(mri_subject, args.eeg_electrodes_excluded_from_mesh)

    if utils.should_run(args, 'save_evoked_to_blender'):
        flags['save_evoked_to_blender'] = save_evoked_to_blender(mri_subject, conditions, args, evoked)

    if utils.should_run(args, 'calc_minmax'):
        flags['calc_minmax'] = calc_minmax(mri_subject, args)

    if not op.isfile(meg.COR):
        eeg_cor = op.join(meg.SUBJECT_MEG_FOLDER, '{}-cor-trans.fif'.format(subject))
        if not op.isfile(eeg_cor):
            raise Exception("Can't find head-MRI transformation matrix. Should be in {} or in {}".format(meg.COR, eeg_cor))
        meg.COR = eeg_cor
    flags = meg.calc_fwd_inv_wrapper(subject, conditions, args, flags, mri_subject)
    flags = meg.calc_stc_per_condition_wrapper(subject, conditions, inverse_method, args, flags)
    return flags


def read_cmd_args(argv=None):
    args = meg.read_cmd_args(argv)
    args.pick_meg = False
    args.pick_eeg = True
    args.reject = False
    args.fwd_usingMEG = False
    args.fwd_usingEEG = True
    return args


if __name__ == '__main__':
    from src.utils import preproc_utils as pu
    from itertools import product

    args = read_cmd_args()
    subjects_itr = product(zip(args.subject, args.mri_subject), args.inverse_method)
    subject_func = lambda x:x[0][1]
    pu.run_on_subjects(args, main, subjects_itr, subject_func)
    print('finish!')
