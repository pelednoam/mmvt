import nibabel as nib
import os.path as op
from src.utils import utils
import numpy as np
import os

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
FMRI_DIR = utils.get_link_dir(LINKS_DIR, 'fMRI')
DTI_DIR = op.join(FMRI_DIR, 'DTI')
BLENDER_ROOT_DIR = op.join(LINKS_DIR, 'mmvt')

AFFINE_TRANS = np.genfromtxt('/homes/5/npeled/space3/fMRI/DTI/hc008/dmri/xfms/diff2anatorig.bbr.mat')

# http://freesurfer.net/fswiki/FsTutorial/Diffusion

def load_tracula_merged(subject):
    dti_fname = op.join(SUBJECTS_DIR, subject, 'dti', 'merged_avg33_mni_bbr.mgz')
    dti_file = nib.load(dti_fname)
    dti_data = dti_file.get_data()
    dti_header = dti_file.get_header()
    print('sdf')


def load_tracula_trk(subject):
    tracks_fols = utils.get_subfolders(op.join(DTI_DIR, subject, 'dpath'))
    output_fol = op.join(BLENDER_ROOT_DIR, subject, 'dti', 'tracula')
    utils.make_dir(output_fol)
    for track_fol in tracks_fols:
        track_fol_name = os.path.basename(track_fol)
        print('Reading {}'.format(track_fol_name))
        track_gen, hdr = nib.trackvis.read(op.join(track_fol, 'path.pd.trk'), as_generator=True, points_space='rasmm')
        hdr = convert_header(hdr)
        vox2ras_trans = get_vox2ras_trans(subject)
        tracks = read_tracks(track_gen, hdr, vox2ras_trans)
        output_fname = op.join(output_fol, '{}.pkl'.format(track_fol_name))
        utils.save(tracks, output_fname)
        print('Save in {}'.format(output_fname))


def convert_header(hdr):
    props = ['id_string', 'dim', 'voxel_size', 'origin', 'n_scalars', 'scalar_name', 'n_properties', 'property_name',
        'vox_to_ras', 'reserved', 'voxel_order', 'pad2', 'image_orientation_patient', 'pad1', 'invert_x', 'invert_y',
        'invert_x', 'swap_xy', 'swap_yz', 'swap_zx', 'n_count', 'version', 'hdr_size']
    hdr_dict = {}
    for prop in props:
        hdr_dict[prop] = hdr[prop]
    return hdr_dict


def get_vox2ras_trans(subject):
    aseg_fname = op.join(SUBJECTS_DIR, subject, 'mri', 'aseg.mgz')
    aseg = nib.load(aseg_fname)
    aseg_hdr = aseg.get_header()
    trans = aseg_hdr.get_vox2ras_tkr()
    return trans


def read_tracks(track_gen, hdr, vox2ras_trans):
    from mne.transforms import apply_trans
    tracks = []
    # zooms = hdr['voxel_size']
    # affine = hdr['vox_to_ras']
    # tv2vx = np.diag((1. / zooms).tolist() + [1])
    # tv2mm = np.dot(affine, tv2vx).astype('f4')

    while True:
        try:
            track = next(track_gen)
        except (StopIteration, TypeError):
            break
        track = track[0]
        # track = apply_trans(hdr['vox_to_ras'], track[0])
        # track = nib.affines.apply_affine(AFFINE_TRANS, track)
        # track = nib.affines.apply_affine(hdr['vox_to_ras'], track)
        # zoom = np.diag([1., 1., 0.5])
        # track = np.dot(track, zoom)
        tracks.append(track)
    return tracks


if __name__ == '__main__':
    subject = 'hc008'
    # load_tracula_merged(subject)
    load_tracula_trk(subject)
