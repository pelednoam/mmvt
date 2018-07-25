import os.path as op
import glob
import numpy as np
from src.utils import utils
from src.preproc import anatomy as anat
from src.preproc import meg
from src.preproc import fMRI as fmri
from src.preproc import connectivity
from src.utils import freesurfer_utils as fu

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
FMRI_DIR = utils.get_link_dir(utils.get_links_dir(), 'fMRI')
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')


def init_anatomy(args):
    args = anat.read_cmd_args(dict(
        subject=args.subject,
        remote_subject_dir=args.remote_subject_dir,
        exclude='create_new_subject_blend_file',
        ignore_missing=True
    ))
    anat.call_main(args)


def init_meg(subject):
    utils.make_dir(op.join(MEG_DIR, subject))
    utils.make_link(op.join(args.remote_subject_dir.format(subject=subject), 'bem'),
                    op.join(MEG_DIR, subject, 'bem'))
    utils.make_link(op.join(MEG_DIR, subject, 'bem'), op.join(SUBJECTS_DIR, subject, 'bem'))


def get_meg_empty_fnames(subject, remote_fol, args):
    csv_fname = op.join(remote_fol, 'cfg.txt')
    if not op.isfile(csv_fname):
        print('No cfg file!')
        return '', '', ''
    day, empty_fname, cor_fname, local_rest_raw_fname = '', '', '', ''
    for line in utils.csv_file_reader(csv_fname, ' '):
        if line[4].lower() == 'resting':
            day = line[2]
            remote_rest_raw_fname = op.join(remote_fol, line[0].zfill(3), line[-1])
            if not op.isfile(remote_rest_raw_fname):
                raise Exception('rest file does not exist! {}'.format(remote_rest_raw_fname))
            local_rest_raw_fname = op.join(MEG_DIR, subject, '{}_resting_raw.fif'.format(subject))
            if not op.isfile(local_rest_raw_fname):
                utils.make_link(remote_rest_raw_fname, local_rest_raw_fname)
            break
    if day == '':
        print('Couldn\'t find the resting day in the cfg!')
        return '', '', ''
    for line in utils.csv_file_reader(csv_fname, ' '):
        if line[4] == 'empty':
            empty_fname = op.join(MEG_DIR, subject, '{}_empty_raw.fif'.format(subject))
            if op.isfile(empty_fname):
                continue
            if line[2] == day:
                remote_empty_fname = op.join(remote_fol, line[0].zfill(3), line[-1])
                if not op.isfile(remote_empty_fname):
                    raise Exception('empty file does not exist! {}'.format(remote_empty_fname))
                utils.make_link(remote_empty_fname, empty_fname)
    cor_dir = op.join(args.remote_subject_dir.format(subject=subject), 'mri', 'T1-neuromag', 'sets')
    if op.isfile(op.join(cor_dir, 'COR-{}-resting.fif'.format(subject))):
        cor_fname = op.join(cor_dir, 'COR-{}-resting.fif'.format(subject))
    elif op.isfile(op.join(cor_dir, 'COR-{}-day{}.fif'.format(subject, day))):
        cor_fname = op.join(cor_dir, 'COR-{}-day{}.fif'.format(subject, day))
    return local_rest_raw_fname, empty_fname, cor_fname


def get_fMRI_rest_fol(subject, remote_root):
    remote_fol = op.join(remote_root, '{}_01'.format(subject.upper()))
    csv_fname = op.join(remote_fol, 'cfg.txt')
    if not op.isfile(csv_fname):
        print('No cfg file!')
        return '', '', ''
    num = None
    for line in utils.csv_file_reader(csv_fname, '\t'):
        if line[1].lower() == 'resting':
            num = line[0]
            break
    if num is None:
        raise Exception('Can\'t find rest in the cfg file for {}!'.format(subject))
    subject_folders = glob.glob(op.join(remote_root, '{}_*'.format(subject.upper())))
    rest_fols = []
    for subject_fol in subject_folders:
        rest_fols = glob.glob(op.join(subject_fol, '**', num.zfill(3)), recursive=True)
        if len(rest_fols) == 1:
            break
    if len(rest_fols) == 0:
        raise Exception('Can\'t find rest in the cfg file for {}!'.format(subject))
    return rest_fols[0]


def convert_rest_dicoms_to_mgz(subject, rest_fol):
    output_fname = op.join(FMRI_DIR, subject, '{}_rest.mgz'.format(subject))
    if op.isfile(output_fname):
        return output_fname
    dicom_files = glob.glob(op.join(rest_fol, 'MR*'))
    dicom_files.sort(key=op.getmtime)
    fu.mri_convert(dicom_files[0], output_fname)
    if op.isfile(output_fname):
        return output_fname
    else:
        raise Exception('Can\'t find {}!'.format(output_fname))


def analyze_meg(args):
    subjects = args.subject
    for subject, mri_subject in zip(subjects, args.mri_subject):
        init_meg(subject)
        local_rest_raw_fname, empty_fname, cor_fname = get_meg_empty_fnames(
            subject, args.remote_meg_dir.format(subject=subject.upper()), args)
        if not op.isfile(empty_fname) or not op.isfile(cor_fname):
            print('{}: Can\'t find empty, raw, or cor files!'.format(subject))
            continue
        args = meg.read_cmd_args(dict(
            subject=subject,
            mri_subject=mri_subject,
            atlas=args.atlas,
            function='rest_functions',
            task='rest',
            reject=True, # Should be True here, unless you are dealling with bad data...
            remove_power_line_noise=True,
            l_freq=3, h_freq=80,
            windows_length=500,
            windows_shift=100,
            inverse_method='MNE',
            raw_fname=local_rest_raw_fname,
            cor_fname=cor_fname,
            empty_fname=empty_fname,
            remote_subject_dir=args.remote_subject_dir,
            # This properties are set automatically if task=='rest'
            # calc_epochs_from_raw=True,
            # single_trial_stc=True,
            # use_empty_room_for_noise_cov=True,
            # windows_num=10,
            # baseline_min=0,
            # baseline_max=0,
        ))
        meg.call_main(args)


def calc_meg_connectivity(args):
    args = connectivity.read_cmd_args(utils.Bag(
        subject=args.subject,
        atlas='laus125',
        function='calc_lables_connectivity',
        connectivity_modality='meg',
        connectivity_method='pli',
        windows_num=1,
        # windows_length=500,
        # windows_shift=100,
        recalc_connectivity=True,
        n_jobs=args.n_jobs
    ))
    connectivity.call_main(args)


def analyze_rest_fmri(args):
    for subject in args.mri_subject:
        remote_rest_fol = get_fMRI_rest_fol(subject, args.remote_fmri_dir)
        local_rest_fname = convert_rest_dicoms_to_mgz(subject, remote_rest_fol)
        if not op.isfile(local_rest_fname):
            print('{}: Can\'t find {}!'.format(subject, local_rest_fname))
            continue
        args = fmri.read_cmd_args(dict(
            subject=subject,
            atlas=args.atlas,
            remote_subject_dir=args.remote_subject_dir,
            function='clean_4d_data',
            fmri_file_template=local_rest_fname,
        ))
        fmri.call_main(args)

        args = fmri.read_cmd_args(dict(
            subject=args.subject,
            atlas=args.atlas,
            function='analyze_4d_data',
            fmri_file_template='rest.sm6.{subject}.{hemi}.mgz',
            labels_extract_mode='mean',
            overwrite_labels_data=False
        ))
        fmri.call_main(args)

        args = connectivity.read_cmd_args(dict(
            subject=args.subject,
            atlas=args.atlas,
            function='calc_lables_connectivity',
            connectivity_modality='fmri',
            connectivity_method='corr',
            labels_extract_mode='mean',
            identifier='',
            save_mmvt_connectivity=True,
            calc_subs_connectivity=False,
            recalc_connectivity=True,
            n_jobs=args.n_jobs
        ))
        connectivity.call_main(args)


def merge_connectivity(args):
    for subject in args.mri_subject:
        output_fname = op.join(MMVT_DIR, subject, 'connectivity', 'meg_fmri.npz')
        meg_con = np.abs(np.load(op.join(MMVT_DIR, subject, 'connectivity', 'meg_static_pli.npy')).squeeze())
        fmri_con = np.abs(np.load(op.join(MMVT_DIR, subject, 'connectivity', 'fmri_static_corr.npy')).squeeze())
        meg_con[np.triu_indices(meg_con.shape[0])] = 0
        fmri_con[np.triu_indices(fmri_con.shape[0])] = 0
        meg_top_k = utils.top_n_indexes(meg_con, args.top_k)
        fmri_top_k = utils.top_n_indexes(fmri_con, args.top_k)
        if len(set(fmri_top_k).intersection(set(meg_top_k))):
            print('fmri and meg top k intersection!')
            continue
        con_meg = np.zeros(meg_con.shape)
        for meg_top in meg_top_k:
            con_meg[meg_top] = meg_con[meg_top]
        con_meg /= np.max(con_meg)
        con_fmri = np.zeros(meg_con.shape)
        for fmri_top in fmri_top_k:
            con_fmri[fmri_top] = fmri_con[fmri_top]
        con_fmri /= np.max(con_fmri)
        con = con_fmri - con_meg
        if len(np.where(con)[0]) != args.top_k * 2:
            print('Wrong number of values in the conn matrix!'.format(len(np.where(con)[0])))
            continue
        d = utils.Bag(np.load(op.join(MMVT_DIR, subject, 'connectivity', 'meg_static_pli.npz')))
        con_vertices_fname = op.join(MMVT_DIR, subject, 'connectivity', 'meg_fmri_vertices.pkl')
        conn_args = connectivity.read_cmd_args(dict(subject=subject, atlas=args.atlas, norm_by_percentile=False))
        connectivity.save_connectivity(
            subject, con, 'cor-pli', connectivity.ROIS_TYPE, d.labels, d.conditions, output_fname, conn_args,
            con_vertices_fname)


# def normalize_data(x, min_x):
#     x -=

if __name__ == '__main__':
    import argparse
    from src.utils import args_utils as au
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-m', '--mri_subject', help='subject name', required=False, default='')
    parser.add_argument('-a', '--atlas', required=False, default='laus125')
    parser.add_argument('-f', '--function', help='function name', required=False, default='analyze_meg')
    parser.add_argument('--top_k', required=False, default=10)
    parser.add_argument('--remote_meg_dir', required=False,
                        default='/autofs/space/lilli_003/users/DARPA-TRANSFER/meg/{subject}')
    parser.add_argument('--remote_fmri_dir', required=False,
                        default='/autofs/space/lilli_003/users/DARPA-TRANSFER/mri')
    parser.add_argument('--remote_subject_dir', required=False,
                        default='/autofs/space/lilli_001/users/DARPA-Recons/{subject}')
    parser.add_argument('--n_jobs', help='cpu num', required=False, default=-1)
    args = utils.Bag(au.parse_parser(parser))
    if args.mri_subject == '':
        args.mri_subject = args.subject
    locals()[args.function](args)
