import os.path as op
import glob
from src.utils import utils
from src.preproc import meg
from src.preproc import fMRI as fmri
from src.preproc import connectivity as con

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')


def get_empty_fnames(subject, args):
    utils.make_dir(op.join(MEG_DIR, subject))
    utils.make_link(op.join(args.remote_subject_dir.format(subject=subject), 'bem'),
                    op.join(MEG_DIR, subject, 'bem'))
    utils.make_link(op.join(MEG_DIR, subject, 'bem'), op.join(SUBJECTS_DIR, subject, 'bem'))

    remote_meg_fol = args.remote_meg_dir.format(subject=subject)
    csv_fname = op.join(remote_meg_fol, 'cfg.txt')
    if not op.isfile(csv_fname):
        print('No cfg file!')
        return '', '', ''
    day, empty_fname, cor_fname, local_rest_raw_fname = '', '', '', ''
    for line in utils.csv_file_reader(csv_fname, ' '):
        if line[4].lower() == 'resting':
            day = line[2]
            remote_rest_raw_fname = op.join(remote_meg_fol, line[0].zfill(3), line[-1])
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
                remote_empty_fname = op.join(remote_meg_fol, line[0].zfill(3), line[-1])
                if not op.isfile(remote_empty_fname):
                    raise Exception('empty file does not exist! {}'.format(remote_empty_fname))
                utils.make_link(remote_empty_fname, empty_fname)
    cor_dir = op.join(args.remote_subject_dir.format(subject=subject), 'mri', 'T1-neuromag', 'sets')
    if op.isfile(op.join(cor_dir, 'COR-{}-resting.fif'.format(subject))):
        cor_fname = op.join(cor_dir, 'COR-{}-resting.fif'.format(subject))
    elif op.isfile(op.join(cor_dir, 'COR-{}-day{}.fif'.format(subject, day))):
        cor_fname = op.join(cor_dir, 'COR-{}-day{}.fif'.format(subject, day))
    return local_rest_raw_fname, empty_fname, cor_fname


def analyze_meg(args):
    subjects = args.subject
    for subject, mri_subject in zip(subjects, args.mri_subject):
        local_rest_raw_fname, empty_fname, cor_fname = get_empty_fnames(subject, args)
        if not op.isfile(empty_fname) or not op.isfile(cor_fname):
            print('{}: Can\'t find empty, raw, or cor files!'.format(subject))
            continue
        args = meg.read_cmd_args(dict(
            subject=subject,
            mri_subject=mri_subject,
            atlas='laus125',
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
    args = con.read_cmd_args(utils.Bag(
        subject=args.subject,
        atlas='laus125',
        function='calc_lables_connectivity',
        connectivity_modality='meg',
        connectivity_method='pli',
        windows_length=500,
        windows_shift=100,
        # sfreq=1000.0,
        # fmin=10,
        # fmax=100
        # recalc_connectivity=True,
        # max_windows_num=100,
        recalc_connectivity=True,
        n_jobs=args.n_jobs
    ))
    con.call_main(args)


def analyze_rest_fmri(args):
    args = fmri.read_cmd_args(dict(
        subject=args.subject,
        atlas=args.atlas,
        function='clean_4d_data',
        fmri_file_template='rest.nii*',
        fsd='rest_linda'
        # template_brain='fsaverage5',
    ))
    pu.run_on_subjects(args, fmri.main)

    args = fmri.read_cmd_args(dict(
        subject=args.subject,
        atlas=args.atlas,
        function='analyze_4d_data',
        # fmri_file_template='fmcpr.up.sm6.{subject}.{hemi}.nii.gz',
        fmri_file_template='{subject}_bld???_rest_reorient_skip_faln_mc_g1000000000_bpss_resid_{hemi}.mgz',
        # template_brain='fsaverage5',
        # template_brain='fsaverage6',
        # labels_extract_mode='mean,pca,pca_2,pca_4,pca_8',
        labels_extract_mode='mean',
        overwrite_labels_data=True
    ))
    pu.run_on_subjects(args, fmri.main)




if __name__ == '__main__':
    import argparse
    from src.utils import args_utils as au
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-m', '--mri_subject', help='subject name', required=False, default='')
    parser.add_argument('-f', '--function', help='function name', required=False, default='analyze_meg')
    parser.add_argument('--remote_meg_dir', required=False,
                        default='/autofs/space/lilli_003/users/DARPA-TRANSFER/meg/{subject}')
    parser.add_argument('--remote_subject_dir', required=False,
                        default='/autofs/space/lilli_001/users/DARPA-Recons/{subject}')
    parser.add_argument('--n_jobs', help='cpu num', required=False, default=-1)
    args = utils.Bag(au.parse_parser(parser))
    if args.mri_subject == '':
        args.mri_subject = args.subject
    locals()[args.function](args)
