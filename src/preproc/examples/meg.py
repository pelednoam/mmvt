import argparse
from src.preproc import meg as meg
from src.utils import utils
from src.utils import args_utils as au


def read_epoches_and_calc_activity(subject, mri_subject):
    args = meg.read_cmd_args(['-s', subject, '-m', mri_subject])
    args.function = ['calc_stc_per_condition', 'calc_labels_avg_per_condition', 'smooth_stc', 'save_activity_map']
    args.pick_ori = 'normal'
    args.colors_map = 'jet'
    meg.run_on_subjects(args)


def calc_single_trial_labels_msit(subject, mri_subject):
    args = meg.read_cmd_args(['-s', subject, '-m', mri_subject])
    args.task = 'MSIT'
    args.atlas = 'laus250'
    args.function = 'calc_stc_per_condition,calc_single_trial_labels_per_condition'
    args.t_tmin = -0.5
    args.t_tmax = 2
    args.single_trial_stc = True
    args.fwd_no_cond = False
    args.files_includes_cond = True
    args.constrast = 'interference'
    meg.run_on_subjects(args)


def calc_msit_evoked(subject, mri_subject):
    args = meg.read_cmd_args(['-s', subject, '-m', mri_subject])
    args.task = 'MSIT'
    args.atlas = 'laus250'
    args.function = 'calc_evoked'
    args.t_tmin = -0.5
    args.t_tmax = 2
    args.calc_epochs_from_raw = True
    args.read_events_from_file = True
    args.remote_subject_meg_dir = '/autofs/space/sophia_002/users/DARPA-MEG/project_orig_msit/events'
    args.events_file_name = '{subject}_msit_nTSSS_interference-eve.txt'
    args.reject = False
    args.pick_eeg = True
    meg.run_on_subjects(args)


def crop_stc_no_baseline(subject, mri_subject):
    args = meg.read_cmd_args(['-s', subject, '-m', mri_subject])
    args.fname_format = '{subject}_02_f2-35_all_correct_combined'
    args.inv_fname_format = '{subject}_02_f2-35-ico-5-meg-eeg'
    args.stc_t_min = -0.1
    args.stc_t_max = 0.15
    args.base_line_max = None
    meg.run_on_subjects(args)


def check_files_names(subject, mri_subject):
    args = meg.read_cmd_args(['-s', subject, '-m', mri_subject])
    args.fname_format = '{subject}_02_f2-35_all_correct_combined'
    args.inv_fname_format = '{subject}_02_f2-35-ico-5-meg-eeg'
    args.function = 'print_names'
    meg.run_on_subjects(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-m', '--mri_subject', help='mri subject name', required=False, default=None,
                        type=au.str_arr_type)
    parser.add_argument('-f', '--function', help='function name', required=True)
    args = utils.Bag(au.parse_parser(parser))
    if not args.mri_subject:
        args.mri_subject = args.subject
    for subject, mri_subject in zip(args.subject, args.mri_subject):
        locals()[args.function](subject, mri_subject)