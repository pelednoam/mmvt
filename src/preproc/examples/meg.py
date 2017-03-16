import argparse

from src.preproc import meg as meg
from src.utils import utils
from src.utils import args_utils as au
from src.utils import preproc_utils as pu


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


def calc_msit(args):
    args = meg.read_cmd_args(dict(
        subject=args.subject,
        mri_subject=args.mri_subject,
        task='MSIT',
        atlas='laus250',
        contrast='interference',
        cleaning_method='nTSSS',
        t_min=-0.5,
        t_max=2,
        calc_epochs_from_raw=True,
        read_events_from_file=True,
        remote_subject_meg_dir='/autofs/space/sophia_002/users/DARPA-MEG/project_orig_msit',
        events_file_name='{subject}_msit_nTSSS_interference-eve.txt',
        reject=False,
        pick_eeg=True
    ))
    meg.call_main(args)


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


def calc_subcorticals(subject, mri_subject):
    '''-s ep001 -m mg78 -f calc_evoked -t MSIT --contrast interference --cleaning_method nTSSS --data_per_task 1 --read_events_from_file 1 --t_min -0.5 t_max 2.0
    -s ep001 -m mg78 -f make_forward_solution,calc_inverse_operator -t MSIT --contrast interference --cleaning_method nTSSS --data_per_task 1 --fwd_calc_subcorticals 1 --inv_calc_subcorticals 1 --remote_subject_dir="/autofs/space/lilli_001/users/DARPA-Recons/ep001"
    -s ep001 -m mg78 -f calc_sub_cortical_activity,save_subcortical_activity_to_blender -t MSIT -i lcmv --contrast interference --cleaning_method nTSSS --data_per_task 1
    '''
    pass


def calc_rest(args):
    # '-s hc029 -a laus125 -t rest -f calc_evoked,make_forward_solution,calc_inverse_operator --reject 0 --remove_power_line_noise 0 --windows_length 1000 --windows_shift 500 --remote_subject_dir "/autofs/space/lilli_001/users/DARPA-Recons/hc029"''
    # '-s hc029 -a laus125 -t rest -f calc_stc_per_condition,calc_labels_avg_per_condition --single_trial_stc 1 --remote_subject_dir "/autofs/space/lilli_001/users/DARPA-Recons/hc029"'
    # '-s subject-name -a atlas-name -t rest -f rest_functions' --l_freq 8 --h_freq 13 --windows_length 500 --windows_shift 100
    args = meg.read_cmd_args(dict(
        subject=args.subject,
        mri_subject=args.mri_subject,
        atlas='laus125',
        function='rest_functions',
        task='rest',
        cleaning_method='tsss',
        reject=False, # Should be True here, unless you are dealling with bad data...
        remove_power_line_noise=True,
        l_freq=3, h_freq=80,
        windows_length=500,
        windows_shift=100,
        inverse_method='MNE',
        remote_subject_dir='/autofs/space/lilli_001/users/DARPA-Recons/{subject}',
        # This properties are set automatically if task=='rest'
        # calc_epochs_from_raw=True,
        # single_trial_stc=True,
        # use_empty_room_for_noise_cov=True,
        # windows_num=10,
        # baseline_min=0,
        # baseline_max=0,
    ))
    meg.call_main(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-m', '--mri_subject', help='mri subject name', required=False, default=None,
                        type=au.str_arr_type)
    parser.add_argument('-i', '--inverse_method', help='inverse_method', required=False, default='MNE',
                        type=au.str_arr_type)
    parser.add_argument('-f', '--function', help='function name', required=True)
    args = utils.Bag(au.parse_parser(parser))
    if not args.mri_subject:
        args.mri_subject = args.subject
    locals()[args.function](args)
    # for subject, mri_subject in zip(args.subject, args.mri_subject):
    #     locals()[args.function](subject, mri_subject)