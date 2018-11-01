import argparse
from src.preproc import eeg as eeg
from src.utils import utils
from src.utils import args_utils as au


def setup_args(**kwargs):
    kwargs = utils.Bag(kwargs)
    args = eeg.read_cmd_args(['-s', kwargs.subject, '-m', kwargs.mri_subject])
    args.atlas = kwargs.atlas
    return args


# def calc_msit_evoked(**kwargs):
#     # -s ep001 -m mg78 -a laus250 -t MSIT --contrast interference --files_includes_cond 1 --read_events_from_file 1 --calc_epochs_from_raw 1 -f calc_evoked
#     args = setup_args(**kwargs)
#     args.function = 'calc_evoked,save_evoked_to_blender'
#     args.task = 'MSIT'
#     args.constrast = 'interference'
#     args.t_tmin = -0.5
#     args.t_tmax = 2
#     args.files_includes_cond = True
#     args.read_events_from_file = True
#     args.calc_epochs_from_raw = True
#     eeg.run_on_subjects(args)


def calc_msit_evoked(args):
    # python -m src.preproc.eeg -s ep001 -m mg78 -a laus125 -f calc_epochs,calc_evokes -t MSIT
    #   --contrast interference --t_max 2 --t_min -0.5 --data_per_task 1 --read_events_from_file 1
    #   --events_file_name {subject}_msit_nTSSS_interference-eve.txt --cleaning_method nTSSS
    args = eeg.read_cmd_args(dict(
        subject=args.subject,
        mri_subject=args.mri_subject,
        task='MSIT',
        function='calc_epochs,calc_evokes',
        data_per_task=True,
        atlas='laus125',
        contrast='interference',
        cleaning_method='nTSSS',
        t_min=-0.5,
        t_max=2,
        read_events_from_file=True,
        normalize_data=False,
        events_file_name='{subject}_msit_nTSSS_interference-eve.txt',
    ))
    eeg.call_main(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-m', '--mri_subject', help='mri subject name', required=False, default=None,
                        type=au.str_arr_type)
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='aparc.DKTatlas')
    parser.add_argument('-f', '--function', help='function name', required=True)
    args = utils.Bag(au.parse_parser(parser))
    if not args.mri_subject:
        args.mri_subject = args.subject
    for subject, mri_subject in zip(args.subject, args.mri_subject):
        kwargs = utils.Bag(dict(args))
        kwargs['subject'], kwargs['mri_subject'] = subject, mri_subject
        locals()[args.function](kwargs)