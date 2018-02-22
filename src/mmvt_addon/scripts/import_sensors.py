import sys
import os
import os.path as op

try:
    from src.mmvt_addon.scripts import scripts_utils as su
except:
    # Add current folder the imports path
    sys.path.append(os.path.split(__file__)[0])
    import scripts_utils as su

STAT_AVG, STAT_DIFF = range(2)


def wrap_blender_call(subject='', modality='', load_data=None, run_in_background=True, debug=None):
    args = read_args()
    if args is None:
        sys.argv = [__file__ , '-s', subject]
        args = read_args()
    if subject != '':
        args.subject = subject
    if modality != '':
        args.modality = modality
    if load_data is not None:
        args.load_data = load_data
    if debug is not None:
        args.debug = debug
    if args.modality not in ['meg', 'eeg']:
        print('args.modality should be eeg or meg!')
    su.call_script(__file__, args, run_in_background=run_in_background, err_pipe=sys.stdin)


def read_args(argv=None):
    parser = su.add_default_args()
    parser.add_argument('--stat', help='conds stat', required=False, default=STAT_DIFF)
    parser.add_argument('--modality', help='modality', required=False, default='meg')
    parser.add_argument('--load_data', help='load_data', required=False, default=1, type=su.is_true)
    return su.parse_args(parser, argv)


def wrap_mmvt_calls(subject_fname):
    args = read_args(su.get_python_argv())
    if args.debug:
        su.debug()
    mmvt = su.init_mmvt_addon()
    if args.modality == 'meg':
        mmvt.import_meg_sensors()
        if args.load_data:
            mmvt.add_data_to_meg_sensors(args.stat)
    elif args.modality == 'eeg':
        mmvt.import_eeg_sensors()
        if args.load_data:
            mmvt.add_data_to_eeg_sensors()
    else:
        print('args.modality should be eeg or meg!')
    su.save_blend_file(subject_fname)
    su.exit_blender()


if __name__ == '__main__':
    import sys
    if op.isfile(sys.argv[0]) and sys.argv[0][-2:] == 'py':
        wrap_blender_call()
    else:
        wrap_mmvt_calls(sys.argv[1])