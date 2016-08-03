import sys
import os
import os.path as op
import glob

try:
    from src.mmvt_addon.scripts import scripts_utils as su
except:
    # Add current folder the imports path
    sys.path.append(os.path.split(__file__)[0])
    import scripts_utils as su


def wrap_blender_call():
    args = read_args()
    if isinstance(args.pos_file, str):
        args.pos_file = [args.pos_file]
    for subject, pos_file in zip(args.subjects, args.pos_file):
        args.subjects = []
        args.subject = subject
        args.pos_file = pos_file
        su.call_script(__file__, args)


def read_args(argv=None):
    parser = su.add_default_args()
    parser.add_argument('-r', '--radius', help='radius', required=False, default=0.15, type=float)
    parser.add_argument('-p', '--pos_file', help='position file', required=False, default='', type=su.str_arr_type)
    args = su.parse_args(parser, argv)
    if len(args.subjects) == 0:
        args.subjects = [args.subject]
    if len(args.pos_file) == 0:
        args.pos_file = [''] * len(args.subjects)
    pos_files = []
    for subject, pos_file in zip(args.subjects, args.pos_file):
        args.subject = subject
        pos_files_fol = op.join(su.get_mmvt_dir(), args.subject, 'electrodes')
        if pos_file == '':
            pos_file_options = glob.glob(op.join(pos_files_fol, 'electrodes*positions*.npz'))
            if args.bipolar:
                pos_file_options = [fname for fname in pos_file_options if 'bipolar' in fname]
            else:
                pos_file_options = [fname for fname in pos_file_options if 'bipolar' not in fname]
            if len(pos_file_options) == 0:
                raise Exception('No electrodes position files in {}!'.format(pos_files_fol))
            elif len(pos_file_options) == 1:
                pos_file = pos_file_options[0]
                print('electrodes file: {}'.format(pos_file))
            else:
                raise Exception('More than one electrodes positions files in {}'.format(pos_files_fol) +
                                'please indicate which one using the -p flag')
        pos_files.append(pos_file)
    args.pos_file = pos_files
    if len(args.pos_file) == 1:
        args.pos_file = args.pos_file[0]
    return args


def import_electrodes(subject_fname):
    args = read_args(su.get_python_argv())
    mmvt = su.init_mmvt_addon()
    mmvt.import_electrodes(args.pos_file, args.bipolar, args.radius)
    mmvt.set_render_output_path = su.get_figures_dir(args)
    su.save_blend_file(subject_fname)
    su.exit_blender()


if __name__ == '__main__':
    import sys
    subject_fname = sys.argv[1]
    if sys.argv[2] == '--background':
        import_electrodes(subject_fname)
    else:
        wrap_blender_call()
