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
    parser.add_argument('--no_mni', help='no mni pos files', required=False, default=True, type=su.is_true)
    args = su.parse_args(parser, argv)
    if len(args.subjects) == 0:
        args.subjects = args.subject.split(',')
        # args.subjects = [args.subject]
    if len(args.pos_file) == 0:
        args.pos_file = [''] * len(args.subjects)
    args.subjects = su.decode_subjects(args.subjects)
    pos_files = []
    for subject, pos_file in zip(args.subjects, args.pos_file):
        args.subject = subject
        pos_files_fol = op.join(su.get_mmvt_dir(), args.subject, 'electrodes')
        if pos_file == '':
            pos_file_template = op.join(pos_files_fol, 'electrodes*positions*.npz')
            pos_file_options = glob.glob(pos_file_template)
            if args.bipolar:
                pos_file_options = [fname for fname in pos_file_options if 'bipolar' in fname]
            else:
                pos_file_options = [fname for fname in pos_file_options if 'bipolar' not in fname]
            if args.no_mni:
                pos_file_options = [fname for fname in pos_file_options if '_mni' not in su.namebase(fname)]
            if len(pos_file_options) == 0:
                raise Exception('No electrodes position files ({}) in {}!'.format(
                    op.join(pos_files_fol, 'electrodes*positions*.npz'), pos_files_fol))
            elif len(pos_file_options) == 1:
                pos_file = pos_file_options[0]
                print('electrodes file: {}'.format(pos_file))
            else:
                pos_file = su.select_one_file(pos_file_options, pos_file_template, 'electrodes positions files')
        else:
            pos_file = op.join(pos_files_fol, '{}.npz'.format(su.namebase(pos_file)))
        if not op.isfile(pos_file):
            raise ("Can't find pos file! {}".format(pos_file))
        pos_files.append(pos_file)
    args.pos_file = pos_files
    if len(args.pos_file) == 1:
        args.pos_file = args.pos_file[0]
    return args


def wrap_mmvt_calls(subject_fname):
    args = read_args(su.get_python_argv())
    if args.debug:
        su.debug()
    mmvt = su.init_mmvt_addon()
    mmvt.utils.write_to_stderr('{} Importing electrodes...'.format(args.subject))
    mmvt.import_electrodes(args.pos_file, mmvt.ELECTRODES_LAYER, args.bipolar, args.radius)
    mmvt.set_render_output_path = su.get_figures_dir(args)
    su.save_blend_file(subject_fname)
    mmvt.utils.write_to_stderr('{} Done!'.format(args.subject))
    su.exit_blender()


if __name__ == '__main__':
    import sys
    if op.isfile(sys.argv[0]) and sys.argv[0][-2:] == 'py':
        wrap_blender_call()
    else:
        wrap_mmvt_calls(sys.argv[1])
