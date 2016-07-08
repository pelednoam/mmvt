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
    su.call_script(__file__, args)


def read_args(argv=None):
    parser = su.add_default_args()
    parser.add_argument('-b', '--bipolar', help='bipolar', required=True, type=su.is_true)
    parser.add_argument('-r', '--radius', help='radius', required=False, default=0.15, type=float)
    parser.add_argument('-p', '--pos_file', help='position file', required=False, default='')
    args = su.parse_args(parser, argv)
    pos_files_fol = op.join(su.get_mmvt_dir(), args.subject, 'electrodes')
    if args.pos_file == '':
        pos_files = glob.glob(op.join(pos_files_fol, 'electrodes*positions*.npz'))
        if args.bipolar:
            pos_files = [fname for fname in pos_files if 'bipolar' in fname]
        else:
            pos_files = [fname for fname in pos_files if 'bipolar' not in fname]
        if len(pos_files) == 0:
            raise Exception('No electrodes position files in {}!'.format(pos_files_fol))
        elif len(pos_files) == 1:
            args.pos_file = pos_files[0]
            print('electrodes file: {}'.format(args.pos_file))
        else:
            raise Exception('More than one electrodes positions files in {}'.format(pos_files_fol) +
                            'please indicate which one using the -p flag')
    return args


def import_meg(subject_fname):
    args = read_args(su.get_python_argv())
    mmvt = su.init_mmvt_addon()
    mmvt.import_electrodes(args.pos_file, args.radius)
    su.save_blend_file(subject_fname)
    su.exit_blender()


if __name__ == '__main__':
    import sys
    subject_fname = sys.argv[1]
    if sys.argv[2] == '--background':
        import_meg(subject_fname)
    else:
        wrap_blender_call()
