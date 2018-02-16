import sys
import os
import os.path as op
import shutil

try:
    from src.mmvt_addon.scripts import scripts_utils as su
except:
    # Add current folder the imports path
    sys.path.append(os.path.split(__file__)[0])
    import scripts_utils as su


def create_new_subject(subject, atlas, overwrite_blend=False):
    sys.argv = [__file__, '-s', subject, '-a', atlas, '--overwrite_blend', overwrite_blend]
    return wrap_blender_call()


def wrap_blender_call():
    args = read_args()
    create_new_subject_file(args)
    args.log_fname = op.join(su.get_logs_fol(args.subject), 'create_new_subject_script.log')
    if op.isfile(args.log_fname):
        os.remove(args.log_fname)
    if args.import_in_blender:
        su.call_script(__file__, args, run_in_background=False, err_pipe=sys.stdin)
        su.waits_for_file(args.log_fname)
    return args


def create_new_subject_file(args):
    # Create a file for the new subject
    if len(args.subjects) == 0:
        args.subjects = [args.subject]
    for subject in args.subjects:
        args.subject = subject
        new_fname = su.get_subject_fname(args)
        empty_subject_fname = op.join(su.get_mmvt_dir(), 'empty_subject.blend')
        if not op.isfile(empty_subject_fname):
            shutil.copy(op.join(su.get_resources_dir(), 'empty_subject.blend'), empty_subject_fname)
        if op.isfile(new_fname) and not args.overwrite_blend:
            overwrite = input('The file {} already exist, do you want to overwrite? '.format(new_fname))
            if su.is_true(overwrite):
               os.remove(new_fname)
               shutil.copy(op.join(su.get_mmvt_dir(), 'empty_subject.blend'), new_fname)
        else:
            shutil.copy(op.join(su.get_mmvt_dir(), 'empty_subject.blend'), new_fname)


def read_args(argv=None):
    parser = su.add_default_args()
    parser.add_argument('--overwrite_blend', help='', required=False, default=0, type=su.is_true)
    parser.add_argument('--import_in_blender', help='', required=False, default=1, type=su.is_true)
    parser.add_argument('--log_fname', help='For inner usage', required=False, default='', type=str)
    return su.parse_args(parser, argv)


def wrap_mmvt_calls(subject_fname):
    args = read_args(su.get_python_argv())
    mmvt = su.init_mmvt_addon()
    mmvt.import_brain()
    mmvt.set_render_output_path = su.get_figures_dir(args)
    su.save_blend_file(subject_fname)
    with open(args.log_fname, 'w') as text_file:
        print(args, file=text_file)
    try:
        su.exit_blender()
    except:
        pass


if __name__ == '__main__':
    import sys
    if op.isfile(sys.argv[0]) and sys.argv[0][-2:] == 'py':
        wrap_blender_call()
    else:
        wrap_mmvt_calls(sys.argv[1])
