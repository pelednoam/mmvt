import sys
import os
import os.path as op
import argparse

try:
    from src.mmvt_addon.scripts import scripts_utils as su
except:
    # Add current folder the imports path
    sys.path.append(os.path.split(__file__)[0])
    import scripts_utils as su

try:
    from src.utils import args_utils as au
except:
    su.add_utils_to_import_path()
    import args_utils as au


def wrap_blender_call():
    args = read_args()
    su.call_script(__file__, args)


def read_args(argv=None):
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True)
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='dkt')
    parser.add_argument('--blender_fol', help='blender folder', required=False, default='')
    args = su.Bag(au.parse_parser(parser, argv))
    return args


def import_meg(subject_fname):
    argv = su.get_python_argv()
    args = read_args(argv)
    mmvt = su.init_mmvt_addon()
    # Call mmvt functions
    su.save_blend_file(subject_fname)
    su.exit_blender()


if __name__ == '__main__':
    import sys
    subject_fname = sys.argv[1]
    if sys.argv[2] == '--background':
        import_meg(subject_fname)
    else:
        wrap_blender_call()
