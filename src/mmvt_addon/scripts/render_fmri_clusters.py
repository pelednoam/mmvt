import sys
import os
import os.path as op


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
    # Add more args here
    return su.parse_args(parser, argv)


def run_script(subject_fname):
    args = read_args(su.get_python_argv())
    mmvt = su.init_mmvt_addon()
    # Call mmvt functions

    su.save_blend_file(subject_fname)
    su.exit_blender()


if __name__ == '__main__':
    import sys
    subject_fname = sys.argv[1]
    if sys.argv[2] == '--background':
        run_script(subject_fname)
    else:
        wrap_blender_call()
