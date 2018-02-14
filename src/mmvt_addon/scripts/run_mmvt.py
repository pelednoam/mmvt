import sys
import os
import os.path as op

try:
    from src.mmvt_addon.scripts import scripts_utils as su
except:
    # Add current folder the imports path
    sys.path.append(os.path.split(__file__)[0])
    import scripts_utils as su


def run(subject='', atlas='', run_in_background=False, debug=None):
    args = read_args()
    if subject != '':
        args.subject = subject
    if atlas != '':
        args.atlas = atlas
    if debug is not None:
        args.debug = debug
    su.call_script(__file__, args, run_in_background=run_in_background)


def read_args(argv=None):
    parser = su.add_default_args()
    # Add more args here
    return su.parse_args(parser, argv, raise_exception_if_subject_is_empty=False)


def init_mmvt_addon(subject_fname):
    args = read_args(su.get_python_argv())
    if args.debug:
        su.debug()
    su.init_mmvt_addon()
    print('Finish init MMVT!')


if __name__ == '__main__':
    if op.isfile(sys.argv[0]) and sys.argv[0][-2:] == 'py':
        run()
    else:
        init_mmvt_addon(sys.argv[1])
