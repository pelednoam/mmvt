import sys
import os


try:
    from src.mmvt_addon.scripts import scripts_utils as su
except:
    # Add current folder the imports path
    sys.path.append(os.path.split(__file__)[0])
    import scripts_utils as su


def run(subject='', atlas='', run_in_background=False):
    args = read_args(dict(
        subject=subject,
        atlas=atlas,
    ))
    if subject != '':
        args.subject = subject
    if atlas != '':
        args.atlas = atlas
    su.call_script(__file__, args, run_in_background=run_in_background)


def read_args(argv=None):
    parser = su.add_default_args()
    # Add more args here
    return su.parse_args(parser, argv)


def init_mmvt_addon(subject_fname):
    # args = read_args(su.get_python_argv())
    # if args.debug:
    #     su.debug()
    su.init_mmvt_addon()
    print('Finish init MMVT!')


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 2 and sys.argv[2] in ['--python', '--background']:
        subject_fname = sys.argv[1]
        init_mmvt_addon(subject_fname)
    elif len(sys.argv) >= 2:
        subject = sys.argv[1]
        atlas = sys.argv[2]
        run_in_background = su.is_true(sys.argv[3]) if len(sys.argv) >= 4 else False
        run(subject, atlas, run_in_background)
    else:
        print('Not enough parameters were sent!')
