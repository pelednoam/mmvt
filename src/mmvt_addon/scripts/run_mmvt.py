import sys
import os
import os.path as op

try:
    from src.mmvt_addon.scripts import scripts_utils as su
except:
    # Add current folder the imports path
    sys.path.append(os.path.split(__file__)[0])
    import scripts_utils as su


def run(subject='', atlas='dkt', run_in_background=False, debug=None, raise_exp=True, run_blender=True):
    args = None
    if sys.argv[0] == __file__:
        args = read_args(raise_exp=raise_exp)
    if args is None:
        sys.argv = [__file__ , '-s', subject, '-a', atlas]
        args = read_args()
    if subject != '':
        args.subject = subject
    if atlas != '':
        args.atlas = atlas
    if debug is not None:
        args.debug = debug
    if run_blender:
        su.call_script(__file__, args, run_in_background=run_in_background)
    mmvt_agent = su.get_mmvt_object(args.subject)
    if mmvt_agent is not None:
        print('We got the mmvt object ({})!'.format(list(mmvt_agent._proxy_agent.connections.keys())[0]))
    return MMVT(mmvt_agent)


class MMVT(object):
    def __init__(self, mmvt_agent):
        self.mmvt_agent = mmvt_agent

    def __getattr__(self, item):
        if item in ['scripts', 'appearance', 'show_hide', 'coloring', 'render', 'transparency', 'play', 'utils']:
            return self
        else:
            return self.mmvt_agent.__getattr__(item)

    def __setattr__(self, item, value):
        if item == 'mmvt_agent':
            super().__setattr__(item, value)
        else:
            return self.mmvt_agent.__setattr__(item, value)



def read_args(argv=None, raise_exp=True):
    parser = su.add_default_args()
    # Add more args here
    # print('argv: {}'.format(sys.argv))
    if raise_exp:
        return su.parse_args(parser, argv, raise_exception_if_subject_is_empty=False)
    else:
        try:
            args = su.parse_args(parser, argv, raise_exception_if_subject_is_empty=False)
        except:
            args = None
    return args


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
        # su.debug()
        init_mmvt_addon(sys.argv[1])
