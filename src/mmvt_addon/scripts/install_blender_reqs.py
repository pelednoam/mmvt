import sys
import os
import os.path as op


try:
    from src.mmvt_addon.scripts import scripts_utils as su
except:
    # Add current folder the imports path
    sys.path.append(os.path.split(__file__)[0])
    import scripts_utils as su


def wrap_blender_call(only_verbose=False):
    sys.argv = [sys.argv[0]]
    args = read_args(dict(subject='empty_subject'))
    su.call_script(__file__, args, blend_fname='empty_subject.blend', only_verbose=only_verbose)


def read_args(argv=None):
    parser = su.add_default_args()
    # Add more args here
    return su.parse_args(parser, argv)


def install_blender_reqs():
    import pip
    pip.main(['install', '--upgrade', 'pip'])
    libs = ['zmq', 'pizco', 'scipy', 'mne', 'joblib']
    for lib in libs:
        try:
            pip.main(['install', lib])
        except:
            print('Error in installing {}!'.format(lib))
    su.exit_blender()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[2] == '--background':
        install_blender_reqs()
    else:
        wrap_blender_call()
