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
    parser.add_argument('--script_name', required=True)
    parser.add_argument('--script_params', required=False, default='', type=su.str_arr_type)
    return su.parse_args(parser, argv)


def do_something(subject_fname):
    args = read_args(su.get_python_argv())
    if args.debug:
        su.debug()
    mmvt = su.init_mmvt_addon()
    mu = mmvt.utils
    # Call mmvt functions
    lib, (run_func, init_func, draw_func, params) = mmvt.scripts.check_script(
        mu.namebase(args.script_name), return_all=True)
    for param_tup in args.script_params:
        if len(param_tup.split(':')) != 2:
            continue
        param_name, param_val = param_tup.split(':')
        mu.set_prop(param_name, param_val)
        # setter_name = 'set_{}'.format(param_name)
        # if hasattr(lib, setter_name):
        #     getattr(lib, setter_name)(param_val)
        # if bpy.context.scene.__getattribute__(param_name):
        #    bpy.context.scene.__setattr__(param_name, param_val)

    run_func(mmvt)
    su.exit_blender()


if __name__ == '__main__':
    import sys
    if op.isfile(sys.argv[0]) and sys.argv[0][-2:] == 'py':
        wrap_blender_call()
    else:
        do_something(sys.argv[1])
