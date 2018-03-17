import sys
import os


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


def wrap_blender_call(only_verbose=False):
    # Empty argv, otherwise cannot be called from setup (which already has args)
    sys.argv = [sys.argv[0]]
    args = read_args()
    args.subject = 'empty_subject'
    su.call_script(__file__, args, blend_fname='empty_subject.blend', only_verbose=only_verbose)#, call_args='-python_cmd {}'.format(sys.executable))


def read_args(argv=None):
    parser = su.add_default_args()
    parser.add_argument('--python_cmd', help='python cmd', required=False, default=sys.executable)
    return su.Bag(au.parse_parser(parser, argv))

    # args = su.Bag(au.parse_parser(parser, argv))
    # args.subject = 'empty_subject'
    # args.python_cmd = sys.executable
    # return args


def install_mmvt_addon():
    import bpy
    import os.path as op
    args = read_args(su.get_python_argv())
    print(args)
    print('python cmd: {}'.format(args.python_cmd))
    module = 'mmvt_loader'
    mmvt_folder = su.get_mmvt_addon_dir()
    path = op.join(mmvt_folder, '{}.py'.format(module))
    bpy.ops.wm.addon_install(filepath=path)
    bpy.ops.wm.addon_expand(module=module)
    bpy.ops.wm.addon_enable(module=module)
    addon_prefs = bpy.context.user_preferences.addons[module].preferences
    addon_prefs.mmvt_folder = mmvt_folder
    addon_prefs.python_cmd = args.python_cmd
    addon_prefs.freeview_cmd = 'freeview' if not su.is_windows() else ''
    addon_prefs.freeview_cmd_verbose = not su.is_windows()
    addon_prefs.freeview_cmd_stdin = not su.is_windows()
    bpy.context.user_preferences.system.use_scripts_auto_execute = True
    bpy.ops.wm.save_userpref()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[2] == '--background':
        install_mmvt_addon()
    else:
        wrap_blender_call()
