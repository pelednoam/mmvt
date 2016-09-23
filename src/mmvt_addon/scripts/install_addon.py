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
    su.call_script(__file__, args, blend_fname='empty_subject.blend', call_args='')


def read_args(argv=None):
    from src.utils import args_utils as au
    parser = su.add_default_args()
    args = su.Bag(au.parse_parser(parser, argv))
    args.subject = 'empty_subject'
    return args



def install_mmvt_addon():
    import bpy
    import os.path as op

    module = 'mmvt_loader'
    mmvt_folder = su.get_mmvt_addon_dir()
    path = op.join(mmvt_folder, '{}.py'.format(module))

    bpy.ops.wm.addon_install(filepath=path)
    bpy.ops.wm.addon_expand(module=module)
    bpy.ops.wm.addon_enable(module=module)
    addon_prefs = bpy.context.user_preferences.addons[module].preferences
    addon_prefs.mmvt_folder = mmvt_folder
    addon_prefs.freeview_cmd_verbose = True
    addon_prefs.freeview_cmd_stdin = True
    bpy.ops.wm.save_userpref()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[2] == '--background':
        install_mmvt_addon()
    else:
        wrap_blender_call()
