import sys
import os
import os.path as op
import argparse

try:
    from src.utils import utils
except:
    pass


try:
    import bpy
except:
    pass



def get_parent_fol(curr_dir='', levels=1):
    if curr_dir == '':
        curr_dir = os.path.dirname(os.path.realpath(__file__))
    parent_fol = os.path.split(curr_dir)[0]
    for _ in range(levels - 1):
        parent_fol = get_parent_fol(parent_fol)
    return parent_fol


try:
    from src.mmvt_addon import mmvt_utils as mu
except:
    mmvt_addon_fol = get_parent_fol()
    sys.path.append(mmvt_addon_fol)
    import mmvt_utils as mu


def get_links_dir():
    return op.join(get_parent_fol(levels=4), 'links')


def get_mmvt_dir():
    return op.join(get_links_dir(), 'mmvt')


def get_blender_dir():
    return op.join(get_links_dir(), 'blender')


def get_utils_dir():
    return op.join(get_parent_fol(levels=2), 'utils')


def add_utils_to_import_path():
    sys.path.append(get_utils_dir())


try:
    from src.utils import args_utils as au
except:
    add_utils_to_import_path()
    import args_utils as au

is_true = au.is_true


class Bag( dict ):
    def __init__(self, *args, **kwargs):
        dict.__init__( self, *args, **kwargs )
        self.__dict__ = self


def make_dir(fol):
    if not os.path.isdir(fol):
        os.makedirs(fol)
    return fol


def call_script(script_fname, args, log_name=''):
    if args.blender_fol == '':
        args.blender_fol = get_blender_dir()
    if not op.isdir(args.blender_fol):
        print('No Blender folder!')
        return

    logs_fol = utils.make_dir(op.join(utils.get_parent_fol(__file__, 4), 'logs'))
    if log_name == '':
        log_name = utils.namebase(script_fname)
    if len(args.subjects) == 0:
        args.subjects = [args.subject]
    for subject in args.subjects:
        args.subject = subject
        call_args = create_call_args(args)
        blend_fname = get_subject_fname(args)
        log_fname = op.join(logs_fol, '{}.log'.format(log_name))
        cmd = '{blender_exe} {blend_fname} --background --python {script_fname} {call_args}'.format( # > {log_fname}
            blender_exe=op.join(args.blender_fol, 'blender'),
            blend_fname = blend_fname, script_fname = script_fname, call_args=call_args, log_fname = log_fname)
        mmvt_addon_fol = utils.get_parent_fol(__file__, 2)
        os.chdir(mmvt_addon_fol)
        utils.run_script(cmd)
    print('Finish! For more details look in {}'.format(log_fname))


def get_subject_fname(args):
    mmvt_dir = get_mmvt_dir()
    return op.join(mmvt_dir, '{}_{}.blend'.format(args.subject, args.atlas))


def create_call_args(args):
    call_args = ''
    for arg, value in args.items():
        call_args += '--{} "{}" '.format(arg, value)
    return call_args


# def fix_argv():
#     argv = sys.argv
#     if "--" not in argv:
#         argv = []  # as if no args are passed
#     else:
#         argv = argv[argv.index("--") + 1:]  # get all args after "--"
#     return argv


def init_mmvt_addon(mmvt_addon_fol=''):
    # To run this function from command line:
    # 1) Copy the empty_subject.blend file, and rename to subject-name_atlas-name.blend
    # 2) Change the directory to the mmvt/src/mmvt_addon
    # 3) run: blender_path/blender /mmvt_path/subject-name_atlas-name.blend --background --python scripts/create_new_subject.py
    if mmvt_addon_fol == '':
        mmvt_addon_fol = os.getcwd()
    print('mmvt_addon_fol: {}'.format(mmvt_addon_fol))
    sys.path.append(mmvt_addon_fol)
    import mmvt_addon
    # imp.reload(mmvt_addon)
    addon_prefs = Bag({'python_cmd':'python', 'freeview_cmd':'freeview', 'freeview_cmd_verbose':True,
                       'freeview_cmd_stdin':True})
    mmvt_addon.main(addon_prefs)
    bpy.context.window.screen = bpy.data.screens['Neuro']
    return mmvt_addon


def save_blend_file(blend_fname):
    bpy.ops.wm.save_as_mainfile(filepath=blend_fname)


def exit_blender():
    bpy.ops.wm.quit_blender()


def get_python_argv():
    # Remove the blender argv and return only the python argv
    return sys.argv[5:]


def add_default_args():
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=False, default='')
    parser.add_argument('--subject', help='subjects names', required=False, default='', type=au.str_arr_type)
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='dkt')
    parser.add_argument('--real_atlas', help='atlas name', required=False, default='aparc.DKTatlas40')
    parser.add_argument('--blender_fol', help='blender folder', required=False, default='')
    return parser


def parse_args(parser, argv):
    args = Bag(au.parse_parser(parser, argv))
    args.real_atlas = get_full_atlas_name(args.atlas)
    if (len(args.subjects) == 0 and args.subject == '') or (len(args.subjects) > 0 and args.subject != ''):
        raise Exception('You need to set --subject or --subjects!')
    return args


def get_resources_dir():
    return op.join(get_parent_fol(levels=3), 'resources')


def get_full_atlas_name(atlas):
    return mu.get_real_atlas_name(atlas, get_mmvt_dir())