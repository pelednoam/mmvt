import sys
import os
import os.path as op

try:
    from src.utils import utils
except:
    pass

try:
    import bpy
except:
    pass


class Bag( dict ):
    """ a dict with d.key short for d["key"]
        d = Bag( k=v ... / **dict / dict.items() / [(k,v) ...] )  just like dict
    """
        # aka Dotdict

    def __init__(self, *args, **kwargs):
        dict.__init__( self, *args, **kwargs )
        self.__dict__ = self

    def __getnewargs__(self):  # for cPickle.dump( d, file, protocol=-1)
        return tuple(self)


def make_dir(fol):
    if not os.path.isdir(fol):
        os.makedirs(fol)
    return fol


def get_parent_fol(curr_dir='', levels=1):
    if curr_dir == '':
        curr_dir = os.path.dirname(os.path.realpath(__file__))
    parent_fol = os.path.split(curr_dir)[0]
    for _ in range(levels - 1):
        parent_fol = get_parent_fol(parent_fol)
    return parent_fol


def get_links_dir():
    return op.join(get_parent_fol(levels=4), 'links')


def get_utils_dir():
    return op.join(get_parent_fol(levels=2), 'utils')


def call_script(script_fname, args, log_name=''):
    LINKS_DIR = utils.get_links_dir()
    MMVT_DIR = op.join(LINKS_DIR, 'mmvt')
    BLENDER_DIR = op.join(LINKS_DIR, 'blender')

    if args.blender_fol == '':
        args.blender_fol = BLENDER_DIR
    if not op.isdir(args.blender_fol):
        print('No Blender folder!')
        return

    logs_fol = utils.make_dir(op.join(utils.get_parent_fol(__file__, 4), 'logs'))
    if log_name == '':
        log_name = utils.namebase(script_fname)
    call_args = create_call_args(args)
    cmd = '{blender_exe} {blend_fname} --background --python {script_fname} {call_args}'.format( # > {log_fname}
        blender_exe=op.join(args.blender_fol, 'blender'),
        blend_fname = op.join(MMVT_DIR, '{}_{}.blend'.format(args.subject, args.atlas)),
        script_fname = script_fname, call_args=call_args,
        log_fname = op.join(logs_fol, '{}.log'.format(log_name)))
    mmvt_addon_fol = utils.get_parent_fol(__file__, 2)
    os.chdir(mmvt_addon_fol)
    # utils.run_command_in_new_thread(cmd, queues=True)
    utils.run_script(cmd)
    print('Finish! For more details look in {}'.format(op.join(logs_fol, 'create_new_user.log')))


def create_call_args(args):
    call_args = ''
    for arg, value in args.items():
        call_args += '--{} "{}" '.format(arg, value)
    return call_args


def fix_argv():
    argv = sys.argv
    if "--" not in argv:
        argv = []  # as if no args are passed
    else:
        argv = argv[argv.index("--") + 1:]  # get all args after "--"
    return argv


def init_mmvt_addon(mmvt_addon_fol=''):
    # To run this function from command line:
    # 1) Copy the empty_subject.blend file, and rename to subject-name_atlas-name.blend
    # 2) Change the directory to the mmvt/src/mmvt_addon
    # 3) run: blender_path/blender /mmvt_path/subject-name_atlas-name.blend --background --python scripts/create_new_user.py
    if mmvt_addon_fol == '':
        mmvt_addon_fol = os.getcwd()
    print('mmvt_addon_fol: {}'.format(mmvt_addon_fol))
    sys.path.append(mmvt_addon_fol)
    import mmvt_addon
    # imp.reload(mmvt_addon)
    addon_prefs = Bag({'python_cmd':'python', 'freeview_cmd':'freeview', 'freeview_cmd_verbose':True,
                       'freeview_cmd_stdin':True})
    mmvt_addon.main(addon_prefs)
    return mmvt_addon


def save_blend_file(blend_fname):
    bpy.ops.wm.save_as_mainfile(filepath=blend_fname)


def exit_blender():
    bpy.ops.wm.quit_blender()


def get_python_argv():
    # Remove the blender argv and return only the python argv
    return sys.argv[5:]
