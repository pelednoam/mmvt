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
    cmd = '{blender_exe} {blend_fname} --background --python {script_fname} > {log_fname}'.format(
        blender_exe=op.join(args.blender_fol, 'blender'),
        blend_fname = op.join(MMVT_DIR, '{}_{}.blend'.format(args.subject, args.atlas)),
        script_fname = script_fname,
        log_fname = op.join(logs_fol, '{}.log'.format(log_name)))
    mmvt_addon_fol = utils.get_parent_fol(__file__, 2)
    os.chdir(mmvt_addon_fol)
    utils.run_script(cmd, True)
    print('Finish! For more details look in {}'.format(op.join(logs_fol, 'create_new_user.log')))


def init_mmvt_addon(mmvt_addon_fol=''):
    # To run this function from command line:
    # 1) Copy the empty_subject.blend file, and rename to subject-name_atlas-name.blend
    # 2) Change the directory to the mmvt/src/mmvt_addon
    # 3) run: blender_path/blender /mmvt_path/subject-name_atlas-name.blend --background --python scripts/create_new_user.py
    if mmvt_addon_fol == '':
        mmvt_addon_fol = os.getcwd()
    print(mmvt_addon_fol)
    sys.path.append(mmvt_addon_fol)
    import mmvt_addon
    # imp.reload(mmvt_addon)
    addon_prefs = Bag({'python_cmd':'python', 'freeview_cmd':'freeview', 'freeview_cmd_verbose':True,
                       'freeview_cmd_stdin':True})
    mmvt_addon.main(addon_prefs)
    return mmvt_addon


def save_blend_file(blend_fname):
    bpy.ops.wm.save_as_mainfile(filepath=blend_fname)
