# These utils are for the setup.py, used *before* installing python libs, so only python vanilla can be used here

import os
import os.path as op
from sys import platform as _platform

IS_LINUX = _platform == "linux" or _platform == "linux2"
IS_MAC = _platform == "darwin"
IS_WINDOWS = _platform == "win32"


def is_windows():
    return IS_WINDOWS


def make_dir(fol):
    if not os.path.isdir(fol):
        os.makedirs(fol)
    return fol


def all(arr):
    return list(set(arr))[0] == True


def get_current_fol():
    return op.dirname(op.realpath(__file__))


def get_parent_fol(curr_dir='', levels=1):
    if curr_dir == '':
        curr_dir = get_current_fol()
    parent_fol = op.split(curr_dir)[0]
    for _ in range(levels - 1):
        parent_fol = get_parent_fol(parent_fol)
    return parent_fol


def get_resources_fol():
    return op.join(get_parent_fol(levels=2), 'resources')


def get_link_dir(links_dir, link_name, var_name='', default_val='', throw_exception=False):
    val = op.join(links_dir, link_name)
    # check if this is a windows folder shortcup
    if op.isfile('{}.lnk'.format(val)):
        from src.mmvt_addon.scripts import windows_utils as wu
        sc = wu.MSShortcut('{}.lnk'.format(val))
        return op.join(sc.localBasePath, sc.commonPathSuffix)
        # return read_windows_dir_shortcut('{}.lnk'.format(val))
    if not op.isdir(val) and default_val != '':
        val = default_val
    if not op.isdir(val):
        val = os.environ.get(var_name, '')
    if not op.isdir(val):
        if throw_exception:
            raise Exception('No {} dir!'.format(link_name))
        else:
            print('No {} dir!'.format(link_name))
    return val


def get_links_dir(links_fol_name='links'):
    parent_fol = get_parent_fol(levels=3)
    links_dir = op.join(parent_fol, links_fol_name)
    return links_dir


def is_link(link_path):
    if is_windows():
        try:
            from src.mmvt_addon.scripts import windows_utils as wu
            sc = wu.MSShortcut('{}.lnk'.format(link_path))
            real_folder_path = op.join(sc.localBasePath, sc.commonPathSuffix)
            return op.isdir(real_folder_path)
        except:
            return False
    else:
        return op.islink(link_path)


def create_folder_link(real_fol, link_fol):
    if not is_link(link_fol):
        if is_windows():
            try:
                if not op.isdir(real_fol):
                    print('The target is not a directory!!')
                    return

                import winshell
                from win32com.client import Dispatch
                path = '{}.lnk'.format(link_fol)
                shell = Dispatch('WScript.Shell')
                shortcut = shell.CreateShortCut(path)
                shortcut.Targetpath = real_fol
                shortcut.save()
            except:
                print("Can't create a link to the folder {}!".format(real_fol))
        else:
            os.symlink(real_fol, link_fol)


def message_box(text, title=''):
    if is_windows():
        import ctypes
        return ctypes.windll.user32.MessageBoxW(0, text, title, 1)
    else:
        # print(text)
        from tkinter import Tk, Label
        root = Tk()
        w = Label(root, text=text)
        w.pack()
        root.mainloop()
        return 1


def choose_folder_gui():
    from tkinter.filedialog import askdirectory
    fol = askdirectory()
    if is_windows():
        fol = fol.replace('/', '\\')
    return fol


def should_run(args, func_name):
    if 'exclude' not in args:
        args.exclude = []
    return ('all' in args.function or func_name in args.function) and func_name not in args.exclude


class Bag( dict ):
    def __init__(self, *args, **kwargs):
        dict.__init__( self, *args, **kwargs )
        self.__dict__ = self
