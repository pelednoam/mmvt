# These utils are for the setup.py, used *before* installing python libs, so only python vanilla can be used here

import os
import os.path as op
import traceback
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
    link = op.join(links_dir, link_name)
    # check if this is a windows folder shortcup
    if op.isfile('{}.lnk'.format(link)):
        from src.mmvt_addon.scripts import windows_utils as wu
        sc = wu.MSShortcut('{}.lnk'.format(link))
        return op.join(sc.localBasePath, sc.commonPathSuffix)
        # return read_windows_dir_shortcut('{}.lnk'.format(val))
    ret = op.realpath(link)
    if not op.isdir(ret) and default_val != '':
        ret = default_val
    if not op.isdir(ret):
        ret = os.environ.get(var_name, '')
    if not op.isdir(ret):
        ret = get_link_dir_from_csv(links_dir, link_name)
        if ret == '':
            if throw_exception:
                raise Exception('No {} dir!'.format(link_name))
            else:
                print('No {} dir!'.format(link_name))
    return ret


def get_link_dir_from_csv(links_dir, link_name, csv_file_name='links.csv'):
    csv_fname = op.join(links_dir, csv_file_name)
    if op.isfile(csv_fname):
        for line in csv_file_reader(csv_fname, ','):
            if len(line) < 2:
                continue
            if line[0][0] == '#':
                continue
            if link_name == line[0]:
                link_dir = line[1]
                if not op.isdir(link_dir):
                    print('get_link_dir_from_csv: the dir for link {} does not exist! {}'.format(link_name, link_dir))
                    link_dir = ''
                return link_dir
    else:
        print('No links csv file was found ({})'.format(csv_fname))
    return ''


def csv_file_reader(csv_fname, delimiter=',', skip_header=0):
    import csv
    with open(csv_fname, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        for line_num, line in enumerate(reader):
            if line_num < skip_header:
                continue
            yield [val.strip() for val in line]


def get_links_dir(links_fol_name='links'):
    parent_fol = get_parent_fol(levels=3)
    links_dir = op.join(parent_fol, links_fol_name)
    return links_dir


def is_link(link_path):
    if is_windows():
        link_fname = link_path if link_path[-4:] == '.lnk' else '{}.lnk'.format(link_path)
        if not op.isfile(link_fname):
            return False
        try:
            from src.mmvt_addon.scripts import windows_utils as wu
            sc = wu.MSShortcut(link_fname)
            real_folder_path = op.join(sc.localBasePath, sc.commonPathSuffix)
            return op.isdir(real_folder_path)
        except:
            # print(traceback.format_exc())
            return False
    else:
        return op.islink(link_path)


def create_folder_link(real_fol, link_fol, overwrite=True):
    if not overwrite and is_link(link_fol):
        print('The link {} is already exist'.format(link_fol))
    else:
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


def message_box(text, title='', style=1):
    # if is_windows():
    #     import ctypes
    #     return ctypes.windll.user32.MessageBoxW(0, text, title, style)
    # else:
    import pymsgbox
    buttons = {0: ['Ok'], 1: ['Ok', 'Cancel'], 2: ['Abort', 'No', 'Cancel'], 3: ['Yes', 'No', 'Cancel'],
               4: ['Yes', 'No'], 5: ['Retry', 'No'], 6: ['Cancel', 'Try Again', 'Continue']}
    return pymsgbox.confirm(text=text, title=title, buttons=buttons[style])
        # from tkinter import Tk, Label
        # root = Tk()
        # w = Label(root, text=text)
        # w.pack()
        # root.mainloop()
        # return 1


def choose_folder_gui(initialdir='', title=''):
    import tkinter
    from tkinter.filedialog import askdirectory
    root = tkinter.Tk()
    root.withdraw()  # hide root
    if initialdir != '':
        fol = askdirectory(initialdir=initialdir, title=title)
    else:
        fol = askdirectory(title=title)
    if is_windows():
        fol = fol.replace('/', '\\')
    return fol


def should_run(args, func_name):
    if 'exclude' not in args:
        args.exclude = []
    return ('all' in args.function or func_name in args.function) and func_name not in args.exclude


def namebase(fname):
    return op.splitext(op.basename(fname))[0]


def merge_two_dics(dic1, dic2):
    ret = dic1.copy()
    ret.update(dic2)
    return ret


class Bag( dict ):
    def __init__(self, *args, **kwargs):
        dict.__init__( self, *args, **kwargs )
        self.__dict__ = self


def run_script(cmd, verbose=False):
    import subprocess
    import sys
    try:
        if verbose:
            print('running: {}'.format(cmd))
        if is_windows():
            output = subprocess.call(cmd)
        else:
            output = subprocess.check_output('{} | tee /dev/stderr'.format(cmd), shell=True)
    except:
        print('Error in run_script!')
        print(traceback.format_exc())
        return ''

    output = output.decode(sys.getfilesystemencoding(), 'ignore')
    print(output)
    return output


def get_parent_fol(curr_dir='', levels=1):
    if curr_dir == '':
        curr_dir = get_current_fol()
    parent_fol = op.split(curr_dir)[0]
    for _ in range(levels - 1):
        parent_fol = get_parent_fol(parent_fol)
    return parent_fol
