import sys
import os
import os.path as op
import argparse
from sys import platform as _platform

try:
    from src.mmvt_addon.scripts import call_script_utils as utils
except:
    try:
        import call_script_utils as utils
    except:
        pass


try:
    import bpy
except:
    pass


IS_LINUX = _platform == "linux" or _platform == "linux2"
IS_MAC = _platform == "darwin"
IS_WINDOWS = _platform == "win32"


HEMIS = ['rh', 'lh']

def is_mac():
    return IS_MAC


def is_windows():
    return IS_WINDOWS


def is_linux():
    return IS_LINUX


def get_current_dir():
    return op.dirname(op.realpath(__file__))


def get_parent_fol(curr_dir='', levels=1):
    if curr_dir == '':
        curr_dir = get_current_dir()
    parent_fol = op.split(curr_dir)[0]
    for _ in range(levels - 1):
        parent_fol = get_parent_fol(parent_fol)
    return parent_fol


def chdir_to_mmvt_addon():
    current_dir = op.split(get_current_dir())[1]
    if current_dir == 'scripts':
        code_root_dir = get_mmvt_addon_dir()
        os.chdir(code_root_dir)
    else:
        print("Not in scripts dir! Can't change the current dir to mmvt addon")

#
# try:
#     from src.mmvt_addon import mmvt_utils as mu
# except:
#     mmvt_addon_fol = get_parent_fol()
#     sys.path.append(mmvt_addon_fol)
#     import mmvt_utils as mu

# timeit = mu.timeit


def get_code_root_dir():
    return get_parent_fol(levels=3)


def get_mmvt_addon_dir():
    return get_parent_fol(levels=1)


def get_links_dir():
    return op.join(get_parent_fol(levels=4), 'links')


def get_windows_link(shortcut):
    try:
        from src.mmvt_addon.scripts import windows_utils as wu
    except:
        sys.path.append(op.split(__file__)[0])
        import windows_utils as wu
    sc = wu.MSShortcut('{}.lnk'.format(shortcut))
    return op.join(sc.localBasePath, sc.commonPathSuffix)


def get_link_dir(links_dir, link_name, var_name='', default_val='', throw_exception=False):
    link = op.join(links_dir, link_name)
    # check if this is a windows folder shortcup
    if op.isfile('{}.lnk'.format(link)):
        # try:
        #     from src.mmvt_addon.scripts import windows_utils as wu
        # except:
        #     os.chdir(op.join(get_code_root_dir(), 'src', 'mmvt_addon', 'scripts'))
        #     from . import windows_utils as wu
        import importlib
        add_fol_to_path(get_current_fol())
        import windows_utils as wu
        importlib.reload(wu)

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


def add_fol_to_path(fol):
    if fol not in sys.path:
        sys.path.append(fol)


def get_current_fol():
    return op.dirname(op.realpath(__file__))


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


def get_mmvt_dir():
    return get_link_dir(get_links_dir(), 'mmvt')
    # return op.join(get_links_dir(), 'mmvt')


def get_subjects_dir():
    return get_link_dir(get_links_dir(), 'subjects', 'SUBJECTS_DIR')


def get_blender_dir():
    return get_link_dir(get_links_dir(), 'blender')
    # return op.join(get_links_dir(), 'blender')


def get_utils_dir():
    return op.join(get_parent_fol(levels=2), 'utils')


def get_preproc_dir():
    return op.join(get_parent_fol(levels=2), 'preproc')


def add_utils_to_import_path():
    utils_dir = get_utils_dir()
    if utils_dir not in sys.path:
        sys.path.append(utils_dir)


def add_preproc_to_import_path():
    preproc_dir = get_preproc_dir()
    if preproc_dir not in sys.path:
        sys.path.append(preproc_dir)


try:
    from src.utils import args_utils as au
except:
    add_utils_to_import_path()
    import args_utils as au

is_true = au.is_true
is_true_or_none = au.is_true_or_none
str_arr_type = au.str_arr_type
int_arr_type = au.int_arr_type
float_arr_type = au.float_arr_type
bool_arr_type = au.bool_arr_type


class Bag( dict ):
    def __init__(self, *args, **kwargs):
        dict.__init__( self, *args, **kwargs )
        self.__dict__ = self


def make_dir(fol):
    if not op.isdir(fol):
        os.makedirs(fol)
    return fol


def call_script(script_fname, args, log_name='', blend_fname=None, call_args=None, run_in_background=True,
                only_verbose=False, err_pipe=None, std_pipe=None, stay_alive=True):
    # if args.blender_fol == '':
    #     args.blender_fol = get_blender_dir()
    blender_fol = get_blender_dir()
    if not op.isdir(blender_fol):
        print('No Blender folder!')
        return
    # blend_fname_is_None = True if blend_fname is None else False
    # call_args_is_None = True if call_args is None else False
    if log_name == '':
        log_name = namebase(script_fname)
        if only_verbose:
            print('log name: {}'.format(log_name))
    if len(args.subjects) == 0:
        args.subjects = [args.subject]
    subjects = args.subjects
    for subject in subjects:
        args.subject = subject
        args.subjects = ''
        # print('*********** {} ***********'.format(subject))
        logs_fol = get_logs_fol(subject)
        if op.isfile(op.join(get_mmvt_dir(), subject, 'logs', 'pizco.log')):
            os.remove(op.join(get_mmvt_dir(), subject, 'logs', 'pizco.log'))
        if blend_fname is None:
            blend_fname = get_subject_fname(args)
        else:
            blend_fname = op.join(get_mmvt_dir(), blend_fname)
        if call_args is None:
            call_args = create_call_args(args)
        log_fname = op.join(logs_fol, '{}.log'.format(log_name))
        cmd = '"{blender_exe}" "{blend_fname}" {background} --python "{script_fname}" -- {call_args}'.format( # > {log_fname}
            blender_exe='./blender', background='--background' if run_in_background else '',
            blend_fname=blend_fname, script_fname=script_fname, call_args=call_args, log_fname=log_fname) # op.join(args.blender_fol, 'blender')
        if not only_verbose:
            utils.run_script(
                cmd, stay_alive=stay_alive, log_fname=log_fname, cwd=blender_fol, err_pipe=err_pipe) #mmvt_addon_fol)
        # if blend_fname_is_None:
        #     blend_fname = None
        # if call_args_is_None:
        #     call_args = None
        # call_args, blend_fname = None, None
    # print('Finish! For more details look in {}'.format(log_fname))


def get_logs_fol(subject):
    logs_fol = op.join(get_mmvt_dir(), subject, 'logs')
    make_dir(logs_fol)
    return logs_fol


def get_subject_fname(args):
    mmvt_dir = get_mmvt_dir()
    atlas = get_real_atlas_name(args.atlas, short_name=True)
    new_fname = op.join(mmvt_dir, '{}_{}.blend'.format(args.subject, atlas))
    # return op.join(mmvt_dir, '{}_{}{}.blend'.format(args.subject, 'bipolar_' if args.bipolar else '', args.atlas))
    return new_fname


def create_call_args(args):
    call_args = ''
    for arg, value in args.items():
        if isinstance(value, list):
            value = ','.join(map(str, value))
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
        mmvt_addon_fol = get_mmvt_addon_dir()
    print('mmvt_addon_fol: {}'.format(mmvt_addon_fol))
    sys.path.append(mmvt_addon_fol)
    import mmvt_addon
    if bpy.context.scene.mmvt_initialized:
        print('mmvt was already initialized')
        return mmvt_addon
    # imp.reload(mmvt_addon)
    addon_prefs = Bag({'python_cmd':sys.executable, 'freeview_cmd':'freeview', 'freeview_cmd_verbose':True,
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
    scripts_params_ind = sys.argv.index('--python') + 3 # because of the " -- "s
    return sys.argv[scripts_params_ind:]


class ParserWithHelp(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def add_default_args():
    # parser = argparse.ArgumentParser(description='MMVT')
    parser = ParserWithHelp(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=False, default='')
    parser.add_argument('--subjects', help='subjects names', required=False, default='', type=au.str_arr_type)
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='dkt')
    parser.add_argument('--real_atlas', help='atlas name', required=False, default='aparc.DKTatlas40')
    parser.add_argument('-b', '--bipolar', help='bipolar', required=False, type=au.is_true)
    parser.add_argument('-d', '--debug', help='debug', required=False, default=0, type=au.is_true)
    # parser.add_argument('--blender_fol', help='blender folder', required=False, default='')
    return parser


def parse_args(parser, argv, raise_exception_if_subject_is_empty=True):
    args = Bag(au.parse_parser(parser, argv))
    args.real_atlas = get_real_atlas_name(args.atlas)
    if ((len(args.subjects) == 0 and args.subject == '') or (len(args.subjects) > 0 and args.subject != '')) and \
            raise_exception_if_subject_is_empty:
        # print(args)
        raise Exception('You need to set --subject or --subjects!')
    return args


def get_resources_dir():
    return op.join(get_parent_fol(levels=3), 'resources')


def get_figures_dir(args):
    figures_dir = op.join(get_mmvt_dir(), args.subject, 'figures')
    make_dir(figures_dir)
    return figures_dir


def get_camera_dir(args):
    camera_dir = op.join(get_mmvt_dir(), args.subject, 'camera')
    make_dir(camera_dir)
    return camera_dir


# def get_full_atlas_name(atlas):
#     return mu.get_real_atlas_name(atlas, get_mmvt_dir())

# def get_mmvt_root():
#     return get_parent_fol(get_user_fol())


def get_user_fol():
    root_fol = bpy.path.abspath('//')
    user = get_user()
    return op.join(root_fol, user)


def get_user():
    return namebase(bpy.data.filepath).split('_')[0]


def get_fname_folder(fname):
    return op.sep.join(fname.split(op.sep)[:-1])


def get_real_atlas_name(atlas, csv_fol='', short_name=False):
    if csv_fol == '':
        csv_fol = get_mmvt_dir()
    csv_fname = op.join(csv_fol, 'atlas.csv')
    real_atlas_name = ''
    if op.isfile(csv_fname):
        for line in csv_file_reader(csv_fname, ',', 1):
            if len(line) < 2:
                continue
            if atlas in [line[0], line[1]]:
                real_atlas_name = line[0] if short_name else line[1]
                break
        if real_atlas_name == '':
            print("Can't find the atlas {} in {}! Please add it to the csv file.".format(atlas, csv_fname))
            return atlas
        return real_atlas_name
    else:
        print('No atlas file was found! Please create a atlas file (csv) in {}, where '.format(csv_fname) +
                        'the columns are name in blend file, annot name, description')
        return ''


def namebase(fname):
    return op.splitext(op.basename(fname))[0]


def get_subject_name(subject_fname):
    return namebase(subject_fname).split('_')[0]


def load_camera(mmvt, mmvt_dir, args):
    if op.isfile(args.camera):
        camera_fname = args.camera
        mmvt.load_camera(args.camera)
    elif op.isfile(op.join(args.output_path, 'camera.pkl')):
        camera_fname = op.join(args.output_path, 'camera.pkl')
        mmvt.load_camera(camera_fname)
    elif op.isfile(op.join(mmvt_dir, args.subject, 'camera', 'camera.pkl')):
        camera_fname = op.join(mmvt_dir, args.subject, 'camera', 'camera.pkl')
        mmvt.load_camera(camera_fname)
    else:
        cont = input('No camera file was detected in the output folder, continue?')
        if not is_true(cont):
            return
    # print('The rendering will be using the camera file in {}'.format(camera_fname))
    # mmvt.load_camera(camera_fname)
    # input('Press any ket to continue...')
    return camera_fname


def debug(port=1090):
    pycharm_fol = get_link_dir(get_links_dir(), 'pycharm', throw_exception=True)
    eggpath = op.join(pycharm_fol, 'debug-eggs', 'pycharm-debug-py3k.egg')
    if not any('pycharm-debug' in p for p in sys.path):
        sys.path.append(eggpath)
    import pydevd
    pydevd.settrace('localhost', port=port, stdoutToServer=True, stderrToServer=True)


def stdout_print(str):
    sys.stdout.write(str)
    sys.stdout.write('\n')
    sys.stdout.flush()


def read_list_from_file(fname):
    arr = []
    if not op.isfile(fname):
        return []
    with open(fname, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith('#'):
                continue
            if line != '':
                arr.append(line)
    return arr


def select_one_file(files, template='', files_desc='', print_title=True, is_dir=False, file_func=None):
    if len(files) == 1:
        return files[0]
    elif len(files) == 0:
        print('No {} was found ({})'.format('file' if not is_dir else 'dir', template))
        return ''
    if print_title:
        print('More than one {} {} were found {}, please pick one.'.format(
            files_desc, 'files' if not is_dir else 'dirs',  'in {}'.format(template) if template != '' else ''))
    for ind, fname in enumerate(files):
        print('{}) {}'.format(ind + 1, fname))
        if file_func is not None:
            file_func(fname)
    input_str = 'Which one do you want to pick (1, 2, ...)? Press 0 to cancel'
    file_num = input(input_str)
    while not is_int(file_num):
        print('Please enter a valid integer')
        file_num = input(input_str)
    if file_num == 0:
        return ''
    else:
        file_num = int(file_num) - 1
        return files[file_num]


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def get_hemi_obj(hemi):
    return bpy.data.objects['inflated_{}'.format(hemi)]


def other_hemi(hemi):
    if 'inflated' in hemi:
        return 'inflated_lh' if hemi == 'inflated_rh' else 'inflated_rh'
    else:
        return 'lh' if hemi == 'rh' else 'rh'


def waits_for_file(fname):
    import time
    log_exist = op.isfile(fname)
    while not log_exist:
        log_exist = op.isfile(fname)
        time.sleep(.1)


def get_mmvt_object(subject):
    mmvt = None
    pizco_log_fname = op.join(get_mmvt_dir(), subject, 'logs', 'pizco.log')
    waits_for_file(pizco_log_fname)
    with open(pizco_log_fname, 'r') as log:
        pizco_address = log.read()
    try:
        from pizco import Proxy
        devnull = open(os.devnull, 'w')
        with RedirectStdStreams(stdout=devnull, stderr=devnull):
            mmvt = Proxy(pizco_address)
    except:
        pass
    return mmvt


def decode_subjects(subjects, remote_subject_dir=''):
    import glob
    import re
    for sub in subjects:
        if '*' in sub:
            subjects.remove(sub)
            subjects.extend([utils.namebase(fol) for fol in glob.glob(op.join(get_subjects_dir(), sub))])
            if remote_subject_dir != '':
                for fol in glob.glob(op.join(remote_subject_dir.format(subject=sub))):
                    start_ind = utils.namebase(remote_subject_dir).index('{subject}')
                    end_ind = re.search('[-_,\.!?]', utils.namebase(fol)[start_ind:]).start() + start_ind
                    subjects.append(utils.namebase(fol)[start_ind:end_ind])
                subjects = list(set(subjects))
        elif 'file:' in sub:
            subjects.remove(sub)
            subjects.extend(utils.read_list_from_file(sub[len('file:'):]))
    return subjects



class RedirectStdStreams(object):
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


if __name__ == '__main__':
    # init_mmvt_addon()
    get_mmvt_object()