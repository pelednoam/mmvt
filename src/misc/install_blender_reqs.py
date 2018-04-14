import os
import sys
from sys import platform as _platform
import os.path as op
import traceback
import glob


TITLE = 'Installing Blender reqs'
# todo: Download get_pip from https://bootstrap.pypa.io/get-pip.py and put it in GET_PIP_FOL
GET_PIP_FOL = 'set-this-folder-name'
# todo: update theis list accroding to your requirements, or use a requirements.txt file
REQS = 'matplotlib scipy joblib tqdm'


def is_windows():
    return _platform == "win32"


def is_osx():
    return _platform == "darwin"


def is_linux():
    return _platform == "linux" or _platform == "linux2"


def install_blender_reqs(blender_fol='', gui=True):
    if blender_fol == '':
        blender_fol = find_blender()
    blender_parent_fol = get_parent_fol(blender_fol)

    # Get pip
    bin_template = op.join(get_parent_fol(blender_fol),  'Resources', '2.7?', 'python') if is_osx() else \
        op.join(blender_fol, '2.7?', 'python')
    blender_bin_folders = sorted(glob.glob(bin_template))
    if len(blender_bin_folders) == 0:
        print("Couldn't find Blender's bin folder! ({})".format(bin_template))
        blender_bin_fol = ''
        choose_folder = gui_input('Please choose the Blender bin folder where python file exists', gui) == 'Ok'
        if choose_folder:
            fol = choose_folder_gui(blender_parent_fol, 'Blender bin folder') if gui else input()
            if fol != '':
                blender_bin_fol = glob.glob(op.join(fol, '2.7?', 'python'))[-1]
        if blender_bin_fol == '':
            return
    else:
        # todo: let the user select the folder if more than one
        blender_bin_fol = blender_bin_folders[-1]
    python_exe = 'python.exe' if is_windows() else 'python3.5m'
    current_dir = os.getcwd()
    os.chdir(blender_bin_fol)
    pip_cmd = '{} {}'.format(op.join('bin', python_exe), op.join(GET_PIP_FOL, 'get-pip.py'))
    if not is_windows():
        run_script(pip_cmd)
        install_cmd = '{} install {}'.format(op.join('bin', 'pip'), REQS)
        run_script(install_cmd)
    else:
        install_cmd = '{} install {}'.format(op.join('Scripts', 'pip'), REQS)
        print(
            'Sorry, automatically installing external python libs in python will be implemented in the future.\n' +
            'Meanwhile, you can do the following:\n' +
            '1) Open a terminal window as administrator: ' +
            'Right click on the "Command Prompt" shortcut from the star menu and choose "Run as administrator"\n' +
            '2) Change the directory to "{}".\n'.format(blender_bin_fol) +
            '3) Run "{}"\n'.format(pip_cmd) +
            '4) Run "{}"\nGood luck!'.format(install_cmd))
    os.chdir(current_dir)


def find_blender():
    blender_fol = ''
    if is_windows():
        blender_win_fol = 'Program Files\Blender Foundation\Blender'
        if op.isdir(op.join('C:\\', blender_win_fol)):
            blender_fol = op.join('C:\\', blender_win_fol)
        elif op.isdir(op.join('D:\\', blender_win_fol)):
            blender_fol = op.join('D:\\', blender_win_fol)
    elif is_linux():
        output = run_script("find ~/ -name 'blender' -type d")
        if not isinstance(output, str):
            output = output.decode(sys.getfilesystemencoding(), 'ignore')
        blender_fols = output.split('\n')
        blender_fols = [fol for fol in blender_fols if op.isfile(op.join(
            get_parent_fol(fol), 'blender.svg')) or 'blender.app' in fol]
        if len(blender_fols) == 1:
            blender_fol = get_parent_fol(blender_fols[0])
    elif is_osx():
        blender_fol = '/Applications/Blender/blender.app/Contents/MacOS'
    return blender_fol


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

    if isinstance(output, str):
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


def get_current_fol():
    return op.dirname(op.realpath(__file__))


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


def gui_input(message, gui, style=1):
    if gui:
        ret = message_box(message, TITLE, style)
    else:
        ret = input(message)
    return ret


def message_box(text, title='', style=1):
    import pymsgbox
    buttons = {0: ['Ok'], 1: ['Ok', 'Cancel'], 2: ['Abort', 'No', 'Cancel'], 3: ['Yes', 'No', 'Cancel'],
               4: ['Yes', 'No'], 5: ['Retry', 'No'], 6: ['Cancel', 'Try Again', 'Continue']}
    return pymsgbox.confirm(text=text, title=title, buttons=buttons[style])


if __name__ == '__main__':
    blender_fol = sys.argv[1] if len(sys.argv) > 1 else ''
    is_gui = bool(sys.argv[2]) if len(sys.argv) > 2 else True
    install_blender_reqs(blender_fol, is_gui)