# The setup suppose to run *before* installing python libs, so only python vanilla can be used here

import os
import os.path as op
import shutil
import traceback
from src.utils import setup_utils as utils
import glob

TITLE = 'MMVT Installation'
BLENDER_WIN_DIR = 'C:\Program Files\Blender Foundation\Blender'


def copy_resources_files(mmvt_root_dir, only_verbose=False):
    resource_dir = utils.get_resources_fol()
    utils.make_dir(op.join(op.join(mmvt_root_dir, 'color_maps')))
    files = ['aparc.DKTatlas40_groups.csv', 'atlas.csv', 'sub_cortical_codes.txt', 'empty_subject.blend']
    cm_files = glob.glob(op.join(resource_dir, 'color_maps', '*.npy'))
    all_files_exist = utils.all([op.isfile(op.join(mmvt_root_dir, file_name)) for file_name in files])
    all_cm_files_exist = utils.all([op.isfile(fname) for fname in cm_files])
    if all_files_exist and all_cm_files_exist:
        if only_verbose:
            print('All files exist!')
        return True
    if not all_cm_files_exist:
        for color_map_file in glob.glob(op.join(resource_dir, 'color_maps', '*.npy')):
            new_file_name = op.join(mmvt_root_dir, 'color_maps', color_map_file.split(op.sep)[-1])
            # print('Copy {} to {}'.format(color_map_file, new_file_name))
            if only_verbose:
                print('Coping {} to {}'.format(color_map_file, new_file_name))
            else:
                shutil.copy(color_map_file, new_file_name)
    if not all_files_exist:
        for file_name in files:
            if only_verbose:
                print('Copying {} to {}'.format(op.join(resource_dir, file_name), op.join(mmvt_root_dir, file_name)))
            else:
                shutil.copy(op.join(resource_dir, file_name), op.join(mmvt_root_dir, file_name))
    return utils.all([op.isfile(op.join(mmvt_root_dir, file_name)) for file_name in files])


def create_links(links_fol_name='links', gui=True, only_verbose=False, links_file_name='links.csv'):
    links_fol = utils.get_links_dir(links_fol_name)
    if only_verbose:
        print('making links dir {}'.format(links_fol))
    else:
        utils.make_dir(links_fol)
    links_names = ['blender', 'mmvt', 'eeg', 'meg', 'fMRI', 'electrodes']
    if not utils.is_windows():
        links_names.insert(1, 'subjects')
    all_links_exist = utils.all([utils.is_link(op.join(links_fol, link_name)) for link_name in links_names])
    if all_links_exist:
        print('All links exist!')
        links = {link_name:utils.get_link_dir(links_fol, link_name) for link_name in links_names}
        write_links_into_csv_file(links, links_fol, links_file_name)
        return True
    if not utils.is_windows() and not utils.is_link(op.join(links_fol, 'freesurfer')):
        if os.environ.get('FREESURFER_HOME', '') == '':
            print('If you are going to use FreeSurfer locally, please source it and rerun')
            cont = input("Do you want to continue (y/n)?") # If you choose to continue, you'll need to create a link to FreeSurfer manually")
            if cont.lower() != 'y':
                return
        else:
            freesurfer_fol = os.environ['FREESURFER_HOME']
            if not only_verbose:
                create_real_folder(freesurfer_fol)

    mmvt_message = 'Please select where do you want to put the blend files? '
    subjects_message = 'Please select where do you want to store the FreeSurfer recon-all files neccessary for MMVT?\n' + \
              'It prefered to create a local folder, because MMVT is going to save files to this directory: '
    blender_message = 'Please select where did you install Blender? '
    meg_message = 'Please select where do you want to put the MEG files (Enter/Cancel if you are not going to use MEG data): '
    eeg_message = 'Please select where do you want to put the EEG files (Enter/Cancel if you are not going to use EEG data): '
    fmri_message = 'Please select where do you want to put the fMRI files (Enter/Cancel if you are not going to use fMRI data): '
    electrodes_message = 'Please select where do you want to put the electrodes files (Enter/Cancel if you are not going to use electrodes data): '

    if utils.is_windows() and op.isdir(BLENDER_WIN_DIR):
        utils.create_folder_link(BLENDER_WIN_DIR, op.join(links_fol, 'blender'))
    else:
        ask_and_create_link(links_fol, 'blender',  blender_message, gui)
    create_default_folders = mmvt_input("Would you like to set default links to the MMVT's folders?", gui)

    messages = [mmvt_message, eeg_message, meg_message, fmri_message, electrodes_message]
    if not utils.is_windows():
        messages.insert(0, subjects_message)
    create_default_dirs = [False] * (1 if utils.is_windows() else 2) + [True] * 2 + [False] * 2

    links = {}
    if not only_verbose:
        for link_name, message, create_default_dir in zip(links_names[1:], messages, create_default_dirs):
            if create_default_folders:
                create_default_link(links_fol, link_name, create_default_dir)
            else:
                links[link_name] = ask_and_create_link(links_fol, link_name, message, gui, create_default_dir)

    links = get_all_links(links, links_fol)
    write_links_into_csv_file(links, links_fol, links_file_name)
    return utils.all([utils.is_link(op.join(links_fol, link_name)) for link_name in links_names])


def mmvt_input(message, gui):
    if gui:
        ret = utils.message_box(message, TITLE)
    else:
        ret = input(message)
    return ret


def ask_and_create_link(links_fol, link_name, message, gui=True, create_default_dir=False):
    fol = ''
    if not utils.is_link(op.join(links_fol, link_name)):
        ret = mmvt_input(message, gui)
        if ret == 1:
            fol = utils.choose_folder_gui() if gui else input()
            if fol != '':
                create_real_folder(fol)
                utils.create_folder_link(fol, op.join(links_fol, link_name))
                if create_default_dir:
                    utils.make_dir(op.join(fol, 'default'))
    else:
        fol = utils.get_link_dir(links_fol, link_name)
    return fol


def create_default_link(links_fol, link_name, create_default_dir=False):
    root_fol = utils.get_parent_fol(levels=3)
    fol = op.join(root_fol, link_name)
    create_real_folder(fol)
    utils.create_folder_link(fol, op.join(links_fol, link_name))
    if create_default_dir:
        utils.make_dir(op.join(fol, 'default'))


def get_all_links(links={}, links_fol=None, links_fol_name='links'):
    if links_fol is None:
        links_fol = utils.get_links_dir(links_fol_name)
    all_links = [utils.namebase(f) for f in glob.glob(op.join(links_fol, '*')) if utils.is_link(f)]
    all_links = {link_name:utils.get_link_dir(links_fol, link_name) for link_name in all_links if link_name not in links}
    links = utils.merge_two_dics(links, all_links)
    return links


def write_links_into_csv_file(links, links_fol=None, links_file_name='links.csv', links_fol_name='links'):
    import csv
    if links_fol is None:
        links_fol = utils.get_links_dir(links_fol_name)
    with open(op.join(links_fol, links_file_name), 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        for link_name, link_dir in links.items():
            csv_writer.writerow([link_name, link_dir])


def create_empty_links_csv(links_fol_name='links', links_file_name='links.csv'):
    links_fol = utils.get_links_dir(links_fol_name)
    links_names = ['mmvt', 'subjects', 'blender', 'eeg', 'meg', 'fMRI', 'electrodes']
    links = {link_name:'' for link_name in links_names}
    write_links_into_csv_file(links, links_fol, links_file_name)


def create_real_folder(real_fol):
    try:
        if real_fol == '':
            real_fol = utils.get_resources_fol()
        utils.make_dir(real_fol)
    except:
        print('Error with creating the folder "{}"'.format(real_fol))
        print(traceback.format_exc())


def install_reqs(only_verbose=False):
    import pip
    retcode = 0
    reqs_fname = op.join(utils.get_parent_fol(levels=2), 'requirements.txt')
    with open(reqs_fname, 'r') as f:
        for line in f:
            if only_verbose:
                print('Trying to install {}'.format(line.strip()))
            else:
                pipcode = pip.main(['install', line.strip()])
                retcode = retcode or pipcode
    return retcode


def main(args):
    # 1) Install dependencies from requirements.txt (created using pipreqs)
    if utils.should_run(args, 'install_reqs'):
        install_reqs(args.only_verbose)

    # 2) Create links
    if utils.should_run(args, 'create_links'):
        links_created = create_links(args.links, args.gui, args.only_verbose)
        if not links_created:
            print('Not all the links were created! Make sure all the links are created before running MMVT.')

    # 3) Copy resources files
    if utils.should_run(args, 'copy_resources_files'):
        links_dir = utils.get_links_dir(args.links)
        mmvt_root_dir = utils.get_link_dir(links_dir, 'mmvt')
        resource_file_exist = copy_resources_files(mmvt_root_dir, args.only_verbose)
        if not resource_file_exist:
            print('Not all the resources files were copied to the MMVT folder.\n'.format(mmvt_root_dir) +
                  'Please copy them manually from the mmvt_code/resources folder')

    # 4) Install the addon in Blender (depends on resources and links)
    if utils.should_run(args, 'install_addon'):
        from src.mmvt_addon.scripts import install_addon
        install_addon.wrap_blender_call(args.only_verbose)

    if 'create_links_csv' in args.function:
        create_empty_links_csv()

    if 'create_csv' in args.function:
        write_links_into_csv_file(get_all_links())

    print('Finish!')


def print_help():
    str = 'functions: install_reqs, create_links, copy_resources_files, install_addon, create_links_csv and create_csv'
    print(str)


if __name__ == '__main__':
    import argparse
    from src.utils import args_utils as au
    parser = argparse.ArgumentParser(description='MMVT Setup')
    parser.add_argument('-l', '--links', help='links folder name', required=False, default='links')
    parser.add_argument('-g', '--gui', help='choose folders using gui', required=False, default='1', type=au.is_true)
    parser.add_argument('-v', '--only_verbose', help='only verbose', required=False, default='0', type=au.is_true)
    parser.add_argument('-f', '--function', help='functions to run', required=False, default='all', type=au.str_arr_type)
    args = utils.Bag(au.parse_parser(parser))
    if 'help' in args.function:
        print_help()
    else:
        main(args)
