# The setup suppose to run *before* installing python libs, so only python vanilla can be used here

import os
import os.path as op
import shutil
import traceback
from src.utils import setup_utils as utils
import glob


TITLE = 'MMVT Installation'


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


def create_links(links_fol_name='links', gui=True, only_verbose=False):
    links_fol = utils.get_links_dir(links_fol_name)
    if only_verbose:
        print('making links dir {}'.format(links_fol))
    else:
        utils.make_dir(links_fol)
    links_names = ['mmvt', 'subjects', 'blender', 'meg', 'fMRI', 'electrodes', 'freesurfer']
    all_links_exist = utils.all([op.islink(op.join(links_fol, link_name)) for link_name in links_names])
    if all_links_exist:
        if only_verbose:
            print('All links exist!')
        return True
    if not utils.is_link(op.join(links_fol, 'freesurfer')):
        if not utils.is_windows():
            if os.environ.get('FREESURFER_HOME', '') == '':
                print('If you are going to use FreeSurfer locally, please source it and rerun')
                # cont = input("Do you want to continue (y/n)?") # If you choose to continue, you'll need to create a link to FreeSurfer manually")
                # if cont.lower() != 'y':
                #     return
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

    if not only_verbose:
        create_link(links_fol, 'mmvt', mmvt_message, gui)
        create_link(links_fol, 'subjects', subjects_message , gui)
        create_link(links_fol, 'blender', blender_message, gui)
        create_link(links_fol, 'eeg', eeg_message, gui, True)
        create_link(links_fol, 'meg', meg_message, gui, True)
        create_link(links_fol, 'fMRI', fmri_message, gui)
        create_link(links_fol, 'electrodes', electrodes_message, gui)

    return utils.all([utils.is_link(op.join(links_fol, link_name)) for link_name in links_names])


def create_link(links_fol, link_name, message, gui=True, create_default_dir=False):
    if not utils.is_link(op.join(links_fol, link_name)):
        ret = utils.message_box(message, TITLE)
        if ret == 1:
            fol = utils.choose_folder_gui() if gui else input()
            create_real_folder(fol)
            utils.create_folder_link(fol, op.join(links_fol, link_name))
            if create_default_dir:
                utils.make_dir(op.join(fol, 'default'))


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

    print('Finish!')


if __name__ == '__main__':
    import argparse
    from src.utils import args_utils as au
    parser = argparse.ArgumentParser(description='MMVT Setup')
    parser.add_argument('-l', '--links', help='links folder name', required=False, default='links')
    parser.add_argument('-g', '--gui', help='choose folders using gui', required=False, default='1', type=au.is_true)
    parser.add_argument('-v', '--only_verbose', help='only verbose', required=False, default='0', type=au.is_true)
    parser.add_argument('-f', '--function', help='functions to run', required=False, default='all', type=au.str_arr_type)
    args = utils.Bag(au.parse_parser(parser))
    main(args)
