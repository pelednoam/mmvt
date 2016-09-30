import os
import os.path as op
import shutil
import numpy as np
import traceback
from src.utils import utils

TITLE = 'MMVT Installation'


def copy_resources_files(mmvt_root_dir):
    resource_dir = utils.get_resources_fol()
    files = ['aparc.DKTatlas40_groups.csv', 'atlas.csv', 'sub_cortical_codes.txt',
             'empty_subject.blend']
    for file_name in files:
        shutil.copy(op.join(resource_dir, file_name), op.join(mmvt_root_dir, file_name))
    return np.all([op.isfile(op.join(mmvt_root_dir, file_name)) for file_name in files])


def create_links(links_fol_name='links', gui=True):
    links_fol = utils.get_links_dir(links_fol_name)
    utils.make_dir(links_fol)
    links_names = ['mmvt', 'subjects', 'blender', 'meg', 'fMRI', 'electrodes', 'freesurfer']
    all_links_exist = np.all([op.islink(op.join(links_fol, link_name)) for link_name in links_names])
    if all_links_exist:
        return True
    if not utils.is_link(op.join(links_fol, 'freesurfer')):
        if not utils.is_windows():
            if os.environ.get('FREESURFER_HOME', '') == '':
                print('If you have FreeSurfer installed, please source it and rerun')
                cont = input("Do you want to continue (y/n)?") # If you choose to continue, you'll need to create a link to FreeSurfer manually")
                if cont.lower() != 'y':
                    return
            else:
                freesurfer_fol = os.environ['FREESURFER_HOME']
                create_real_folder(freesurfer_fol)
    if not utils.is_link(op.join(links_fol, 'mmvt')):
        ret = utils.message_box('Please select where do you want to put the blend files? ', TITLE)
        if ret == 1:
            mmvt_fol = utils.choose_folder_gui() if gui else input()
            create_real_folder(mmvt_fol)
            utils.create_folder_link(mmvt_fol, op.join(links_fol, 'mmvt'))
    if not utils.is_link(op.join(links_fol, 'subjects')):
        ret = utils.message_box('Please select where do you want to store the FreeSurfer recon-all files neccessary for MMVT?\n' +
              'It prefered to create a local folder, because MMVT is going to save files to this directory: ', TITLE)
        if ret == 1:
            subjects_fol = utils.choose_folder_gui() if gui else input()
            create_real_folder(subjects_fol)
            utils.create_folder_link(subjects_fol, op.join(links_fol, 'subjects'))
    if not utils.is_link(op.join(links_fol, 'blender')):
        ret = utils.message_box('Please select where did you install Blender? ')
        if ret == 1:
            blender_fol = utils.choose_folder_gui() if gui else input()
            create_real_folder(blender_fol)
            utils.create_folder_link(blender_fol, op.join(links_fol, 'blender'))
    if not utils.is_link(op.join(links_fol, 'meg')):
        ret = utils.message_box('Please select where do you want to put the MEG files (Enter/Cancel if you are not going to use MEG data): ', TITLE)
        if ret == 1:
            meg_fol = utils.choose_folder_gui() if gui else input()
            create_real_folder(meg_fol)
            utils.create_folder_link(meg_fol, op.join(links_fol, 'meg'))
            if meg_fol != utils.get_resources_fol():
                utils.make_dir(op.join(meg_fol, 'default'))
    if not utils.is_link(op.join(links_fol, 'fMRI')):
        ret = utils.message_box('Please select where do you want to put the fMRI files (Enter/Cancel if you are not going to use fMRI data): ', TITLE)
        if ret == 1:
            fmri_fol = utils.choose_folder_gui() if gui else input()
            create_real_folder(fmri_fol)
            utils.create_folder_link(fmri_fol, op.join(links_fol, 'fMRI'))
    if not utils.is_link(op.join(links_fol, 'electrodes')):
        ret = utils.message_box('Please select where do you want to put the electrodes files (Enter/Cancel if you are not going to use electrodes data): ', TITLE)
        if ret == 1:
            electrodes_fol = utils.choose_folder_gui() if gui else input()
            create_real_folder(electrodes_fol)
            utils.create_folder_link(electrodes_fol, op.join(links_fol, 'electrodes'))

    # for real_fol, link_name in zip([mmvt_fol, subjects_fol, blender_fol, meg_fol, fmri_fol, electrodes_fol, freesurfer_fol],
    #         links_names):
    #     try:
    #         # utils.create_folder_link(real_fol, op.join(links_fol, link_name))
    #         # if not op.islink(op.join(links_fol, link_name)):
    #         #     os.symlink(real_fol, op.join(links_fol, link_name))
    #         # Add the default task in meg folder
    #         if link_name == 'meg' and real_fol != utils.get_resources_fol():
    #             utils.make_dir(op.join(real_fol, 'default'))
    #     except:
    #         print('Error with folder {} and link {}'.format(real_fol, link_name))
    #         print(traceback.format_exc())
    return np.all([utils.is_link(op.join(links_fol, link_name)) for link_name in links_names])


def create_real_folder(real_fol):
    try:
        if real_fol == '':
            real_fol = utils.get_resources_fol()
        utils.make_dir(real_fol)
    except:
        print('Error with creating the folder "{}"'.format(real_fol))
        print(traceback.format_exc())


def install_reqs():
    import pip
    retcode = 0
    reqs_fname = op.join(utils.get_parent_fol(levels=2), 'requirements.txt')
    with open(reqs_fname, 'r') as f:
        for line in f:
            print('Trying to install {}'.format(line.strip()))
            pipcode = pip.main(['install', line.strip()])
            retcode = retcode or pipcode
    return retcode


def main(args):
    # 1) Create links
    if utils.should_run(args, 'create_links'):
        links_created = create_links(args.links_fol_name, args.gui)
        if not links_created:
            print('Not all the links were created! Make sure all the links are created before running MMVT.')

    # 2) Copy resources files
    if utils.should_run(args, 'copy_resources_files'):
        links_dir = utils.get_links_dir(args.links_fol_name)
        mmvt_root_dir = utils.get_link_dir(links_dir, 'mmvt')
        resource_file_exist = copy_resources_files(mmvt_root_dir)
        if not resource_file_exist:
            print('Not all the resources files were copied to the MMVT folder.\n'.format(mmvt_root_dir) +
                  'Please copy them manually from the mmvt_code/resources folder')

    # 3) Install the addon in Blender (depends on resources and links)
    if utils.should_run(args, 'install_addon'):
        from src.mmvt_addon.scripts import install_addon
        install_addon.wrap_blender_call()

    # 4) Install dependencies from requirements.txt (created using pipreqs)
    if utils.should_run(args, 'install_reqs'):
        install_reqs()
        print('Finish!')


if __name__ == '__main__':
    import argparse
    from src.utils import args_utils as au
    parser = argparse.ArgumentParser(description='MMVT Setup')
    parser.add_argument('-l', '--links', help='links folder name', required=False, default='links')
    parser.add_argument('-g', '--gui', help='choose folders using gui', required=False, default='1', type=au.is_true)
    parser.add_argument('-f', '--function', help='functions to run', required=False, default='all', type=au.str_arr_type)
    args = utils.Bag(au.parse_parser(parser))
    main(args)
