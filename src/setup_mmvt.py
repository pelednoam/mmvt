import os
import os.path as op
import shutil
import numpy as np
import traceback
from src.utils import utils


def copy_resources_files(mmvt_root_dir):
    resource_dir = utils.get_resources_fol()
    files = ['aparc.DKTatlas40_groups.csv', 'atlas.csv', 'sub_cortical_codes.txt',
             'empty_subject.blend']
    for file_name in files:
        shutil.copy(op.join(resource_dir, file_name), op.join(mmvt_root_dir, file_name))
    return np.all([op.isfile(op.join(mmvt_root_dir, file_name)) for file_name in files])


def create_links(links_fol_name='links', gui=True):
    #todo: Work only on linux (maybe mac also)
    if gui:
        from tkinter.filedialog import askdirectory
    links_fol = utils.get_links_dir(links_fol_name)
    utils.make_dir(links_fol)
    links_names = ['mmvt', 'subjects', 'blender', 'meg', 'fMRI', 'electrodes', 'freesurfer']
    all_links_exist = np.all([op.islink(op.join(links_fol, link_name)) for link_name in links_names])
    if all_links_exist:
        return True
    if not utils.is_windows:
        if os.environ.get('FREESURFER_HOME', '') == '':
            print('If you have FreeSurfer installed, please source it and rerun')
            cont = input("Do you want to continue (y/n)?") # If you choose to continue, you'll need to create a link to FreeSurfer manually")
            if cont.lower() != 'y':
                return
        else:
            freesurfer_fol = os.environ['FREESURFER_HOME']
            create_real_folder(freesurfer_fol)
    print('Where do you want to put the blend files? ')
    mmvt_fol = askdirectory() if gui else input()
    create_real_folder(mmvt_fol)
    print('Where do you want to store the FreeSurfer recon-all files neccessary for MMVT?\n' +
          'It prefered to create a local folder, because MMVT is going to save files to this directory: ')
    subjects_fol = askdirectory() if gui else input()
    create_real_folder(subjects_fol)
    print('Where did you install Blender? ')
    blender_fol = askdirectory() if gui else input()
    create_real_folder(blender_fol)
    print('Where do you want to put the MEG files (Enter/Cancel if you are not going to use MEG data): ')
    meg_fol = askdirectory() if gui else input()
    create_real_folder(meg_fol)
    print('Where do you want to put the fMRI files (Enter/Cancel if you are not going to use fMRI data): ')
    fmri_fol = askdirectory() if gui else input()
    create_real_folder(fmri_fol)
    print('Where do you want to put the electrodes files (Enter/Cancel if you are not going to use electrodes data): ')
    electrodes_fol = askdirectory() if gui else input()
    create_real_folder(electrodes_fol)

    for real_fol, link_name in zip([mmvt_fol, subjects_fol, blender_fol, meg_fol, fmri_fol, electrodes_fol, freesurfer_fol],
            links_names):
        try:
            utils.create_folder_link(real_fol, op.join(links_fol, link_name))
            # if not op.islink(op.join(links_fol, link_name)):
            #     os.symlink(real_fol, op.join(links_fol, link_name))
            # Add the default task in meg folder
            if link_name == 'meg' and real_fol != utils.get_resources_fol():
                utils.make_dir(op.join(real_fol, 'default'))
        except:
            print('Error with folder {} and link {}'.format(real_fol, link_name))
            print(traceback.format_exc())
    return np.all([op.islink(op.join(links_fol, link_name)) for link_name in links_names])


def create_real_folder(real_fol):
    try:
        if real_fol == '':
            real_fol = utils.get_resources_fol()
        utils.make_dir(real_fol)
    except:
        print('Error with creating the folder "{}"'.format(real_fol))
        print(traceback.format_exc())


def main():
    import argparse
    from src.utils import args_utils as au
    parser = argparse.ArgumentParser(description='MMVT Setup')
    parser.add_argument('-l', '--links', help='links folder name', required=False, default='links')
    parser.add_argument('-g', '--gui', help='choose folders using gui', required=False, default='1', type=au.is_true)
    args = utils.Bag(au.parse_parser(parser))

    links_created = create_links(args.links, args.gui)
    if not links_created:
        print('Not all the links were created! Make sure all the links are created before running MMVT.')
    else:
        links_dir = utils.get_links_dir()
        mmvt_root_dir = op.join(links_dir, 'mmvt')
        resource_file_exist = copy_resources_files(mmvt_root_dir)
        if not resource_file_exist:
            print('Not all the resources files were copied to the MMVT () folder.\n'.format(mmvt_root_dir) +
                  'Please copy them manually from the mmvt_code/resources folder')
        else:
            print('Finish!')


if __name__ == '__main__':
    main()