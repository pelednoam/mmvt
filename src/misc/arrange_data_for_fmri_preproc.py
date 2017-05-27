import os
import os.path as op
import glob
import shutil

from src.utils import utils
from src.examples import fMRI_conn_vs_memory_score

root_paths = ['/homes/5/npeled/space1/Documents/memory_task', '/home/npeled/Documents/memory_task/']
root_path = [p for p in root_paths if op.isdir(p)][0]

ana_results_fname = op.join(root_path, 'ana_results.pkl')

preproc_fol = '/homes/5/npeled/space1/memory'
preproc_fol = '/autofs/cluster/scratch/tuesday/noam'
fmri_root_data = op.join(preproc_fol, 'DataProcessed_Noam')
subject_fmri_fold = '/cluster/neuromind/douw/scans/adults'


def arrange_data(subjects, trs):
    files_list = []
    for sub in subjects:
        utils.make_dir(op.join(fmri_root_data, sub))
        sub_root_anat_fol = utils.make_dir(op.join(fmri_root_data, sub, 'anat'))
        sub_root_bold_fol = utils.make_dir(op.join(fmri_root_data, sub, 'bold'))
        anat_fols = [f for f in glob.glob(op.join(subject_fmri_fold, sub, 'anat', '*')) if op.isdir(f)]
        if len(anat_fols) == 0:
            print('No anat folder for {}!'.format(sub))
            continue
        elif len(anat_fols) > 1:
            print('More than one anat folder for {}!'.format(sub))
            continue
        anat_fol = anat_fols[0]
        anat_number = int(utils.namebase(anat_fol))
        anat_files = glob.glob(op.join(anat_fol, '*nii.gz'))
        if len(anat_files) == 1:
            sub_anat_fol = utils.make_dir(op.join(sub_root_anat_fol, utils.namebase(anat_fol)))
            make_link(anat_files[0], op.join(sub_anat_fol, '{}.nii.gz'.format(utils.namebase(anat_files[0]))))
        elif len(anat_files) == 0:
            print('No nii.gz files were found in {}!'.format(anat_fol))
            continue
        else:
            print('Too many nii.gz files were found in {}!'.format(anat_fol))
            continue
        bold_fols = [f for f in glob.glob(op.join(subject_fmri_fold, sub, 'bold', '*')) if op.isdir(f)]
        bolds_numbers = ','.join([str(int(utils.namebase(bold_fol))) for bold_fol in bold_fols])
        for bold_fol in bold_fols:
            fmri_files = glob.glob(op.join(bold_fol, '*_rest_reorient_skip_faln_mc.nii.gz'))
            if len(fmri_files) == 1:
                sub_fmri_fol = utils.make_dir(op.join(sub_root_bold_fol, utils.namebase(bold_fol)))
                target_file = op.join(sub_fmri_fol, '{}_bld{}_rest_reorient.nii.gz'.format(sub, utils.namebase(bold_fol)))
                make_link(fmri_files[0], target_file)
                # remove_link(target_file)
                # if not op.isfile(target_file):
                #     shutil.copy(fmri_files[0], target_file)
            elif len(fmri_files) == 0:
                print('No *_rest_reorient_skip_faln_mc.nii.gz files were found in {}!'.format(bold_fol))
                continue
            else:
                print('Too many *_rest_reorient_skip_faln_mc.nii.gz files were found in {}!'.format(bold_fol))
                continue
        files_list.append('{} {} {} {}'.format(sub, bolds_numbers, str(anat_number).zfill(3), trs[sub]))
    return files_list


def make_link(source, target):
    try:
        os.symlink(source, target)
    except FileExistsError as e:
        print('{} already exist'.format(target))


def remove_link(source):
    try:
        os.unlink(source)
    except:
        pass


def get_trs():
    res = {}
    _, _, trs, _, all_subjects = fMRI_conn_vs_memory_score.read_scoring()
    for tr, sub in zip(trs, all_subjects):
        res[sub] = tr
    return res


if __name__ == '__main__':
    trs = get_trs()
    res = utils.load(ana_results_fname)
    subjects = res[5]
    files_list = arrange_data(subjects, trs)
    list_fol = utils.make_dir(op.join(fmri_root_data, 'Lists'))
    with open(op.join(preproc_fol, 'Lists', 'sub_bold_mpr_tr.txt'), 'w') as f:
        for res in files_list:
            f.write('{}\n'.format(res))


