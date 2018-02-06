import os
import os.path as op
import glob
import shutil

from src.utils import utils
from src.examples import fMRI_conn_vs_memory_score
from src.utils import freesurfer_utils as fu

root_paths = ['/homes/5/npeled/space1/Documents/memory_task', '/home/npeled/Documents/memory_task/']
root_path = [p for p in root_paths if op.isdir(p)][0]

ana_results_fname = op.join(root_path, 'ana_results.pkl')

# preproc_fol = '/homes/5/npeled/space1/memory'
preproc_fol = '/cluster/scratch/tuesday/noam'
fmri_root_data = op.join(preproc_fol, 'DataProcessed_memory')
# subject_fmri_fold = '/cluster/neuromind/douw/scans/adults'
subject_fmri_fold = '/homes/5/npeled/space1/fMRI'
ts_root_fol = '/cluster/neuromind/douw/scans/patients_mri_epochs_final/laus125'


params_file_text = '''#################################################################
# This is a parameter file that lists the specific anatomical
# and functional parameters hat are called upon in the 
# preprocessing and fcMRI scripts.  It should be edited for 
# each subject
#################################################################
#
set subject={subject}

# Number of frames to delete
set target="/space/bidlin4/1/users/Share/tools/current//code/targets/rN12Trio_avg152T1_brain.4dint"
set epitarget="/space/bidlin4/1/users/Share/tools/current//code/templates/volume/EPI.mnc" 
@ skip=4
set TR_vol='{tr}'
set mprs  	        = (4)

#goto process_FC
########## process:
set qc_folder='qc'                # quality control folder
set slab=0                        # 1 = slab registration
set BOLDbasename=$subject"_bld*_*_reorient.nii.gz"
set fieldmap_correction=0         # 1 = fieldmap correction

 
set highres=(4)                    # MPRAGE 
set bold=({bolds})               # all bold runs
set runid=(rest rest)	

set bet_extract         = 1                             # 1 = brain extract (necessary when highres is T1 MPRAGE)
set bet_flags           = "-g -.4"
exit;


process_FC:

set fcbold=({bolds})
set runid=(rest rest)
@ skip = 0
set blur=0.735452
set oh=2
set ol=0
set bh=0.08
set bl=0.0
set ventreg=/space/bidlin4/1/users/Share/tools/current//code/masks/avg152T1_ventricles_MNI
set wmreg=/space/bidlin4/1/users/Share/tools/current//code/masks/avg152T1_WM_MNI
set wbreg=/space/bidlin4/1/users/Share/tools/current//code/masks/avg152T1_brain_MNI
set ppstr=reorient_skip_faln_mc_atl
set mvstr=reorient_skip_faln_mc
set G=1
'''

def arrange_data(subjects):
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
            utils.make_link(anat_files[0], op.join(sub_anat_fol, '{}.nii.gz'.format(utils.namebase(anat_files[0]))))
        elif len(anat_files) == 0:
            print('No nii.gz files were found in {}!'.format(anat_fol))
            continue
        else:
            print('Too many nii.gz files were found in {}!'.format(anat_fol))
            continue
        bold_fols = [f for f in glob.glob(op.join(subject_fmri_fold, sub, 'bold', '*')) if op.isdir(f)]
        bolds_numbers = ','.join([str(int(utils.namebase(bold_fol))) for bold_fol in bold_fols])
        for bold_fol in bold_fols:
            fmri_files = glob.glob(op.join(bold_fol, '*.nii'))# '{}_bld{}_rest.nii'.format(sub, utils.namebase(bold_fol))))
            if len(fmri_files) == 1:
                sub_fmri_fol = utils.make_dir(op.join(sub_root_bold_fol, utils.namebase(bold_fol)))
                # utils.delete_folder_files(sub_fmri_fol)
                # target_file = op.join(sub_fmri_fol, '{}_bld{}_rest_reorient.nii.gz'.format(sub, utils.namebase(bold_fol)))
                target_file = op.join(sub_fmri_fol, '{}_bld{}_rest.nii'.format(sub, utils.namebase(bold_fol)))
                utils.make_link(fmri_files[0], target_file)
                new_target_file = op.join(sub_fmri_fol, '{}_bld{}_rest_reorient.nii.gz'.format(sub, utils.namebase(bold_fol)))
                if not op.isfile(new_target_file):
                    output = utils.run_script('mri_convert {} {}'.format(target_file, new_target_file))
                # utils.remove_link(target_file)
                # if not op.isfile(target_file):
                #     shutil.copy(fmri_files[0], target_file)
            elif len(fmri_files) == 0:
                print('No *_rest_reorient_skip_faln_mc.nii.gz files were found in {}!'.format(bold_fol))
                continue
            else:
                print('Too many *_rest_reorient_skip_faln_mc.nii.gz files were found in {}!'.format(bold_fol))
                continue
        tr = int(fu.get_tr(target_file) * 1000) / 1000.0
        files_list.append('{} {} {} {}'.format(sub, bolds_numbers, str(anat_number).zfill(3), tr))#int(tr))) # int(trs[sub])))
        utils.make_dir(op.join(fmri_root_data, sub, 'scripts'))
        params_fname = op.join(fmri_root_data, sub, 'scripts', '{}.params'.format(sub))
        with open(params_fname, 'w') as f:
            f.write(params_file_text.format(subject=sub, bolds=bolds_numbers.replace(',', ' '), tr=tr))

    return files_list


def get_trs():
    res = {}
    _, _, trs, _, all_subjects = fMRI_conn_vs_memory_score.read_scoring()
    for tr, sub in zip(trs, all_subjects):
        res[sub] = tr
    return res


def copy_fmri_ts_files(subjects, delete_previous_files=True):
    FMRI_DIR = utils.get_link_dir(utils.get_links_dir(), 'fMRI')

    for sub in subjects:
        # ts_files = glob.glob(op.join(ts_root_fol, sub, 'rest', 'laus125_*.txt'))
        ts_files = glob.glob(op.join(ts_root_fol, '{}_laus125_*.txt'.format(sub)))
        if len(ts_files) == 0:
            print('No ts files for {}!'.format(sub))
        for ts_file in ts_files:
            target_fname = op.join(FMRI_DIR, sub, utils.namebase_with_ext(ts_file))
            if delete_previous_files:
                for old_fname in glob.glob(op.join(FMRI_DIR, sub, '*laus125_*.txt')):
                    os.remove(old_fname)
            print('Copy {} to {}'.format(ts_file, target_fname))
            shutil.copy(ts_file, target_fname)


if __name__ == '__main__':
    # trs = get_trs()
    # res = utils.load(ana_results_fname)
    # subjects = res[5]
    # copy_fmri_ts_files(subjects, delete_previous_files=True)
    subjects = ['nmr01013']
    list_name = 'fast_tr.txt' # 'sub_bold_mpr_tr.txt'
    files_list = arrange_data(subjects)#, trs)
    list_fol = utils.make_dir(op.join(fmri_root_data, 'Lists'))
    utils.make_dir(op.join(preproc_fol, 'Lists'))
    with open(op.join(preproc_fol, 'Lists', list_name), 'w') as f:
        for res in files_list:
            f.write('{}\n'.format(res))


