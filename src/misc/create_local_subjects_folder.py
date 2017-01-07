import os
import traceback
import shutil

CACH_SUBJECT_DIR = '/space/huygens/1/users/mia/subjects/{subject}_SurferOutput/'

def prepare_subject_folder(neccesary_files, subject, remote_subject_dir, local_subjects_dir, print_traceback=False):
    local_subject_dir = os.path.join(local_subjects_dir, subject)
    for fol, files in neccesary_files.iteritems():
        if not os.path.isdir(os.path.join(local_subject_dir, fol)):
            os.makedirs(os.path.join(local_subject_dir, fol))
        for file_name in files:
            try:
                if not os.path.isfile(os.path.join(local_subject_dir, fol, file_name)):
                    shutil.copyfile(os.path.join(remote_subject_dir, fol, file_name),
                                os.path.join(local_subject_dir, fol, file_name))
            except:
                if print_traceback:
                    print(traceback.format_exc())
    all_files_exists = True
    for fol, files in neccesary_files.iteritems():
        for file_name in files:
            if not os.path.isfile(os.path.join(local_subject_dir, fol, file_name)):
                print("The file {} doesn't exist in the local subjects folder!!!".format(file_name))
                all_files_exists = False
    if not all_files_exists:
        raise Exception('Not all files exist in the local subject folder!!!')


if __name__ == '__main__':
    neccesary_files = {'mri': ['orig.mgz', 'rawavg.mgz', 'brain.mgz'], 'surf': ['rh.pial', 'lh.pial']}
    local_subjects_dir = ''
    subjects = []
    for subject in subjects:
        remote_subject_dir = CACH_SUBJECT_DIR.format(subject=subject.upper())
        prepare_subject_folder(neccesary_files, subject, remote_subject_dir, local_subjects_dir, print_traceback=True)