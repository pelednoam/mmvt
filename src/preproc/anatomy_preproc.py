import glob
import os
import os.path as op
import shutil
import numpy as np
import scipy.io as sio
from collections import Counter, defaultdict
import mne
import traceback
from src import utils
from src import matlab_utils
from src import labels_utils as lu
from src import freesurfer_utils as fu

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
FREE_SURFER_HOME = utils.get_link_dir(LINKS_DIR, 'freesurfer', 'FREESURFER_HOME')
BLENDER_ROOT_DIR = op.join(LINKS_DIR, 'mmvt')
os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR
BRAINDER_SCRIPTS_DIR = op.join(utils.get_parent_fol(), 'brainder_scripts')
ASEG_TO_SRF = op.join(BRAINDER_SCRIPTS_DIR, 'aseg2srf -s "{}"') # -o {}'
HEMIS = ['rh', 'lh']


def subcortical_segmentation(subject, overwrite_subcortical_objs=False):
    # !!! Can't run in an IDE! must run in the terminal after sourcing freesurfer !!!
    # Remember that you need to have write permissions on SUBJECTS_DIR!!!
    # You can create a local folder for the current subject. The files you need are:
    # mri/aseg.mgz, mri/norm.mgz'
    script_fname = op.join(BRAINDER_SCRIPTS_DIR, 'aseg2srf')
    if not op.isfile(script_fname):
        raise Exception('The subcortical segmentation script is missing! {}'.format(script_fname))
    if not utils.is_exe(script_fname):
        utils.set_exe_permissions(script_fname)

    aseg_to_srf_output_fol = op.join(SUBJECTS_DIR, subject, 'ascii')
    function_output_fol = op.join(SUBJECTS_DIR, subject, 'subcortical_objs')
    renamed_output_fol = op.join(SUBJECTS_DIR, subject, 'subcortical')
    lookup = load_subcortical_lookup_table()
    obj_files = glob.glob(op.join(function_output_fol, '*.srf'))
    if len(obj_files) < len(lookup) or overwrite_subcortical_objs:
        utils.delete_folder_files(function_output_fol)
        utils.delete_folder_files(aseg_to_srf_output_fol)
        utils.delete_folder_files(renamed_output_fol)
        print('Trying to write into {}'.format(aseg_to_srf_output_fol))
        utils.run_script(ASEG_TO_SRF.format(subject, SUBJECTS_DIR, subject))
        os.rename(aseg_to_srf_output_fol, function_output_fol)
    ply_files = glob.glob(op.join(renamed_output_fol, '*.ply'))
    if len(ply_files) < len(lookup) or overwrite_subcortical_objs:
        convert_and_rename_subcortical_files(subject, function_output_fol, renamed_output_fol, lookup)
    flag_ok = len(glob.glob(op.join(renamed_output_fol, '*.ply'))) == len(lookup) and \
        len(glob.glob(op.join(renamed_output_fol, '*.npz'))) == len(lookup)
    return flag_ok


def load_subcortical_lookup_table():
    codes_file = op.join(SUBJECTS_DIR, 'sub_cortical_codes.txt')
    lookup = np.genfromtxt(codes_file, dtype=str, delimiter=',')
    lookup = {int(val):name for name, val in zip(lookup[:, 0], lookup[:, 1])}
    return lookup


def convert_and_rename_subcortical_files(subject, fol, new_fol, lookup):
    obj_files = glob.glob(op.join(fol, '*.srf'))
    utils.delete_folder_files(new_fol)
    for obj_file in obj_files:
        num = int(op.basename(obj_file)[:-4].split('_')[-1])
        new_name = lookup.get(num, '')
        if new_name != '':
            utils.srf2ply(obj_file, op.join(new_fol, '{}.ply'.format(new_name)))
            verts, faces = utils.read_ply_file(op.join(new_fol, '{}.ply'.format(new_name)))
            np.savez(op.join(new_fol, '{}.npz'.format(new_name)), verts=verts, faces=faces)
    blender_fol = op.join(BLENDER_ROOT_DIR, subject, 'subcortical')
    if op.isdir(blender_fol):
        shutil.rmtree(blender_fol)
    shutil.copytree(new_fol, blender_fol)


def rename_cortical(lookup, fol, new_fol):
    ply_files = glob.glob(op.join(fol, '*.ply'))
    utils.delete_folder_files(new_fol)
    for ply_file in ply_files:
        base_name = op.basename(ply_file)
        num = int(base_name.split('.')[-2])
        hemi = base_name.split('.')[0]
        name = lookup[hemi].get(num, num)
        new_name = '{}-{}'.format(name, hemi)
        shutil.copy(ply_file, op.join(new_fol, '{}.ply'.format(new_name)))


def create_labels_lookup(subject, hemi, aparc_name):
    import mne.label
    annot_fname = op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format(hemi, aparc_name))
    if not op.isfile(annot_fname):
        return {}
    annot, ctab, label_names = \
        mne.label._read_annot(annot_fname)
    lookup_key = 1
    lookup = {}
    for label_ind in range(len(label_names)):
        indices_num = len(np.where(annot == ctab[label_ind, 4])[0])
        if indices_num > 0 or label_names[label_ind].astype(str) == 'unknown':
            lookup[lookup_key] = label_names[label_ind].astype(str)
            lookup_key += 1
    return lookup


def freesurfer_surface_to_blender_surface(subject, hemi='both', overwrite=False):
    for hemi in utils.get_hemis(hemi):
        surf_name = op.join(SUBJECTS_DIR, subject, 'surf', '{}.pial'.format(hemi))
        surf_wavefront_name = '{}.asc'.format(surf_name)
        surf_new_name = '{}.srf'.format(surf_name)
        hemi_ply_fname = '{}.ply'.format(surf_name)
        if overwrite or not op.isfile(hemi_ply_fname):
            print('{}: convert srf to asc'.format(hemi))
            utils.run_script('mris_convert {} {}'.format(surf_name, surf_wavefront_name))
            os.rename(surf_wavefront_name, surf_new_name)
            print('{}: convert asc to ply'.format(hemi))
            convert_hemis_srf_to_ply(subject, hemi)
        for hemi in utils.get_hemis(hemi):
            ply_fname = op.join(SUBJECTS_DIR, subject, 'surf', '{}.pial.ply'.format(hemi))
            verts, faces = utils.read_ply_file(ply_fname)
            np.savez(op.join(SUBJECTS_DIR, subject, 'mmvt', '{}.pial'.format(hemi)), verts=verts, faces=faces)
            shutil.copyfile(op.join(SUBJECTS_DIR, subject, 'mmvt', '{}.pial.npz'.format(hemi)),
                            op.join(BLENDER_ROOT_DIR, subject, '{}.pial.npz'.format(hemi)))

    return utils.both_hemi_files_exist(op.join(SUBJECTS_DIR, subject, 'surf', '{hemi}.pial.ply'))


def convert_hemis_srf_to_ply(subject, hemi='both'):
    for hemi in utils.get_hemis(hemi):
        ply_file = utils.srf2ply(op.join(SUBJECTS_DIR, subject, 'surf', '{}.pial.srf'.format(hemi)),
                                 op.join(SUBJECTS_DIR, subject, 'surf', '{}.pial.ply'.format(hemi)))
        shutil.copyfile(ply_file, op.join(BLENDER_ROOT_DIR, subject, '{}.pial.ply'.format(hemi)))


def calc_faces_verts_dic(subject, overwrite=False):
    ply_files = [op.join(SUBJECTS_DIR, subject,'surf', '{}.pial.ply'.format(hemi)) for hemi in HEMIS]
    out_files = [op.join(BLENDER_ROOT_DIR, subject, 'faces_verts_{}.npy'.format(hemi)) for hemi in HEMIS]
    subcortical_plys = glob.glob(op.join(BLENDER_ROOT_DIR, subject, 'subcortical', '*.ply'))
    errors = []
    if len(subcortical_plys) > 0:
        faces_verts_dic_fnames = [op.join(BLENDER_ROOT_DIR, subject, 'subcortical', '{}_faces_verts.npy'.format(
                utils.namebase(ply))) for ply in subcortical_plys]
        ply_files.extend(subcortical_plys)
        out_files.extend(faces_verts_dic_fnames)

    for ply_file, out_file in zip(ply_files, out_files):
        if not overwrite and op.isfile(out_file):
            # print('{} already exist.'.format(out_file))
            continue
        # ply_file = op.join(SUBJECTS_DIR, subject,'surf', '{}.pial.ply'.format(hemi))
        # print('preparing a lookup table for {}'.format(ply_file))
        verts, faces = utils.read_ply_file(ply_file)
        _faces = faces.ravel()
        print('{}: verts: {}, faces: {}, faces ravel: {}'.format(utils.namebase(ply_file), verts.shape[0], faces.shape[0], len(_faces)))
        faces_arg_sort = np.argsort(_faces)
        faces_sort = np.sort(_faces)
        faces_count = Counter(faces_sort)
        max_len = max([v for v in faces_count.values()])
        lookup = np.ones((verts.shape[0], max_len)) * -1
        diff = np.diff(faces_sort)
        n = 0
        for ind, (k, v) in enumerate(zip(faces_sort, faces_arg_sort)):
            lookup[k, n] = v
            n = 0 if ind<len(diff) and diff[ind] > 0 else n+1
        # print('writing {}'.format(out_file))
        np.save(out_file, lookup.astype(np.int))
        print('{} max lookup val: {}'.format(utils.namebase(ply_file), int(np.max(lookup))))
        if len(_faces) != int(np.max(lookup)) + 1:
            errors[utils.namebase(ply_file)] = 'Wrong values in lookup table! ' + \
                'faces ravel: {}, max looup val: {}'.format(len(_faces), int(np.max(lookup)))
    if len(errors) > 0:
        for k, message in errors.items():
            print('{}: {}'.format(k, message))
    return len(errors) == 0


def check_ply_files(subject):
    ply_subject = op.join(SUBJECTS_DIR, subject, 'surf', '{}.pial.ply')
    ply_blender = op.join(BLENDER_ROOT_DIR, subject, '{}.pial.ply')
    ok = True
    for hemi in HEMIS:
        # print('reading {}'.format(ply_subject.format(hemi)))
        verts1, faces1 = utils.read_ply_file(ply_subject.format(hemi))
        # print('reading {}'.format(ply_blender.format(hemi)))
        verts2, faces2 = utils.read_ply_file(ply_blender.format(hemi))
        print('vertices: ply: {}, blender: {}'.format(verts1.shape[0], verts2.shape[0]))
        print('faces: ply: {}, blender: {}'.format(faces1.shape[0], faces2.shape[0]))
        ok = ok and verts1.shape[0] == verts2.shape[0] and faces1.shape[0]==faces2.shape[0]
    return ok


def convert_perecelated_cortex(subject, aparc_name, overwrite_ply_files=False, hemi='both'):
    lookup = {}
    for hemi in utils.get_hemis(hemi):
        lookup[hemi] = create_labels_lookup(subject, hemi, aparc_name)
        if len(lookup[hemi]) == 0:
            continue
        srf_fol = op.join(SUBJECTS_DIR, subject,'{}.pial.{}'.format(aparc_name, hemi))
        ply_fol = op.join(SUBJECTS_DIR, subject,'{}_{}_ply'.format(aparc_name, hemi))
        blender_fol = op.join(BLENDER_ROOT_DIR, subject,'{}.pial.{}'.format(aparc_name, hemi))
        utils.convert_srf_files_to_ply(srf_fol, overwrite_ply_files)
        rename_cortical(lookup, srf_fol, ply_fol)
        utils.rmtree(blender_fol)
        shutil.copytree(ply_fol, blender_fol)
    return lookup


def create_annotation_file_from_fsaverage(subject, aparc_name='aparc250', fsaverage='fsaverage',
        overwrite_annotation=False, overwrite_morphing=False, solve_labels_collisions=False,
        morph_labels_from_fsaverage=True, fs_labels_fol='', n_jobs=6):
    annotations_exist = np.all([op.isfile(op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format(hemi,
        aparc_name))) for hemi in HEMIS])
    existing_freesurfer_annotations = ['aparc.DKTatlas40.annot', 'aparc.annot', 'aparc.a2009s.annot']
    if '{}.annot'.format(aparc_name) in existing_freesurfer_annotations:
        morph_labels_from_fsaverage = False
        solve_labels_collisions = False
        if not annotations_exist:
            fu.create_annotation_file(subject, aparc_name, subjects_dir=SUBJECTS_DIR, freesurfer_home=FREE_SURFER_HOME)
    if morph_labels_from_fsaverage:
        utils.morph_labels_from_fsaverage(subject, SUBJECTS_DIR, aparc_name, n_jobs=n_jobs,
            fsaverage=fsaverage, overwrite=overwrite_morphing, fs_labels_fol=fs_labels_fol)
    if solve_labels_collisions:
        backup_labels_fol = '{}_before_solve_collision'.format(aparc_name, fsaverage)
        lu.solve_labels_collision(subject, SUBJECTS_DIR, aparc_name, backup_labels_fol, n_jobs)
        lu.backup_annotation_files(subject, SUBJECTS_DIR, aparc_name)
    # Note that using the current mne version this code won't work, because of collissions between hemis
    # You need to change the mne.write_labels_to_annot code for that.
    if overwrite_annotation or not annotations_exist:
        utils.labels_to_annot(subject, SUBJECTS_DIR, aparc_name, overwrite=overwrite_annotation)
    return utils.both_hemi_files_exist(op.join(
        SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format('{hemi}', aparc_name)))


def parcelate_cortex(subject, aparc_name, overwrite=False, overwrite_ply_files=False, minimum_labels_num=50):
    files_exist = True
    for hemi in HEMIS:
        blender_labels_fol = op.join(BLENDER_ROOT_DIR, subject,'{}.pial.{}'.format(aparc_name, hemi))
        files_exist = files_exist and op.isdir(blender_labels_fol) and \
            len(glob.glob(op.join(blender_labels_fol, '*.ply'))) > minimum_labels_num

    if overwrite or not files_exist:
        matlab_command = op.join(BRAINDER_SCRIPTS_DIR, 'splitting_cortical.m')
        matlab_command = "'{}'".format(matlab_command)
        sio.savemat(op.join(BRAINDER_SCRIPTS_DIR, 'params.mat'),
            mdict={'subject': subject, 'aparc':aparc_name, 'subjects_dir': SUBJECTS_DIR,
                   'scripts_dir': BRAINDER_SCRIPTS_DIR, 'freesurfer_home': FREE_SURFER_HOME})
        cmd = 'matlab -nodisplay -nosplash -nodesktop -r "run({}); exit;"'.format(matlab_command)
        utils.run_script(cmd)
        # convert the  obj files to ply
        lookup = convert_perecelated_cortex(subject, aparc_name, overwrite_ply_files)
        matlab_labels_vertices = save_matlab_labels_vertices(subject, aparc_name)

    labels_num = sum([len(lookup[hemi]) for hemi in HEMIS])
    labels_files_num = sum([len(glob.glob(op.join(BLENDER_ROOT_DIR, subject,'{}.pial.{}'.format(
        aparc_name, hemi), '*.ply'))) for hemi in HEMIS])
    labels_dic_fname = op.join(BLENDER_ROOT_DIR, subject,'labels_dic_{}_{}.pkl'.format(aparc_name, hemi))
    print('labels_files_num == labels_num: {}'.format(labels_files_num == labels_num))
    print('isfile(labels_dic_fname): {}'.format(op.isfile(labels_dic_fname)))
    print('matlab_labels_vertices files: {}'.format(matlab_labels_vertices))
    return labels_files_num == labels_num and op.isfile(labels_dic_fname) and matlab_labels_vertices


def save_matlab_labels_vertices(subject, aparc_name):
    for hemi in HEMIS:
        matlab_fname = op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot_labels.m'.format(hemi, aparc_name))
        if op.isfile(matlab_fname):
            labels_dic = matlab_utils.matlab_cell_arrays_to_dict(matlab_fname)
            utils.save(labels_dic, op.join(BLENDER_ROOT_DIR, subject, 'labels_dic_{}_{}.pkl'.format(aparc_name, hemi)))
        else:
            return False
    return True


def save_labels_vertices(subject, aparc_name):
    labels_fnames = glob.glob(op.join(SUBJECTS_DIR, subject, 'label', aparc_name, '*.label'))
    if len(labels_fnames) > 0:
        labels = []
        for label_fname in labels_fnames:
            label = mne.read_label(label_fname)
            labels.append(label)
    else:
        # Read from the annotation file
        labels = []
        for hemi in HEMIS:
            annot_fname = op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format(hemi, aparc_name))
            if op.isfile(annot_fname):
                labels_hemi = mne.read_labels_from_annot(subject, aparc_name)
                labels.extend(labels_hemi)
            else:
                print("Can't find the annotation file! {}".format(annot_fname))
                return False
    labels_names, labels_vertices = defaultdict(list), defaultdict(list)
    for label in labels:
        labels_names[label.hemi].append(label.name)
        labels_vertices[label.hemi].append(label.vertices)
    output_fname = op.join(BLENDER_ROOT_DIR, subject, 'labels_vertices_{}.pkl'.format(aparc_name))
    utils.save((labels_names, labels_vertices), output_fname)
    return op.isfile(output_fname)


def create_spatial_connectivity(subject):
    try:
        connectivity_per_hemi = {}
        for hemi in utils.HEMIS:
            d = np.load(op.join(SUBJECTS_DIR, subject, 'mmvt', '{}.pial.npz'.format(hemi)))
            connectivity_per_hemi[hemi] = mne.spatial_tris_connectivity(d['faces'])
        utils.save(connectivity_per_hemi, op.join(BLENDER_ROOT_DIR, subject, 'spatial_connectivity.pkl'))
        success = True
    except:
        print('Error in create_spatial_connectivity!')
        print(traceback.format_exc())
        success = False
    return success


def calc_lavels_center_of_mass(subject, atlas, read_from_annotation=True, surf_name='pial', labels_fol=''):
    if (read_from_annotation):
        labels = mne.read_labels_from_annot(subject, atlas, 'both', surf_name, subjects_dir=SUBJECTS_DIR)
        if len(labels) == 0:
            print('No labels were found in {} annotation file!'.format(atlas))
    else:
        labels = []
        for label_file in glob.glob(op.join(labels_fol, '{}.label')):
            label = mne.read_label(label_file)
            labels.append(label)
        if len(labels) == 0:
            print('No labels were found in {}!'.format(labels_fol))
    if len(labels) > 0:
        center_of_mass = {}
        for label in labels:
            center_of_mass[label.name] = np.mean(label.pos, 0)
        com_fname = op.join(SUBJECTS_DIR, subject, 'label', '{}_center_of_mass.pkl'.format(atlas))
        blend_fname = op.join(BLENDER_ROOT_DIR, subject, '{}_center_of_mass.pkl'.format(atlas))
        utils.save(center_of_mass, com_fname)
        shutil.copyfile(com_fname, blend_fname)
    return len(labels) > 0 and op.isfile(com_fname) and op.isfile(blend_fname)


# def find_hemis_boarders(subject):
#     from scipy.spatial.distance import cdist
#     verts = {}
#     for hemi in utils.HEMIS:
#         ply_file = op.join(SUBJECTS_DIR)
#         verts[hemi], _ = utils.read_ply_file(op.join(SUBJECTS_DIR, subject, 'surf', '{}.pial.ply'.format(hemi)))
#     dists = cdist(verts['rh'], verts['lh'])
#
#
# def find_hemis_axis(subject):
#     from sklearn.decomposition import PCA
#     import matplotlib.pyplot as plt
#     all_verts = []
#     for hemi in utils.HEMIS:
#         ply_file = op.join(SUBJECTS_DIR)
#         verts, _ = utils.read_ply_file(op.join(SUBJECTS_DIR, subject, 'surf', '{}.pial.ply'.format(hemi)))
#         all_verts.extend(verts)
#     pca = PCA(n_components=2)
#     model = pca.fit(all_verts)
#     trans_verts = pca.transform(all_verts)
#     print ('explained variance (first {} components): {:.2f}'.format(2, sum(pca.explained_variance_ratio_)))
#     plt.scatter(trans_verts[:, 0], trans_verts[:, 1])
#     plt.show()
#
#
# def plot_hyperplane(clf, min_x, max_x, linestyle, label):
#     import matplotlib.pyplot as plt
#     # get the separating hyperplane
#     w = clf.coef_[0]
#     a = -w[0] / w[1]
#     xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
#     yy = a * xx - (clf.intercept_[0]) / w[1]
#     plt.plot(xx, yy, linestyle, label=label)

def main(subject, aparc_name, neccesary_files, remote_subject_dir, overwrite_annotation=False, fsaverage='fsaverage',
         overwrite_morphing_labels=False, overwrite_hemis_srf=False, overwrite_labels_ply_files=False,
         overwrite_ply_files=False, overwrite_faces_verts=False, solve_labels_collisions=False,
         morph_labels_from_fsaverage=True, fs_labels_fol='', n_jobs=1):
    # todo: When possivble, read verts and faces from the nzp and not from ply files
    flags = dict()
    utils.make_dir(op.join(SUBJECTS_DIR, subject, 'mmvt'))
    # *) Prepare the local subject's folder
    flags['prepare_local_subjects_folder'] = utils.prepare_local_subjects_folder(
        neccesary_files, subject, remote_subject_dir, SUBJECTS_DIR, print_traceback=False)

    # *) Create annotation file from fsaverage
    flags['annot'] = create_annotation_file_from_fsaverage(subject, aparc_name, fsaverage,
        overwrite_annotation, overwrite_morphing_labels, solve_labels_collisions,
        morph_labels_from_fsaverage, fs_labels_fol, n_jobs)

    # *) convert rh.pial and lh.pial to rh.pial.ply and lh.pial.ply
    flags['hemis'] = freesurfer_surface_to_blender_surface(subject, overwrite=overwrite_hemis_srf)

    # *) Calls Matlab 'splitting_cortical.m' script
    flags['parc_cortex'] = parcelate_cortex(subject, aparc_name, overwrite_labels_ply_files, overwrite_ply_files)

    # *) Create srf files for subcortical structures
    # !!! Should source freesurfer !!!
    # Remember that you need to have write permissions on SUBJECTS_DIR!!!
    flags['subcortical'] = subcortical_segmentation(subject)

    # *) Create a dictionary for verts and faces for both hemis
    flags['faces_verts'] = calc_faces_verts_dic(subject, overwrite_faces_verts)

    # *) Save the labels vertices for meg label plotting
    flags['labels_vertices'] = save_labels_vertices(subject, aparc_name)

    # *) Create the subject's connectivity
    flags['connectivity'] = create_spatial_connectivity(subject)

    # *) Check the pial surfaces
    flags['ply_files'] = check_ply_files(subject)

    # *) Calc the labels center of mass
    flags['center_of_msas'] =  calc_lavels_center_of_mass(
        subject, aparc_name, read_from_annotation=True, surf_name='pial')

    for flag_type, val in flags.items():
        print('{}: {}'.format(flag_type, val))
    return flags


def run_on_subjects(subjects, remote_subjects_dir, overwrite_annotation=False, overwrite_morphing_labels=False,
        solve_labels_collisions=False, overwrite_hemis_srf=False, overwrite_labels_ply_files=False,
        overwrite_faces_verts=False, morph_labels_from_fsaverage=True, fsaverage='fsaverage', fs_labels_fol='', n_jobs=1):
    subjects_flags, subjects_errors = {}, {}
    for subject in subjects:
        remote_subject_dir = op.join(remote_subjects_dir, subject)
        utils.make_dir(op.join(BLENDER_ROOT_DIR, subject))
        try:
            print('*******************************************')
            print('subject: {}, atlas: {}'.format(subject, aparc_name))
            print('*******************************************')
            flags = main(subject, aparc_name, neccesary_files, remote_subject_dir,
                overwrite_annotation=overwrite_annotation, overwrite_morphing_labels=overwrite_morphing_labels,
                overwrite_hemis_srf=overwrite_hemis_srf, overwrite_labels_ply_files=overwrite_labels_ply_files,
                overwrite_faces_verts=overwrite_faces_verts, solve_labels_collisions=solve_labels_collisions,
                morph_labels_from_fsaverage=morph_labels_from_fsaverage, fsaverage=fsaverage,
                fs_labels_fol=fs_labels_fol, n_jobs=n_jobs)
            subjects_flags[subject] = flags
        except:
            subjects_errors[subject] = traceback.format_exc()
            print('Error in subject {}'.format(subject))
            print(traceback.format_exc())

    errors = defaultdict(list)
    for subject, flags in subjects_flags.items():
        print('subject {}:'.format(subject))
        for flag_type, val in flags.items():
            print('{}: {}'.format(flag_type, val))
            if not val:
                errors[subject].append(flag_type)
    print('Errors:')
    for subject, error in errors.items():
        print('{}: {}'.format(subject, error))



if __name__ == '__main__':
    # ******************************************************************
    # Be sure that you have matlab installed on your machine,
    # and you can run it from the terminal by just calling 'matlab'
    # Some of the functions are using freesurfer, so if you want to
    # run main, you need to run it from the terminal (not from an IDE),
    # after sourcing freesurfer.
    # If you want to import the subcortical structures, you need to
    # download the folder 'brainder_scripts' from the git,
    # and put it under the main mmvt folder.
    # ******************************************************************

    # Files needed in the local subject folder
    # subcortical_to_surface: mri/aseg.mgz, mri/norm.mgz
    # freesurfer_surface_to_blender_surface: surf/rh.pial, surf/lh.pial
    # convert_srf_files_to_ply: surf/lh.sphere.reg, surf/rh.sphere.reg
    neccesary_files = {'..': ['sub_cortical_codes.txt'], 'mri': ['aseg.mgz', 'norm.mgz', 'ribbon.mgz'],
        'surf': ['rh.pial', 'lh.pial', 'rh.sphere.reg', 'lh.sphere.reg', 'lh.white', 'rh.white']}

    overwrite_annotation = False
    overwrite_morphing_labels = False
    overwrite_hemis_srf = False
    overwrite_labels_ply_files = False
    overwrite_faces_verts = False
    solve_labels_collisions = False
    morph_labels_from_fsaverage = True
    fs_labels_fol = '/space/lilli/1/users/DARPA-Recons/fscopy/label/arc_april2016'

    fsaverage = 'fscopy'# 'fsaverage'
    aparc_name = 'arc_april2016' #''aparc.DKTatlas40' #'laus250' #
    n_jobs = 6

    # remote_subjects_dir = '/space/huygens/1/users/mia/subjects/{}_SurferOutput/'.format(subject.upper())
    # remote_subjects_dir = CACH_SUBJECT_DIR.format(subject=subject.upper())
    # remote_subjects_dir = op.join('/cluster/neuromind/tools/freesurfer', subject)
    remote_subjects_dir = op.join('/autofs/space/lilli_001/users/DARPA-MEG/freesurfs')
    subjects = ['pp009'] #set(utils.get_all_subjects(SUBJECTS_DIR, 'mg', '_')) - set(['mg96'])
    # run_on_subjects(
    #     subjects, remote_subjects_dir, overwrite_annotation, overwrite_morphing_labels, solve_labels_collisions,
    #     overwrite_hemis_srf, overwrite_labels_ply_files, overwrite_faces_verts, morph_labels_from_fsaverage, fsaverage,
    #     fs_labels_fol, n_jobs)

    # freesurfer_surface_to_blender_surface('fscopy', overwrite=overwrite_hemis_srf)
    calc_lavels_center_of_mass('pp009', aparc_name, read_from_annotation=True)
    # aparc_name = 'laus250'
    # users_flags = {}
    # subjects = ['mg78']
    # for subject in subjects:
    #     subcortical_segmentation(subject)
    #     users_flags[subject] = {}
    #     users_flags[subject]['parc_cortex'] = parcelate_cortex(subject, aparc_name, overwrite=True, overwrite_ply_files=True)
    #     users_flags[subject]['labels_vertices'] = save_labels_vertices(subject, aparc_name)
    # for subject in subjects:
    #     for flag_type, val in users_flags[subject].items():
    #         print('{}: {} {}'.format(subject, flag_type, val))
    print('finish!')