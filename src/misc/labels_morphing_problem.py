import os.path as op
import os
import glob
import numpy as np
import mne


def morph_labels_from_fsaverage(
        subject, subjects_dir, atlas='laus125', fsaverage='fsaverage', hemi='both', surf_name='pial',overwrite=False,
        n_jobs=6):
    subject_dir = op.join(subjects_dir, subject)
    labels_fol = op.join(subjects_dir, fsaverage, 'label', atlas)
    morphed_labels_fol = op.join(subject_dir, 'label', atlas) 
    if not op.isdir(morphed_labels_fol):
        os.makedirs(morphed_labels_fol)
    labels = mne.read_labels_from_annot(
        fsaverage, atlas, subjects_dir=subjects_dir, surf_name=surf_name, hemi=hemi)
    if len(labels) == 0:
        raise Exception('morph_labels_from_fsaverage: No labels for {}, {}'.format(fsaverage, atlas))
    # Make sure we have a morph map, and if not, create it here, and not in the parallel function
    mne.surface.read_morph_map(subject, fsaverage, subjects_dir=subjects_dir)
    verts = load_surf(subject, subjects_dir)
    indices = np.array_split(np.arange(len(labels)), n_jobs)
    chunks = [([labels[ind] for ind in chunk_indices], subject, fsaverage, labels_fol, morphed_labels_fol, verts,
               subjects_dir, overwrite) for chunk_indices in indices]
    results = run_parallel(_morph_labels_parallel, chunks, n_jobs)
    morphed_labels = []
    for chunk_morphed_labels in results:
        morphed_labels.extend(chunk_morphed_labels)
    return morphed_labels


def labels_to_annot(subject, labels, subjects_dir, atlas='laus125', overwrite=True):
    subject_dir = op.join(subjects_dir, subject)
    annot_files_exist = both_hemi_files_exist(
        op.join(subject_dir, 'label', '{}.{}.annot'.format('{hemi}', atlas)))
    if annot_files_exist and not overwrite:
        return True
    if len(labels) == 0:
        labels_files = glob.glob(op.join(subject_dir, 'label', atlas, '*.label'))
        if len(labels_files) == 0:
            raise Exception('labels_to_annot: No labels files!')
        for label_file in labels_files:
            if 'unknown' in namebase(label_file):
                continue
            label = mne.read_label(label_file)
            labels.append(label)
        labels.sort(key=lambda l: l.name)
    mne.write_labels_to_annot(
        subject=subject, labels=labels, parc=atlas, overwrite=overwrite, subjects_dir=subjects_dir)
    return both_hemi_files_exist(op.join(subject_dir, 'label', '{}.{}.annot'.format('{hemi}', atlas)))


def read_labels_parallel(subject, subjects_dir, atlas, labels_fol='', n_jobs=6):
    labels_fol = op.join(subjects_dir, subject, 'label', atlas) if labels_fol == '' else labels_fol
    labels_files = glob.glob(op.join(labels_fol, '*.label'))
    indices = np.array_split(np.arange(len(labels_files)), n_jobs)
    files_chunks = [[labels_files[ind] for ind in chunk_indices] for chunk_indices in indices]
    results = run_parallel(_read_labels_parallel, files_chunks, njobs=n_jobs)
    labels = []
    for labels_chunk in results:
        labels.extend(labels_chunk)
    return labels


def _read_labels_parallel(files_chunk):
    labels = []
    for label_fname in files_chunk:
        label = mne.read_label(label_fname)
        labels.append(label)
    return labels


def _morph_labels_parallel(p):
    labels, subject, fsaverage, labels_fol, morphed_labels_fol, verts, subjects_dir, overwrite = p
    if len(labels) == 0:
        raise Exception('_morph_labels_parallel: no labels!')
    morphed_labels = []
    for fs_label in labels:
        fs_label.values.fill(1.0)
        morphed_label = fs_label.morph(fsaverage, subject, grade=None, n_jobs=1, subjects_dir=subjects_dir)
        morphed_label.pos = verts[morphed_label.hemi][morphed_label.vertices]
        morphed_labels.append(morphed_label)
    return morphed_labels


def run_parallel(func, params, njobs=1):
    import multiprocessing
    if njobs == 1:
        results = [func(p) for p in params]
    else:
        pool = multiprocessing.Pool(processes=njobs)
        results = pool.map(func, params)
        pool.close()
    return results


def load_surf(subject, subjects_dir):
    import nibabel as nib
    verts = {}
    for hemi in ['rh', 'lh']:
        hemi_verts, _ = nib.freesurfer.read_geometry(op.join(subjects_dir, subject, 'surf', '{}.pial'.format(hemi)))
        verts[hemi] = hemi_verts
    return verts


def both_hemi_files_exist(file_template):
    if '*' not in file_template:
        return op.isfile(file_template.format(hemi='rh')) and op.isfile(file_template.format(hemi='lh'))
    else:
        return len(glob.glob(file_template.format(hemi='rh'))) == 1 and \
               len(glob.glob(file_template.format(hemi='lh'))) == 1


def namebase(fname):
    return op.splitext(op.basename(fname))[0]


def get_n_jobs(n_jobs):
    import multiprocessing
    cpu_num = multiprocessing.cpu_count()
    n_jobs = int(n_jobs)
    if n_jobs > cpu_num:
        n_jobs = cpu_num
    elif n_jobs < 0:
        n_jobs = cpu_num + n_jobs
    if n_jobs < 1:
        n_jobs == 1
    return n_jobs


if __name__ == '__main__':
    subjects_dir = os.environ['SUBJECTS_DIR']
    subject = 'sample'
    atlas = 'laus125' # 'aparc.DKTatlas'
    n_jobs = get_n_jobs(-1)
    morphed_labels = morph_labels_from_fsaverage(subject, subjects_dir, atlas=atlas, n_jobs=n_jobs)
    ret = labels_to_annot(subject, morphed_labels, subjects_dir, atlas, overwrite=True)
    print(ret)