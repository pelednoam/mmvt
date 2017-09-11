import numpy as np
import os.path as op

from src.utils import utils
from src.utils import labels_utils as lu

links_dir = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(links_dir, 'subjects', 'SUBJECTS_DIR')
MMVT_DIR = utils.get_link_dir(links_dir, 'mmvt',)

def mode(arr):
    return np.bincount(arr).argmax()
    #from collections import Counter
    #return Counter(arr).most_common(1)


# @utils.profileit(root_folder=op.join(MMVT_DIR, 'profileit'))
def parcelate(subject, atlas, hemi, surface_type, n_jobs=6):
    output_fol = op.join(MMVT_DIR, subject, 'labels', '{}.{}.{}'.format(atlas, surface_type, hemi))
    utils.make_dir(output_fol)
    vtx, fac = utils.read_ply_file(op.join(MMVT_DIR, subject, 'surf', '{}.{}.pkl'.format(hemi, surface_type)))
    vertices_labels_ids_lookup = lu.create_vertices_labels_lookup(subject, atlas, True)[hemi]
    labels = lu.read_labels(subject, SUBJECTS_DIR, atlas, hemi=hemi)
    # labels_ids_lookup = {label_name:label_ind for label_ind, label_name in enumerate(labels)}

    nV = vtx.shape[0]
    nF = fac.shape[0]
    #dpx = dpxread(dpxfile);
    #nX = numel(dpx)

    # udpx = [l_id for l_id in range]
    nL = len(labels)
    # uidx = range(nL)
    print('The number of unique labels is {}'.format(nL))
    # dpxidx = np.zeros(nV)
    # for lab_id, lab in enumerate(labels):
    #     dpxidx(dpx == udpx(lab)) = uidx(lab) # Replace labels by indices

    vtxL = [[] for _ in range(nL)]
    facL = [[] for _ in range(nL)]

    for f in range (nF):
        # Current face & labels
        Cfac = fac[f]
        Cidx = [vertices_labels_ids_lookup[vert_ind] for vert_ind in Cfac if vert_ind in vertices_labels_ids_lookup]
        # Depending on how many vertices of the current face
        # are in different labels, behave differently
        # nuCidx = len(np.unique(Cidx))
        # if nuCidx == 1: # If all vertices share same label
        # if Cidx[0] == Cidx[1] == Cidx[2]:
        same_label = utils.all_items_equall(Cidx)
        if same_label:
            # Add the current face to the list of faces of the
            # respective label, and don't create new faces
            facL[Cidx[0]] += [Cfac.tolist()]
        else: # If 2 or 3 vertices are in different labels
            # Create 3 new vertices at the midpoints of the 3 edges
            vtxCfac = vtx[Cfac]
            vtxnew = (vtxCfac + vtxCfac[[1, 2, 0]]) / 2
            vtx = np.concatenate((vtx, vtxnew))
            # Define 4 new faces, with care preserve normals (all CCW)
            facnew = [
                [Cfac[0], nV, nV + 2],
                [nV,  Cfac[1], nV + 1],
                [nV + 2,  nV + 1, Cfac[2]],
                [nV, nV + 1, nV + 2]]
            # Update nV for the next loop
            nV = vtx.shape[0]
            # Add the new faces to their respective labels
            facL[Cidx[0]] += [facnew[0]]
            facL[Cidx[1]] += [facnew[1]]
            facL[Cidx[2]] += [facnew[2]]
            freq_Cidx = mode(Cidx)
            facL[freq_Cidx] += [facnew[3]] # central face

    # Having defined new faces and assigned all faces to labels, now
    # select the vertices and redefine faces to use the new vertex indices
    # Also, create the file for the indices
    # fidx = fopen(sprintf('%s.index.csv', srfprefix), 'w');

    # params = []
    # for lab in range(nL):
    #     facL_lab = facL[lab]
    #     facL_lab_flat = utils.list_flatten(facL_lab)
    #     vidx = list(set(facL_lab_flat))
    #     vtxL_lab = vtx[vidx]
    #     params.append((facL_lab, vtxL_lab, vidx, nV, labels[lab].name, hemi, output_fol))
    # utils.run_parallel(writing_ply_files_parallel, params, njobs=n_jobs)
    #
    for lab in range(nL):
        writing_ply_files(lab, facL[lab], vtx, vtxL, labels, hemi, output_fol)


def writing_ply_files_parallel(p):
    facL_lab, vtxL_lab, vidx, nV, label_name, hemi, output_fol = p
    # Reindex the faces
    tmp = np.zeros(nV, dtype=np.int)
    tmp[vidx] = np.arange(len(vidx))
    facL_lab_flat = utils.list_flatten(facL_lab)
    facL_lab = np.reshape(tmp[facL_lab_flat], (len(facL_lab), len(facL_lab[0])))
    # Save the resulting surface
    label_name = '{}-{}.ply'.format(lu.get_label_hemi_invariant_name(label_name), hemi)
    # print('Writing {}'.format(op.join(output_fol, label_name)))
    utils.write_ply_file(vtxL_lab, facL_lab, op.join(output_fol, label_name), True)


# @utils.profileit(root_folder=op.join(MMVT_DIR, 'profileit'))
def writing_ply_files(lab, facL_lab, vtx, vtxL, labels, hemi, output_fol):
    # Vertices for the current label
    nV = vtx.shape[0]
    facL_lab_flat = utils.list_flatten(facL_lab)
    vidx = list(set(facL_lab_flat))
    vtxL[lab] = vtx[vidx]
    # Reindex the faces
    tmp = np.zeros(nV, dtype=np.int)
    tmp[vidx] = np.arange(len(vidx))
    facL_lab = np.reshape(tmp[facL_lab_flat], (len(facL_lab), len(facL_lab[0])))
    # Save the resulting surface
    label_name = '{}-{}.ply'.format(lu.get_label_hemi_invariant_name(labels[lab].name), hemi)
    # print('Writing {}'.format(op.join(output_fol, label_name)))
    utils.write_ply_file(vtxL[lab], facL_lab, op.join(output_fol, label_name), True)

    # Add the corresponding line to the index file
    # fprintf(fidx, '%s,%g\n', fname, udpx(lab));


def create_labels_lookup(subject, hemi, aparc_name):
    import mne.label
    annot_fname = op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format(hemi, aparc_name))
    if not op.isfile(annot_fname):
        return {}
    annot, ctab, label_names = mne.label._read_annot(annot_fname)
    lookup_key = 1
    lookup = {}
    for label_ind in range(len(label_names)):
        indices_num = len(np.where(annot == ctab[label_ind, 4])[0])
        if indices_num > 0 or label_names[label_ind].astype(str) == 'unknown':
            lookup[lookup_key] = label_names[label_ind].astype(str)
            lookup_key += 1
    return lookup


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
