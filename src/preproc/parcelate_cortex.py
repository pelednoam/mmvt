import numpy as np
import os.path as op
import glob
import time
import traceback

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
def parcelate(subject, atlas, hemi, surface_type, vertices_labels_ids_lookup=None,
              overwrite_vertices_labels_lookup=False):
    output_fol = op.join(MMVT_DIR, subject, 'labels', '{}.{}.{}'.format(atlas, surface_type, hemi))
    utils.make_dir(output_fol)
    vtx, fac = utils.read_ply_file(op.join(MMVT_DIR, subject, 'surf', '{}.{}.ply'.format(hemi, surface_type)))
    if vertices_labels_ids_lookup is None or overwrite_vertices_labels_lookup:
        vertices_labels_ids_lookup = lu.create_vertices_labels_lookup(
            subject, atlas, True, overwrite_vertices_labels_lookup)[hemi]
    labels = lu.read_labels(subject, SUBJECTS_DIR, atlas, hemi=hemi)
    if 'unknown-{}'.format(hemi) not in [l.name for l in labels]:
        labels.append(lu.Label([], name='unknown-{}'.format(hemi), hemi=hemi))

    nV = vtx.shape[0]
    nF = fac.shape[0]
    nL = len(labels)
    # print('The number of unique labels is {}'.format(nL))

    vtxL = [[] for _ in range(nL)]
    facL = [[] for _ in range(nL)]

    now = time.time()
    for f in range(nF):
        utils.time_to_go(now, f, nF, runs_num_to_print=5000)
        # Current face & labels
        Cfac = fac[f]
        Cidx = [vertices_labels_ids_lookup[vert_ind] for vert_ind in Cfac]
        # Depending on how many vertices of the current face
        # are in different labels, behave differently
        # nuCidx = len(np.unique(Cidx))
        # if nuCidx == 1: # If all vertices share same label
        # same_label = utils.all_items_equall(Cidx)
        # if same_label:
        if Cidx[0] == Cidx[1] == Cidx[2]:
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
    ret = True
    for lab in range(nL):
        ret = ret and writing_ply_files(subject, lab, facL[lab], vtx, vtxL, labels, hemi, output_fol)

    if ret:
        labels_files_num = len(glob.glob(op.join(MMVT_DIR, subject, 'labels', '{}.{}.{}'.format(
            atlas, surface_type, hemi), '*.ply')))
        # print(atlas, surface_type, hemi, labels_files_num, len(labels))
        # if labels_files_num != len(labels):
        #     print('labels_files_num = {}, but len(labels) = {}'.format(labels_files_num, len(labels)))
        # todo: should check the the -1 is becase the unknowns weren't written
        return labels_files_num <= len(labels) -1
    else:
        return False


def writing_ply_files(subject, lab, facL_lab, vtx, vtxL, labels, hemi, output_fol):
    # Vertices for the current label
    nV = vtx.shape[0]
    facL_lab_flat = utils.list_flatten(facL_lab)
    if len(facL_lab_flat) == 0:
        print("Cant write {}, no vertices!".format(labels[lab]))
        return True
    vidx = list(set(facL_lab_flat))
    vtxL[lab] = vtx[vidx]
    # Reindex the faces
    tmp = np.zeros(nV, dtype=np.int)
    tmp[vidx] = np.arange(len(vidx))
    try:
        facL_lab = np.reshape(tmp[facL_lab_flat], (len(facL_lab), len(facL_lab[0])))
    except:
        print(traceback.format_exc())
        dumps_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'dumps'))
        utils.save((lab, facL_lab, vtx, vtxL, labels, hemi, output_fol),
                   op.join(dumps_fol, 'parcelate_cortex_writing_ply_files.pkl'))
        return False

    # Save the resulting surface
    label_name = '{}-{}.ply'.format(lu.get_label_hemi_invariant_name(labels[lab].name), hemi)
    # print('Writing {}'.format(op.join(output_fol, label_name)))
    utils.write_ply_file(vtxL[lab], facL_lab, op.join(output_fol, label_name), True)
    return op.isfile(op.join(output_fol, label_name))


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


if __name__ == '__main__':
    subject = 'nmr01121'
    dumps_fol = op.join(MMVT_DIR, subject, 'dumps')
    (lab, facL_lab, vtx, vtxL, labels, hemi, output_fol) = utils.load(
        op.join(dumps_fol, 'parcelate_cortex_writing_ply_files.pkl'))
    writing_ply_files(subject, lab, facL_lab, vtx, vtxL, labels, hemi, output_fol)