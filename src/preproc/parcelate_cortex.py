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


def parcelate(subject, atlas, hemi, surface_type, srfprefix):
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
    print('The number of unique labels is %d\n', nL)
    # dpxidx = np.zeros(nV)
    # for lab_id, lab in enumerate(labels):
    #     dpxidx(dpx == udpx(lab)) = uidx(lab) # Replace labels by indices

    vtxL = [[]] * nL
    facL = [[]] * nL

    for f in range (nF):
        # Current face & labels
        Cfac = fac[f]
        Cidx = [vertices_labels_ids_lookup[vert_ind] for vert_ind in Cfac]
        # Depending on how many vertices of the current face
        # are in different labels, behave differently
        # nuCidx = len(np.unique(Cidx))
        # if nuCidx == 1: # If all vertices share same label
        same_label = utils.all_items_equall(Cidx)
        if same_label:
            # Add the current face to the list of faces of the
            # respective label, and don't create new faces
            facL[Cidx[0]] += Cfac.tolist()
        else: # If 2 or 3 vertices are in different labels
            # Create 3 new vertices at the midpoints of the 3 edges
            vtxCfac = vtx[Cfac]
            vtxnew = (vtxCfac + vtxCfac[[1, 2, 0]]) / 2
            vtx = np.concatenate((vtx, vtxnew))
            # Define 4 new faces, with care preserve normals (all CCW)
            facnew = [
                [Cfac[0], nV + 1, nV + 3],
                [nV + 1,  Cfac[1], nV + 2],
                [nV + 3,  nV + 2, Cfac[3]],
                [nV + 1, nV + 2, nV + 3]]
            # Update nV for the next loop
            nV = vtx.size[0]
            # Add the new faces to their respective labels
            facL[Cidx[0]] = [facL[Cidx[0]], facnew[0]]
            facL[Cidx[1]] = [facL[Cidx[1]], facnew[1]]
            facL[Cidx[2]] = [facL[Cidx[2]], facnew[2]]
            facL[mode(Cidx)] = [facL[mode(Cidx)], facnew[3]] # central face

    # Having defined new faces and assigned all faces to labels, now
    # select the vertices and redefine faces to use the new vertex indices
    # Also, create the file for the indices
    # fidx = fopen(sprintf('%s.index.csv', srfprefix), 'w');
    for lab in range(nL):
        # Vertices for the current label
        vidx = np.unique(facL[lab])
        vtxL[lab] = vtx[vidx]
        # Reindex the faces
        tmp = np.zeros(nV)
        tmp[vidx] = range(len(vidx))
        facL[lab] = np.reshape(tmp[facL[lab]], len(facL[lab]))
        # Save the resulting surface
        utils.write_ply_file(vtxL[lab], facL[lab],
            '{}.{:.4f}.ply'.format((srfprefix, lab)), True)

        # Add the corresponding line to the index file
        #fprintf(fidx, '%s,%g\n', fname, udpx(lab));


