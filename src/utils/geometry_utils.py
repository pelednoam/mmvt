import os.path as op
import nibabel as nib
import numpy as np


def get_dural_surface(subject_fol, do_calc_normals=False):
    verts, faces, norms = {}, {}, {}
    for hemi_ind, hemi in enumerate(['rh', 'lh']):
        surf_fname = op.join(subject_fol, 'surf', '{}.dural'.format(hemi))
        if op.isfile(surf_fname):
            verts[hemi], faces[hemi] = nib.freesurfer.read_geometry(surf_fname)
            if do_calc_normals:
                norms[hemi] = calc_normals(verts[hemi], faces[hemi])
        else:
            print("Couldn't find the dural surface! {}".format(surf_fname))
            print('You can create it using the following command:')
            subject = op.splitext(op.basename(subject_fol))[0]
            print('python -m src.misc.dural.create_dural -s {}'.format(subject))
            return (None, None, None) if do_calc_normals else (None, None)
    if do_calc_normals:
        return verts, faces, norms
    else:
        return verts, faces


def calc_normals(vertices, faces):
    # https://sites.google.com/site/dlampetest/python/calculating-normals-of-a-triangle-mesh-using-numpy
    #Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    #Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    #Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    n = normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:,0]] += n
    norm[faces[:,1]] += n
    norm[faces[:,2]] += n
    norm = normalize_v3(norm)
    return norm


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2)
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def point_in_mesh(point, closest_vert, closeset_vert_normal, sigma=0, sigma_in=None):
    # https://blender.stackexchange.com/questions/31693/how-to-find-if-a-point-is-inside-a-mesh
    p2 = point - closest_vert
    v = p2.dot(closeset_vert_normal) + sigma
    return not(v < 0.0)
