import scipy.io as sio
import numpy as np
import os.path as op
from collections import Counter
import traceback

PLY_HEADER = 'ply\nformat ascii 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nelement face {}\nproperty list uchar int vertex_index\nend_header\n'


def write_ply_file(verts, faces, ply_fname):
    try:
        verts_num = verts.shape[0]
        faces_num = faces.shape[0]
        verts -= verts.mean(axis=0)
        faces = np.rint(faces).astype(np.int)
        faces_for_ply = np.hstack((np.ones((faces_num, 1)) * faces.shape[1], faces))
        with open(ply_fname, 'w') as f:
            f.write(PLY_HEADER.format(verts_num, faces_num))
        with open(ply_fname, 'ab') as f:
            np.savetxt(f, verts, fmt='%.5f', delimiter=' ')
            np.savetxt(f, faces_for_ply, fmt='%d', delimiter=' ')
        return op.isfile(ply_fname)
    except:
        print('Error in write_ply_file! ({})'.format(ply_fname))
        print(traceback.format_exc())
        return False


def read_ply_file(ply_fname):
    with open(ply_fname, 'r') as f:
        lines = f.readlines()
        verts_num = int(lines[3].split(' ')[-1])
        faces_num = int(lines[10].split(' ')[-1])
        verts_lines = lines[13:13 + verts_num]
        faces_lines = lines[13 + verts_num:]
        verts = np.array([list(map(float, l.strip().split(' '))) for l in verts_lines])
        faces = np.array([list(map(int, l.strip().split(' '))) for l in faces_lines])[:,1:]
    return verts, faces


def calc_ply_faces_verts(verts, faces, out_file):
    _faces = faces.ravel()
    faces_arg_sort = np.argsort(_faces)
    faces_sort = np.sort(_faces)
    faces_count = Counter(faces_sort)
    max_len = max([v for v in faces_count.values()])
    lookup = np.ones((verts.shape[0], max_len)) * -1
    diff = np.diff(faces_sort)
    n = 0
    for ind, (k, v) in enumerate(zip(faces_sort, faces_arg_sort)):
        lookup[k, n] = v
        n = 0 if ind < len(diff) and diff[ind] > 0 else n+1
    np.save(out_file, lookup.astype(np.int))
    if len(_faces) != int(np.max(lookup)) + 1:
        raise Exception('Wrong values in lookup table! ' +
                        'faces ravel: {}, max looup val: {}'.format(len(_faces), int(np.max(lookup))))
    return op.isfile(out_file)


def create_ply(root):
    verts = sio.loadmat(op.join(root, 'V.mat'))['V']
    faces = sio.loadmat(op.join(root, 'F.mat'))['F']
    colors = sio.loadmat(op.join(root, 'C.mat'))['C']
    print(verts.shape, faces.shape, colors.shape)
    ply_fname = op.join(root, 'raza.ply')
    write_ply_file(verts, faces, ply_fname)


def load_ply(root, ply_name, colors_name):
    verts, faces = read_ply_file(op.join(root, '{}.ply'.format(ply_name)))
    colors = sio.loadmat(op.join(root, colors_name))['C']
    np.save(op.join(root, '{}_faces_verts.npy'.format(ply_name)), (verts, faces, colors))
    calc_ply_faces_verts(verts, faces, op.join(root, '{}_faces_verts_lookup.npy'.format(ply_name)))


def save_vertices_values(root, colors_name):
    import scipy.io as sio
    colors = sio.loadmat(op.join(root, colors_name))['C']
    np.save(op.join(root, 'colors.npy'), colors)


def load_csv_filedata_fname(data_fname):
    x = np.genfromtxt(data_fname, delimiter=',', skip_header=1)
    xyz = x[:, 1:4]
    vals = x[:, 4]
    return xyz, vals


if __name__ == '__main__':
    root = '/homes/5/npeled/space1/raza'
    # load_ply(root, 'raza', 'C.mat')
    # save_vertices_values(root, 'C.mat')
    load_csv_filedata_fname(op.join(root, 'MBH.csv'))
    print('finish!')