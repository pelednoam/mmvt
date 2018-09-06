from pymesh import obj
import os.path as op
from src.utils import utils

MMVT_DIR = utils.get_link_dir(utils.get_links_dir(), 'mmvt')


def create_eeg_mesh_from_obj(subject, obj_fname):
    mesh_ply_fname = op.join(MMVT_DIR, subject, 'eeg', 'eeg_helmet.ply')
    # mesh = obj.Obj(obj_fname)
    # verts, faces = mesh.vertices, mesh.faces
    verts, faces = utils.read_obj_file(obj_fname)
    utils.write_ply_file(verts, faces, mesh_ply_fname)
    faces_verts_out_fname = op.join(MMVT_DIR, subject, 'eeg', 'eeg_faces_verts.npy')
    utils.calc_ply_faces_verts(verts, faces, faces_verts_out_fname, True,
                               utils.namebase(faces_verts_out_fname))

def recreate_mesh_faces_verts(subject, ply_fname):
    verts, faces = utils.read_ply_file(ply_fname)
    faces_verts_out_fname = op.join(MMVT_DIR, subject, 'eeg', 'eeg_faces_verts.npy')
    utils.calc_ply_faces_verts(verts, faces, faces_verts_out_fname, True,
                               utils.namebase(faces_verts_out_fname))


if __name__ == '__main__':
    subject = 'ep001'
    # create_eeg_mesh_from_obj(subject, '/homes/5/npeled/space1/mmvt/ep001/eeg/eeg_helmet.obj')
    recreate_mesh_faces_verts(subject, '/homes/5/npeled/space1/mmvt/ep001/eeg/eeg_helmet.ply')
    print('Finish!')