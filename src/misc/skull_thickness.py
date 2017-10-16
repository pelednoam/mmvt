# import bpy
import os.path as op
import nibabel.freesurfer as nib_fs
from src.utils import utils

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
MMVT_DIR = op.join(LINKS_DIR, 'mmvt')


def create_skull_plys(subject):
    for skull_surf in ['inner_skull', 'outer_skull']:
        ply_fname = op.join(SUBJECTS_DIR, subject, 'bem', '{}.ply'.format(skull_surf))
        surf_fname = op.join(SUBJECTS_DIR, subject, 'bem', '{}.surf'.format(skull_surf))
        verts, faces = nib_fs.read_geometry(surf_fname)
        utils.write_ply_file(verts, faces, ply_fname, True)


def create_faces_verts(subject):
    skull_fol = op.join(MMVT_DIR, subject, 'skull')
    errors = {}
    for skull_surf in ['inner_skull', 'outer_skull']:
        ply_fname = op.join(skull_fol, '{}.ply'.format(skull_surf))
        verts, faces = utils.read_ply_file(ply_fname)
        faces_verts_fname = op.join(skull_fol, 'faces_verts_{}.npy'.format(skull_surf))
        errors = utils.calc_ply_faces_verts(verts, faces, faces_verts_fname, False, utils.namebase(ply_fname), errors)
    if len(errors) > 0:
        for k, message in errors.items():
            print('{}: {}'.format(k, message))

if __name__ == '__main__':
    subject = 'mg78'
    # create_skull_plys(subject)
    create_faces_verts(subject)