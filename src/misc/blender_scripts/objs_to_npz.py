import os.path as op
import os
import glob
# Only in python2!!!
import pywavefront
import numpy as np
import traceback

def namebase(fname):
    return op.splitext(op.basename(fname))[0]

def parent_fol(curr_dir):
    return op.split(curr_dir)[0]

def get_file_type(fname):
    return op.splitext(op.basename(fname))[1][1:]

fol = '/home/npeled/Angelique/Niles/objects'
if len(glob.glob(op.join(fol, '*.'))) > 0:
    raise Exception('Files without suffix!', glob.glob(op.join(fol, '*.')))
for obj_fname in glob.glob(op.join(fol, '*.*')):
    if ' ' in namebase(obj_fname):
        new_base_name = op.join(fol, namebase(obj_fname).replace(' ', '_'))
        file_type = get_file_type(obj_fname)
        org_base_fname = op.join(fol, '{}.{}'.format(namebase(obj_fname), file_type))
        os.rename(org_base_fname, op.join(fol, '{}.{}'.format(new_base_name, file_type)))

for obj_fname in glob.glob(op.join(fol, '*.obj')):
    npz_fname = op.join(fol, '{}.npz'.format(namebase(obj_fname)))
    if not op.isfile(npz_fname):
        try:
            meshes = pywavefront.Wavefront(obj_fname)
            vertices = np.array(meshes.vertices)
            faces = np.array(meshes.faces)
            print(vertices.shape, faces.shape)
            if len(faces) > 0:
                print('Saving {}'.format(npz_fname))
                np.savez(npz_fname, vertices=np.array(meshes.vertices), faces=np.array(meshes.faces))
            else:
                print('!! Not saving {}, no faces!'.format(npz_fname))
        except:
            print(traceback.format_exc())
            meshes = pywavefront.Wavefront(obj_fname)
            print('Error in {}'.format(obj_fname))