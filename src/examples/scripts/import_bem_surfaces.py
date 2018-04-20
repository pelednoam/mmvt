import os.path as op
import numpy as np


def run(mmvt):
    surf_fol = op.join(mmvt.utils.get_user_fol(), 'surf')
    watershed_files = ['inner_skull_surface', 'outer_skull_surface', 'outer_skin_surface']
    bem_fnames = [op.join(surf_fol, '{}.ply'.format(watershed_name)) for watershed_name in watershed_files]
    trans = {'inner_skull_surface': 0, 'outer_skull_surface': 0.8, 'outer_skin_surface': 1}
    colors = {'inner_skull_surface': 'red', 'outer_skull_surface': 'lightcyan', 'outer_skin_surface': 'white'}
    if not all([op.isfile(f) for f in bem_fnames]):
        print('Not all bem surfaces exist, trying to create them')
        mmvt.utils.run_mmvt_func('src.preproc.anatomy', 'load_bem_surfaces')
        return
    for bem_fname, watershed_name in zip(bem_fnames, watershed_files):
        obj = mmvt.utils.get_obj(watershed_name)
        material_name = '{}_mat'.format(watershed_name)
        if obj is None:
            obj = mmvt.data.load_ply(bem_fname, watershed_name, new_material_name=material_name)
        mmvt.appearance.set_transparency(material_name, trans[watershed_name])
        mmvt.appearance.set_layers_depth_trans(material_name, 10)
        faces_verts = np.load(op.join(surf_fol, '{}_faces_verts.npy'.format(watershed_name)))
        data = np.ones((len(obj.data.vertices), 4))
        data[:, 1:] = mmvt.colors.name_to_rgb(colors[watershed_name])
        mmvt.coloring.activity_map_obj_coloring(watershed_name, data, faces_verts)

