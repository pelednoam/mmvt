import os.path as op

TRANS = [0, 0.8, 1]
COLORS = ['red', 'lightcyan', 'white']


def run(mmvt):
    surf_fol = op.join(mmvt.utils.get_user_fol(), 'surf')
    watershed_files = ['inner_skull_surface', 'outer_skull_surface', 'outer_skin_surface']
    watershed_fnames = [op.join(surf_fol, '{}.ply'.format(watershed_name)) for watershed_name in watershed_files]
    if not all([op.isfile(f) for f in watershed_fnames]):
        print('Not all bem surfaces exist, trying to create them')
        mmvt.utils.run_mmvt_func('src.preproc.anatomy', 'load_bem_surfaces')
        return
    for ind, (bem_fname, watershed_name) in enumerate(zip(watershed_fnames, watershed_files)):
        material_name = '{}_mat'.format(watershed_name)
        surf_obj = mmvt.utils.get_obj(watershed_name)
        if surf_obj is None:
            surf_obj = mmvt.data.load_ply(bem_fname, watershed_name, new_material_name=material_name)
        mmvt.appearance.set_transparency(material_name, TRANS[ind])
        mmvt.appearance.set_layers_depth_trans(material_name, 10)
        data = mmvt.coloring.get_obj_color_data(surf_obj, COLORS[ind])
        mmvt.coloring.activity_map_obj_coloring(watershed_name, data)

