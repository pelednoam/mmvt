import os.path as op
import numpy as np


def run(mmvt):
    import importlib
    mu = mmvt.utils
    mu.add_fol_to_path(mu.get_current_fol())
    import plot_vertices
    importlib.reload(plot_vertices)

    data = {'rh':[], 'lh':[], 'cb_title':'Vertices data example'}
    N = 500
    for hemi in mu.HEMIS:
        verts = np.load(op.join(mu.get_user_fol(), 'surf', '{}.pial.npz'.format(hemi)))['verts']
        selected_verts = np.random.randint(0, len(verts), N)
        data[hemi] = np.vstack(
            (selected_verts, np.random.randint(1, 11, len(selected_verts)))).T
    for hemi in ['Left', 'Right']:
        verts = np.load(op.join(mu.get_user_fol(), 'subcortical', 'Left-Cerebellum-Cortex.npz'))['verts']
        selected_verts = np.random.randint(0, len(verts), N)
        data['{}-Cerebellum-Cortex'.format(hemi)] = np.vstack(
            (selected_verts, np.random.randint(1, 11, len(selected_verts)))).T
    fol = mu.make_dir(op.join(mu.get_user_fol(), 'vertices_data'))
    np.savez(op.join(fol, 'vertices_data.npz'), **data)
    plot_vertices.run(mmvt)