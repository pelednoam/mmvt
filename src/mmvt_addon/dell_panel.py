import bpy
import bpy_extras
import os.path as op
import glob
import numpy as np
import importlib
import shutil
import os
import random
import traceback
import time
import csv
from itertools import cycle
import mmvt_utils as mu
from scripts import scripts_utils as su

try:
    from dell import find_electrodes_in_ct as fect
    importlib.reload(fect)
    DELL_EXIST = True
except:
    print(traceback.format_exc())
    DELL_EXIST = False

try:
    import nibabel as nib
    NIBABEL_EXIST = True
except:
    NIBABEL_EXIST = False


def _addon():
    return DellPanel.addon


# def dell_ct_n_groups_update(self, context):
#     if bpy.context.scene.dell_ct_n_groups > 0:
#         DellPanel.colors = mu.get_distinct_colors(bpy.context.scene.dell_ct_n_groups)


def ct_mark_noise_update(self, context):
    for p in DellPanel.noise:
        bpy.data.objects[DellPanel.names[p]].hide = not bpy.context.scene.ct_mark_noise


def ct_plot_lead_update(self, context):
    mu.show_hide_hierarchy(bpy.context.scene.ct_plot_lead, 'leads', also_parent=False, select=False)


def dell_move_elec_update(self, context):
    if not DellPanel.update_position:
        return
    elc = context.active_object
    coo = (bpy.context.scene.dell_move_x, bpy.context.scene.dell_move_y, bpy.context.scene.dell_move_z)
    _addon().set_ct_coo(coo)
    tkreg_ras = _addon().get_tkreg_ras()
    for k in range(3):
        elc.location[k] = tkreg_ras[k] * 0.1
    _addon().create_slices(pos=tkreg_ras)
    elc_ind = DellPanel.names.index(elc.name)
    DellPanel.pos[elc_ind] = tkreg_ras


# @mu.profileit('cumtime', op.join(mu.get_user_fol()))
def find_electrodes_pipeline():
    find_how_many_electrodes_above_threshold()
    print('import_electrodes...')
    _addon().import_electrodes(elecs_pos=DellPanel.pos, elecs_names=DellPanel.names, bipolar=False,
                               parnet_name='Deep_electrodes')
    print('finish importing the elecctrodes!')
    _addon().show_electrodes()


def find_how_many_electrodes_above_threshold():
    local_maxima_fname = op.join(DellPanel.output_fol, 'local_maxima_{}.npy'.format(
        bpy.context.scene.dell_ct_threshold))
    if bpy.context.scene.dell_binary_erosion:
        DellPanel.ct_data = fect.binary_erosion(DellPanel.ct_data, bpy.context.scene.dell_ct_threshold)
    if not op.isfile(local_maxima_fname):
        print('find_voxels_above_threshold ({})'.format(bpy.context.scene.dell_ct_threshold))
        ct_voxels = fect.find_voxels_above_threshold(DellPanel.ct_data, bpy.context.scene.dell_ct_threshold)
        print('{} voxels were found above {}'.format(len(ct_voxels), bpy.context.scene.dell_ct_threshold))
        print('Finding local maxima')
        ct_voxels = fect.find_all_local_maxima(
            DellPanel.ct_data, ct_voxels, bpy.context.scene.dell_ct_threshold,
            find_nei_maxima=bpy.context.scene.dell_find_nei_maxima, max_iters=100)
        np.save(local_maxima_fname, ct_voxels)
    else:
        ct_voxels = np.load(local_maxima_fname)
    print('{} local maxima were found'.format(len(ct_voxels)))
    ct_voxels = fect.remove_neighbors_voxels(DellPanel.ct_data, ct_voxels)
    print('{} local maxima after removing neighbors'.format(len(ct_voxels)))
    print('mask_voxels_outside_brain...')
    ct_electrodes, _ = fect.mask_voxels_outside_brain(
        ct_voxels, DellPanel.ct.header, DellPanel.brain, mu.get_user_fol(), bpy.context.scene.dell_brain_mask_sigma)
    print('{} voxels in the brain were found'.format(len(ct_electrodes)))
    DellPanel.pos = fect.ct_voxels_to_t1_ras_tkr(ct_electrodes, DellPanel.ct.header, DellPanel.brain.header)
    DellPanel.hemis = fect.find_electrodes_hemis(
        mu.get_user_fol(), DellPanel.pos, groups=None, sigma=bpy.context.scene.dell_brain_mask_sigma)
    DellPanel.names = name_electrodes(DellPanel.hemis)
    output_fname = op.join(DellPanel.output_fol, '{}_electrodes.pkl'.format(int(bpy.context.scene.dell_ct_threshold)))
    bpy.context.scene.dell_how_many_electrodes_above_threshold = str(len(DellPanel.names))
    mu.save((DellPanel.pos, DellPanel.names, DellPanel.hemis, DellPanel.groups, DellPanel.noise,
             bpy.context.scene.dell_ct_threshold), output_fname)


def save():
    output_fname = op.join(DellPanel.output_fol, '{}_electrodes.pkl'.format(int(bpy.context.scene.dell_ct_threshold)))
    mu.save((DellPanel.pos, DellPanel.names, DellPanel.hemis, DellPanel.groups, DellPanel.noise,
             bpy.context.scene.dell_ct_threshold), output_fname)
    print('dell objects were saved to {}'.format(output_fname))


@mu.tryit()
def init_groups():
    parent = bpy.data.objects.get('Deep_electrodes', None)
    if parent is None or len(parent.children) == 0:
        return
    for p in DellPanel.noise:
        if p >= len(DellPanel.names):
            continue
        _addon().object_coloring(bpy.data.objects[DellPanel.names[p]], tuple(bpy.context.scene.dell_ct_noise_color))
    for ind, group in enumerate(DellPanel.groups):
        color = next(DellPanel.colors)
        if np.all(color == tuple(bpy.context.scene.dell_ct_noise_color)):
            color = next(DellPanel.colors)
        for elc_ind in group:
            _addon().object_coloring(bpy.data.objects[DellPanel.names[elc_ind]], tuple(color))
    if len(DellPanel.groups) == 0:
        clear_electrodes_color()
    # log_fname = op.join(DellPanel.output_fol, '{}_log.pkl'.format(
    #     int(bpy.context.scene.dell_ct_threshold)))
    # if op.isfile(log_fname):
    #     DellPanel.log = mu.load(log_fname)
    # DellPanel.current_log = []


def init_noise():
    for p in DellPanel.noise:
        if p >= len(DellPanel.names):
            continue
        group_inds = [k for k, g in enumerate(DellPanel.groups) if p in g]
        if len(group_inds) > 0:
            continue
        _addon().object_coloring(bpy.data.objects[DellPanel.names[p]], (1, 1, 1))
    DellPanel.noise = set()
    init_groups()


def load_electrodes_from_file():
    log_fname = op.join(DellPanel.output_fol, '{}_electrodes.pkl'.format(int(bpy.context.scene.dell_ct_threshold)))
    if not op.isfile(log_fname):
        return False
    print('Loading electrodes names and pos from {}'.format(log_fname))
    objects = bpy.data.objects['Deep_electrodes'].children
    (DellPanel.pos, DellPanel.names, DellPanel.hemis, DellPanel.groups, DellPanel.noise,
     bpy.context.scene.dell_ct_threshold) = mu.load(log_fname)
    extra_electrodes = set(DellPanel.names) - set([o.name for o in objects])
    for extra_elec_name in extra_electrodes:
        elc_ind = DellPanel.names.index(extra_elec_name)
        _addon().data.create_electrode(DellPanel.pos[elc_ind], extra_elec_name)
    init_groups()
    return True


def refresh_pos_and_names(overwrite=False):
    print('Refreshing the pos and names')
    output_fname = op.join(DellPanel.output_fol, '{}_electrodes.pkl'.format(int(bpy.context.scene.dell_ct_threshold)))
    objects = bpy.data.objects['Deep_electrodes'].children
    if op.isfile(output_fname) and not overwrite:
        load_electrodes_from_file()
    else:
        DellPanel.names = [o.name for o in objects]
        DellPanel.pos = np.array([np.array(o.location) for o in objects]) * 10
        DellPanel.groups = []
        # subject_fol = op.join(mu.get_subjects_dir(), mu.get_user())
        DellPanel.hemis = fect.find_electrodes_hemis(
            mu.get_user_fol(), DellPanel.pos, None, 1, DellPanel.verts_dural, DellPanel.normals_dural)
        DellPanel.names = name_electrodes(DellPanel.hemis)
        for new_name, obj in zip(DellPanel.names, objects):
            obj.name = new_name
        for obj in objects:
            obj.name = obj.name.replace('.001', '')
        clear_electrodes_color()
        mu.save((DellPanel.pos, DellPanel.names, DellPanel.hemis, DellPanel.groups, DellPanel.noise,
                 bpy.context.scene.dell_ct_threshold), output_fname)


def export_electrodes(group_hemi_default='G'):
    from collections import Counter
    output_fol = op.join(mu.get_user_fol(), 'electrodes')
    subject = mu.get_user()

    delete_noise_electrodes()
    if len(DellPanel.groups) > 0:
        groups = DellPanel.groups
    else:
        parent_obj = bpy.data.objects.get('Deep_electrodes') # _addon().electrodes_panel_parent_obj
        groups = [[k for k, o in enumerate(parent_obj.children)]]
        if len(DellPanel.pos) != len(groups[0]):
            print('len(DellPanel.pos) ({}) != len(groups[0]) ({})!'.format(len(DellPanel.pos), len(groups[0])))
            return
        group_hemi = group_hemi_default
    csv_fname = op.join(output_fol, '{}_RAS.csv'.format(subject))
    for fname in glob.glob(op.join(output_fol, '*.dat')):
        os.remove(fname)
    with open(csv_fname, 'w') as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_NONE)
        wr.writerow(['Electrode Name','R','A','S'])
        groups_inds = {'R':0, 'L':0}
        for group in groups:
            # DellPanel.hemis
            # group_hemi = 'R' if group_hemi == 'rh' else 'L' if group_hemi == 'lh' else group_hemi_default
            group_hemi = Counter([DellPanel.names[group[k]][0] for k in range(len(group))]).most_common()[0][0]
            if group_hemi in groups_inds:
                group_name = '{}G{}'.format(group_hemi, chr(ord('A') + groups_inds[group_hemi]))
            else:
                group_name = group_hemi_default
            elcs_new_names = ['{}{}'.format(group_name, k+1) for k in range(len(group))]
            group_elcs_names = [DellPanel.names[g] for g in group]
            for elc_ind, elc_name, elc_new_name in zip(group, group_elcs_names, elcs_new_names):
                bpy.data.objects[elc_name].name = elc_new_name
                DellPanel.names[elc_ind] = elc_new_name
            for ind, elc_ind in enumerate(group):
                wr.writerow([elcs_new_names[ind], *['{:.2f}'.format(loc) for loc in DellPanel.pos[elc_ind]]])
            if group_hemi in groups_inds:
                groups_inds[group_hemi] += 1

    _addon().electrodes.init(_addon(), False)
    print('The electrodes were exported to {}'.format(csv_fname))


def delete_noise_electrodes():
    bpy.ops.object.select_all(action='DESELECT')
    noise_elctrodes = [DellPanel.names[elc_ind] for elc_ind in DellPanel.noise]
    for noise_elc in noise_elctrodes:
        bpy.data.objects[noise_elc].select = True
    bpy.ops.object.delete()


def mark_electrode_lead_as_noise():
    elc_name = bpy.context.selected_objects[0].name
    elc_ind = DellPanel.names.index(elc_name)
    group_inds = [k for k, g in enumerate(DellPanel.groups) if elc_ind in g]
    if len(group_inds) == 0:
        print('Selected electrode has no group')
        return
    print('Setting group {} as noise'.format(group_inds[0]))
    g = DellPanel.groups[group_inds[0]]
    for elc_ind in g:
        mark_elc_as_noise(None, elc_ind)
        # DellPanel.noise.add(elc_ind)
        # _addon().object_coloring(bpy.data.objects[DellPanel.names[elc_ind]], tuple(bpy.context.scene.dell_ct_noise_color))
    lead_obj = bpy.data.objects.get('{}-{}'.format(DellPanel.names[g[0]], DellPanel.names[g[-1]]))
    if lead_obj is not None:
        bpy.data.objects.remove(lead_obj, True)
        # lead_obj.hide = True
    del DellPanel.groups[group_inds[0]]
    # save_dell_objects()


def save_dell_objects():
    output_fname = op.join(DellPanel.output_fol, '{}_electrodes.pkl'.format(int(bpy.context.scene.dell_ct_threshold)))
    mu.save((DellPanel.pos, DellPanel.names, DellPanel.hemis, DellPanel.groups, DellPanel.noise,
             bpy.context.scene.dell_ct_threshold), output_fname)


def import_electrodes_from_dell():
    _addon().import_electrodes(elecs_pos=DellPanel.pos, elecs_names=DellPanel.names, bipolar=False,
                               parnet_name='Deep_electrodes')
    print('finish importing the elecctrodes!')
    _addon().show_electrodes()


def mark_selected_electrodes_as_noise():
    for elc_obj in bpy.context.selected_objects:
        elc_name = elc_obj.name
        if elc_name not in DellPanel.names:
            continue
        elc_ind = DellPanel.names.index(elc_name)
        group_inds = [k for k, g in enumerate(DellPanel.groups) if elc_ind in g]
        if len(group_inds) == 1:
            DellPanel.groups[group_inds[0]].remove(elc_ind)
        print('Removing {} from Dell objects'.format(elc_name))
        mark_elc_as_noise(elc_obj, elc_ind)
    DellPanel.groups = [g for g in DellPanel.groups if len(g) > 0]
    # save_dell_objects()


def mark_selected_noise_as_electrodes():
    names = [o.name for o in bpy.context.selected_objects]
    for elc_name in names:
        if elc_name not in DellPanel.names:
            continue
        elc_ind = DellPanel.names.index(elc_name)
        DellPanel.noise.discard(elc_ind)
        _addon().object_coloring(bpy.data.objects[elc_name], (0.8, 0.8, 0.8))


def mark_non_group_electrodes_as_noise():
    # bad_electrodes = []
    for elc_obj in bpy.data.objects['Deep_electrodes'].children:
        # elc_obj.select = False
        elc_name = elc_obj.name
        # if elc_name not in DellPanel.names:
        #     bad_electrodes.append(elc_obj)
        #     continue
        elc_ind = DellPanel.names.index(elc_name)
        group_inds = [k for k, g in enumerate(DellPanel.groups) if elc_ind in g]
        if len(group_inds) == 0:
            mark_elc_as_noise(elc_obj, elc_ind)
    #         bad_electrodes.append(elc_obj)
    #         if elc_name in DellPanel.names:
    #             elc_ind = DellPanel.names.index(elc_name)
    #             group_inds = [k for k, g in enumerate(DellPanel.groups) if elc_ind in g]
    #             if len(group_inds) == 1:
    #                 DellPanel.groups[group_inds[0]].remove(elc_ind)
    #             DellPanel.names.remove(elc_name)
    #
    # for bad_elec in bad_electrodes:
    #     bad_elec.select = True
    #     bpy.ops.object.delete()


def mark_elc_as_noise(elc_obj=None, elc_ind=-1):
    if elc_obj is None and elc_ind < len(DellPanel.names):
        elc_obj = bpy.data.objects.get(DellPanel.names[elc_ind])
        if elc_obj is None:
            print('No object for {}!'.format(DellPanel.names[elc_ind]))
    if elc_ind == -1:
        if elc_obj.name in DellPanel.names:
            elc_ind = DellPanel.names.index(elc_obj.name)
        else:
            print('{} is not in DellPanel.names!'.format(elc_obj.name))
    DellPanel.noise.add(elc_ind)
    if elc_obj is not None:
        elc_obj.hide = not bpy.context.scene.ct_mark_noise
        _addon().object_coloring(elc_obj, tuple(bpy.context.scene.dell_ct_noise_color))


def create_new_electrode():
    new_pos = _addon().where_am_i.get_tkreg_ras()
    name = bpy.context.selected_objects[0].name if len(bpy.context.selected_objects) == 1 else ''
    if len(bpy.context.selected_objects) == 1 and name in DellPanel.names:
        elc_ind = DellPanel.names.index(name)
        hemi = DellPanel.hemis[elc_ind]
    else:
        elc_ind = 0
        hemi = fect.find_electrodes_hemis(
            mu.get_user_fol(), [new_pos], None, bpy.context.scene.dell_brain_mask_sigma, DellPanel.verts_dural,
            DellPanel.normals_dural)[0]
        name = '{}UNx'.format('L' if hemi == 'lh' else 'R' if hemi == 'rh' else 'U')
    elecs_group = [int(n[3:]) for n in DellPanel.names if n.startswith(name[:3])]
    new_num = max(elecs_group) + 1 if len(elecs_group) > 0 else 1
    new_name = '{}{}'.format(name[:3], new_num)
    DellPanel.names.append(new_name)
    group_inds = [k for k, g in enumerate(DellPanel.groups) if elc_ind in g]
    if len(group_inds) == 1:
        new_index = DellPanel.names.index(new_name)
        DellPanel.groups[group_inds[0]].append(new_index)
    DellPanel.pos = np.vstack((DellPanel.pos, new_pos))
    DellPanel.hemis.append(hemi)
    _addon().data.create_electrode(new_pos, new_name)
    color = bpy.data.objects['Deep_electrodes'].children[0].active_material.diffuse_color
    _addon().object_coloring(bpy.data.objects[new_name], tuple(color))
    # save_dell_objects()


def create_new_electrode_between():
    names = [o.name for o in bpy.context.selected_objects]
    elc_inds = [DellPanel.names.index(names[k]) for k in range(2)]
    pos = [DellPanel.pos[elc_ind] for elc_ind in elc_inds]
    new_pos = np.mean(pos, 0)
    hemi = DellPanel.hemis[elc_inds[0]]
    group_ind = [k for k,g in enumerate(DellPanel.groups) if elc_inds[0] in g][0]
    new_num = max([int(name[3:]) for name in DellPanel.names if name.startswith(names[0][:3])]) + 1
    new_name = '{}{}'.format(names[0][:3], new_num)
    color = bpy.context.selected_objects[0].active_material.diffuse_color
    DellPanel.names.append(new_name)
    new_index = DellPanel.names.index(new_name)
    DellPanel.pos = np.vstack((DellPanel.pos, new_pos))
    DellPanel.hemis.append(hemi)
    DellPanel.groups[group_ind].append(new_index)
    _addon().data.create_electrode(new_pos, new_name)
    _addon().object_coloring(bpy.data.objects[new_name], tuple(color))
    # save_dell_objects()


# @mu.profileit('cumtime', op.join(mu.get_user_fol()))
def find_electrode_lead():
    if len(bpy.context.selected_objects) == 1 and bpy.context.selected_objects[0].name in DellPanel.names:
        selected_elc = bpy.context.selected_objects[0].name
        elc_ind = DellPanel.names.index(selected_elc)
        if elc_ind in set(mu.flat_list_of_lists(DellPanel.groups)):
            print('{} is already in a group!'.format(DellPanel.names[elc_ind]))
        else:
            group = _find_electrode_lead(elc_ind)
    elif len(bpy.context.selected_objects) == 2 and \
            all(bpy.context.selected_objects[k].name in DellPanel.names for k in range(2)):
        selected_elcs = [bpy.context.selected_objects[k].name for k in range(2)]
        elc_inds = [DellPanel.names.index(elc) for elc in selected_elcs]
        all_groups = set(mu.flat_list_of_lists(DellPanel.groups))
        if any([elc_ind in all_groups for elc_ind in elc_inds]):
            print('Choose electrodes that are not in an existing group')
        else:
            group = _find_electrode_lead(*elc_inds)
    else:
        print("You should first select an electrode")


def _find_electrode_lead(elc_ind, elc_ind2=-1, debug=True):
    if elc_ind2 == -1:
        group, noise, DellPanel.dists, dists_to_cylinder, gof, best_elc_ind = fect.find_electrode_group(
            elc_ind, DellPanel.pos, DellPanel.hemis, DellPanel.groups, bpy.context.scene.dell_ct_error_radius,
            bpy.context.scene.dell_ct_min_elcs_for_lead, bpy.context.scene.dell_ct_max_dist_between_electrodes,
            bpy.context.scene.dell_ct_min_distance, bpy.context.scene.dell_do_post_search)
        # DellPanel.ct_data, bpy.context.scene.dell_ct_threshold, DellPanel.ct.header, DellPanel.brain.header
    else:
        group, noise, DellPanel.dists, dists_to_cylinder, gof, best_elc_ind = fect.find_group_between_pair(
            elc_ind, elc_ind2, DellPanel.pos, bpy.context.scene.dell_ct_error_radius,
            bpy.context.scene.dell_ct_min_distance)
    if not isinstance(group, list):
        group = group.tolist()
    if len(group) == 0:
        print('No group was found for {}!'.format(DellPanel.names[elc_ind]))
        DellPanel.noise.add(elc_ind)
        return []

    DellPanel.dists_to_cylinder = {DellPanel.names[p]:d for p, d in dists_to_cylinder.items()}
    if bpy.context.scene.ct_plot_lead:
        create_lead(group[0], group[-1])
    DellPanel.groups.append(group)
    for p in noise:
        _addon().object_coloring(bpy.data.objects[DellPanel.names[p]], tuple(bpy.context.scene.dell_ct_noise_color))
        DellPanel.noise.add(p)
    color = next(DellPanel.colors)
    for elc_ind in group:
        _addon().object_coloring(bpy.data.objects[DellPanel.names[elc_ind]], tuple(color))
    return group


def find_electrode_group_on_dural():
    if len(bpy.context.selected_objects) == 2:
        selected_elcs = [bpy.context.selected_objects[k].name for k in range(2)]
        _find_electrode_group_on_dural(selected_elcs)
    elif len(bpy.context.selected_objects) == 0:
        for pt1, pt2 in zip(['G38', 'G64', 'G19', 'G49', 'G37', 'G46', 'G7', 'G22'][::-1],
                            ['G26', 'G52', 'G56', 'G41', 'G20', 'G47', 'G15', 'G48'][::-1]):
            _find_electrode_group_on_dural((pt1, pt2))
        for pt1, pt2 in zip(['G38', 'G25', 'G61', 'G36', 'G4', 'G39', 'G23', 'G26'],
                            ['G22', 'G58', 'G35', 'G1', 'G63', 'G30', 'G13', 'G48']):
            _find_electrode_group_on_dural((pt1, pt2))


def _find_electrode_group_on_dural(selected_elcs):
    elc_inds = [DellPanel.names.index(elc) for elc in selected_elcs]
    points, points_inside = fect.find_points_path_on_dural_surface(
        elc_inds[0], elc_inds[1], DellPanel.hemis, DellPanel.pos, DellPanel.verts_dural, DellPanel.verts_dural_nei,
        DellPanel.names, 5, (0.1, 0.5))
    color = next(DellPanel.colors)
    for elc_ind in points_inside:
        _addon().object_coloring(bpy.data.objects[DellPanel.names[elc_ind]], tuple(color))
    print([DellPanel.names[p] for p in points_inside])
    for p1, p2 in zip(points[:-1], points[1:]):
        create_lead(p1, p2, inds_are_points=True)


# @mu.profileit('cumtime', op.join(mu.get_user_fol()))
def find_all_groups(runs_num=100):
    clear_groups()
    if bpy.context.scene.dell_find_all_group_using_timer:
        start_timer()
    else:
        group_found = find_random_group()
        run_num = 0
        while group_found and run_num < runs_num:
            g = DellPanel.groups[-1]
            print('{}) group {}-{} found!'.format(run_num + 1, DellPanel.names[g[0]], DellPanel.names[g[-1]]))
            group_found = find_random_group()
            run_num += 1
    save_dell_objects()
    print('Done!')


def find_random_group(runs_num=100):
    elcs = list(set(range(len(DellPanel.names))) - set(mu.flat_list_of_lists(DellPanel.groups)) - DellPanel.noise)
    group, run_num, log = [], 0, None
    while len(elcs) > 0 and len(group) == 0 and run_num < runs_num:
        elc_ind = random.choice(elcs)
        group = _find_electrode_lead(elc_ind)
        if len(group) == 0:
            elcs = list(set(range(len(DellPanel.names))) - set(mu.flat_list_of_lists(DellPanel.groups)) - DellPanel.noise)
            run_num += 1
        else:
            _addon().render.save_image(add_colorbar=False)
        #     log = (DellPanel.names[elc_ind], [DellPanel.names[ind] for ind in group])
        #     DellPanel.log.append(log)

    # if log is not None:
    #     DellPanel.current_log = [log]
    #     mu.save(DellPanel.log, op.join(DellPanel.output_fol, '{}_log.pkl'.format(
    #         int(bpy.context.scene.dell_ct_threshold))))
    print('find_random_group: exit afer {} runs'.format(run_num))
    if len(group) == 0:
        bpy.context.scene.dell_all_groups_were_found = True
    return len(group) > 0


def project_electrodes_on_leads():
    for g in DellPanel.groups:
        group_elecs_pos = [DellPanel.pos[g[k]] for k in range(len(g))]
        group_elecs_names = [DellPanel.names[g[k]] for k in range(len(g))]
        group_elecs_new_pos = mu.move_electrodes_to_line(DellPanel.pos[g[0]], DellPanel.pos[g[-1]], group_elecs_pos)
        for elc_name, elc_new_pos in zip(group_elecs_names, group_elecs_new_pos):
            bpy.data.objects[elc_name].location = elc_new_pos * 0.1


def save_ct_neighborhood():
    if len(bpy.context.selected_objects) == 1 and bpy.context.selected_objects[0].name in DellPanel.names:
        selected_elc = bpy.context.selected_objects[0].name
        elc_ind = DellPanel.names.index(selected_elc)
        ct_vox = fect.t1_ras_tkr_to_ct_voxels(DellPanel.pos[elc_ind], DellPanel.ct.header, DellPanel.brain.header)
        ct_vals = fect.get_voxel_neighbors_ct_values(DellPanel.ct_data, ct_vox)
        output_fname = op.join(mu.get_user_fol(), 'ct', 'voxel_neighbors_ct_values_{}.npy'.format(
            DellPanel.names[elc_ind]))
        print(ct_vals)
        np.save(output_fname, ct_vals)
        print('CT values around the electrode were saved in {}'.format(output_fname))
    else:
        print('You need to select an electrode first!')


def name_electrodes(elctrodes_hemis):
    elcs_nums = {'rh':1, 'lh':1, 'un':1}
    names = []
    for elc_hemi in elctrodes_hemis:
        names.append('{}UN{}'.format('R' if elc_hemi == 'rh' else 'L' if elc_hemi == 'lh' else 'M', elcs_nums[elc_hemi]))
        elcs_nums[elc_hemi] += 1
    return names


def delete_electrodes():
    clear_groups()
    mu.delete_hierarchy('Deep_electrodes')


def delete_electrodes_from_selection():
    for obj in bpy.context.selected_objects:
        if obj.parent.name != 'Deep_electrodes':
            obj.selected = False
    selected_electrodes_indices = [DellPanel.names.index(obj.name) for obj in bpy.context.selected_objects]
    elcs_pos = [DellPanel.pos[ind] for ind in selected_electrodes_indices]
    elcs_ct_voxels = np.rint(fect.t1_ras_tkr_to_ct_voxels(elcs_pos, DellPanel.ct.header, DellPanel.brain.header)).astype(int)
    ct_int = [DellPanel.ct_data[tuple(vox)] for vox in elcs_ct_voxels]
    print('CT int of selected electrodes: {}'.format(ct_int))
    max_ct_int_ind = np.argmin(ct_int)
    print('{} has the highest CT int: {}'.format(DellPanel.names[max_ct_int_ind], max(ct_int)))
    bpy.context.selected_objects[max_ct_int_ind].select = False
    bpy.ops.object.delete()


def create_new_group_from_selected_electrodes():
    new_group = [DellPanel.names.index(obj.name) for obj in bpy.context.selected_objects]
    DellPanel.groups.append(new_group)
    color = next(DellPanel.colors)
    for elc_ind in new_group:
        _addon().object_coloring(bpy.data.objects[DellPanel.names[elc_ind]], tuple(color))


def calc_threshold_precentile():
    bpy.context.scene.dell_ct_threshold = np.percentile(
        DellPanel.ct_data, bpy.context.scene.dell_ct_threshold_percentile)


def prev_ct_electrode():
    group, in_group_ind = find_select_electrode_group()
    if in_group_ind != -1:
        prec_elc_ind = group[in_group_ind - 1] if in_group_ind > 0 else group[-1]
        prev_elc_name = DellPanel.names[prec_elc_ind]
        select_new_electrode(prev_elc_name)


def next_ct_electrode():
    group, in_group_ind = find_select_electrode_group()
    if in_group_ind != -1:
        next_elc_ind = group[in_group_ind + 1] if in_group_ind + 1 < len(group) else group[0]
        next_elc_name = DellPanel.names[next_elc_ind]
        select_new_electrode(next_elc_name)


def find_select_electrode_group(selected_elc=''):
    in_group_ind, group = -1, []
    if selected_elc == '' and len(bpy.context.selected_objects) == 1:
        selected_elc = bpy.context.selected_objects[0].name
    if len(bpy.context.selected_objects) == 1 and bpy.context.selected_objects[0].name in DellPanel.names:
        elc_ind = DellPanel.names.index(selected_elc)
        groups_mask = [(elc_ind in g) for g in DellPanel.groups]
        if sum(groups_mask) == 1:
            group = [g for g, m in zip(DellPanel.groups, groups_mask) if m][0]
            in_group_ind = group.index(elc_ind)
    return group, in_group_ind


def select_new_electrode(new_electrode_name):
    bpy.data.objects[bpy.context.selected_objects[0].name].select = False
    bpy.data.objects[new_electrode_name].select = True
    bpy.context.scene.objects.active = bpy.data.objects[new_electrode_name]
    _addon().electode_was_manually_selected(new_electrode_name)
    bpy.context.scene.dell_move_x = bpy.context.scene.ct_voxel_x
    bpy.context.scene.dell_move_y = bpy.context.scene.ct_voxel_y
    bpy.context.scene.dell_move_z = bpy.context.scene.ct_voxel_z


def create_lead(ind1, ind2, radius=0.05, inds_are_points=False):
    layers = [False] * 20
    lead_layer = _addon().ELECTRODES_LAYER
    layers[lead_layer] = True
    parent_name = 'leads'
    mu.create_empty_if_doesnt_exists(parent_name, _addon().BRAIN_EMPTY_LAYER, None)

    if not inds_are_points:
        p1, p2 = DellPanel.pos[ind1] * 0.1, DellPanel.pos[ind2] * 0.1
    else:
        p1, p2 = ind1 * 0.1, ind2 * 0.1
    mu.cylinder_between(p1, p2, radius, layers)
    if not inds_are_points:
        lead_name = '{}-{}'.format(DellPanel.names[ind1], DellPanel.names[ind2])
    else:
        lead_name = mu.rand_letters(5)
    color = tuple(np.concatenate((bpy.context.scene.dell_ct_lead_color, [1])))
    mu.create_material('{}_mat'.format(lead_name), color, 1)
    cur_obj = bpy.context.active_object
    cur_obj.name = lead_name
    cur_obj.parent = bpy.data.objects[parent_name]
    bpy.data.objects[lead_name].select = False


def clear_groups():
    for p in DellPanel.noise:
        if bpy.data.objects.get(DellPanel.names[p], None) is not None:
            bpy.data.objects[DellPanel.names[p]].hide = False
    DellPanel.groups = []
    DellPanel.noise = set()
    groups_fname = op.join(DellPanel.output_fol, '{}_groups.pkl'.format(
        int(bpy.context.scene.dell_ct_threshold)))
    if op.isfile(groups_fname):
        shutil.copy(groups_fname, '{}_backup{}'.format(*op.splitext(groups_fname)))
        os.remove(groups_fname)
    bpy.context.scene.ct_mark_noise = True
    clear_electrodes_color()
    mu.delete_hierarchy('leads')
    log_fname = op.join(DellPanel.output_fol, '{}_log.pkl'.format(
        int(bpy.context.scene.dell_ct_threshold)))
    if op.isfile(log_fname):
        shutil.copy(log_fname, '{}_backup{}'.format(*op.splitext(log_fname)))
        os.remove(log_fname)
    # DellPanel.log = []
    bpy.context.scene.dell_all_groups_were_found = False
    DellPanel.is_playing = False


def dell_ct_electrode_was_selected(elc_name):
    if not DellPanel.init:
        return
    group, in_group_ind = find_select_electrode_group(elc_name)
    group = [DellPanel.names[g] for g in group]
    # DellPanel.current_log = [(elc, g) for (elc, g) in DellPanel.log if set(g) == set(group)]
    DellPanel.update_position = False
    tkreg_ras = _addon().electrodes.get_electrode_pos(elc_name) * 10# bpy.data.objects[elc_name].location * 10
    _addon().where_am_i.set_tkreg_ras(tkreg_ras)
    # tkreg_ras2 = _addon().get_tkreg_ras()
    ct_vox = _addon().get_ct_voxel()
    bpy.context.scene.dell_move_x = ct_vox[0]# bpy.data.objects[elc_name].location[0]
    bpy.context.scene.dell_move_y = ct_vox[1]# bpy.data.objects[elc_name].location[1]
    bpy.context.scene.dell_move_z = ct_vox[2]# bpy.data.objects[elc_name].location[2]
    DellPanel.update_position = True


def clear_electrodes_color():
    for elc_name in DellPanel.names:
        if bpy.data.objects.get(elc_name) is not None:
            _addon().object_coloring(bpy.data.objects[elc_name], (1, 1, 1))


def open_interactive_ct_viewer():
    elc_name = bpy.context.selected_objects[0].name
    if elc_name in DellPanel.names:
        t1_vox = DellPanel.pos[DellPanel.names.index(elc_name)]
        ct_vox = fect.t1_ras_tkr_to_ct_voxels(t1_vox, DellPanel.ct.header, DellPanel.brain.header)
        voxel = ','.join(map(str, tuple(ct_vox)))
        cmd = '{} -m src.preproc.ct -s {} '.format(bpy.context.scene.python_cmd, mu.get_user()) + \
              '-f save_electrode_ct_pics --voxel "{}" --elc_name {} '.format(voxel, elc_name) + \
              '--interactive 1 --ignore_missing 1'
        mu.run_command_in_new_thread(cmd, False)


def check_if_outside_pial():
    # aseg = None if not bpy.context.scene.dell_brain_mask_use_aseg else DellPanel.aseg
    # subject_fol = op.join(mu.get_subjects_dir(), mu.get_user())
    voxels = fect.t1_ras_tkr_to_ct_voxels(DellPanel.pos, DellPanel.ct.header, DellPanel.brain.header)
    voxels_in, voxels_in_indices = fect.mask_voxels_outside_brain(
        voxels, DellPanel.ct.header, DellPanel.brain, mu.get_user_fol(),
        bpy.context.scene.dell_brain_mask_sigma)
    indices_outside_brain = set(range(len(voxels))) - set(voxels_in_indices)
    for ind in indices_outside_brain:
        _addon().object_coloring(bpy.data.objects[DellPanel.names[ind]], (0, 1, 0))


def save_ct_electrodes_figures():
    group, in_group_ind = find_select_electrode_group()
    electrodes_names = ','.join([DellPanel.names[g] for g in group])
    t1_vox = np.array([DellPanel.pos[g] for g in group])
    ct_vox = fect.t1_ras_tkr_to_ct_voxels(t1_vox, DellPanel.ct.header, DellPanel.brain.header)
    voxels = ';'.join([','.join(map(str, tuple(vox))) for vox in ct_vox])
    cmd = '{} -m src.preproc.ct -s {} '.format(bpy.context.scene.python_cmd, mu.get_user()) + \
          '-f save_electrodes_group_ct_pics --voxels "{}" '.format(voxels) + \
          '--electrodes_names "{}" --interactive 0 --ignore_missing 1'.format(electrodes_names)
    mu.run_command_in_new_thread(cmd, False)


def run_ct_preproc():
    cmd = '{} -m src.preproc.ct -s {} -f save_subject_ct_trans,save_images_data_and_header --ignore_missing 1'.format(
        bpy.context.scene.python_cmd, mu.get_user())
    mu.run_command_in_new_thread(cmd, False)


def install_dell_reqs():
    from scripts import scripts_utils as su
    from scripts import call_script_utils as sutils

    links_dir = su.get_links_dir()
    blender_fol = su.get_link_dir(links_dir, 'blender')
    blender_bin_fol = glob.glob(op.join(blender_fol, '2.7?', 'python'))[0]
    cmd = '{} install sklearn nibabel'.format(op.join(blender_bin_fol, 'bin', 'pip'))
    sutils.run_script(cmd)


# def dell_draw_presentation(self, context):
#     layout = self.layout
#     parent = bpy.data.objects.get('Deep_electrodes', None)
#     if parent is None or len(parent.children) == 0:
#         row = layout.row(align=0)
#         row.prop(context.scene, 'dell_ct_threshold', text="Threshold")
#         row.prop(context.scene, 'dell_ct_threshold_percentile', text='Percentile')
#         row.operator(CalcThresholdPercentile.bl_idname, text="Calc threshold", icon='STRANDS')
#         layout.prop(context.scene, 'dell_brain_mask_sigma', text='Brain mask sigma')
#         layout.prop(context.scene, 'dell_find_nei_maxima', text='Find local nei maxima')
#         layout.operator(FindHowManyElectrodesAboveThrshold.bl_idname, text="Find how many electrodes", icon='NODE_SEL')
#         layout.operator(GetElectrodesAboveThrshold.bl_idname, text="Import electrodes", icon='ROTATE')
#         if bpy.context.scene.dell_how_many_electrodes_above_threshold != '':
#             layout.label(text='Found {} potential electrodes'.format(
#                 bpy.context.scene.dell_how_many_electrodes_above_threshold))
#     else:
#         # layout.prop(context.scene, 'dell_ct_min_elcs_for_lead', text="Min electrodes per lead")
#         if len(bpy.context.selected_objects) == 2 and all(bpy.context.selected_objects[k].name in DellPanel.names for k in range(2)):
#             layout.operator(CreateNewElectrodeBetween.bl_idname, text="Create new electrode between",
#                             icon='PARTICLE_DATA')
#             # layout.operator(FindElectrodeLead.bl_idname, text="Find lead between selected electrodes", icon='PARTICLE_DATA')
#         if not bpy.context.scene.dell_all_groups_were_found:
#             layout.operator(FindRandomLead.bl_idname, text="Find a group", icon='POSE_HLT')
#             layout.operator(FindAllLeads.bl_idname, text="Find all groups", icon='LAMP_SUN')
#         layout.operator(ProjectElectrodesOnLeads.bl_idname, text="Project electrodes on leads", icon='SURFACE_DATA')
#         # if len(bpy.context.selected_objects) == 2 and all(bpy.context.selected_objects[k].name in DellPanel.names for k in range(2)):
#         #     layout.operator(CreateNewElectrodeBetween.bl_idname, text="Create new electrode between",
#         #                     icon='WORLD_DATA')
#         layout.prop(context.scene, 'ct_mark_noise', text='Show noise')
#         if len(bpy.context.selected_objects) == 1:
#            if bpy.context.selected_objects.name in DellPanel.names:
#                layout.label(text='{} ind: {}'.format(
#                    bpy.context.selected_objects.name, DellPanel.names.index(bpy.context.selected_objects.name)))
#            else:
#                layout.label(text='{} not in DellPanel.names!'.format(bpy.context.selected_objects.name))
#         if len(DellPanel.groups) > 0:
#             layout.label(text='#Groups found: {}'.format(len(DellPanel.groups)))
#             # rd = context.scene.render
#             # row = layout.row()
#             # row.template_list("DellGroupItem", "", rd, "layers", rd.layers, "active_index", rows=2)
#             box = layout.box()
#             col = box.column()
#             for g in DellPanel.groups:
#                 mu.add_box_line(col, '{}-{}'.format(DellPanel.names[g[0]], DellPanel.names[g[-1]]), str(len(g)), 0.8)
#         layout.operator(ClearGroups.bl_idname, text="Clear groups", icon='GHOST_DISABLED')
#     if parent is not None and len(parent.children) > 0:
#         layout.operator(ExportDellElectrodes.bl_idname, text="Rename & Export", icon='EXPORT')
#         layout.prop(context.scene, 'dell_delete_electrodes', text='Delete electrodes')
#         if bpy.context.scene.dell_delete_electrodes:
#             layout.operator(DeleteElectrodes.bl_idname, text="Delete electrodes", icon='CANCEL')
#         row = layout.row(align=True)
#         row.prop(context.scene, 'dell_move_x')
#         row.prop(context.scene, 'dell_move_y')
#         row.prop(context.scene, 'dell_move_z')


class DellGroupItem(bpy.types.UIList):
    # http://sinestesia.co/blog/tutorials/using-uilists-in-blender/
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layer = item
        g = DellPanel.groups[index]
        layout.label(text=DellPanel.names[g[0]])
        layout.label(text=DellPanel.names[g[-1]])
        layout.label(text=str(len(g)))
        layout.prop(layer, "use", text="", index=index)


def dell_draw(self, context):
    layout = self.layout
    parent = bpy.data.objects.get('Deep_electrodes', None)
    electrode_with_group_selected = \
        len(bpy.context.selected_objects) == 1 and bpy.context.selected_objects[0].name in DellPanel.names and \
        DellPanel.names.index(bpy.context.selected_objects[0].name) in mu.flat_list_of_lists(DellPanel.groups)
    if not NIBABEL_EXIST or not DELL_EXIST:
        layout.operator(InstallReqs.bl_idname, text="Install reqs", icon='ROTATE')
    elif not DellPanel.ct_found:
        layout.operator(ChooseCTFile.bl_idname, text="Load CT", icon='PLUGIN').filepath=op.join(
            mu.get_user_fol(), 'ct', '*.mgz')
    elif parent is None or len(parent.children) == 0:
        row = layout.row(align=0)
        if len(glob.glob(op.join(DellPanel.output_fol, '{}_electrodes.pkl'.format(
                int(bpy.context.scene.dell_ct_threshold))))):
            layout.operator(ImportElectrodesFromDell.bl_idname, text="Import electrodes", icon='EDIT')
        row.prop(context.scene, 'dell_ct_threshold', text="Threshold")
        row.prop(context.scene, 'dell_ct_threshold_percentile', text='Percentile')
        row.operator(CalcThresholdPercentile.bl_idname, text="Calc threshold", icon='STRANDS')
        layout.prop(context.scene, 'dell_brain_mask_sigma', text='Brain mask sigma')
        layout.prop(context.scene, 'dell_find_nei_maxima', text='Find local nei maxima')
        # layout.prop(context.scene, 'use_only_brain_mask', text='Use only the brain mask')
        # layout.prop(context.scene, 'dell_binary_erosion', text='USe Binary Erosion')
        layout.operator(FindHowManyElectrodesAboveThrshold.bl_idname, text="Find how many electrodes", icon='NODE_SEL')
        layout.operator(GetElectrodesAboveThrshold.bl_idname, text="Import electrodes", icon='ROTATE')
        if bpy.context.scene.dell_how_many_electrodes_above_threshold != '':
            layout.label(text='Found {} potential electrodes'.format(
                bpy.context.scene.dell_how_many_electrodes_above_threshold))
    else:
        # row = layout.row(align=0)
        # row.prop(context.scene, 'dell_ct_n_components', text="n_components")
        # row.prop(context.scene, 'dell_ct_n_groups', text="n_groups")
        row = layout.row(align=0)
        row.prop(context.scene, 'dell_ct_error_radius', text="Error radius")
        row.prop(context.scene, 'dell_ct_min_elcs_for_lead', text="Min for lead")
        row = layout.row(align=0)
        row.prop(context.scene, 'dell_ct_max_dist_between_electrodes', text="Max dist between")
        row.prop(context.scene, 'dell_ct_min_distance', text="Min dist between")
        layout.prop(context.scene, 'dell_debug', text='debug')
        row = layout.row(align=0)
        row.prop(context.scene, 'ct_mark_noise', text='Mark noise')
        row.operator(InitNoise.bl_idname, text="Init noise", icon='RECOVER_LAST')
        if bpy.context.scene.ct_mark_noise:
            row.prop(context.scene, 'dell_ct_noise_color', text='')
        row = layout.row(align=0)
        row.prop(context.scene, 'ct_plot_lead', text='Plot lead')
        if bpy.context.scene.ct_plot_lead:
            row.prop(context.scene, 'dell_ct_lead_color', text='')
        select_elc_group = -1
        if len(bpy.context.selected_objects) == 1 and bpy.context.selected_objects[0].name in DellPanel.names:
            name = bpy.context.selected_objects[0].name
            ind = DellPanel.names.index(name)
            groups = [k for k, g in enumerate(DellPanel.groups) if ind in g]
            if len(groups) == 1:
                select_elc_group = groups[0]
                group_str = 'group {}, {} in group'.format(select_elc_group , len(DellPanel.groups[groups[0]]))
            else:
                group_str = ''
            pos = DellPanel.pos[ind]
            layout.label(text='{} index: {} hemi: {} {}'.format(name, ind, DellPanel.hemis[ind], group_str))
            if select_elc_group >= 0:
                layout.prop(context.scene, "dell_selected_electrode_group_name", text="Group name")
            layout.label(text='pos: {}'.format(pos))
        # if len(bpy.context.selected_objects) == 1:
        #     layout.operator(OpenInteractiveCTViewer.bl_idname, text="Open interactive CT viewer", icon='LOGIC')
        # if len(DellPanel.dell_files) > 1:
        #     layout.prop(context.scene, 'dell_files', text='Dell files')
        row = layout.row(align=True)
        row.operator(SaveElectrodesObjects.bl_idname, text="Save", icon='SAVE_PREFS')
        row.operator(RefreshElectrodesObjects.bl_idname, text="Load", icon='OUTLINER_OB_FORCE_FIELD')
        # row.prop(context.scene, "dell_refresh_pos_and_names_overwrite", text='Overwrite')
        layout.operator(CreateNewElectrode.bl_idname, text="Create new electrode", icon='OUTLINER_OB_META')
        if len(bpy.context.selected_objects) == 1 and bpy.context.selected_objects[0].name in DellPanel.names:
            if not electrode_with_group_selected:
                layout.operator(FindElectrodeLead.bl_idname, text="Find selected electrode's lead", icon='PARTICLE_DATA')
            else:
                layout.operator(SaveCTElectrodesFigures.bl_idname, text="Save CT figures", icon='OUTLINER_OB_FORCE_FIELD')
            layout.operator(MarkElectrodeLeadAsNoise.bl_idname, text="Mark electrode's lead as noise",
                            icon='PANEL_CLOSE')
        layout.operator(ProjectElectrodesOnLeads.bl_idname, text="Project electrodes on leads", icon='SURFACE_DATA')
        if len(bpy.context.selected_objects) == 2 and all(bpy.context.selected_objects[k].name in DellPanel.names for k in range(2)):
            layout.operator(CreateNewElectrodeBetween.bl_idname, text="Create new electrode between",
                            icon='WORLD_DATA')
            layout.operator(FindElectrodeLead.bl_idname, text="Find lead between selected electrodes", icon='PARTICLE_DATA')
            layout.operator(FindGroupBetweenTwoElectrodesOnDural.bl_idname, text="Find group on dural", icon='EXPORT')
        if len(bpy.context.selected_objects) > 0:
            layout.operator(MarkSelectedElectrodesAsNoise.bl_idname, text="Mark selected electrodes as noise", icon='PANEL_CLOSE')
            layout.operator(MarkSelectedNoiseAsElectrodes.bl_idname, text="Mark selected noise as electrodes", icon='EYEDROPPER')
            layout.operator(CreateNewGroupFromSelectedElectrodes.bl_idname, text="Create New group from selection",
                            icon='EDIT')

        # if len(bpy.context.selected_objects) == 4 and all(
        #         bpy.context.selected_objects[k].name in DellPanel.names for k in range(4)):
        if len(bpy.context.selected_objects) == 0:
            layout.operator(FindGroupBetweenTwoElectrodesOnDural.bl_idname, text="Find grid on dural", icon='EXPORT')
        if not bpy.context.scene.dell_all_groups_were_found:
            layout.operator(FindRandomLead.bl_idname, text="Find a group", icon='POSE_HLT')
            layout.operator(FindAllLeads.bl_idname, text="Find all groups", icon='LAMP_SUN')
            layout.prop(context.scene, 'dell_do_post_search', text='Do post search')
            layout.prop(context.scene, 'dell_find_all_group_using_timer', text='Use timer')
        # layout.operator(SaveCTNeighborhood.bl_idname, text="Save CT neighborhood", icon='EDIT')
        layout.label(text='#Groups found: {}'.format(len(DellPanel.groups)))
        if len(DellPanel.groups) > 0:
            box = layout.box()
            col = box.column()
            for g in DellPanel.groups:
                mu.add_box_line(col, '{}-{}'.format(DellPanel.names[g[0]], DellPanel.names[g[-1]]), str(len(g)), 0.8)
        layout.operator(ClearGroups.bl_idname, text="Clear groups", icon='GHOST_DISABLED')
    if parent is not None and len(parent.children) > 0:
        layout.operator(MarkNonGroupElectrodesAsNoise.bl_idname, text="Mark non group electrodes as noise", icon='RADIO')
        layout.prop(context.scene, 'dell_delete_electrodes', text='Delete electrodes')
        if bpy.context.scene.dell_delete_electrodes:
            layout.operator(DeleteElectrodes.bl_idname, text="Delete electrodes", icon='CANCEL')
        layout.operator(ExportDellElectrodes.bl_idname, text="Export electrodes", icon='EXPORT')
        row = layout.row(align=True)
        row.prop(context.scene, 'dell_move_x')
        row.prop(context.scene, 'dell_move_y')
        row.prop(context.scene, 'dell_move_z')
    if len(bpy.context.selected_objects) > 1:
        layout.operator(DeleteElectrodesFromGroup.bl_idname, text="Leave highest CT int", icon='CANCEL')
    if electrode_with_group_selected:
        row = layout.row(align=0)
        row.operator(NextCTElectrode.bl_idname, text="", icon='PREV_KEYFRAME')
        row.operator(PrevCTElectrode.bl_idname, text="", icon='NEXT_KEYFRAME')
        row.label(text=bpy.context.selected_objects[0].name)
        if bpy.context.selected_objects[0].name in DellPanel.dists_to_cylinder:
            row.label(text='Dist to cylinder: {:.2f}'.format(DellPanel.dists_to_cylinder[bpy.context.selected_objects[0].name]))
    if len(DellPanel.current_log) > 0:
        layout.label(text='Selected Electrode and its group:')
        box = layout.box()
        col = box.column()
        elc, group = DellPanel.current_log[0]
        row = col.split(percentage=0.3, align=True)
        row.label(text=elc)
        row.label(text='{}-{}'.format(group[0], group[-1]))
        row.label(text=str(len(group)))
    if len(bpy.context.selected_objects) == 2 and all(
            bpy.context.selected_objects[k].name in DellPanel.names for k in range(2)):
        layout.label(text='Distance between {} and {}: {:.2f}'.format(
            bpy.context.selected_objects[0].name, bpy.context.selected_objects[1].name,
            np.linalg.norm(bpy.context.selected_objects[0].location - bpy.context.selected_objects[1].location) * 10))
    # row = layout.row(align=0)
    # row.operator(CheckIfElectrodeOutsidePial.bl_idname, text="Find outer electrodes", icon='ROTATE')
    # row.prop(context.scene, 'dell_brain_mask_sigma', text='Brain mask sigma')
    # row.prop(context.scene, 'dell_brain_mask_use_aseg', text='Use aseg')
    layout.prop(context.scene, 'dell_ct_print_distances', text='Show distances within group')
    if bpy.context.scene.dell_ct_print_distances and len(DellPanel.dists) > 0 and len(DellPanel.groups) > 0:
        layout.label(text='Group inner distances:')
        box = layout.box()
        col = box.column()
        last_group = DellPanel.groups[-1]
        for elc1, elc2, dist in zip([DellPanel.names[k] for k in last_group[:-1]], [DellPanel.names[k] for k in last_group[1:]], DellPanel.dists):
            mu.add_box_line(col, '{}-{}'.format(elc1, elc2), '{:.2f}'.format(dist), 0.8)


def start_timer():
    DellPanel.is_playing = True
    DellPanel.init_play = True
    if DellPanel.first_time:
        print('Starting the timer!')
        DellPanel.first_time = False
        bpy.ops.wm.modal_dell_timer_operator()


class ModalDellTimerOperator(bpy.types.Operator):
    """Operator which runs its self from a timer"""
    bl_idname = "wm.modal_dell_timer_operator"
    bl_label = "Modal Timer Operator"

    _timer = None
    _time = time.time()

    def modal(self, context, event):
        # First frame initialization:
        if DellPanel.init_play:
            # Do some timer init
            pass

        if not DellPanel.is_playing:
            return {'PASS_THROUGH'}

        if event.type in {'ESC'}:
            print('Stop!')
            self.cancel(context)
            return {'PASS_THROUGH'}

        if event.type == 'TIMER':
            if time.time() - self._time > DellPanel.play_time_step:
                self._time = time.time()
                try:
                    run_num = 0
                    group_found = find_random_group()
                    while not group_found and run_num < DellPanel.max_finding_group_tries:
                        group_found = find_random_group()
                        run_num += 1
                    if run_num == DellPanel.max_finding_group_tries:
                        self.cancel(context)
                except:
                    print(traceback.format_exc())
                    print('Error in plotting at {}!'.format(self.limits))

        return {'PASS_THROUGH'}

    def execute(self, context):
        wm = context.window_manager
        self.cancel(context)
        self._timer = wm.event_timer_add(time_step=0.05, window=context.window)
        self._time = time.time()
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        DellPanel.is_playing = False
        bpy.context.scene.update()
        if self._timer:
            try:
                wm = context.window_manager
                wm.event_timer_remove(self._timer)
            except:
                pass


class ChooseCTFile(bpy.types.Operator, bpy_extras.io_utils.ExportHelper):
    bl_idname = "mmvt.choose_ct_file"
    bl_label = "Choose CT file"

    filename_ext = '.mgz'
    filter_glob = bpy.props.StringProperty(default='*.mgz', options={'HIDDEN'}, maxlen=255)

    def execute(self, context):
        ct_fname = self.filepath
        ct_name = 'ct_reg_to_mr.mgz' # mu.namebase_with_ext(ct_fname)
        user_fol = mu.get_user_fol()
        ct_fol = mu.get_fname_folder(ct_fname)
        if ct_fol != op.join(user_fol, 'ct'):
            mu.make_dir(op.join(user_fol, 'ct'))
            shutil.copy(ct_fname, op.join(user_fol, 'ct', ct_name))
        run_ct_preproc()
        init(_addon(), ct_name)
        return {'FINISHED'}


class OpenInteractiveCTViewer(bpy.types.Operator):
    bl_idname = "mmvt.open_interactive_ct_viewer"
    bl_label = "open_interactive_ct_viewer"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        open_interactive_ct_viewer()
        return {'PASS_THROUGH'}


class SaveElectrodesObjects(bpy.types.Operator):
    bl_idname = "mmvt.save_pos_and_names"
    bl_label = "save_pos_and_names"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        save()
        return {'PASS_THROUGH'}


class RefreshElectrodesObjects(bpy.types.Operator):
    bl_idname = "mmvt.refresh_pos_and_names"
    bl_label = "refresh_pos_and_names"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        refresh_pos_and_names()#bpy.context.scene.dell_refresh_pos_and_names_overwrite)
        return {'PASS_THROUGH'}


class FindGroupBetweenTwoElectrodesOnDural(bpy.types.Operator):
    bl_idname = "mmvt.find_electrode_group_on_dural"
    bl_label = "find_electrode_group_on_dural"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        find_electrode_group_on_dural()
        return {'PASS_THROUGH'}


class CreateNewElectrodeBetween(bpy.types.Operator):
    bl_idname = "mmvt.create_new_electrode_between"
    bl_label = "create_new_electrode_between"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        create_new_electrode_between()
        return {'PASS_THROUGH'}


class InitNoise(bpy.types.Operator):
    bl_idname = "mmvt.dell_init_noise"
    bl_label = "dell_init_noise"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        init_noise()
        return {'PASS_THROUGH'}


class CreateNewElectrode(bpy.types.Operator):
    bl_idname = "mmvt.create_new_electrode"
    bl_label = "create_new_electrode"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        create_new_electrode()
        return {'PASS_THROUGH'}


class FindElectrodeLead(bpy.types.Operator):
    bl_idname = "mmvt.find_electrode_lead"
    bl_label = "find_electrode_lead"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        find_electrode_lead()
        return {'PASS_THROUGH'}


class FindRandomLead(bpy.types.Operator):
    bl_idname = "mmvt.find_random_lead"
    bl_label = "find_random_lead"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        find_random_group()
        return {'PASS_THROUGH'}


class FindAllLeads(bpy.types.Operator):
    bl_idname = "mmvt.find_all_leads"
    bl_label = "find_all_leads"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        find_all_groups()
        return {'PASS_THROUGH'}


class ProjectElectrodesOnLeads(bpy.types.Operator):
    bl_idname = "mmvt.project_electrodes_on_leads"
    bl_label = "project_electrodes_on_leads"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        project_electrodes_on_leads()
        return {'PASS_THROUGH'}


class SaveCTNeighborhood(bpy.types.Operator):
    bl_idname = "mmvt.save_ct_neighborhood"
    bl_label = "save_ct_neighborhood"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        save_ct_neighborhood()
        return {'PASS_THROUGH'}


class GetElectrodesAboveThrshold(bpy.types.Operator):
    bl_idname = "mmvt.find_electrodes_pipeline"
    bl_label = "find_electrodes_pipeline"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        find_electrodes_pipeline()
        return {'PASS_THROUGH'}


class ImportElectrodesFromDell(bpy.types.Operator):
    bl_idname = "mmvt.import_electrodes_from_dell"
    bl_label = "import_electrodes_from_dell"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        import_electrodes_from_dell()
        return {'PASS_THROUGH'}


class FindHowManyElectrodesAboveThrshold(bpy.types.Operator):
    bl_idname = "mmvt.find_how_many_electrodes_above_threshold"
    bl_label = "find_how_many_electrodes_above_threshold"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        find_how_many_electrodes_above_threshold()
        return {'PASS_THROUGH'}


class PrevCTElectrode(bpy.types.Operator):
    bl_idname = 'mmvt.prev_ct_electrode'
    bl_label = 'prev_ct_electrode'
    bl_options = {'UNDO'}

    def invoke(self, context, event=None):
        prev_ct_electrode()
        return {'FINISHED'}


class NextCTElectrode(bpy.types.Operator):
    bl_idname = 'mmvt.next_ct_electrode'
    bl_label = 'next_ct_electrode'
    bl_options = {'UNDO'}

    def invoke(self, context, event=None):
        next_ct_electrode()
        return {'FINISHED'}


class CheckIfElectrodeOutsidePial(bpy.types.Operator):
    bl_idname = 'mmvt.check_if_outside_pial'
    bl_label = 'check_if_outside_pial'
    bl_options = {'UNDO'}

    def invoke(self, context, event=None):
        check_if_outside_pial()
        return {'FINISHED'}



class SaveCTElectrodesFigures(bpy.types.Operator):
    bl_idname = 'mmvt.save_ct_electrodes_figures'
    bl_label = 'save_ct_electrodes_figures'
    bl_options = {'UNDO'}

    def invoke(self, context, event=None):
        save_ct_electrodes_figures()
        return {'FINISHED'}


class ExportDellElectrodes(bpy.types.Operator):
    bl_idname = "mmvt.export_dell_electrodes"
    bl_label = "export_dell_electrodes"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        export_electrodes()
        return {'PASS_THROUGH'}


class MarkNonGroupElectrodesAsNoise(bpy.types.Operator):
    bl_idname = "mmvt.mark_non_group_electrodes_as_noise"
    bl_label = "mark_non_group_electrodes_as_noise"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        mark_non_group_electrodes_as_noise()
        return {'PASS_THROUGH'}


class MarkSelectedNoiseAsElectrodes(bpy.types.Operator):
    bl_idname = "mmvt.mark_selected_noise_as_electrodes"
    bl_label = "mark_selected_noise_as_electrodes"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        mark_selected_noise_as_electrodes()
        return {'PASS_THROUGH'}


class MarkSelectedElectrodesAsNoise(bpy.types.Operator):
    bl_idname = "mmvt.mark_selected_electrodes_as_noise"
    bl_label = "mark_selected_electrodes_as_noise"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        mark_selected_electrodes_as_noise()
        return {'PASS_THROUGH'}


class MarkElectrodeLeadAsNoise(bpy.types.Operator):
    bl_idname = "mmvt.mark_electrode_lead_as_noise"
    bl_label = "mark_electrode_lead_as_noise"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        mark_electrode_lead_as_noise()
        return {'PASS_THROUGH'}


class CreateNewGroupFromSelectedElectrodes(bpy.types.Operator):
    bl_idname = "mmvt.create_new_group_from_selected_electrodes"
    bl_label = "create_new_group_from_selected_electrodes"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        create_new_group_from_selected_electrodes()
        return {'PASS_THROUGH'}


class DeleteElectrodes(bpy.types.Operator):
    bl_idname = "mmvt.delete_electrodes"
    bl_label = "delete_electrodes"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        delete_electrodes()
        return {'PASS_THROUGH'}


class DeleteElectrodesFromGroup(bpy.types.Operator):
    bl_idname = "mmvt.delete_electrodes_from_selection"
    bl_label = "delete_electrodes_from_selection"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        delete_electrodes_from_selection()
        return {'PASS_THROUGH'}


class InstallReqs(bpy.types.Operator):
    bl_idname = "mmvt.install_dell_reqs"
    bl_label = "install_dell_reqs"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        install_dell_reqs()
        return {'PASS_THROUGH'}


class CalcThresholdPercentile(bpy.types.Operator):
    bl_idname = "mmvt.calc_threshold_precentile"
    bl_label = "calc_threshold_precentile"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        calc_threshold_precentile()
        return {'PASS_THROUGH'}


class ClearGroups(bpy.types.Operator):
    bl_idname = "mmvt.clear_groups"
    bl_label = "clear_groups"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        clear_groups()
        return {'PASS_THROUGH'}


bpy.types.Scene.dell_ct_threshold = bpy.props.FloatProperty(default=0.5, min=0, description="")
bpy.types.Scene.dell_ct_threshold_percentile = bpy.props.FloatProperty(default=99.9, min=0, max=100, description="")
# bpy.types.Scene.dell_ct_n_components = bpy.props.IntProperty(min=0, description='')
# bpy.types.Scene.dell_ct_n_groups = bpy.props.IntProperty(min=0, description='', update=dell_ct_n_groups_update)
bpy.types.Scene.dell_ct_error_radius = bpy.props.FloatProperty(min=1, default=2)
bpy.types.Scene.dell_ct_min_elcs_for_lead = bpy.props.IntProperty(min=2, max=20, default=4)
bpy.types.Scene.dell_ct_max_dist_between_electrodes = bpy.props.FloatProperty(default=15, min=1, max=100)
bpy.types.Scene.dell_ct_min_distance = bpy.props.FloatProperty(default=3, min=0, max=100)
bpy.types.Scene.dell_ct_noise_color = bpy.props.FloatVectorProperty(
    name="object_color", subtype='COLOR', default=(0, 0.5, 0), min=0.0, max=1.0, description="color picker")
bpy.types.Scene.dell_ct_lead_color = bpy.props.FloatVectorProperty(
    name="object_color", subtype='COLOR', default=(0.5, 0.175, 0.02), min=0.0, max=1.0, description="color picker")
bpy.types.Scene.ct_mark_noise = bpy.props.BoolProperty(default=True, update=ct_mark_noise_update)
bpy.types.Scene.ct_plot_lead = bpy.props.BoolProperty(default=True, update=ct_plot_lead_update)
bpy.types.Scene.dell_ct_print_distances = bpy.props.BoolProperty(default=False)
bpy.types.Scene.dell_delete_electrodes = bpy.props.BoolProperty(default=False)
bpy.types.Scene.dell_find_all_group_using_timer = bpy.props.BoolProperty(default=False)
bpy.types.Scene.dell_do_post_search = bpy.props.BoolProperty(default=True)
bpy.types.Scene.dell_brain_mask_sigma = bpy.props.FloatProperty(min=0, max=5, default=1)
bpy.types.Scene.dell_brain_mask_use_aseg = bpy.props.BoolProperty(default=True)
# bpy.types.Scene.use_only_brain_mask = bpy.props.BoolProperty(default=False)
bpy.types.Scene.dell_binary_erosion = bpy.props.BoolProperty(default=True)
bpy.types.Scene.dell_find_nei_maxima = bpy.props.BoolProperty(default=True)
bpy.types.Scene.dell_refresh_pos_and_names_overwrite = bpy.props.BoolProperty(default=True)
bpy.types.Scene.dell_debug = bpy.props.BoolProperty(default=True)
bpy.types.Scene.dell_move_x = bpy.props.IntProperty(default=0, step=1, name='x', update=dell_move_elec_update)
bpy.types.Scene.dell_move_y = bpy.props.IntProperty(default=0, step=1, name='y', update=dell_move_elec_update)
bpy.types.Scene.dell_move_z = bpy.props.IntProperty(default=0, step=1, name='z', update=dell_move_elec_update)
bpy.types.Scene.dell_selected_electrode_group_name = bpy.props.StringProperty()
bpy.types.Scene.dell_how_many_electrodes_above_threshold = bpy.props.StringProperty()
bpy.types.Scene.dell_files = bpy.props.EnumProperty(items=[], description="Dell files")
bpy.types.Scene.dell_all_groups_were_found = bpy.props.BoolProperty(default=True)


class DellPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmodmde"
    bl_category = "mmvt"
    bl_label = "Dell"
    addon = None
    init = False
    ct_found = False
    ct = None
    brain = None
    output_fol = ''
    names, colors, groups, dists, dists_to_cylinder = [], [], [], [], []
    noise = set()
    init_play, is_playing, first_time = False, False, True
    play_time_step = 0.7
    max_finding_group_tries = 10
    log, current_log = [], []
    debug_fol = ''
    update_position = True

    def draw(self, context):
        if DellPanel.init:
            dell_draw(self, context)
            # dell_draw_presentation(self, context)


def init(addon, ct_name='ct_reg_to_mr.mgz', brain_mask_name='brain.mgz', aseg_name='aseg.mgz', debug=True):
    DellPanel.addon = addon
    try:
        if not DELL_EXIST:
            print("Can't find find_electrodes_in_ct!")
            DellPanel.init = False
            return
        ret = init_dural()
        if not ret:
            print('No dural surface, Can\'t init dell panel')
            DellPanel.init = False
            return
        DellPanel.output_fol = op.join(mu.get_user_fol(), 'ct', 'finding_electrodes_in_ct')
        DellPanel.ct_found = init_ct(ct_name, brain_mask_name, aseg_name)
        if DellPanel.ct_found:
            # init_electrodes()
            DellPanel.colors = cycle(mu.get_distinct_colors(10))
            # intput_files = glob.glob(op.join(DellPanel.output_fol, '*_electrodes.pkl'))
            # if len(intput_files) == 1:
            #     input_fname = intput_files[0]
            # else:
            #     # todo: let the user choose which one
            #     input_fname = op.join(
            #         DellPanel.output_fol, '{}_electrodes.pkl'.format(int(bpy.context.scene.dell_ct_threshold)))
            # # files = glob.glob(op.join(DellPanel.output_fol, '*_electrodes.pkl'))
            # # if len(files) > 0:
            # if op.isfile(input_fname):
            #     try:
            #         (DellPanel.pos, DellPanel.names, DellPanel.hemis, DellPanel.groups, DellPanel.noise,
            #          bpy.context.scene.dell_ct_threshold) = mu.load(input_fname)
            #         print('{} groups were loaded from {}'.format(len(DellPanel.groups), input_fname))
            #     except:
            #         # support old files
            #         (DellPanel.pos, DellPanel.names, DellPanel.hemis,
            #          bpy.context.scene.dell_ct_threshold) = mu.load(input_fname)
                # (DellPanel.pos, DellPanel.names, DellPanel.hemis, bpy.context.scene.dell_ct_threshold) = mu.load(files[0])
            parent = bpy.data.objects.get('Deep_electrodes')
            if parent is None:
                parent = _addon().create_empty_if_doesnt_exists(
                    'Deep_electrodes', _addon().BRAIN_EMPTY_LAYER, [False] * 20, 'Deep_electrodes')
            loaded = load_electrodes_from_file()
            if not loaded:
                bpy.context.scene.dell_ct_threshold_percentile = 99.9
                bpy.context.scene.dell_ct_threshold = np.percentile(
                    DellPanel.ct_data, bpy.context.scene.dell_ct_threshold_percentile)
            # init_groups()
        bpy.context.scene.dell_ct_error_radius = 2.5
        bpy.context.scene.dell_ct_min_elcs_for_lead = 6
        bpy.context.scene.dell_ct_max_dist_between_electrodes = 15
        bpy.context.scene.dell_ct_min_distance = 1.5
        bpy.context.scene.dell_brain_mask_sigma = 1
        bpy.context.scene.dell_delete_electrodes = False
        bpy.context.scene.dell_find_all_group_using_timer = False
        bpy.context.scene.dell_do_post_search = True
        # bpy.context.scene.use_only_brain_mask = False
        bpy.context.scene.dell_binary_erosion = False
        bpy.context.scene.dell_debug = False
        bpy.context.scene.dell_how_many_electrodes_above_threshold = ''
        bpy.context.scene.dell_all_groups_were_found = False
        if bpy.context.scene.dell_debug:
            DellPanel.debug_fol = mu.make_dir(op.join(DellPanel.output_fol, mu.rand_letters(5)))
        if not DellPanel.init:
            DellPanel.init = True
            register()
    except:
        print(traceback.format_exc())
        DellPanel.init = False


def init_ct(ct_name='ct_reg_to_mr.mgz', brain_mask_name='brain.mgz', aseg_name='aseg.mgz'):
    user_fol = mu.get_user_fol()
    mu.make_dir(op.join(user_fol, 'ct', 'finding_electrodes_in_ct'))
    ct_fname = op.join(user_fol, 'ct', ct_name)
    if op.isfile(ct_fname):
        print('Dell panel: loading {}'.format(ct_fname))
        DellPanel.ct = nib.load(ct_fname)
        DellPanel.ct_data = DellPanel.ct.get_data()
    else:
        print("Dell: Can't find the ct!")
        # return False
    # subjects_dir = su.get_subjects_dir()
    mu.make_dir(op.join(mu.get_user_fol(), 'mri'))
    brain_mask_fname = op.join(mu.get_user_fol(), 'mri', brain_mask_name)
    if not op.isfile(brain_mask_fname):
        brain_mask_fname_subjects = op.join(mu.get_subjects_dir(), mu.get_user(), 'mri', brain_mask_name)
        if op.isfile(brain_mask_fname_subjects):
            shutil.copyfile(brain_mask_fname_subjects, brain_mask_fname)
    if op.isfile(brain_mask_fname):
        DellPanel.brain = nib.load(brain_mask_fname)
        DellPanel.brain_mask = DellPanel.brain.get_data()
    else:
        print("Dell: Can't find brain.mgz!")
        # return False
    aseg_fname = op.join(mu.get_user_fol(), 'mri', aseg_name)
    DellPanel.aseg = nib.load(aseg_fname).get_data() if op.isfile(aseg_fname) else None
    return True


def init_dural():
    try:
        user_fol = mu.get_user_fol()
        verts_dural_neighbors_fname = op.join(user_fol, 'verts_neighbors_dural_{hemi}.pkl')
        DellPanel.verts_dural_nei = {hemi:mu.load(verts_dural_neighbors_fname.format(hemi=hemi)) \
            if op.isfile(verts_dural_neighbors_fname.format(hemi=hemi)) else None for hemi in mu.HEMIS}
        DellPanel.verts_dural, DellPanel.faces_dural = fect.read_surf_verts(user_fol, 'dural', True)
        if DellPanel.verts_dural['rh'] is None or DellPanel.faces_dural['rh'] is None:
            return False
        DellPanel.normals_dural = {hemi:fect.calc_normals(DellPanel.verts_dural[hemi], DellPanel.faces_dural[hemi])
                                   for hemi in mu.HEMIS}
        return True
    except:
        print(traceback.format_exc())
        return False


def register():
    try:
        unregister()
        bpy.utils.register_class(DellPanel)
        bpy.utils.register_class(InstallReqs)
        bpy.utils.register_class(ChooseCTFile)
        bpy.utils.register_class(CalcThresholdPercentile)
        bpy.utils.register_class(GetElectrodesAboveThrshold)
        bpy.utils.register_class(FindHowManyElectrodesAboveThrshold)
        bpy.utils.register_class(ImportElectrodesFromDell)
        bpy.utils.register_class(OpenInteractiveCTViewer)
        bpy.utils.register_class(FindElectrodeLead)
        bpy.utils.register_class(FindRandomLead)
        bpy.utils.register_class(FindAllLeads)
        bpy.utils.register_class(ProjectElectrodesOnLeads)
        bpy.utils.register_class(SaveCTNeighborhood)
        bpy.utils.register_class(PrevCTElectrode)
        bpy.utils.register_class(NextCTElectrode)
        bpy.utils.register_class(SaveCTElectrodesFigures)
        bpy.utils.register_class(ClearGroups)
        bpy.utils.register_class(CheckIfElectrodeOutsidePial)
        bpy.utils.register_class(DeleteElectrodes)
        bpy.utils.register_class(CreateNewGroupFromSelectedElectrodes)
        bpy.utils.register_class(MarkSelectedElectrodesAsNoise)
        bpy.utils.register_class(MarkSelectedNoiseAsElectrodes)
        bpy.utils.register_class(MarkNonGroupElectrodesAsNoise)
        bpy.utils.register_class(MarkElectrodeLeadAsNoise)
        bpy.utils.register_class(DeleteElectrodesFromGroup)
        bpy.utils.register_class(ExportDellElectrodes)
        bpy.utils.register_class(CreateNewElectrodeBetween)
        bpy.utils.register_class(InitNoise)
        bpy.utils.register_class(CreateNewElectrode)
        bpy.utils.register_class(FindGroupBetweenTwoElectrodesOnDural)
        bpy.utils.register_class(SaveElectrodesObjects)
        bpy.utils.register_class(RefreshElectrodesObjects)
        bpy.utils.register_class(ModalDellTimerOperator)
        bpy.utils.register_class(DellGroupItem)
    except:
        print("Can't register Dell Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(DellPanel)
        bpy.utils.unregister_class(InstallReqs)
        bpy.utils.unregister_class(ChooseCTFile)
        bpy.utils.unregister_class(CalcThresholdPercentile)
        bpy.utils.unregister_class(GetElectrodesAboveThrshold)
        bpy.utils.unregister_class(FindHowManyElectrodesAboveThrshold)
        bpy.utils.unregister_class(OpenInteractiveCTViewer)
        bpy.utils.unregister_class(FindElectrodeLead)
        bpy.utils.unregister_class(FindRandomLead)
        bpy.utils.unregister_class(FindAllLeads)
        bpy.utils.unregister_class(SaveCTNeighborhood)
        bpy.utils.unregister_class(PrevCTElectrode)
        bpy.utils.unregister_class(NextCTElectrode)
        bpy.utils.unregister_class(CreateNewGroupFromSelectedElectrodes)
        bpy.utils.unregister_class(SaveCTElectrodesFigures)
        bpy.utils.unregister_class(ClearGroups)
        bpy.utils.unregister_class(CheckIfElectrodeOutsidePial)
        bpy.utils.unregister_class(DeleteElectrodes)
        bpy.utils.unregister_class(MarkSelectedElectrodesAsNoise)
        bpy.utils.unregister_class(MarkNonGroupElectrodesAsNoise)
        bpy.utils.unregister_class(MarkElectrodeLeadAsNoise)
        bpy.utils.unregister_class(MarkSelectedNoiseAsElectrodes)
        bpy.utils.unregister_class(DeleteElectrodesFromGroup)
        bpy.utils.unregister_class(ExportDellElectrodes)
        bpy.utils.unregister_class(CreateNewElectrodeBetween)
        bpy.utils.unregister_class(CreateNewElectrode)
        bpy.utils.unregister_class(InitNoise)
        bpy.utils.unregister_class(FindGroupBetweenTwoElectrodesOnDural)
        bpy.utils.unregister_class(SaveElectrodesObjects)
        bpy.utils.unregister_class(RefreshElectrodesObjects)
        bpy.utils.unregister_class(ModalDellTimerOperator)
        bpy.utils.unregister_class(DellGroupItem)
        bpy.utils.unregister_class(ProjectElectrodesOnLeads)
    except:
        pass


