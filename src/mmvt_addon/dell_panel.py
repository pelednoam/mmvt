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
    # print(traceback.format_exc())
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


@mu.profileit('cumtime', op.join(mu.get_user_fol()))
def find_electrodes_pipeline():
    user_fol = mu.get_user_fol()
    subject_fol = op.join(mu.get_subjects_dir(), mu.get_user())
    local_maxima_fname = op.join(DellPanel.output_fol, 'local_maxima.npy')
    if bpy.context.scene.dell_binary_erosion:
        DellPanel.ct_data = fect.binary_erosion(DellPanel.ct_data, bpy.context.scene.dell_ct_threshold)
    if True: #not op.isfile(local_maxima_fname):
        print('find_voxels_above_threshold...')
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
        ct_voxels, DellPanel.ct.header, DellPanel.brain, subject_fol, bpy.context.scene.dell_brain_mask_sigma,
        bpy.context.scene.use_only_brain_mask)
    print('{} voxels in the brain were found'.format(len(ct_electrodes)))
    DellPanel.pos = fect.ct_voxels_to_t1_ras_tkr(ct_electrodes, DellPanel.ct.header, DellPanel.brain.header)
    print('find_electrodes_hemis...')
    # DellPanel.hemis, _ = fect.find_electrodes_hemis(user_fol, DellPanel.pos)
    DellPanel.hemis = fect.find_electrodes_hemis(
        subject_fol, DellPanel.pos, groups=None, sigma=bpy.context.scene.dell_brain_mask_sigma)
    DellPanel.names = name_electrodes(DellPanel.hemis)
    mu.save((DellPanel.pos, DellPanel.names, DellPanel.hemis, bpy.context.scene.dell_ct_threshold),
            op.join(DellPanel.output_fol, '{}_electrodes.pkl'.format(int(bpy.context.scene.dell_ct_threshold))))
    print('import_electrodes...')
    _addon().import_electrodes(elecs_pos=DellPanel.pos, elecs_names=DellPanel.names, bipolar=False,
                               parnet_name='Deep_electrodes')
    _addon().show_electrodes()


def refresh_pos_and_names():
    objects = bpy.data.objects['Deep_electrodes'].children
    DellPanel.names = [o.name for o in objects]
    DellPanel.pos = np.array([np.array(o.location) for o in objects]) * 10
    DellPanel.groups = []
    subject_fol = op.join(mu.get_subjects_dir(), mu.get_user())
    DellPanel.hemis = fect.find_electrodes_hemis(
        subject_fol, DellPanel.pos, None, 1, DellPanel.verts_dural, DellPanel.normals_dural)
    DellPanel.names = name_electrodes(DellPanel.hemis)
    for new_name, obj in zip(DellPanel.names, objects):
        obj.name = new_name
    for obj in objects:
        obj.name = obj.name.replace('.001', '')
    mu.save((DellPanel.pos, DellPanel.names, DellPanel.hemis, bpy.context.scene.dell_ct_threshold),
            op.join(DellPanel.output_fol, '{}_electrodes.pkl'.format(int(bpy.context.scene.dell_ct_threshold))))


def export_electrodes(group_hemi_default='G'):
    from collections import Counter
    output_fol = op.join(mu.get_user_fol(), 'electrodes')
    subject = mu.get_user()
    if len(DellPanel.groups) > 0:
        groups = DellPanel.groups
    else:
        groups = [[k for k, o in enumerate(_addon().electrodes_panel_parent_obj.children)]]
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
            elcs_names = ['{}{}'.format(group_name, k+1) for k in range(len(group))]
            for ind, elc_ind in enumerate(group):
                wr.writerow([elcs_names[ind], *['{:.2f}'.format(loc) for loc in DellPanel.pos[elc_ind]]])
            if group_hemi in groups_inds:
                groups_inds[group_hemi] += 1
    print('The electrodes were exported to {}'.format(csv_fname))


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
    DellPanel.pos = np.vstack((DellPanel.pos, new_pos))
    DellPanel.hemis.append(hemi)
    DellPanel.groups[group_ind].append(new_num)
    _addon().create_electrode(new_pos, new_name)
    _addon().object_coloring(bpy.data.objects[new_name], tuple(color))


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

    if debug:
        if DellPanel.debug_fol == '':
            DellPanel.debug_fol = mu.make_dir(op.join(DellPanel.output_fol, mu.rand_letters(5)))
        mu.save((elc_ind, DellPanel.pos, DellPanel.hemis, DellPanel.groups, bpy.context.scene.dell_ct_error_radius,
            bpy.context.scene.dell_ct_min_elcs_for_lead, bpy.context.scene.dell_ct_max_dist_between_electrodes,
            bpy.context.scene.dell_ct_min_distance, bpy.context.scene.dell_do_post_search), op.join(
            DellPanel.debug_fol, '_find_electrode_lead_{}-{}_{}_{}.pkl'.format(
                group[0], group[-1], elc_ind, int(bpy.context.scene.dell_ct_threshold))))

    DellPanel.dists_to_cylinder = {DellPanel.names[p]:d for p, d in dists_to_cylinder.items()}
    log = (DellPanel.names[best_elc_ind], [DellPanel.names[ind] for ind in group])
    DellPanel.log.append(log)
    DellPanel.current_log = [log]
    mu.save(DellPanel.log, op.join(DellPanel.output_fol, '{}_log.pkl'.format(
        int(bpy.context.scene.dell_ct_threshold))))
    if bpy.context.scene.ct_plot_lead:
        create_lead(group[0], group[-1])
    DellPanel.groups.append(group)
    for p in noise:
        # print('Marking {} as noise for group {}-{}'.format(DellPanel.names[p], DellPanel.names[group[0]], DellPanel.names[group[-1]]))
        _addon().object_coloring(bpy.data.objects[DellPanel.names[p]], tuple(bpy.context.scene.dell_ct_noise_color))
        DellPanel.noise.add(p)
    mu.save((DellPanel.groups, DellPanel.noise), op.join(DellPanel.output_fol, '{}_groups.pkl'.format(
        int(bpy.context.scene.dell_ct_threshold))))
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
            log = (DellPanel.names[elc_ind], [DellPanel.names[ind] for ind in group])
            DellPanel.log.append(log)

    if log is not None:
        DellPanel.current_log = [log]
        mu.save(DellPanel.log, op.join(DellPanel.output_fol, '{}_log.pkl'.format(
            int(bpy.context.scene.dell_ct_threshold))))
    print('find_random_group: exit afer {} runs'.format(run_num))
    return len(group) > 0


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


def find_select_electrode_group():
    in_group_ind, group = -1, []
    if len(bpy.context.selected_objects) == 1 and bpy.context.selected_objects[0].name in DellPanel.names:
        selected_elc = bpy.context.selected_objects[0].name
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
    DellPanel.log = []
    DellPanel.is_playing = False


def dell_ct_electrode_was_selected(elc_name):
    if not DellPanel.init:
        return
    group, in_group_ind = find_select_electrode_group()
    group = [DellPanel.names[g] for g in group]
    DellPanel.current_log = [(elc, g) for (elc, g) in DellPanel.log if set(g) == set(group)]
    DellPanel.update_position = False
    # tkreg_ras = bpy.data.objects[elc_name].location * 10
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
    aseg = None if not bpy.context.scene.dell_brain_mask_use_aseg else DellPanel.aseg
    subject_fol = op.join(mu.get_subjects_dir(), mu.get_user())
    voxels = fect.t1_ras_tkr_to_ct_voxels(DellPanel.pos, DellPanel.ct.header, DellPanel.brain.header)
    voxels_in, voxels_in_indices = fect.mask_voxels_outside_brain(
        voxels, DellPanel.ct.header, DellPanel.brain, subject_fol,
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
        row.prop(context.scene, 'dell_ct_threshold', text="Threshold")
        row.prop(context.scene, 'dell_ct_threshold_percentile', text='Percentile')
        row.operator(CalcThresholdPercentile.bl_idname, text="Calc threshold", icon='STRANDS')
        layout.prop(context.scene, 'dell_brain_mask_sigma', text='Brain mask sigma')
        layout.prop(context.scene, 'dell_find_nei_maxima', text='Find local nei maxima')
        layout.prop(context.scene, 'use_only_brain_mask', text='Use only the brain mask')
        layout.prop(context.scene, 'dell_binary_erosion', text='USe Binary Erosion')
        layout.operator(GetElectrodesAboveThrshold.bl_idname, text="Find electrodes", icon='ROTATE')
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
        if bpy.context.scene.ct_mark_noise:
            row.prop(context.scene, 'dell_ct_noise_color', text='')
        row = layout.row(align=0)
        row.prop(context.scene, 'ct_plot_lead', text='Plot lead')
        if bpy.context.scene.ct_plot_lead:
            row.prop(context.scene, 'dell_ct_lead_color', text='')
        if len(bpy.context.selected_objects) == 1 and bpy.context.selected_objects[0].name in DellPanel.names:
            name = bpy.context.selected_objects[0].name
            ind = DellPanel.names.index(name)
            layout.label(text='{} index: {} hemi: {}'.format(name, ind, DellPanel.hemis[ind]))
        layout.label(text='#Groups found: {}'.format(len(DellPanel.groups)))
        if len(DellPanel.groups) > 0:
            box = layout.box()
            col = box.column()
            for g in DellPanel.groups:
                mu.add_box_line(col, '{}-{}'.format(DellPanel.names[g[0]], DellPanel.names[g[-1]]), str(len(g)), 0.8)
        # if len(bpy.context.selected_objects) == 1:
        #     layout.operator(OpenInteractiveCTViewer.bl_idname, text="Open interactive CT viewer", icon='LOGIC')
        layout.operator(RefreshElectrodesObjects.bl_idname, text="Refresh", icon='OUTLINER_OB_FORCE_FIELD')
        if len(bpy.context.selected_objects) == 1 and bpy.context.selected_objects[0].name in DellPanel.names:
            if not electrode_with_group_selected:
                layout.operator(FindElectrodeLead.bl_idname, text="Find selected electrode's lead", icon='PARTICLE_DATA')
            else:
                layout.operator(SaveCTElectrodesFigures.bl_idname, text="Save CT figures", icon='OUTLINER_OB_FORCE_FIELD')
        if len(bpy.context.selected_objects) == 2 and all(bpy.context.selected_objects[k].name in DellPanel.names for k in range(2)):
            layout.operator(CreateNewElectrodeBetween.bl_idname, text="Create new electrode between",
                            icon='WORLD_DATA')
            layout.operator(FindElectrodeLead.bl_idname, text="Find lead between selected electrodes", icon='PARTICLE_DATA')
            layout.operator(FindGroupBetweenTwoElectrodesOnDural.bl_idname, text="Find group on dural", icon='EXPORT')
        # if len(bpy.context.selected_objects) == 4 and all(
        #         bpy.context.selected_objects[k].name in DellPanel.names for k in range(4)):
        if len(bpy.context.selected_objects) == 0:
            layout.operator(FindGroupBetweenTwoElectrodesOnDural.bl_idname, text="Find grid on dural", icon='EXPORT')
        layout.operator(FindRandomLead.bl_idname, text="Find a group", icon='POSE_HLT')
        layout.operator(FindAllLeads.bl_idname, text="Find all groups", icon='LAMP_SUN')
        layout.prop(context.scene, 'dell_do_post_search', text='Do post search')
        layout.prop(context.scene, 'dell_find_all_group_using_timer', text='Use timer')
        # layout.operator(SaveCTNeighborhood.bl_idname, text="Save CT neighborhood", icon='EDIT')
        layout.operator(ClearGroups.bl_idname, text="Clear groups", icon='GHOST_DISABLED')
    if parent is not None and len(parent.children) > 0:
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


class RefreshElectrodesObjects(bpy.types.Operator):
    bl_idname = "mmvt.refresh_pos_and_names"
    bl_label = "refresh_pos_and_names"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        refresh_pos_and_names()
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
bpy.types.Scene.dell_do_post_search = bpy.props.BoolProperty(default=False)
bpy.types.Scene.dell_brain_mask_sigma = bpy.props.FloatProperty(min=0, max=5, default=1)
bpy.types.Scene.dell_brain_mask_use_aseg = bpy.props.BoolProperty(default=True)
bpy.types.Scene.use_only_brain_mask = bpy.props.BoolProperty(default=False)
bpy.types.Scene.dell_binary_erosion = bpy.props.BoolProperty(default=True)
bpy.types.Scene.dell_find_nei_maxima = bpy.props.BoolProperty(default=True)
bpy.types.Scene.dell_debug = bpy.props.BoolProperty(default=True)
bpy.types.Scene.dell_move_x = bpy.props.IntProperty(default=0, step=1, name='x', update=dell_move_elec_update)
bpy.types.Scene.dell_move_y = bpy.props.IntProperty(default=0, step=1, name='y', update=dell_move_elec_update)
bpy.types.Scene.dell_move_z = bpy.props.IntProperty(default=0, step=1, name='z', update=dell_move_elec_update)


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


def init(addon, ct_name='ct_reg_to_mr.mgz', brain_mask_name='brain.mgz', aseg_name='aseg.mgz', debug=True):
    DellPanel.addon = addon
    try:
        if not DELL_EXIST:
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
            init_electrodes()
            DellPanel.colors = cycle(mu.get_distinct_colors(10))
            files = glob.glob(op.join(DellPanel.output_fol, '*_electrodes.pkl'))
            if len(files) > 0:
                (DellPanel.pos, DellPanel.names, DellPanel.hemis, bpy.context.scene.dell_ct_threshold) = mu.load(files[0])
                parent = bpy.data.objects.get('Deep_electrodes')
                if parent is None:
                    parent = _addon().create_empty_if_doesnt_exists(
                        'Deep_electrodes', _addon().BRAIN_EMPTY_LAYER, [False] * 20, 'Deep_electrodes')
                elcs_names = [o.name for o in parent.children]
                if len(DellPanel.names) != len(parent.children) or set(DellPanel.names) != set(elcs_names):
                    DellPanel.names = elcs_names
            else:
                bpy.context.scene.dell_ct_threshold_percentile = 99.9
            bpy.context.scene.dell_ct_threshold = np.percentile(
                DellPanel.ct_data, bpy.context.scene.dell_ct_threshold_percentile)
            init_groups()
        bpy.context.scene.dell_ct_error_radius = 2
        bpy.context.scene.dell_ct_min_elcs_for_lead = 4
        bpy.context.scene.dell_ct_max_dist_between_electrodes = 15
        bpy.context.scene.dell_ct_min_distance = 2.5
        bpy.context.scene.dell_brain_mask_sigma = 1
        bpy.context.scene.dell_delete_electrodes = False
        bpy.context.scene.dell_find_all_group_using_timer = False
        bpy.context.scene.dell_do_post_search = False
        bpy.context.scene.use_only_brain_mask = False
        bpy.context.scene.dell_binary_erosion = False
        bpy.context.scene.dell_debug = False
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
    if not op.isfile(ct_fname):
        print("Dell: Can't find the ct!")
        return False
    subjects_dir = su.get_subjects_dir()
    brain_mask_fname = op.join(subjects_dir, mu.get_user(), 'mri', brain_mask_name)
    if not op.isfile(brain_mask_fname):
        print("Dell: Can't find brain.mgz!")
        return False
    print('Dell panel: loading {}'.format(ct_fname))
    DellPanel.ct = nib.load(ct_fname)
    DellPanel.ct_data = DellPanel.ct.get_data()
    DellPanel.brain = nib.load(brain_mask_fname)
    DellPanel.brain_mask = DellPanel.brain.get_data()
    aseg_fname = op.join(subjects_dir, mu.get_user(), 'mri', aseg_name)
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


def init_electrodes():
    elcs_files = glob.glob(op.join(mu.get_user_fol(), 'electrodes', '*electrodes_positions.npz'))
    if len(elcs_files) == 1:
        elcs_dict = mu.Bag(np.load(elcs_files[0]))
        bipolar = '-' in elcs_dict.names[0]
        groups = set([mu.elec_group(elc_name, bipolar) for elc_name in elcs_dict.names])
        # bpy.context.scene.dell_ct_n_components = len(elcs_dict.names)
        # bpy.context.scene.dell_ct_n_groups = len(groups)


@mu.tryit()
def init_groups():
    groups_fname = op.join(DellPanel.output_fol, '{}_groups.pkl'.format(
        int(bpy.context.scene.dell_ct_threshold)))
    DellPanel.groups, DellPanel.noise = mu.load(groups_fname) if op.isfile(groups_fname) else ([], set())
    DellPanel.groups = [list(l) for l in DellPanel.groups]
    parent = bpy.data.objects.get('Deep_electrodes', None)
    if parent is None or len(parent.children) == 0:
        return
    for ind, group in enumerate(DellPanel.groups):
        color = next(DellPanel.colors)
        for elc_ind in group:
            _addon().object_coloring(bpy.data.objects[DellPanel.names[elc_ind]], tuple(color))
    for p in DellPanel.noise:
        _addon().object_coloring(bpy.data.objects[DellPanel.names[p]], tuple(bpy.context.scene.dell_ct_noise_color))
    if len(DellPanel.groups) == 0:
        clear_electrodes_color()
    log_fname = op.join(DellPanel.output_fol, '{}_log.pkl'.format(
        int(bpy.context.scene.dell_ct_threshold)))
    if op.isfile(log_fname):
        DellPanel.log = mu.load(log_fname)
    DellPanel.current_log = []


def register():
    try:
        unregister()
        bpy.utils.register_class(DellPanel)
        bpy.utils.register_class(InstallReqs)
        bpy.utils.register_class(ChooseCTFile)
        bpy.utils.register_class(CalcThresholdPercentile)
        bpy.utils.register_class(GetElectrodesAboveThrshold)
        bpy.utils.register_class(OpenInteractiveCTViewer)
        bpy.utils.register_class(FindElectrodeLead)
        bpy.utils.register_class(FindRandomLead)
        bpy.utils.register_class(FindAllLeads)
        bpy.utils.register_class(SaveCTNeighborhood)
        bpy.utils.register_class(PrevCTElectrode)
        bpy.utils.register_class(NextCTElectrode)
        bpy.utils.register_class(SaveCTElectrodesFigures)
        bpy.utils.register_class(ClearGroups)
        bpy.utils.register_class(CheckIfElectrodeOutsidePial)
        bpy.utils.register_class(DeleteElectrodes)
        bpy.utils.register_class(DeleteElectrodesFromGroup)
        bpy.utils.register_class(ExportDellElectrodes)
        bpy.utils.register_class(CreateNewElectrodeBetween)
        bpy.utils.register_class(FindGroupBetweenTwoElectrodesOnDural)
        bpy.utils.register_class(RefreshElectrodesObjects)
        bpy.utils.register_class(ModalDellTimerOperator)
    except:
        print("Can't register Dell Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(DellPanel)
        bpy.utils.unregister_class(InstallReqs)
        bpy.utils.unregister_class(ChooseCTFile)
        bpy.utils.unregister_class(CalcThresholdPercentile)
        bpy.utils.unregister_class(GetElectrodesAboveThrshold)
        bpy.utils.unregister_class(OpenInteractiveCTViewer)
        bpy.utils.unregister_class(FindElectrodeLead)
        bpy.utils.unregister_class(FindRandomLead)
        bpy.utils.unregister_class(FindAllLeads)
        bpy.utils.unregister_class(SaveCTNeighborhood)
        bpy.utils.unregister_class(PrevCTElectrode)
        bpy.utils.unregister_class(NextCTElectrode)
        bpy.utils.unregister_class(SaveCTElectrodesFigures)
        bpy.utils.unregister_class(ClearGroups)
        bpy.utils.unregister_class(CheckIfElectrodeOutsidePial)
        bpy.utils.unregister_class(DeleteElectrodes)
        bpy.utils.unregister_class(DeleteElectrodesFromGroup)
        bpy.utils.unregister_class(ExportDellElectrodes)
        bpy.utils.unregister_class(CreateNewElectrodeBetween)
        bpy.utils.unregister_class(FindGroupBetweenTwoElectrodesOnDural)
        bpy.utils.unregister_class(RefreshElectrodesObjects)
        bpy.utils.unregister_class(ModalDellTimerOperator)
    except:
        pass


