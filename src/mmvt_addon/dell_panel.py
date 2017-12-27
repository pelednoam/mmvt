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

import mmvt_utils as mu
from scripts import scripts_utils as su

try:
    from dell import find_electrodes_in_ct as fect
    importlib.reload(fect)
    DELL_EXIST = True
except:
    DELL_EXIST = False

try:
    import nibabel as nib
    NIBABEL_EXIST = True
except:
    NIBABEL_EXIST = False


def _addon():
    return DellPanel.addon


def dell_ct_n_groups_update(self, context):
    if bpy.context.scene.dell_ct_n_groups > 0:
        DellPanel.colors = mu.get_distinct_colors(bpy.context.scene.dell_ct_n_groups)


def get_electrodes_above_threshold():
    user_fol = mu.get_user_fol()
    print('find_voxels_above_threshold...')
    ct_voxels = fect.find_voxels_above_threshold(DellPanel.ct_data, bpy.context.scene.dell_ct_threshold)
    print('mask_voxels_outside_brain...')
    ct_voxels = fect.mask_voxels_outside_brain(ct_voxels, DellPanel.ct.header, DellPanel.brain, DellPanel.aseg)
    print('Finding local maxima')
    ct_electrodes = fect.find_all_local_maxima(DellPanel.ct_data, ct_voxels, bpy.context.scene.dell_ct_threshold, max_iters=100)
    # print('clustering...')
    # ct_electrodes, _ = fect.clustering(
    #     ct_voxels, DellPanel.ct_data, bpy.context.scene.dell_ct_n_components, True, 'knn', #DellPanel.output_fol,
    #     threshold=bpy.context.scene.dell_ct_threshold)
    DellPanel.pos = fect.ct_voxels_to_t1_ras_tkr(ct_electrodes, DellPanel.ct.header, DellPanel.brain.header)
    print('find_electrodes_hemis...')
    DellPanel.hemis, _ = fect.find_electrodes_hemis(user_fol, DellPanel.pos)
    DellPanel.names = name_electrodes(DellPanel.hemis)
    mu.save((DellPanel.pos, DellPanel.names, DellPanel.hemis, bpy.context.scene.dell_ct_threshold),
            op.join(DellPanel.output_fol, '{}_electrodes.pkl'.format(int(bpy.context.scene.dell_ct_threshold))))
    print('import_electrodes...')
    _addon().import_electrodes(elecs_pos=DellPanel.pos, elecs_names=DellPanel.names, bipolar=False,
                               parnet_name='Deep_electrodes')
    _addon().show_electrodes()


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


def _find_electrode_lead(elc_ind, elc_ind2=-1):
    if elc_ind2 == -1:
        group, noise, DellPanel.dists = fect.find_electrode_group(
            elc_ind, DellPanel.pos, DellPanel.hemis, DellPanel.groups, bpy.context.scene.dell_ct_error_radius,
            bpy.context.scene.dell_ct_min_elcs_for_lead, bpy.context.scene.dell_ct_max_dist_between_electrodes,
            bpy.context.scene.dell_ct_min_distance)
    else:
        group, noise, DellPanel.dists = fect.find_group_between_pair(
            elc_ind, elc_ind2, DellPanel.pos, bpy.context.scene.dell_ct_error_radius,
            bpy.context.scene.dell_ct_min_distance)
    if len(group) == 0:
        print('No group was found for {}!'.format(DellPanel.names[elc_ind]))
        DellPanel.noise.add(elc_ind)
        return []
    DellPanel.groups.append(group)
    for p in noise:
        print('Marking {} as noise'.format(DellPanel.names[p]))
        _addon().object_coloring(bpy.data.objects[DellPanel.names[p]], tuple(bpy.context.scene.ct_noise_color))
        DellPanel.noise.add(p)
    mu.save(DellPanel.groups, op.join(DellPanel.output_fol, '{}_groups.pkl'.format(
        int(bpy.context.scene.dell_ct_threshold))))
    color = DellPanel.colors[len(DellPanel.groups) - 1]
    for elc_ind in group:
        _addon().object_coloring(bpy.data.objects[DellPanel.names[elc_ind]], tuple(color))
    return group


def find_random_group():
    elcs = list(set(range(len(DellPanel.names))) - set(mu.flat_list_of_lists(DellPanel.groups)) - DellPanel.noise)
    group, run_num = [], 0
    while len(elcs) > 0 and len(group) == 0 and run_num < 10:
        elc_ind = random.choice(elcs)
        group = _find_electrode_lead(elc_ind)
        if len(group) == 0:
            elcs = list(set(range(len(DellPanel.names))) - set(mu.flat_list_of_lists(DellPanel.groups)) - DellPanel.noise)
        run_num += 1


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
    elcs_nums = {'rh':1, 'lh':1}
    names = []
    for elc_hemi in elctrodes_hemis:
        names.append('{}UN{}'.format('R' if elc_hemi == 'rh' else 'L', elcs_nums[elc_hemi]))
        elcs_nums[elc_hemi] += 1
    return names


def delete_electrodes():
    mu.delete_hierarchy('Deep_electrodes')


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
    in_group_ind = -1
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


def clear_groups():
    DellPanel.groups = []
    groups_fname = op.join(DellPanel.output_fol, '{}_groups.pkl'.format(
        int(bpy.context.scene.dell_ct_threshold)))
    if op.isfile(groups_fname):
        shutil.copy(groups_fname, '{}_backup{}'.format(*op.splitext(groups_fname)))
        os.remove(groups_fname)
    clear_electrodes_color()


def clear_electrodes_color():
    for elc_name in DellPanel.names:
        if bpy.data.objects.get(elc_name) is not None:
            _addon().object_coloring(bpy.data.objects[elc_name], (1, 1, 1))


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
        layout.operator(GetElectrodesAboveThrshold.bl_idname, text="Find electrodes", icon='ROTATE')
    else:
        row = layout.row(align=0)
        row.prop(context.scene, 'dell_ct_n_components', text="n_components")
        row.prop(context.scene, 'dell_ct_n_groups', text="n_groups")
        row = layout.row(align=0)
        row.prop(context.scene, 'dell_ct_error_radius', text="Error radius")
        row.prop(context.scene, 'dell_ct_min_elcs_for_lead', text="Min for lead")
        row = layout.row(align=0)
        row.prop(context.scene, 'dell_ct_max_dist_between_electrodes', text="Max dist between")
        row.prop(context.scene, 'dell_ct_min_distance', text="Min dist between")
        layout.label(text='#Groups found: {}'.format(len(DellPanel.groups)))
        layout.prop(context.scene, 'ct_noise_color', text='Noise color')
        if len(bpy.context.selected_objects) == 1 and bpy.context.selected_objects[0].name in DellPanel.names:
            layout.operator(FindElectrodeLead.bl_idname, text="Find selected electrode's lead", icon='PARTICLE_DATA')
        if len(bpy.context.selected_objects) == 2 and all(bpy.context.selected_objects[k].name in DellPanel.names for k in range(2)):
            layout.operator(FindElectrodeLead.bl_idname, text="Find lead between selected electrodes", icon='PARTICLE_DATA')
        layout.operator(FindRandomLead.bl_idname, text="I feel lucky", icon='LAMP_SUN')
        # layout.operator(SaveCTNeighborhood.bl_idname, text="Save CT neighborhood", icon='EDIT')
        layout.operator(ClearGroups.bl_idname, text="Clear groups", icon='GHOST_DISABLED')
    if parent is not None and len(parent.children) > 0:
        layout.operator(DeleteElectrodes.bl_idname, text="Delete electrodes", icon='CANCEL')
    if len(bpy.context.selected_objects) == 1 and bpy.context.selected_objects[0].name in DellPanel.names and len(DellPanel.groups) > 0:
        row = layout.row(align=0)
        row.operator(NextCTElectrode.bl_idname, text="", icon='PREV_KEYFRAME')
        row.operator(PrevCTElectrode.bl_idname, text="", icon='NEXT_KEYFRAME')
        row.label(text=bpy.context.selected_objects[0].name)
    if len(DellPanel.dists) > 0 and len(DellPanel.groups) > 0:
        layout.label(text='Group inner distances:')
        box = layout.box()
        col = box.column()
        last_group = DellPanel.groups[-1]
        for elc1, elc2, dist in zip([DellPanel.names[k] for k in last_group[:-1]], [DellPanel.names[k] for k in last_group[1:]], DellPanel.dists):
            mu.add_box_line(col, '{}-{}'.format(elc1, elc2), '{:.2f}'.format(dist), 0.8)


class ChooseCTFile(bpy.types.Operator, bpy_extras.io_utils.ExportHelper):
    bl_idname = "mmvt.choose_ct_file"
    bl_label = "Choose CT file"

    filename_ext = '.mgz'
    filter_glob = bpy.props.StringProperty(default='*.mgz', options={'HIDDEN'}, maxlen=255)

    def execute(self, context):
        ct_fname = self.filepath
        ct_name = 'ct_reg_to_mr.mgz' # mu.namesbase_with_ext(ct_fname)
        user_fol = mu.get_user_fol()
        ct_fol = mu.get_fname_folder(ct_fname)
        if ct_fol != op.join(user_fol, 'ct'):
            mu.make_dir(op.join(user_fol, 'ct'))
            shutil.copy(ct_fname, op.join(user_fol, 'ct', ct_name))
        run_ct_preproc()
        init(_addon(), ct_name)
        return {'FINISHED'}


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


class SaveCTNeighborhood(bpy.types.Operator):
    bl_idname = "mmvt.save_ct_neighborhood"
    bl_label = "save_ct_neighborhood"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        save_ct_neighborhood()
        return {'PASS_THROUGH'}


class GetElectrodesAboveThrshold(bpy.types.Operator):
    bl_idname = "mmvt.get_electrodes_above_threshold"
    bl_label = "get_electrodes_above_threshold"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        get_electrodes_above_threshold()
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


class DeleteElectrodes(bpy.types.Operator):
    bl_idname = "mmvt.delete_electrodes"
    bl_label = "delete_electrodes"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        delete_electrodes()
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
bpy.types.Scene.dell_ct_n_components = bpy.props.IntProperty(min=0, description='')
bpy.types.Scene.dell_ct_n_groups = bpy.props.IntProperty(min=0, description='', update=dell_ct_n_groups_update)
bpy.types.Scene.dell_ct_error_radius = bpy.props.FloatProperty(min=1, max=8, default=2)
bpy.types.Scene.dell_ct_min_elcs_for_lead = bpy.props.IntProperty(min=2, max=20, default=4)
bpy.types.Scene.dell_ct_max_dist_between_electrodes = bpy.props.FloatProperty(default=15, min=1, max=100)
bpy.types.Scene.dell_ct_min_distance = bpy.props.FloatProperty(default=3, min=0, max=100)
bpy.types.Scene.ct_noise_color = bpy.props.FloatVectorProperty(
    name="object_color", subtype='COLOR', default=(0, 0.5, 0), min=0.0, max=1.0, description="color picker")



class DellPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "Dell"
    addon = None
    init = False
    ct_found = False
    ct = None
    brain = None
    output_fol = ''
    colors = []
    groups = []
    noise = set()
    dists = []

    def draw(self, context):
        if DellPanel.init:
            dell_draw(self, context)


def init(addon, ct_name='ct_reg_to_mr.mgz', brain_mask_name='brain.mgz', aseg_name='aseg.mgz'):
    DellPanel.addon = addon
    try:
        DellPanel.output_fol = op.join(mu.get_user_fol(), 'ct', 'finding_electrodes_in_ct')
        DellPanel.ct_found = init_ct(ct_name, brain_mask_name, aseg_name)
        if DellPanel.ct_found:
            init_electrodes()
            if bpy.context.scene.dell_ct_n_groups > 0:
                DellPanel.colors = mu.get_distinct_colors(bpy.context.scene.dell_ct_n_groups)
            files = glob.glob(op.join(DellPanel.output_fol, '*_electrodes.pkl'))
            if len(files) > 0:
                (DellPanel.pos, DellPanel.names, DellPanel.hemis, bpy.context.scene.dell_ct_threshold) = mu.load(files[0])
            else:
                bpy.context.scene.dell_ct_threshold_percentile = 99.9
            bpy.context.scene.dell_ct_threshold = np.percentile(
                DellPanel.ct_data, bpy.context.scene.dell_ct_threshold_percentile)
            init_groups()
        bpy.context.scene.dell_ct_error_radius = 2
        bpy.context.scene.dell_ct_min_elcs_for_lead = 4
        bpy.context.scene.dell_ct_max_dist_between_electrodes = 15
        bpy.context.scene.dell_ct_min_distance = 3
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
    DellPanel.ct = nib.load(ct_fname)
    DellPanel.ct_data = DellPanel.ct.get_data()
    DellPanel.brain = nib.load(brain_mask_fname)
    DellPanel.brain_mask = DellPanel.brain.get_data()
    aseg_fname = op.join(subjects_dir, mu.get_user(), 'mri', aseg_name)
    DellPanel.aseg = nib.load(aseg_fname).get_data() if op.isfile(aseg_fname) else None
    return True


def init_electrodes():
    elcs_files = glob.glob(op.join(mu.get_user_fol(), 'electrodes', '*electrodes_positions.npz'))
    if len(elcs_files) == 1:
        elcs_dict = mu.Bag(np.load(elcs_files[0]))
        bipolar = '-' in elcs_dict.names[0]
        groups = set([mu.elec_group(elc_name, bipolar) for elc_name in elcs_dict.names])
        bpy.context.scene.dell_ct_n_components = len(elcs_dict.names)
        bpy.context.scene.dell_ct_n_groups = len(groups)


def init_groups():
    groups_fname = op.join(DellPanel.output_fol, '{}_groups.pkl'.format(
        int(bpy.context.scene.dell_ct_threshold)))
    DellPanel.groups = mu.load(groups_fname) if op.isfile(groups_fname) else []
    DellPanel.groups = [list(l) for l in DellPanel.groups]
    parent = bpy.data.objects.get('Deep_electrodes', None)
    if parent is None or len(parent.children) == 0:
        return
    for ind, group in enumerate(DellPanel.groups):
        color = DellPanel.colors[ind]
        for elc_ind in group:
            _addon().object_coloring(bpy.data.objects[DellPanel.names[elc_ind]], tuple(color))
    if len(DellPanel.groups) == 0:
        clear_electrodes_color()


def register():
    try:
        unregister()
        bpy.utils.register_class(DellPanel)
        bpy.utils.register_class(InstallReqs)
        bpy.utils.register_class(ChooseCTFile)
        bpy.utils.register_class(CalcThresholdPercentile)
        bpy.utils.register_class(GetElectrodesAboveThrshold)
        bpy.utils.register_class(FindElectrodeLead)
        bpy.utils.register_class(FindRandomLead)
        bpy.utils.register_class(SaveCTNeighborhood)
        bpy.utils.register_class(PrevCTElectrode)
        bpy.utils.register_class(NextCTElectrode)
        bpy.utils.register_class(ClearGroups)
        bpy.utils.register_class(DeleteElectrodes)
    except:
        print("Can't register Dell Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(DellPanel)
        bpy.utils.unregister_class(InstallReqs)
        bpy.utils.unregister_class(ChooseCTFile)
        bpy.utils.unregister_class(CalcThresholdPercentile)
        bpy.utils.unregister_class(GetElectrodesAboveThrshold)
        bpy.utils.unregister_class(FindElectrodeLead)
        bpy.utils.unregister_class(FindRandomLead)
        bpy.utils.unregister_class(SaveCTNeighborhood)
        bpy.utils.unregister_class(PrevCTElectrode)
        bpy.utils.unregister_class(NextCTElectrode)
        bpy.utils.unregister_class(ClearGroups)
        bpy.utils.unregister_class(DeleteElectrodes)
    except:
        pass


