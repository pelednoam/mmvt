import bpy
import os.path as op
import glob
import numpy as np
import importlib
import shutil
import os

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
    print('clustering...')
    ct_electrodes, clusters = fect.clustering(
        ct_voxels, DellPanel.ct_data, bpy.context.scene.dell_ct_n_components, 'knn', #DellPanel.output_fol,
        threshold=bpy.context.scene.dell_ct_threshold)
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


@mu.profileit('cumtime', op.join(mu.get_user_fol()))
def find_electrode_lead():
    if len(bpy.context.selected_objects) == 1 and bpy.context.selected_objects[0].name in DellPanel.names:
        selected_elc = bpy.context.selected_objects[0].name
        elc_ind = DellPanel.names.index(selected_elc)
        if elc_ind in set(mu.flat_list_of_lists(DellPanel.groups)):
            print('{} is already in a group!'.format(selected_elc))
            return
        group = fect.find_electrode_group(
            elc_ind, DellPanel.pos, DellPanel.hemis, DellPanel.groups, error_radius=3, min_elcs_for_lead=4,
            max_dist_between_electrodes=15, min_distance=2)
        if group is None:
            print('No group was found for {}!'.format(selected_elc))
            return
        DellPanel.groups.append(group)
        mu.save(DellPanel.groups, op.join(DellPanel.output_fol, '{}_groups.pkl'.format(
            int(bpy.context.scene.dell_ct_threshold))))
        color = DellPanel.colors[len(DellPanel.groups)]
        for elc_ind in group:
            _addon().object_coloring(bpy.data.objects[DellPanel.names[elc_ind]], tuple(color))
        print(group)
    else:
        print("You should first select an electrode")


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


def clear_groups():
    DellPanel.groups = []
    groups_fname = op.join(DellPanel.output_fol, '{}_groups.pkl'.format(
        int(bpy.context.scene.dell_ct_threshold)))
    shutil.copy(groups_fname, '{}_backup{}'.format(*op.splitext(groups_fname)))
    os.remove(groups_fname)
    clear_electrodes_color()


def clear_electrodes_color():
    for elc_name in DellPanel.names:
        _addon().object_coloring(bpy.data.objects[elc_name], (1, 1, 1))


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
    if not NIBABEL_EXIST or not DELL_EXIST:
        layout.operator(InstallReqs.bl_idname, text="Install reqs", icon='ROTATE')
        return
    layout.prop(context.scene, 'dell_ct_n_components', text="n_components")
    layout.prop(context.scene, 'dell_ct_n_groups', text="n_groups")
    layout.prop(context.scene, 'dell_ct_threshold', text="Threshold")
    row = layout.row(align=0)
    row.prop(context.scene, 'dell_ct_threshold_percentile', text='')
    row.operator(CalcThresholdPercentile.bl_idname, text="Calc threshold", icon='STRANDS')
    parent = bpy.data.objects.get('Deep_electrodes', None)
    if parent is None or len(parent.children) == 0:
        layout.operator(GetElectrodesAboveThrshold.bl_idname, text="Find electrodes", icon='ROTATE')
    else:
        layout.operator(FindElectrodeLead.bl_idname, text="Find selected electrode's lead", icon='PARTICLE_DATA')
        layout.operator(ClearGroups.bl_idname, text="Clear groups", icon='GHOST_DISABLED')
        layout.operator(DeleteElectrodes.bl_idname, text="Delete electrodes", icon='CANCEL')


class FindElectrodeLead(bpy.types.Operator):
    bl_idname = "mmvt.find_electrode_lead"
    bl_label = "find_electrode_lead"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        find_electrode_lead()
        return {'PASS_THROUGH'}


class GetElectrodesAboveThrshold(bpy.types.Operator):
    bl_idname = "mmvt.get_electrodes_above_threshold"
    bl_label = "get_electrodes_above_threshold"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        get_electrodes_above_threshold()
        return {'PASS_THROUGH'}


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


class DellPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "Dell"
    addon = None
    init = False
    ct = None
    brain = None
    output_fol = ''
    colors = []
    groups = []

    def draw(self, context):
        if DellPanel.init:
            dell_draw(self, context)


def init(addon):
    DellPanel.addon = addon
    if not init_ct():
        return
    init_electrodes()
    if bpy.context.scene.dell_ct_n_groups > 0:
        DellPanel.colors = mu.get_distinct_colors(bpy.context.scene.dell_ct_n_groups)
    DellPanel.output_fol = op.join(mu.get_user_fol(), 'ct', 'finding_electrodes_in_ct')
    files = glob.glob(op.join(DellPanel.output_fol, '*_electrodes.pkl'))
    if len(files) > 0:
        (DellPanel.pos, DellPanel.names, DellPanel.hemis, bpy.context.scene.dell_ct_threshold) = mu.load(files[0])
    else:
        bpy.context.scene.dell_ct_threshold_percentile = 99.9
    bpy.context.scene.dell_ct_threshold = np.percentile(
        DellPanel.ct_data, bpy.context.scene.dell_ct_threshold_percentile)
    init_groups()
    register()
    DellPanel.init = True


def init_ct():
    user_fol = mu.get_user_fol()
    ct_fname = op.join(user_fol, 'ct', 'ct_reg_to_mr.mgz')
    if not op.isfile(ct_fname):
        print("Dell: Can't find the ct!")
        return False
    subjects_dir = su.get_subjects_dir()
    brain_mask_fname = op.join(subjects_dir, mu.get_user(), 'mri', 'brain.mgz')
    if not op.isfile(brain_mask_fname):
        print("Dell: Can't find brain.mgz!")
        return False
    DellPanel.ct = nib.load(ct_fname)
    DellPanel.ct_data = DellPanel.ct.get_data()
    DellPanel.brain = nib.load(brain_mask_fname)
    DellPanel.brain_mask = DellPanel.brain.get_data()
    DellPanel.aseg = nib.load(op.join(subjects_dir, mu.get_user(), 'mri', 'aseg.mgz')).get_data()
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
        bpy.utils.register_class(CalcThresholdPercentile)
        bpy.utils.register_class(GetElectrodesAboveThrshold)
        bpy.utils.register_class(FindElectrodeLead)
        bpy.utils.register_class(ClearGroups)
        bpy.utils.register_class(DeleteElectrodes)
    except:
        print("Can't register Dell Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(DellPanel)
        bpy.utils.unregister_class(InstallReqs)
        bpy.utils.unregister_class(CalcThresholdPercentile)
        bpy.utils.unregister_class(GetElectrodesAboveThrshold)
        bpy.utils.unregister_class(FindElectrodeLead)
        bpy.utils.unregister_class(ClearGroups)
        bpy.utils.unregister_class(DeleteElectrodes)
    except:
        pass

