import bpy
import os.path as op
import glob
import mmvt_utils as mu
import nibabel as nib

from dell import find_electrodes_in_ct as fect


def _addon():
    return DellPanel.addon


def find_electrode_lead():
    pass


def get_electrodes_above_threshold():
    ct_voxels = fect.find_voxels_above_threshold(DellPanel.ct_data, bpy.context.scene.dell_ct_threshold)
    ct_voxels = fect.mask_voxels_outside_brain(ct_voxels, DellPanel.ct.header, DellPanel.brain)


def dell_draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'dell_ct_n_components', text="n_components")
    layout.prop(context.scene, 'dell_ct_n_groups', text="n_groups")
    layout.prop(context.scene, 'dell_ct_threshold', text="Threshold")
    layout.operator(FindElectrodeLead.bl_idname, text="Do something", icon='ROTATE')


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


bpy.types.Scene.dell_ct_threshold = bpy.props.FloatProperty(default=0.5, min=0, description="")
bpy.types.Scene.dell_ct_n_components = bpy.props.IntProperty(min=0, description="")
bpy.types.Scene.dell_ct_n_groups = bpy.props.IntProperty(min=0, description="")


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

    def draw(self, context):
        if DellPanel.init:
            dell_draw(self, context)


def init(addon):
    DellPanel.addon = addon
    if init_ct():
        register()
        DellPanel.init = True


def init_ct():
    user_fol = mu.get_user_fol()
    ct_fname = op.join(user_fol, 'freeview', 'ct.mgz')
    if not op.isfile(ct_fname):
        print("Dell: Can't find the ct!")
        return False
    brain_mask_fname = op.join(user_fol, 'freeview', 'brain.mgz')
    if not op.isfile(brain_mask_fname):
        print("Dell: Can't find brain.mgz!")
        return False
    DellPanel.ct = nib.load(ct_fname)
    DellPanel.ct_data = DellPanel.ct.get_data()
    DellPanel.brain = nib.load(brain_mask_fname)
    DellPanel.brain_mask = DellPanel.brain.get_data()
    return True


def register():
    try:
        unregister()
        bpy.utils.register_class(DellPanel)
        bpy.utils.register_class(GetElectrodesAboveThrshold)
        bpy.utils.register_class(FindElectrodeLead)
    except:
        print("Can't register Dell Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(DellPanel)
        bpy.utils.unregister_class(GetElectrodesAboveThrshold)
        bpy.utils.unregister_class(FindElectrodeLead)
    except:
        pass
