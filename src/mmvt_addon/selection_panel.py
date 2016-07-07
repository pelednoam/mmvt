import bpy
import mmvt_utils as mu
import numpy as np
import colors_utils as cu
import connections_panel
import electrodes_panel


def deselect_all():
    for obj in bpy.data.objects:
        obj.select = False
    if bpy.data.objects.get(' '):
        bpy.data.objects[' '].select = True
        bpy.context.scene.objects.active = bpy.data.objects[' ']


def select_all_rois():
    select_brain_objects('Brain', bpy.data.objects['Cortex-lh'].children + bpy.data.objects['Cortex-rh'].children)


def select_only_subcorticals():
    select_brain_objects('Subcortical_structures', bpy.data.objects['Subcortical_structures'].children)


def select_all_electrodes():
    select_brain_objects('Deep_electrodes', bpy.data.objects['Deep_electrodes'].children)


def select_all_connections():
    select_brain_objects('connections', bpy.data.objects['connections'].children)


def conditions_selection_update(self, context):
    mu.filter_graph_editor(bpy.context.scene.conditions_selection)
    SelectionMakerPanel.addon.clear_and_recolor()


def select_brain_objects(parent_obj_name, children):
    parent_obj = bpy.data.objects[parent_obj_name]
    if bpy.context.scene.selection_type == 'diff':
        if parent_obj.animation_data is None:
            print('parent_obj.animation_data is None!')
        else:
            mu.show_hide_obj_and_fcurves(children, False)
            parent_obj.select = True
            for fcurve in parent_obj.animation_data.action.fcurves:
                fcurve.hide = False
                fcurve.select = True
    else:
        mu.show_hide_obj_and_fcurves(children, True)
        parent_obj.select = False


def set_conditions_enum(conditions):
    conditions = mu.unique_save_order(conditions)
    selection_items = [(c, '{}'.format(c), '', ind) for ind, c in enumerate(conditions)]
    bpy.types.Scene.conditions_selection = bpy.props.EnumProperty(
        items=selection_items, description="Condition Selection", update=conditions_selection_update)


def set_selection_type(selection_type):
    bpy.context.scene.selection_type = selection_type


class SelectAllRois(bpy.types.Operator):
    bl_idname = "ohad.roi_selection"
    bl_label = "select2 ROIs"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        select_all_rois()
        mu.view_all_in_graph_editor(context)
        if bpy.context.scene.selection_type == 'diff':
            mu.change_fcurves_colors([bpy.data.objects['Brain']])
        else:
            corticals_labels = mu.get_corticals_labels()
            mu.change_fcurves_colors(corticals_labels)
        return {"FINISHED"}


class SelectAllSubcorticals(bpy.types.Operator):
    bl_idname = "ohad.subcorticals_selection"
    bl_label = "select only subcorticals"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        select_only_subcorticals()
        mu.view_all_in_graph_editor(context)
        if bpy.context.scene.selection_type == 'diff':
            mu.change_fcurves_colors([bpy.data.objects['Subcortical_structures']])
        else:
            mu.change_fcurves_colors(bpy.data.objects['Subcortical_structures'].children)
        return {"FINISHED"}


class SelectAllElectrodes(bpy.types.Operator):
    bl_idname = "ohad.electrodes_selection"
    bl_label = "select2 Electrodes"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        select_all_electrodes()
        mu.unfilter_graph_editor()
        if bpy.context.scene.selection_type == 'diff':
            mu.change_fcurves_colors([bpy.data.objects['Deep_electrodes']])
        elif bpy.context.scene.selection_type == 'spec_cond':
            mu.filter_graph_editor(bpy.context.scene.conditions_selection)
        else:
            mu.change_fcurves_colors(bpy.data.objects['Deep_electrodes'].children)
        mu.view_all_in_graph_editor(context)
        return {"FINISHED"}


class SelectAllConnections(bpy.types.Operator):
    bl_idname = "ohad.connections_selection"
    bl_label = "select connections"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        select_all_connections()
        mu.view_all_in_graph_editor(context)
        return {"FINISHED"}


class ClearSelection(bpy.types.Operator):
    bl_idname = "ohad.clear_selection"
    bl_label = "deselect all"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        for obj in bpy.data.objects:
            obj.select = False
        if bpy.data.objects.get(' '):
            bpy.data.objects[' '].select = True
            bpy.context.scene.objects.active = bpy.data.objects[' ']

        return {"FINISHED"}


class FitSelection(bpy.types.Operator):
    bl_idname = "ohad.fit_selection"
    bl_label = "Fit selection"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        mu.view_all_in_graph_editor(context)
        return {"FINISHED"}


bpy.types.Scene.selection_type = bpy.props.EnumProperty(
    items=[("diff", "Conditions difference", "", 1), ("conds", "All conditions", "", 2),
           ("spec_cond", "Specific condition", "", 3)], description="Selection type")
bpy.types.Scene.conditions_selection = bpy.props.EnumProperty(items=[], description="Condition Selection",
                                                              update=conditions_selection_update)


class SelectionMakerPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
    bl_label = "Selection Panel"
    addon = None

    @staticmethod
    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene, "selection_type", text="")
        if bpy.context.scene.selection_type == 'spec_cond':
            layout.prop(context.scene, "conditions_selection", text="")
        layout.operator(SelectAllRois.bl_idname, text="Select all cortical ROIs", icon='BORDER_RECT')
        layout.operator(SelectAllSubcorticals.bl_idname, text="Select all subcorticals", icon = 'BORDER_RECT' )
        if bpy.data.objects.get(electrodes_panel.PARENT_OBJ):
            layout.operator(SelectAllElectrodes.bl_idname, text="Select all Electrodes", icon='BORDER_RECT')
        if bpy.data.objects.get(connections_panel.PARENT_OBJ):
            layout.operator(SelectAllConnections.bl_idname, text="Select all Connections", icon='BORDER_RECT')
        layout.operator(ClearSelection.bl_idname, text="Deselect all", icon='PANEL_CLOSE')
        layout.operator(FitSelection.bl_idname, text="Fit graph window", icon='MOD_ARMATURE')


def init(addon):
    SelectionMakerPanel.addon = addon
    register()


def register():
    try:
        unregister()
        bpy.utils.register_class(SelectionMakerPanel)
        bpy.utils.register_class(FitSelection)
        bpy.utils.register_class(ClearSelection)
        bpy.utils.register_class(SelectAllConnections)
        bpy.utils.register_class(SelectAllElectrodes)
        bpy.utils.register_class(SelectAllSubcorticals)
        bpy.utils.register_class(SelectAllRois)
        # print('Selection Panel was registered!')
    except:
        print("Can't register Selection Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(SelectionMakerPanel)
        bpy.utils.unregister_class(FitSelection)
        bpy.utils.unregister_class(ClearSelection)
        bpy.utils.unregister_class(SelectAllConnections)
        bpy.utils.unregister_class(SelectAllElectrodes)
        bpy.utils.unregister_class(SelectAllSubcorticals)
        bpy.utils.unregister_class(SelectAllRois)
    except:
        pass
