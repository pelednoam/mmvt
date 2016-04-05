bl_info = {
    "name": "Multi-modal visualization tool",
    "author": "Ohad Felsenstein & Noam Peled",
    "version": (1, 2),
    "blender": (2, 7, 2),
    "api": 33333,
    "location": "View3D > Add > Mesh > Say3D",
    "description": "Multi-modal visualization tool",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Add Mesh"}


import bpy
import mathutils
import numpy as np
import os
import os.path as op
import sys
import time
import glob
import math
import importlib
import numbers
import itertools
import traceback
from collections import defaultdict

import mmvt_utils
importlib.reload(mmvt_utils)
import colors_utils
importlib.reload(colors_utils)

import coloring_panel
importlib.reload(coloring_panel)
import connections_panel
importlib.reload(connections_panel)
import play_panel
importlib.reload(play_panel)
import dti_panel
importlib.reload(dti_panel)
import electrodes_panel
importlib.reload(electrodes_panel)
import freeview_panel
importlib.reload(freeview_panel)
import search_panel
importlib.reload(search_panel)
import appearance_panel
importlib.reload(appearance_panel)
import where_am_i_panel
importlib.reload(where_am_i_panel)
import fMRI_panel
importlib.reload(fMRI_panel)
import render_panel
importlib.reload(render_panel)
import listener_panel
importlib.reload(listener_panel)
import data_panel
importlib.reload(data_panel)
import selection_panel
importlib.reload(selection_panel)
import vertex_data_panel
importlib.reload(vertex_data_panel)


print("Neuroscience add on started!")
# todo: should change that in the code!!!
# Should be here bpy.types.Scene.maximal_time_steps
T = 2500

bpy.types.Scene.atlas = bpy.props.StringProperty(name='atlas', default='laus250')
bpy.context.scene.atlas = mmvt_utils.get_atlas()
bpy.types.Scene.bipolar = bpy.props.BoolProperty(default=False, description="Bipolar electrodes")
bpy.types.Scene.electrode_radius = bpy.props.FloatProperty(default=0.15, description="Electrodes radius", min=0.01, max=1)
bpy.context.scene.electrode_radius = 0.15

# LAYERS
(CONNECTIONS_LAYER, ELECTRODES_LAYER, ROIS_LAYER, ACTIVITY_LAYER, LIGHTS_LAYER,
    BRAIN_EMPTY_LAYER, EMPTY_LAYER) = 3, 1, 10, 11, 12, 5, 14

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Selection links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
select_brain_objects = selection_panel.select_brain_objects
select_all_connections = selection_panel.select_all_connections
select_all_electrodes = selection_panel.select_all_electrodes
select_only_subcorticals = selection_panel.select_only_subcorticals
select_all_rois = selection_panel.select_all_rois
deselect_all = selection_panel.deselect_all
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Coloring links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
object_coloring = coloring_panel.object_coloring
clear_subcortical_fmri_activity = coloring_panel.clear_subcortical_fmri_activity
clear_cortex = coloring_panel.clear_cortex
clear_object_vertex_colors = coloring_panel.clear_object_vertex_colors
color_object_homogeneously = coloring_panel.color_object_homogeneously
init_activity_map_coloring = coloring_panel.init_activity_map_coloring
load_faces_verts = coloring_panel.load_faces_verts
load_meg_subcortical_activity = coloring_panel.load_meg_subcortical_activity
activity_map_coloring = coloring_panel.activity_map_coloring
meg_labels_coloring = coloring_panel.meg_labels_coloring
meg_labels_coloring_hemi = coloring_panel.meg_labels_coloring_hemi
plot_activity = coloring_panel.plot_activity
fmri_subcortex_activity_color = coloring_panel.fmri_subcortex_activity_color
activity_map_obj_coloring = coloring_panel.activity_map_obj_coloring
color_manually = coloring_panel.color_manually
color_subcortical_region = coloring_panel.color_subcortical_region
clear_subcortical_regions = coloring_panel.clear_subcortical_regions
clear_colors_from_parent_childrens = coloring_panel.clear_colors_from_parent_childrens
default_coloring = coloring_panel.default_coloring
get_fMRI_activity = coloring_panel.get_fMRI_activity
get_faces_verts = coloring_panel.get_faces_verts
clear_colors = coloring_panel.clear_colors
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Coloring links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Filter Panel ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
bpy.types.Scene.closest_curve_str = ''
bpy.types.Scene.filter_is_on = False


def find_obj_with_val():
    cur_objects = []
    for obj in bpy.data.objects:
        if obj.select is True:
            cur_objects.append(obj)

    for ii in range(len(bpy.data.screens['Neuro'].areas)):
        if bpy.data.screens['Neuro'].areas[ii].type == 'GRAPH_EDITOR':
            for jj in range(len(bpy.data.screens['Neuro'].areas[ii].spaces)):
                if bpy.data.screens['Neuro'].areas[ii].spaces[jj].type == 'GRAPH_EDITOR':
                    # print(dir(bpy.data.screens['Neuro'].areas[ii].spaces[jj]))
                    target = bpy.data.screens['Neuro'].areas[ii].spaces[jj].cursor_position_y

    values, names, obj_names = [], [], []
    for cur_obj in cur_objects:
        # if cur_obj.animation_data is None:
        #     continue
        # for fcurve in cur_obj.animation_data.action.fcurves:
        #     val = fcurve.evaluate(bpy.context.scene.frame_current)
        #     name = mmvt_utils.fcurve_name(fcurve)
        for name, val in cur_obj.items():
            if isinstance(val, numbers.Number):
                values.append(val)
                names.append(name)
                obj_names.append(cur_obj.name)
            # print(name)
    np_values = np.array(values) - target
    try:
        index = np.argmin(np.abs(np_values))
        closest_curve_name = names[index]
        closet_object_name = obj_names[index]
    except ValueError:
        closest_curve_name = ''
        closet_object_name = ''
        print('ERROR - Make sure you select all objects in interest')
    # print(closest_curve_name, closet_object_name)
    bpy.types.Scene.closest_curve_str = closest_curve_name
    # object_name = closest_curve_str
    # if bpy.data.objects.get(object_name) is None:
    #     object_name = object_name[:object_name.rfind('_')]
    print('object name: {}, curve name: {}'.format(closet_object_name, closest_curve_name))
    parent_obj = bpy.data.objects[closet_object_name].parent
    # print('parent: {}'.format(bpy.data.objects[object_name].parent))
    # try:
    if parent_obj.name == 'Deep_electrodes':
        print('filtering electrodes')
        filter_electrode_func(closet_object_name, closest_curve_name)
    elif parent_obj.name == connections_panel.PARENT_OBJ:
        connections_panel.find_connections_closest_to_target_value(closet_object_name, closest_curve_name, target)
    else:
        filter_roi_func(closet_object_name, closest_curve_name)
    # except KeyError:
    #     filter_roi_func(object_name)


class FindCurveClosestToCursor(bpy.types.Operator):
    bl_idname = "ohad.curve_close_to_cursor"
    bl_label = "curve close to cursor"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        find_obj_with_val()
        return {"FINISHED"}


def filter_draw(self, context):
    layout = self.layout
    layout.prop(context.scene, "filter_topK", text="Top K")
    row = layout.row(align=0)
    row.prop(context.scene, "filter_from", text="From")
    # row.label(str(GrabFromFiltering.value))
    row.operator(GrabFromFiltering.bl_idname, text="", icon='BORDERMOVE')
    # row.operator("ohad.grab_from", text="", icon = 'BORDERMOVE')
    row.prop(context.scene, "filter_to", text="To")
    row.operator(GrabToFiltering.bl_idname, text="", icon='BORDERMOVE')
    layout.prop(context.scene, "filter_curves_type", text="")
    layout.prop(context.scene, "filter_curves_func", text="")
    layout.operator("ohad.filter", text="Filter " + bpy.context.scene.filter_curves_type, icon='BORDERMOVE')
    if bpy.types.Scene.filter_is_on:
        layout.operator("ohad.filter_clear", text="Clear Filtering", icon='PANEL_CLOSE')
    col = layout.column(align=0)
    col.operator("ohad.curve_close_to_cursor", text="closest curve to cursor", icon='SNAP_SURFACE')
    col.label(text=bpy.types.Scene.closest_curve_str)

    # bpy.context.area.type = 'GRAPH_EDITOR'
    # filter_to = bpy.context.scence.frame_preview_end


files_names = {'MEG': 'labels_data_{hemi}.npz', 'Electrodes': 'electrodes_data_{stat}.npz'}

bpy.types.Scene.closest_curve = bpy.props.StringProperty(description="Find closest curve to cursor", update=filter_draw)
#bpy.types.Scene.filter_topK = bpy.props.IntProperty(default=1, min=0, description="The top K elements to be shown")
bpy.types.Scene.filter_topK = bpy.props.IntProperty(default=1, min=0, description="The top K elements to be shown")
bpy.types.Scene.filter_from = bpy.props.IntProperty(default=0, min=0, description="When to filter from")
# bpy.types.Scene.filter_to = bpy.props.IntProperty(default=bpy.data.scenes['Scene'].frame_preview_end, min=0, description="When to filter to")
bpy.types.Scene.filter_to = bpy.props.IntProperty(default=bpy.context.scene.frame_end, min=0,
                                                  description="When to filter to")
bpy.types.Scene.filter_curves_type = bpy.props.EnumProperty(
    items=[("MEG", "MEG time course", "", 1), ("Electrodes", " Electrodes time course", "", 2)],
    description="Type of curve to be filtered", update=filter_draw)
bpy.types.Scene.filter_curves_func = bpy.props.EnumProperty(
    items=[("RMS", "RMS", "RMS between the two conditions", 1), ("SumAbs", "SumAbs", "Sum of the abs values", 2),
           ("threshold", "Above threshold", "", 3)],
    description="Filtering function", update=filter_draw)


class FilteringMakerPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
    bl_label = "Filter number of curves"

    def draw(self, context):
        filter_draw(self, context)


class GrabFromFiltering(bpy.types.Operator):
    bl_idname = "ohad.grab_from"
    bl_label = "grab from"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        # print(bpy.context.scene.frame_current)
        context.scene.filter_from = bpy.context.scene.frame_current
        # print(bpy.context.scene.filter_from)
        bpy.data.scenes['Scene'].frame_preview_start = context.scene.frame_current
        return {"FINISHED"}


class GrabToFiltering(bpy.types.Operator):
    bl_idname = "ohad.grab_to"
    bl_label = "grab to"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        # print(bpy.context.scene.frame_current)
        context.scene.filter_to = bpy.context.scene.frame_current
        # print(bpy.context.scene.filter_to)
        bpy.data.scenes['Scene'].frame_preview_end = context.scene.frame_current
        return {"FINISHED"}


class ClearFiltering(bpy.types.Operator):
    bl_idname = "ohad.filter_clear"
    bl_label = "filter clear"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        clear_filtering()
        type_of_filter = bpy.context.scene.filter_curves_type
        if type_of_filter == 'MEG':
            select_all_rois()
        elif type_of_filter == 'Electrodes':
            select_all_electrodes()
        bpy.data.scenes['Scene'].frame_preview_end = get_max_time_steps()
        bpy.data.scenes['Scene'].frame_preview_start = 1
        bpy.types.Scene.closest_curve_str = ''
        bpy.types.Scene.filter_is_on = False
        return {"FINISHED"}


def clear_filtering():
    for subhierarchy in bpy.data.objects['Brain'].children:
        new_mat = bpy.data.materials['unselected_label_Mat_cortex']
        if subhierarchy.name == 'Subcortical_structures':
            new_mat = bpy.data.materials['unselected_label_Mat_subcortical']
        for obj in subhierarchy.children:
            obj.active_material = new_mat

    if bpy.data.objects.get('Deep_electrodes'):
        for obj in bpy.data.objects['Deep_electrodes'].children:
            de_select_electrode(obj)


def de_select_electrode(obj, call_create_and_set_material=True):
    obj.active_material.node_tree.nodes["Layer Weight"].inputs[0].default_value = 1
    # safety check, if something happened to the electrode's material
    if call_create_and_set_material:
        mmvt_utils.create_and_set_material(obj)
    # Sholdn't change to color here. If user plot the electrodes, we don't want to change it back to white.
    # obj.active_material.node_tree.nodes["RGB"].outputs[0].default_value = (1, 1, 1, 1)


def get_max_time_steps():
    # Check if maximal_time_steps is in bpy.types.Scene
    try:
        return bpy.types.Scene.maximal_time_steps
    except:
        print('No preperty maximal_time_steps in bpy.types.Scene')

    # Check if there is animation data in MEG
    try:
        hemi = bpy.data.objects['Cortex-lh']
        # Takes the first child first condition fcurve
        fcurves = hemi.children[0].animation_data.action.fcurves[0]
        return len(fcurves.keyframe_points) - 3
    except:
        print('No MEG data')

    try:
        elec = bpy.data.objects['Deep_electrodes'].children[0]
        fcurves = elec.animation_data.action.fcurves[0]
        return len(fcurves.keyframe_points) - 2
    except:
        print('No deep electrodes data')

    # Bad fallback...
    return T


def filter_roi_func(closet_object_name, closest_curve_name=None):
    if bpy.context.scene.selection_type == 'conds':
        bpy.data.objects[closet_object_name].select = True

    bpy.context.scene.objects.active = bpy.data.objects[closet_object_name]
    if bpy.data.objects[closet_object_name].active_material == bpy.data.materials['unselected_label_Mat_subcortical']:
        bpy.data.objects[closet_object_name].active_material = bpy.data.materials['selected_label_Mat_subcortical']
    else:
        bpy.data.objects[closet_object_name].active_material = bpy.data.materials['selected_label_Mat']
    bpy.types.Scene.filter_is_on = True


def filter_electrode_func(closet_object_name, closest_curve_name=None):
    bpy.data.objects[closet_object_name].active_material.node_tree.nodes["Layer Weight"].inputs[0].default_value = 0.3
    # todo: selecting the electrode will show both of their conditions time series
    # We don't want it to happen if selection_type == 'conds'...
    if bpy.context.scene.selection_type == 'conds':
        bpy.data.objects[closet_object_name].select = True
    bpy.context.scene.objects.active = bpy.data.objects[closet_object_name]
    bpy.types.Scene.filter_is_on = True


class Filtering(bpy.types.Operator):
    bl_idname = "ohad.filter"
    bl_label = "Filter deep elctrodes"
    bl_options = {"UNDO"}
    topK = -1
    filter_from = 100000
    filter_to = -100000
    current_activity_path = ''
    type_of_filter = None
    type_of_func = None
    current_file_to_upload = ''
    current_root_path = mmvt_utils.get_user_fol() #bpy.context.scene.conf_path

    def get_object_to_filter(self, source_files):
        data, names = [], []
        for input_file in source_files:
            try:
                f = np.load(input_file)
                data.append(f['data'])
                names.extend([name.astype(str) for name in f['names']])
            except:
                mmvt_utils.message(self, "Can't load {}!".format(input_file))

        print('filtering {}-{}'.format(self.filter_from, self.filter_to))

        t_range = range(self.filter_from, self.filter_to + 1)

        print(self.type_of_func)
        d = np.vstack((d for d in data))
        print('%%%%%%%%%%%%%%%%%%%' + str(len(d[0, :, 0])))
        t_range = range(max(self.filter_from, 1), min(self.filter_to, len(d[0, :, 0])) - 1)
        if self.type_of_func == 'RMS':
            dd = np.squeeze(np.diff(d[:, t_range, :], axis=2)) # d[:, t_range, 0] - d[:, t_range, 1]
            dd = np.sqrt(np.sum(np.power(dd, 2), 1))
        elif self.type_of_func == 'SumAbs':
            dd = np.sum(abs(d[:, t_range, :]), (1, 2))
        elif self.type_of_func == 'threshold':
            dd = np.max(np.abs(np.squeeze(np.diff(d[:, t_range, :], axis=2))), axis=1)

        if self.topK > 0:
            self.topK = min(self.topK, len(names))
        else:
            self.topK = sum(dd > 0)

        if self.type_of_func == 'threshold':
            indices = np.where(dd > bpy.context.scene.coloring_threshold)[0]
            objects_to_filtter_in = sorted(indices, key=lambda i:dd[i])[::-1][:self.topK]
            # objects_to_filtter_in = np.argsort(dd[indices])[::-1][:self.topK]
        else:
            objects_to_filtter_in = np.argsort(dd)[::-1][:self.topK]
        print(dd[objects_to_filtter_in])
        return objects_to_filtter_in, names

    def filter_electrodes(self, current_file_to_upload):
        print('filter_electrodes')
        source_files = [op.join(self.current_activity_path, current_file_to_upload)]
        objects_indices, names = self.get_object_to_filter(source_files)

        for obj in bpy.data.objects:
            obj.select = False

        deep_electrodes_obj = bpy.data.objects['Deep_electrodes']
        for obj in deep_electrodes_obj.children:
            obj.active_material.node_tree.nodes["Layer Weight"].inputs[0].default_value = 1

        if bpy.context.scene.selection_type == 'diff':
            filter_obj_names = [names[ind] for ind in objects_indices]
            for fcurve in deep_electrodes_obj.animation_data.action.fcurves:
                con_name = mmvt_utils.fcurve_name(fcurve)
                fcurve.hide = con_name not in filter_obj_names
                fcurve.select = not fcurve.hide
            deep_electrodes_obj.select = True
        else:
            deep_electrodes_obj.select = False

        for ind in range(min(self.topK, len(objects_indices)) - 1, -1, -1):
            if bpy.data.objects.get(names[objects_indices[ind]]):
                orig_name = bpy.data.objects[names[objects_indices[ind]]].name
                filter_electrode_func(orig_name)
            else:
                print("Can't find {}!".format(names[objects_indices[ind]]))

    def filter_rois(self, current_file_to_upload):
        print('filter_ROIs')
        set_appearance_show_rois_layer(bpy.context.scene, True)
        source_files = [op.join(self.current_activity_path, current_file_to_upload.format(hemi=hemi)) for hemi
                        in mmvt_utils.HEMIS]
        objects_indices, names = self.get_object_to_filter(source_files)
        for obj in bpy.data.objects:
            obj.select = False
            if obj.parent == bpy.data.objects['Subcortical_structures']:
                obj.active_material = bpy.data.materials['unselected_label_Mat_subcortical']
            elif obj.parent == bpy.data.objects['Cortex-lh'] or obj.parent == bpy.data.objects['Cortex-rh']:
                obj.active_material = bpy.data.materials['unselected_label_Mat_cortex']

        if bpy.context.scene.selection_type == 'diff':
            filter_obj_names = [names[ind] for ind in objects_indices]
            brain_obj = bpy.data.objects['Brain']
            for fcurve in brain_obj.animation_data.action.fcurves:
                con_name = mmvt_utils.fcurve_name(fcurve)
                fcurve.hide = con_name not in filter_obj_names
                fcurve.select = not fcurve.hide
            brain_obj.select = True

        for ind in range(min(self.topK, len(objects_indices)) - 1, -1, -1):
            if bpy.data.objects.get(names[objects_indices[ind]]):
                orig_name = bpy.data.objects[names[objects_indices[ind]]].name
                filter_roi_func(orig_name)
            else:
                print("Can't find {}!".format(names[objects_indices[ind]]))
            # print(orig_name)
            # # new_name = '*'+orig_name
            # # print(new_name)
            # # bpy.data.objects[orig_name].name = new_name
            # bpy.data.objects[orig_name].select = True
            # bpy.context.scene.objects.active = bpy.data.objects[orig_name]
            # # if bpy.data.objects[orig_name].parent != bpy.data.objects[orig_name]:
            # if bpy.data.objects[orig_name].active_material == bpy.data.materials['unselected_label_Mat_subcortical']:
            #     bpy.data.objects[orig_name].active_material = bpy.data.materials['selected_label_Mat_subcortical']
            # else:
            #     bpy.data.objects[orig_name].active_material = bpy.data.materials['selected_label_Mat']

    def invoke(self, context, event=None):
        change_view3d()
        setup_layers()
        self.topK = bpy.context.scene.filter_topK
        self.filter_from = bpy.context.scene.filter_from
        self.filter_to = bpy.context.scene.filter_to
        self.current_activity_path = mmvt_utils.get_user_fol() # bpy.path.abspath(bpy.context.scene.conf_path)
        # self.current_activity_path = bpy.path.abspath(bpy.context.scene.activity_path)
        self.type_of_filter = bpy.context.scene.filter_curves_type
        self.type_of_func = bpy.context.scene.filter_curves_func
        current_file_to_upload = files_names[self.type_of_filter]

        # print(self.current_root_path)
        # source_files = ["/homes/5/npeled/space3/ohad/mg79/electrodes_data.npz"]
        if self.type_of_filter == 'Electrodes':
            current_file_to_upload = current_file_to_upload.format(
                stat='avg' if bpy.context.scene.selection_type == 'conds' else 'diff')
            self.filter_electrodes(current_file_to_upload)
        elif self.type_of_filter == 'MEG':
            self.filter_rois(current_file_to_upload)

        # bpy.context.screen.areas[2].spaces[0].dopesheet.filter_fcurve_name = '*'
        return {"FINISHED"}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Show / Hide objects ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def show_hide_hierarchy(do_hide, obj):
    if bpy.data.objects.get(obj) is not None:
        bpy.data.objects[obj].hide = do_hide
        for child in bpy.data.objects[obj].children:
            child.hide = do_hide
            child.hide_render = do_hide


def show_hide_hemi(val, obj_func_name, obj_brain_name):
    if bpy.data.objects.get(obj_func_name) is not None:
        bpy.data.objects[obj_func_name].hide = val
        bpy.data.objects[obj_func_name].hide_render = val
    show_hide_hierarchy(val, obj_brain_name)


def show_hide_rh(self, context):
    show_hide_hemi(bpy.context.scene.objects_show_hide_rh, "rh", "Cortex-rh")


def show_hide_lh(self, context):
    show_hide_hemi(bpy.context.scene.objects_show_hide_lh, "lh", "Cortex-lh")


def show_hide_sub_cortical_update(self, context):
    show_hide_sub_corticals(bpy.context.scene.objects_show_hide_sub_cortical)


def show_hide_sub_corticals(val):
    show_hide_hierarchy(val, "Subcortical_structures")
    # show_hide_hierarchy(bpy.context.scene.objects_show_hide_sub_cortical, "Subcortical_activity_map")
    # We split the activity map into two types: meg for the same activation for the each structure, and fmri
    # for a better resolution, like on the cortex.
    # todo: can't display both subcortical activity
    show_hide_hierarchy(val, "Subcortical_fmri_activity_map")
    show_hide_hierarchy(val, "Subcortical_meg_activity_map")


bpy.types.Scene.objects_show_hide_lh = bpy.props.BoolProperty(default=True, description="Show left hemisphere",
                                                              update=show_hide_lh)
bpy.types.Scene.objects_show_hide_rh = bpy.props.BoolProperty(default=True, description="Show right hemisphere",
                                                              update=show_hide_rh)
bpy.types.Scene.objects_show_hide_sub_cortical = bpy.props.BoolProperty(default=True, description="Show sub cortical",
                                                                        update=show_hide_sub_cortical_update)

class ShowHideObjectsPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
    bl_label = "Show Hide Objects"

    def draw(self, context):
        col1 = self.layout.column(align=True)
        col1.prop(context.scene, 'objects_show_hide_lh', text="Left Hemisphere", icon='RESTRICT_VIEW_OFF')
        col1.prop(context.scene, 'objects_show_hide_rh', text="Right Hemisphere", icon='RESTRICT_VIEW_OFF')
        col1.prop(context.scene, 'objects_show_hide_sub_cortical', text="Sub Cortical", icon='RESTRICT_VIEW_OFF')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Show / Hide objects ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Appearance links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
setup_layers = appearance_panel.setup_layers
change_view3d = appearance_panel.change_view3d
show_rois = appearance_panel.show_rois
show_activity = appearance_panel.show_activity
show_electrodes = appearance_panel.show_electrodes
change_to_rendered_brain = appearance_panel.change_to_rendered_brain
change_to_solid_brain = appearance_panel.change_to_solid_brain
make_brain_solid_or_transparent = appearance_panel.make_brain_solid_or_transparent
update_layers = appearance_panel.update_layers
update_solidity = appearance_panel.update_solidity

get_appearance_show_electrodes_layer = appearance_panel.get_appearance_show_electrodes_layer
set_appearance_show_electrodes_layer = appearance_panel.set_appearance_show_electrodes_layer
get_appearance_show_activity_layer = appearance_panel.get_appearance_show_activity_layer
set_appearance_show_activity_layer = appearance_panel.set_appearance_show_activity_layer
get_appearance_show_rois_layer = appearance_panel.get_appearance_show_rois_layer
set_appearance_show_rois_layer = appearance_panel.set_appearance_show_rois_layer
get_appearance_show_connections_layer = appearance_panel.get_appearance_show_connections_layer
set_appearance_show_connections_layer = appearance_panel.set_appearance_show_connections_layer
get_filter_view_type = appearance_panel.get_filter_view_type
set_filter_view_type = appearance_panel.set_filter_view_type

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Appearance links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Transparency Panel ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TransparencyPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
    bl_label = "Transparency"

    def draw(self, context):
        transparency_draw(self, context)


def transparency_draw(self, context):
    if context.scene.filter_view_type == '1' and bpy.context.scene.appearance_show_activity_layer is True:
    # if context.scene.filter_view_type == 'RENDERED' and bpy.context.scene.appearance_show_activity_layer is True:
        layout = self.layout
        layout.prop(context.scene, 'appearance_solid_slider', text="Show solid brain")
        split2 = layout.split()
        split2.prop(context.scene, 'appearance_depth_Bool', text="Show cortex deep layers")
        split2.prop(context.scene, 'appearance_depth_slider', text="Depth")
        layout.operator("ohad.appearance_update", text="Update")


class UpdateAppearance(bpy.types.Operator):
    bl_idname = "ohad.appearance_update"
    bl_label = "filter clear"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        if context.scene.filter_view_type == '1' and bpy.context.scene.appearance_show_activity_layer is True:
            make_brain_solid_or_transparent()
            update_layers()
        else:
            self.report({'ERROR'}, 'You should change the view to Rendered Brain first.')
        return {"FINISHED"}

bpy.types.Scene.appearance_solid_slider = bpy.props.FloatProperty(default=0.0, min=0, max=1, description="")
bpy.types.Scene.appearance_depth_slider = bpy.props.IntProperty(default=1, min=1, max=10, description="")
bpy.types.Scene.appearance_depth_Bool = bpy.props.BoolProperty(default=False, description="")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Transparency Panel ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main(addon_prefs=None):
    bpy.context.scene.appearance_show_electrodes_layer = False
    bpy.context.scene.appearance_show_activity_layer = False
    bpy.context.scene.appearance_show_ROIs_layer = True
    bpy.context.scene.appearance_show_connections_layer = False
    setup_layers()
    try:
        current_module = sys.modules[__name__]
        coloring_panel.init(current_module)
        connections_panel.init(current_module)
        play_panel.init(current_module)
        dti_panel.init(current_module)
        electrodes_panel.init(current_module)
        freeview_panel.init(current_module, addon_prefs)
        search_panel.init(current_module)
        where_am_i_panel.init(current_module)
        appearance_panel.init(current_module)
        fMRI_panel.init(current_module)
        render_panel.init(current_module)
        listener_panel.init(current_module)
        data_panel.init(current_module)
        selection_panel.init(current_module)
        vertex_data_panel.init(current_module)

        bpy.utils.register_class(UpdateAppearance)
        bpy.utils.register_class(Filtering)
        bpy.utils.register_class(FindCurveClosestToCursor)
        bpy.utils.register_class(GrabFromFiltering)
        bpy.utils.register_class(GrabToFiltering)
        bpy.utils.register_class(ClearFiltering)

        bpy.utils.register_class(TransparencyPanel)
        bpy.utils.register_class(ShowHideObjectsPanel)
        bpy.utils.register_class(FilteringMakerPanel)
    except:
        print('The classes are already registered!')
        print(traceback.format_exc())


if __name__ == "__main__":
    main()

# ###############################################################
