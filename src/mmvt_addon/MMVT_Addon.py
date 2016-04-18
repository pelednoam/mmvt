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
import filter_panel
importlib.reload(filter_panel)

print("Neuroscience add on started!")
# todo: should change that in the code!!!
# Should be here bpy.types.Scene.maximal_time_steps
T = 2500

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
get_obj_color = coloring_panel.get_obj_color
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
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Filtering links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
find_obj_with_val = filter_panel.find_obj_with_val
filter_draw = filter_panel.filter_draw
clear_filtering = filter_panel.clear_filtering
de_select_electrode = filter_panel.de_select_electrode
filter_roi_func = filter_panel.filter_roi_func
filter_electrode_func = filter_panel.filter_electrode_func
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Filtering links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Rendering links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
render_image = render_panel.render_image
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Rendering links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    return 2500

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
    # todo: might cause some problems in the future
    show_hide_hierarchy(True, "Subcortical_fmri_activity_map")
    show_hide_hierarchy(True if not val else False, "Subcortical_meg_activity_map")


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
show_activity_layer = appearance_panel.show_activity_layer
# get_appearance_show_electrodes_layer = appearance_panel.get_appearance_show_electrodes_layer
# set_appearance_show_electrodes_layer = appearance_panel.set_appearance_show_electrodes_layer
# get_appearance_show_activity_layer = appearance_panel.get_appearance_show_activity_layer
# set_appearance_show_activity_layer = appearance_panel.set_appearance_show_activity_layer
# get_appearance_show_rois_layer = appearance_panel.get_appearance_show_rois_layer
# set_appearance_show_rois_layer = appearance_panel.set_appearance_show_rois_layer
# get_appearance_show_connections_layer = appearance_panel.get_appearance_show_connections_layer
# set_appearance_show_connections_layer = appearance_panel.set_appearance_show_connections_layer
# get_filter_view_type = appearance_panel.get_filter_view_type
# set_filter_view_type = appearance_panel.set_filter_view_type

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
    if context.scene.filter_view_type == 'rendered' and bpy.context.scene.appearance_show_rois_activity == 'activity':
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
        # if context.scene.filter_view_type == 'rendered' and bpy.context.scene.appearance_show_activity_layer is True:
        make_brain_solid_or_transparent()
        update_layers()
        # else:
        #     self.report({'ERROR'}, 'You should change the view to Rendered Brain first.')
        return {"FINISHED"}

bpy.types.Scene.appearance_solid_slider = bpy.props.FloatProperty(default=0.0, min=0, max=1, description="")
bpy.types.Scene.appearance_depth_slider = bpy.props.IntProperty(default=1, min=1, max=10, description="")
bpy.types.Scene.appearance_depth_Bool = bpy.props.BoolProperty(default=False, description="")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Transparency Panel ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

_listener_in_queue, _listener__out_queue = None, None
def start_listener():
    cmd = 'python {}'.format(op.join(mmvt_utils.current_path(), 'addon_listener.py'))
    listener_in_queue, listener__out_queue = mmvt_utils.run_command_in_new_thread(cmd)
    return listener_in_queue, listener__out_queue


def main(addon_prefs=None):
    bpy.context.scene.appearance_show_electrodes_layer = False
    bpy.context.scene.appearance_show_activity_layer = False
    bpy.context.scene.appearance_show_ROIs_layer = True
    bpy.context.scene.appearance_show_connections_layer = False
    try:
        # _listener_in_queue, _listener__out_queue = start_listener()
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
        fMRI_panel.init(current_module, addon_prefs)
        render_panel.init(current_module)
        listener_panel.init(current_module)
        data_panel.init(current_module)
        selection_panel.init(current_module)
        vertex_data_panel.init(current_module)
        filter_panel.init(current_module)

        bpy.utils.register_class(UpdateAppearance)

        bpy.utils.register_class(TransparencyPanel)
        bpy.utils.register_class(ShowHideObjectsPanel)
    except:
        print('The classes are already registered!')
        print(traceback.format_exc())
    # setup_layers()

if __name__ == "__main__":
    main()

# ###############################################################
