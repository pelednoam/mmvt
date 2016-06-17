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
import show_hide_panel
importlib.reload(show_hide_panel)
import transparency_panel
importlib.reload(transparency_panel)
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
import stim_panel
importlib.reload(stim_panel)

print("MMVT addon started!")
# todo: should change that in the code!!!
# Should be here bpy.types.Scene.maximal_time_steps
T = 2500

# LAYERS
(CONNECTIONS_LAYER, ELECTRODES_LAYER, ROIS_LAYER, ACTIVITY_LAYER, LIGHTS_LAYER,
    BRAIN_EMPTY_LAYER, EMPTY_LAYER) = 3, 1, 10, 11, 12, 5, 14

bpy.types.Scene.python_cmd = bpy.props.StringProperty(name='python cmd', default='python')

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
color_electrodes_sources = coloring_panel.color_electrodes_sources
get_elecctrodes_sources = coloring_panel.get_elecctrodes_sources
clear_colors_from_parent_childrens = coloring_panel.clear_colors_from_parent_childrens
default_coloring = coloring_panel.default_coloring
get_fMRI_activity = coloring_panel.get_fMRI_activity
get_faces_verts = coloring_panel.get_faces_verts
clear_colors = coloring_panel.clear_colors
clear_and_recolor = coloring_panel.clear_and_recolor
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Filtering links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
find_obj_with_val = filter_panel.find_obj_with_val
filter_draw = filter_panel.filter_draw
clear_filtering = filter_panel.clear_filtering
de_select_electrode = filter_panel.de_select_electrode
filter_roi_func = filter_panel.filter_roi_func
filter_electrode_func = filter_panel.filter_electrode_func
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Rendering links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
render_image = render_panel.render_image
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Show Hide links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
show_hide_hierarchy = show_hide_panel.show_hide_hierarchy
show_hide_hemi = show_hide_panel.show_hide_hemi
# show_hide_rh = show_hide_panel.show_hide_rh
# show_hide_lh = show_hide_panel.show_hide_lh
show_hide_sub_corticals = show_hide_panel.show_hide_sub_corticals
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Appearance links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
setup_layers = appearance_panel.setup_layers
change_view3d = appearance_panel.change_view3d
show_rois = appearance_panel.show_rois
show_activity = appearance_panel.show_activity
show_electrodes = appearance_panel.show_electrodes
show_connections = appearance_panel.show_connections
change_to_rendered_brain = appearance_panel.change_to_rendered_brain
change_to_solid_brain = appearance_panel.change_to_solid_brain
make_brain_solid_or_transparent = appearance_panel.make_brain_solid_or_transparent
update_layers = appearance_panel.update_layers
update_solidity = appearance_panel.update_solidity


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


_listener_in_queue, _listener_out_queue = None, None
def start_listener():
    cmd = 'python {}'.format(op.join(mmvt_utils.current_path(), 'addon_listener.py'))
    listener_in_queue, listener_out_queue = mmvt_utils.run_command_in_new_thread(cmd)
    return listener_in_queue, listener_out_queue


def main(addon_prefs=None):
    show_activity()
    show_electrodes(False)
    show_connections(False)
    bpy.context.scene.atlas = mmvt_utils.get_atlas()
    bpy.context.scene.python_cmd = addon_prefs.python_cmd
    code_fol = mmvt_utils.get_parent_fol(mmvt_utils.get_parent_fol())
    os.chdir(code_fol)

    try:
        # _listener_in_queue, _listener__out_queue = start_listener()
        current_module = sys.modules[__name__]
        show_hide_panel.init(current_module)
        transparency_panel.init(current_module)
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
        filter_panel.init(current_module)
        stim_panel.init(current_module)
    except:
        print('The classes are already registered!')
        print(traceback.format_exc())
    # setup_layers()

if __name__ == "__main__":
    main()

