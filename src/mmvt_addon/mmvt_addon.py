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
import os
import os.path as op
import sys
import importlib
import traceback
import logging

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
import streaming_panel
importlib.reload(streaming_panel)
import colorbar_panel
importlib.reload(colorbar_panel)
# import load_results_panel
# importlib.reload(load_results_panel)
import pizco_panel
importlib.reload(pizco_panel)
import load_results_panel
importlib.reload(load_results_panel)
import list_panel
importlib.reload(list_panel)


print("mmvt addon started!")
# todo: should change that in the code!!!
# Should be here bpy.types.Scene.maximal_time_steps
# T = 2500
# bpy.types.Scene.maximal_time_steps = T

# LAYERS
(TARGET_LAYER, LIGHTS_LAYER, EMPTY_LAYER, BRAIN_EMPTY_LAYER, ROIS_LAYER, ACTIVITY_LAYER, INFLATED_ROIS_LAYER,
 INFLATED_ACTIVITY_LAYER, ELECTRODES_LAYER, CONNECTIONS_LAYER, EEG_LAYER, MEG_LAYER) = range(12)

bpy.types.Scene.python_cmd = bpy.props.StringProperty(name='python cmd', default='python')
settings = None
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Data links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import_brain = data_panel.import_brain
add_data_to_parent_obj = data_panel.add_data_to_parent_obj
add_data_to_brain = data_panel.add_data_to_brain
import_electrodes = data_panel.import_electrodes
eeg_data_and_meta = data_panel.eeg_data_and_meta
load_electrodes_data = data_panel.load_electrodes_data
load_electrodes_dists = data_panel.load_electrodes_dists
load_eeg_data = data_panel.load_eeg_data
load_meg_sensors_data = data_panel.load_meg_sensors_data
import_meg_sensors = data_panel.import_meg_sensors
add_data_to_meg_sensors = data_panel.add_data_to_meg_sensors
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Selection links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
select_brain_objects = selection_panel.select_brain_objects
select_all_connections = selection_panel.select_all_connections
select_all_electrodes = selection_panel.select_all_electrodes
select_all_eeg = selection_panel.select_all_eeg
select_only_subcorticals = selection_panel.select_only_subcorticals
select_all_rois = selection_panel.select_all_rois
select_all_meg_sensors = selection_panel.select_all_meg_sensors
deselect_all = selection_panel.deselect_all
set_selection_type = selection_panel.set_selection_type
conditions_diff = selection_panel.conditions_diff
both_conditions = selection_panel.both_conditions
spec_condition = selection_panel.spec_condition
fit_selection = selection_panel.fit_selection
select_roi = selection_panel.select_roi
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Coloring links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
object_coloring = coloring_panel.object_coloring
color_objects = coloring_panel.color_objects
get_obj_color = coloring_panel.get_obj_color
clear_subcortical_fmri_activity = coloring_panel.clear_subcortical_fmri_activity
clear_cortex = coloring_panel.clear_cortex
clear_object_vertex_colors = coloring_panel.clear_object_vertex_colors
color_objects_homogeneously = coloring_panel.color_objects_homogeneously
init_activity_map_coloring = coloring_panel.init_activity_map_coloring
load_faces_verts = coloring_panel.load_faces_verts
load_meg_subcortical_activity = coloring_panel.load_meg_subcortical_activity
activity_map_coloring = coloring_panel.activity_map_coloring
meg_labels_coloring = coloring_panel.meg_labels_coloring
labels_coloring_hemi = coloring_panel.labels_coloring_hemi
plot_activity = coloring_panel.plot_activity
fmri_subcortex_activity_color = coloring_panel.fmri_subcortex_activity_color
activity_map_obj_coloring = coloring_panel.activity_map_obj_coloring
color_manually = coloring_panel.color_manually
color_subcortical_region = coloring_panel.color_subcortical_region
clear_subcortical_regions = coloring_panel.clear_subcortical_regions
color_electrodes = coloring_panel.color_electrodes
color_electrodes_sources = coloring_panel.color_electrodes_sources
get_elecctrodes_sources = coloring_panel.get_elecctrodes_sources
clear_colors_from_parent_childrens = coloring_panel.clear_colors_from_parent_childrens
default_coloring = coloring_panel.default_coloring
get_fMRI_activity = coloring_panel.get_fMRI_activity
get_faces_verts = coloring_panel.get_faces_verts
clear_colors = coloring_panel.clear_colors
clear_and_recolor = coloring_panel.clear_and_recolor
set_threshold = coloring_panel.set_threshold
create_inflated_curv_coloring = coloring_panel.create_inflated_curv_coloring
color_eeg_helmet = coloring_panel.color_eeg_helmet
calc_colors = coloring_panel.calc_colors
init_meg_labels_coloring_type = coloring_panel.init_meg_labels_coloring_type
color_connections = coloring_panel.color_connections
plot_meg = coloring_panel.plot_meg
plot_stc_t = coloring_panel.plot_stc_t
plot_stc = coloring_panel.plot_stc
init_meg_activity_map = coloring_panel.init_meg_activity_map
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Filtering links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
find_obj_with_val = filter_panel.find_obj_with_val
filter_draw = filter_panel.filter_draw
clear_filtering = filter_panel.clear_filtering
de_select_electrode_and_sensor = filter_panel.de_select_electrode_and_sensor
filter_roi_func = filter_panel.filter_roi_func
filter_electrode_or_sensor = filter_panel.filter_electrode_or_sensor
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Rendering links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
load_camera = render_panel.load_camera
grab_camera = render_panel.grab_camera
set_render_quality = render_panel.set_render_quality
set_render_output_path = render_panel.set_render_output_path
set_render_smooth_figure = render_panel.set_render_smooth_figure
render_image = render_panel.render_image
render_lateral_medial_split_brain = render_panel.render_lateral_medial_split_brain
render_in_queue = render_panel.render_in_queue
finish_rendering = render_panel.finish_rendering
update_camera_files = render_panel.update_camera_files
set_background_color = render_panel.set_background_color
set_lighting = render_panel.set_lighting
get_rendering_in_the_background = render_panel.get_rendering_in_the_background
set_rendering_in_the_background = render_panel.set_rendering_in_the_background
init_rendering = render_panel.init_rendering
camera_mode = render_panel.camera_mode
save_image = render_panel.save_image
set_to_camera_view = render_panel.set_to_camera_view
exit_from_camera_view = render_panel.exit_from_camera_view
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Show Hide links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
show_hide_hierarchy = show_hide_panel.show_hide_hierarchy
show_hide_hemi = show_hide_panel.show_hide_hemi
rotate_brain = show_hide_panel.rotate_brain
start_rotating = show_hide_panel.start_rotating
stop_rotating = show_hide_panel.stop_rotating
zoom = show_hide_panel.zoom
view_all= show_hide_panel.view_all
show_hide_sub_corticals = show_hide_panel.show_hide_sub_corticals
hide_subcorticals = show_hide_panel.hide_subcorticals
show_subcorticals = show_hide_panel.show_subcorticals
show_sagital = show_hide_panel.show_sagital
show_coronal = show_hide_panel.show_coronal
show_axial = show_hide_panel.show_axial
split_view = show_hide_panel.split_view
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Appearance links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
setup_layers = appearance_panel.setup_layers
change_view3d = appearance_panel.change_view3d
show_rois = appearance_panel.show_rois
show_activity = appearance_panel.show_activity
show_electrodes = appearance_panel.show_electrodes
show_hide_electrodes = appearance_panel.show_electrodes
show_hide_eeg = appearance_panel.show_hide_eeg
show_hide_meg_sensors = appearance_panel.show_hide_meg_sensors
show_hide_connections = appearance_panel.show_hide_connections
change_to_rendered_brain = appearance_panel.change_to_rendered_brain
change_to_solid_brain = appearance_panel.change_to_solid_brain
make_brain_solid_or_transparent = appearance_panel.make_brain_solid_or_transparent
update_layers = appearance_panel.update_layers
update_solidity = appearance_panel.update_solidity
connections_visible = appearance_panel.connections_visible
show_hide_connections = appearance_panel.show_hide_connections
show_pial = appearance_panel.show_pial
show_inflated = appearance_panel.show_inflated
set_inflated_ratio = appearance_panel.set_inflated_ratio
get_inflated_ratio = appearance_panel.get_inflated_ratio
is_pial = appearance_panel.is_pial
is_inflated = appearance_panel.is_inflated
is_activity = appearance_panel.is_activity
is_rois = appearance_panel.is_rois
is_solid = appearance_panel.is_solid
is_rendered = appearance_panel.is_rendered
set_closest_vertex_and_mesh_to_cursor = appearance_panel.set_closest_vertex_and_mesh_to_cursor
get_closest_vertex_and_mesh_to_cursor = appearance_panel.get_closest_vertex_and_mesh_to_cursor
clear_closet_vertex_and_mesh_to_cursor = appearance_panel.clear_closet_vertex_and_mesh_to_cursor
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Play links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
set_play_type = play_panel.set_play_type
set_play_from = play_panel.set_play_from
set_play_to = play_panel.set_play_to
set_play_dt = play_panel.set_play_dt
capture_graph = play_panel.capture_graph
render_movie = play_panel.render_movie
get_current_t = play_panel.get_current_t
set_current_t = play_panel.set_current_t
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ electrodes links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
color_the_relevant_lables = electrodes_panel.color_the_relevant_lables
get_leads = electrodes_panel.get_leads
get_lead_electrodes = electrodes_panel.get_lead_electrodes
set_current_electrode = electrodes_panel.set_current_electrode
set_electrodes_labeling_file = electrodes_panel.set_electrodes_labeling_file
set_show_only_lead = electrodes_panel.set_show_only_lead
is_current_electrode_marked = electrodes_panel.is_current_electrode_marked
get_electrodes_names = electrodes_panel.get_electrodes_names
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ colorbar links~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
show_cb_in_render = colorbar_panel.show_cb_in_render
set_colorbar_max_min = colorbar_panel.set_colorbar_max_min
set_colorbar_max = colorbar_panel.set_colorbar_max
set_colorbar_min = colorbar_panel.set_colorbar_min
set_colorbar_title = colorbar_panel.set_colorbar_title
get_cm = colorbar_panel.get_cm
get_colorbar_max_min = colorbar_panel.get_colorbar_max_min
get_colorbar_max = colorbar_panel.get_colorbar_max
get_colorbar_min = colorbar_panel.get_colorbar_min
get_colorbar_title = colorbar_panel.get_colorbar_title
get_colormap_name = colorbar_panel.get_colormap_name
colorbar_values_are_locked = colorbar_panel.colorbar_values_are_locked
lock_colorbar_values = colorbar_panel.lock_colorbar_values
set_colormap = colorbar_panel.set_colormap
set_colorbar_prec = colorbar_panel.set_colorbar_prec
get_colorbar_prec = colorbar_panel.get_colorbar_prec
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ fMRI links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fMRI_clusters_files_exist = fMRI_panel.fMRI_clusters_files_exist
find_closest_cluster = fMRI_panel.find_closest_cluster
find_fmri_files_min_max = fMRI_panel.find_fmri_files_min_max
get_clusters_file_names = fMRI_panel.get_clusters_file_names
get_clusters_files = fMRI_panel.get_clusters_files
plot_all_blobs = fMRI_panel.plot_all_blobs
load_fmri_cluster = fMRI_panel.load_fmri_cluster
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ connections links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
connections_exist = connections_panel.connections_exist
connections_data = connections_panel.connections_data
plot_connections = connections_panel.plot_connections
vertices_selected = connections_panel.vertices_selected
create_connections = connections_panel.create_connections
filter_nodes = connections_panel.filter_nodes
get_connections_parent_name = connections_panel.get_connections_parent_name
select_connection = connections_panel.select_connection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ utils links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
view_all_in_graph_editor = mmvt_utils.view_all_in_graph_editor
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ transparency links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
set_brain_transparency = transparency_panel.set_brain_transparency
set_light_layers_depth = transparency_panel.set_light_layers_depth
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ where_am_i_panel links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
set_ras_coo = where_am_i_panel.set_ras_coo
set_tkreg_ras_coo = where_am_i_panel.set_tkreg_ras_coo
find_closest_obj = where_am_i_panel.find_closest_obj
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ vertex_data_panel links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
find_vertex_index_and_mesh_closest_to_cursor = vertex_data_panel.find_vertex_index_and_mesh_closest_to_cursor
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ freeview_panel links ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
save_cursor_position = freeview_panel.save_cursor_position


def get_max_time_steps():
    # Check if there is animation data in MEG
    # try:
    #     return bpy.context.scene.maximal_time_steps
    # except:
    #     print('No preperty maximal_time_steps in bpy.types.Scene')
    found = False
    try:
        hemi = bpy.data.objects['Cortex-lh']
        # Takes the first child first condition fcurve
        fcurves = hemi.children[0].animation_data.action.fcurves[0]
        bpy.types.Scene.maximal_time_steps = len(fcurves.keyframe_points) - 3
        found = True
    except:
        print('No MEG data')

    if not found:
        try:
            fcurves = bpy.data.objects['fMRI'].animation_data.action.fcurves[0]
            bpy.types.Scene.maximal_time_steps = len(fcurves.keyframe_points) - 2
            found = True
        except:
            print('No dynamic fMRI data')

    if not found:
        try:
            parent_obj = bpy.data.objects['Deep_electrodes']
            if not parent_obj.animation_data is None:
                fcurves = parent_obj.animation_data.action.fcurves
            else:
                fcurves = parent_obj.children[1].animation_data.action.fcurves
            # else:
            #     fcurves = parent_obj.children[0].animation_data.action.fcurves[0]
            bpy.types.Scene.maximal_time_steps = len(fcurves[0].keyframe_points) - 2
            found = True
        except:
            print('No deep electrodes data')

    try:
        if found:
            print('max time steps: {}'.format(bpy.types.Scene.maximal_time_steps))
            return bpy.types.Scene.maximal_time_steps
    except:
        print('No preperty maximal_time_steps in bpy.types.Scene')

    # Bad fallback...
    return 2500


_listener_in_queue, _listener_out_queue = None, None


def start_listener():
    cmd = 'python {}'.format(op.join(mmvt_utils.current_path(), 'addon_listener.py'))
    listener_in_queue, listener_out_queue = mmvt_utils.run_command_in_new_thread(cmd)
    return listener_in_queue, listener_out_queue


# def init_pizco(mmvt):
#     try:
#         from pizco import Server
#         mmvt.c = mmvt_utils.get_graph_context()
#         mmvt.s = mmvt.c['scene']
#         Server(mmvt, 'tcp://127.0.0.1:8000')
#     except:
#         print('No pizco')


def make_all_fcurve_visible():
    for obj in bpy.data.objects:
        try:
            for cur_fcurve in obj.animation_data.action.fcurves:
                cur_fcurve.hide = False
        except:
            pass


def init(addon_prefs):
    global settings
    run_faulthandler()
    set_play_to(get_max_time_steps())
    mmvt_utils.view_all_in_graph_editor(bpy.context)
    bpy.context.window.screen = bpy.data.screens['Neuro']
    bpy.context.scene.atlas = mmvt_utils.get_atlas()
    bpy.context.scene.python_cmd = addon_prefs.python_cmd
    # bpy.data.screens['Neuro'].areas[1].spaces[0].region_3d.view_rotation = [1, 0, 0, 0]
    make_all_fcurve_visible()
    # set default values
    figures_fol = op.join(mmvt_utils.get_user_fol(), 'figures')
    mmvt_utils.make_dir(figures_fol)
    set_render_output_path(figures_fol)
    set_render_quality(60)
    mmvt_utils.set_show_textured_solid()
    mmvt_utils.hide_relationship_lines()
    code_fol = mmvt_utils.get_parent_fol(mmvt_utils.get_parent_fol())
    settings = mmvt_utils.read_config_ini()
    os.chdir(code_fol)


def run_faulthandler():
    import faulthandler
    logs = op.join(mmvt_utils.get_user_fol(), 'logs')
    mmvt_utils.make_dir(logs)
    fault_handler = open(op.join(logs, 'faulthandler_{}.txt'.format(mmvt_utils.rand_letters(5))), 'w')
    faulthandler.enable(fault_handler)


@mmvt_utils.tryit()
def fix_scale():
    for hemi in mmvt_utils.HEMIS:
        hemi_obj = bpy.data.objects[hemi]
        for i in range(3):
            hemi_obj.scale[i] = 0.1
        for label_obj in bpy.data.objects['Cortex-{}'.format(hemi)].children:
            for i in range(3):
                label_obj.scale[i] = 0.1
    for sub_obj in bpy.data.objects['Subcortical_structures'].children:
        for i in range(3):
            sub_obj.scale[i] = 0.1


def main(addon_prefs=None):
    init(addon_prefs)
    try:
        mmvt = sys.modules[__name__]
        appearance_panel.init(mmvt)
        show_hide_panel.init(mmvt)
        selection_panel.init(mmvt)
        coloring_panel.init(mmvt)
        electrodes_panel.init(mmvt)
        play_panel.init(mmvt)
        filter_panel.init(mmvt)
        freeview_panel.init(mmvt, addon_prefs)
        render_panel.init(mmvt)
        fMRI_panel.init(mmvt)
        streaming_panel.init(mmvt)
        colorbar_panel.init(mmvt)
        search_panel.init(mmvt)
        transparency_panel.init(mmvt)
        where_am_i_panel.init(mmvt)
        data_panel.init(mmvt)
        stim_panel.init(mmvt)
        dti_panel.init(mmvt)
        connections_panel.init(mmvt)
        vertex_data_panel.init(mmvt)
        load_results_panel.init(mmvt)
        pizco_panel.init(mmvt)
        # list_panel.init(mmvt)
        # _listener_in_queue, _listener__out_queue = start_listener()
        # listener_panel.init(mmvt)
        pass
    except:
        print('The classes are already registered!')
        print(traceback.format_exc())

    fix_scale()
    split_view(0)
    show_electrodes(False)
    show_hide_connections(False)
    # show_pial()
    mmvt_utils.select_layer(BRAIN_EMPTY_LAYER, False)
    mmvt_utils.unfilter_graph_editor()


if __name__ == "__main__":
    main()


