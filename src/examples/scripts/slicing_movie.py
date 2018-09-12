import bpy
import numpy as np
import os.path as op
import time


def run(mmvt):
    mu = mmvt.utils
    output_fol = mu.make_dir(op.join(mu.get_user_fol(), 'figures', 'slicing_movie'))
    mmvt.render.set_output_path(output_fol)
    mmvt.render.set_render_quality(bpy.context.scene.quality)
    mmvt.transparency.set_brain_transparency(0)

    mmvt.appearance.show_hide_meg_sensors(False)
    mmvt.appearance.show_hide_eeg_sensors(False)
    mmvt.appearance.show_hide_connections(False)
    mmvt.show_hide.hide_head()
    mmvt.show_hide.show_hemis()
    mmvt.show_hide.show_subcorticals()
    mmvt.show_hide.hide_cerebellum()
    mmvt.show_hide.show_coronal(show_frontal=True)
    mmvt.show_hide.show_head()

    mmvt.transparency.set_brain_transparency(0)
    mmvt.transparency.set_head_transparency(1)

    # Show electrodes, plot groups coloring amd add electrodes leads
    mmvt.appearance.show_hide_electrodes(True)
    mmvt.coloring.color_manually('electrodes_groups_coloring')
    mmvt.electrodes.set_show_electrodes_groups_leads(True)

    coordinates = mu.get_cursor_location()
    bpy.context.scene.slicing_movie_from_y = coordinates[1]
    mmvt.where_am_i_panel.create_slices()
    mmvt.slicer.set_slices_plot_cross(False)
    mmvt.slicer.set_slicer_cut_type(mmvt.slicer.CUT_CORONAL)
    slicing_range = np.arange(bpy.context.scene.slicing_movie_from_y, bpy.context.scene.slicing_movie_to_y,
                              bpy.context.scene.slicing_movie_dy)
    now, N = time.time(), len(slicing_range)
    for run, y in enumerate(slicing_range):
        mu.time_to_go(now, run, N, 1, do_write_to_stderr=True)
        cut_pos = [0, y, 0]
        coordinates[1] = y
        mmvt.slicer.slice_brain(cut_pos, save_image=bpy.context.scene.slicing_save_movie, render_image=False)
        if bpy.context.scene.slicing_render_movie:
            mmvt.render.render_image('slicing_{}.{}'.format(
                run, mmvt.render.get_figure_format()), set_to_camera_mode=True)
        mmvt.slicer.clear_slice()
        mmvt.where_am_i_panel.create_slices(pos=coordinates)


bpy.types.Scene.slicing_movie_from_y = bpy.props.FloatProperty(default=8)
bpy.types.Scene.slicing_movie_to_y = bpy.props.FloatProperty(default=-9.2)
bpy.types.Scene.slicing_movie_dy = bpy.props.FloatProperty(default=-0.1)
bpy.types.Scene.slicing_render_movie = bpy.props.BoolProperty(default=False, description='Renders each frame')
bpy.types.Scene.slicing_save_movie = bpy.props.BoolProperty(default=False, description='Saves each frame')


def draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'quality', text='Rendering quality')
    layout.prop(context.scene, 'slicing_movie_from_y', text='from')
    layout.prop(context.scene, 'slicing_movie_to_y', text='to')
    layout.prop(context.scene, 'slicing_movie_dy', text='dy')
    layout.prop(context.scene, 'slicing_render_movie', text='Render frames')
    layout.prop(context.scene, 'slicing_save_movie', text='Save frames')

