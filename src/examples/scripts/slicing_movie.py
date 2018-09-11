import bpy
import numpy as np
import os.path as op


def run(mmvt):
    mu = mmvt.utils
    output_fol = mu.make_dir(op.join(mu.get_user_fol(), 'figures', 'slicing_movie'))
    mmvt.render.set_output_path(output_fol)
    mmvt.render.set_render_quality(60)
    mmvt.transparency.set_brain_transparency(0)

    mmvt.appearance.show_hide_meg_sensors(False)
    mmvt.appearance.show_hide_eeg_sensors(False)
    mmvt.appearance.show_hide_electrodes(False)
    mmvt.appearance.show_hide_connections(False)
    mmvt.show_hide.hide_head()
    mmvt.show_hide.show_hemis()
    mmvt.show_hide.show_subcorticals()
    mmvt.show_hide.show_coronal(show_frontal=True)

    coordinates = mu.get_cursor_location()
    bpy.context.scene.slicing_movie_from_y = coordinates[1]
    mmvt.where_am_i_panel.create_slices()
    mmvt.slicer.set_slices_plot_cross(False)
    mmvt.slicer.set_slicer_cut_type(mmvt.slicer.CUT_CORONAL)
    for y in np.arange(bpy.context.scene.slicing_movie_from_y, bpy.context.scene.slicing_movie_to_y,
                       bpy.context.scene.slicing_movie_dy):
        cut_pos = [0, y, 0]
        coordinates[1] = y
        mmvt.slicer.slice_brain(cut_pos, save_image=bpy.context.scene.slicing_save_movie,
                                render_image=bpy.context.scene.slicing_render_movie)
        mmvt.slicer.clear_slice()
        mmvt.where_am_i_panel.create_slices(pos=coordinates)
        return

bpy.types.Scene.slicing_movie_from_y = bpy.props.FloatProperty(default=8)
bpy.types.Scene.slicing_movie_to_y = bpy.props.FloatProperty(default=-9.2)
bpy.types.Scene.slicing_movie_dy = bpy.props.FloatProperty(default=-0.1)
bpy.types.Scene.slicing_render_movie = bpy.props.BoolProperty(default=False, description='Renders each frame')
bpy.types.Scene.slicing_save_movie = bpy.props.BoolProperty(default=False, description='Saves each frame')


def draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'slicing_movie_from_y', text='from')
    layout.prop(context.scene, 'slicing_movie_to_y', text='to')
    layout.prop(context.scene, 'slicing_movie_dy', text='dz')
    layout.prop(context.scene, 'slicing_render_movie', text='Render frames')
    layout.prop(context.scene, 'slicing_save_movie', text='Save frames')

