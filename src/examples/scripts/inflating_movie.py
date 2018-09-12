import bpy
import numpy as np
import os.path as op
import time


def run(mmvt):
    mu = mmvt.utils
    output_fol = mu.make_dir(op.join(mu.get_user_fol(), 'figures', 'inflating_movie'))
    mmvt.render.set_output_path(output_fol)
    mmvt.render.set_render_quality(60)
    mmvt.transparency.set_brain_transparency(1)
    mmvt.transparency.set_layer_weight(0.2)

    mmvt.appearance.show_hide_meg_sensors(False)
    mmvt.appearance.show_hide_eeg_sensors(False)
    mmvt.appearance.show_hide_electrodes(False)
    mmvt.appearance.show_hide_connections(False)
    mmvt.show_hide.hide_head()
    mmvt.show_hide.show_hemis()
    mmvt.show_hide.show_subcorticals()
    mmvt.show_hide.show_coronal(show_frontal=True)

    if mmvt.appearance.flat_map_exists():
        inf_range = np.concatenate((np.arange(-1, 1, 0.01), np.arange(1, -1, -0.01)))
    else:
        inf_range = np.concatenate((np.arange(-1, 0, 0.01), np.arange(0, -1, -0.01)))
    dz = 360 / (len(inf_range))
    now, N = time.time(), len(inf_range)
    for run, inflating in enumerate(inf_range):
        mu.time_to_go(now, run, N, 1, do_write_to_stderr=True)
        mmvt.appearance.set_inflated_ratio(inflating)
        if bpy.context.scene.inflating_save_movie:
            mmvt.render.save_image('inflating', bpy.context.scene.save_selected_view)
        if bpy.context.scene.inflating_render_movie:
            mmvt.render.render_image('inflating_{}.{}'.format(
                run, mmvt.render.get_figure_format()), set_to_camera_mode=True)
        mmvt.render.camera_mode('ORTHO')
        mmvt.show_hide.rotate_brain(0, 0, dz)
        mmvt.render.camera_mode('CAMERA')


bpy.types.Scene.inflating_render_movie = bpy.props.BoolProperty(default=False, description='Renders each frame')
bpy.types.Scene.inflating_save_movie = bpy.props.BoolProperty(default=False, description='Saves each frame')


def draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'inflating_render_movie', text='Render frames')
    layout.prop(context.scene, 'inflating_save_movie', text='Save frames')

