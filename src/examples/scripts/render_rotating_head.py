import bpy
import os.path as op
import numpy as np


def run(mmvt):
    mu = mmvt.utils
    from_to = bpy.context.scene.render_rot_head_to - bpy.context.scene.render_rot_head_from
    output_fol = mu.make_dir(op.join(mu.get_user_fol(), 'figures', bpy.context.scene.render_rot_head_type))

    mmvt.appearance.show_hide_meg_sensors(bpy.context.scene.render_rot_head_type in ['meg_helmet', 'meg_helmet_source'])
    mmvt.appearance.show_hide_eeg_sensors(bpy.context.scene.render_rot_head_type in ['eeg_helmet', 'eeg_helmet_source'])
    mmvt.appearance.show_hide_electrodes(False)
    mmvt.appearance.show_hide_connections(False)
    mmvt.show_hide.show_hemis()
    mmvt.show_hide.show_subcorticals()
    mmvt.show_hide.show_coronal(show_frontal=True)
    mmvt.show_hide.show_head()

    mmvt.coloring.set_lower_threshold(2)
    mmvt.coloring.set_current_time(0)
    mmvt.colorbar.set_colorbar_min_max(
        -bpy.context.scene.render_rot_head_cb_min, bpy.context.scene.render_rot_head_cb_max)
    mmvt.colorbar.set_colormap(bpy.context.scene.render_rot_head_cm)

    mmvt.render.set_output_path(output_fol)
    mmvt.render.set_render_quality(10)

    mmvt.transparency.set_brain_transparency(0)
    mmvt.transparency.set_head_transparency(0.5)

    if from_to > 360:
        dz = 360 / from_to * bpy.context.scene.render_rot_head_dt
        mu.write_to_stderr('Setting dz to {}'.format(dz))
        mmvt.show_hide.set_rotate_brain(dz=dz)
        mmvt.play.render_movie(
            bpy.context.scene.render_rot_head_type, bpy.context.scene.render_rot_head_from,
            bpy.context.scene.render_rot_head_to, play_dt=bpy.context.scene.render_rot_head_dt, rotate_brain=True)
    else:
        plot_chunks = np.array_split(np.arange(0, 360, 1), from_to)
        for plot_chunk in plot_chunks:
            for ind, frame in enumerate(plot_chunk):
                if ind == 0:
                    mmvt.play.plot_something(cur_frame=frame)
                bpy.context.scene.frame_current = frame
                mmvt.render.render_image('{}_{}.{}'.format(
                    bpy.context.scene.render_rot_head_type, frame, mmvt.render.get_figure_format()))
                mmvt.render.camera_mode('ORTHO')
                mmvt.show_hide.rotate_brain(0, 0, 1)
                mmvt.render.camera_mode('CAMERA')


items_names = [
    ("meg", "MEG activity"),("meg_helmet", "MEG helmet"), ("eeg_helmet", "EEG helmet"),
    ('meg_helmet_source', 'MEG helmet & source'), ('eeg_helmet_source', 'EEG helmet & source')]
items = [(n[0], n[1], '', ind) for ind, n in enumerate(items_names)]
bpy.types.Scene.render_rot_head_type = bpy.props.EnumProperty(items=items)
# bpy.types.Scene.render_rot_head_type = bpy.props.StringProperty(default='meg_helmet_source')
bpy.types.Scene.render_rot_head_from = bpy.props.IntProperty(min=0, default=0)
bpy.types.Scene.render_rot_head_to = bpy.props.IntProperty(min=0, default=0)
bpy.types.Scene.render_rot_head_dt = bpy.props.IntProperty(min=1, default=1)
bpy.types.Scene.render_rot_head_cb_min = bpy.props.FloatProperty(default=-1)
bpy.types.Scene.render_rot_head_cb_max = bpy.props.FloatProperty(default=1)
bpy.types.Scene.render_rot_head_cm = bpy.props.StringProperty(default='BuPu-YlOrRd')


def draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'render_rot_head_type', text='')
    layout.prop(context.scene, 'render_rot_head_from', text='from')
    layout.prop(context.scene, 'render_rot_head_to', text='to')
    layout.prop(context.scene, 'render_rot_head_dt', text='dt')
