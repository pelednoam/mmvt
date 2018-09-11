import bpy
import os.path as op


def run(mmvt):
    mu = mmvt.utils
    output_fol = mu.make_dir(op.join(mu.get_user_fol(), 'figures', 'meg_sensors_and_source'))
    dz = 360/ (bpy.context.scene.render_rot_head_to - bpy.context.scene.render_rot_head_from) * \
         bpy.context.scene.render_rot_head_dt
    mu.write_to_stderr('Setting dz to {}'.format(dz))

    mmvt.appearance.show_hide_meg_sensors(bpy.context.scene.render_rot_head_type in ['meg_helmet', 'meg_helmet_source'])
    mmvt.appearance.show_hide_eeg_sensors(bpy.context.scene.render_rot_head_type in ['eeg_helmet', 'eeg_helmet_source'])
    mmvt.appearance.show_hide_electrodes(False)
    mmvt.appearance.show_hide_connections(False)
    mmvt.show_hide.show_hemis()
    mmvt.show_hide.show_subcorticals()
    mmvt.show_hide.show_coronal(show_frontal=True)
    mmvt.show_hide.set_rotate_brain(dz=dz)
    mmvt.show_hide.show_head()

    mmvt.coloring.set_lower_threshold(2)
    mmvt.coloring.set_current_time(0)
    mmvt.colorbar.set_colorbar_min_max(-bpy.context.scene.render_rot_head_cb, bpy.context.scene.render_rot_head_cb)
    mmvt.colorbar.set_colormap('BuPu-YlOrRd')

    mmvt.render.set_output_path(output_fol)
    mmvt.render.set_render_quality(60)

    mmvt.transparency.set_brain_transparency(0)
    mmvt.transparency.set_head_transparency(0.5)

    mmvt.play.render_movie(
        bpy.context.scene.render_rot_head_type, bpy.context.scene.render_rot_head_from,
        bpy.context.scene.render_rot_head_to, play_dt=bpy.context.scene.render_rot_head_dt, rotate_brain=True)


bpy.types.Scene.render_rot_head_type = bpy.props.StringProperty(default='meg_helmet_source')
bpy.types.Scene.render_rot_head_from = bpy.props.IntProperty(min=0, default=0)
bpy.types.Scene.render_rot_head_to = bpy.props.IntProperty(min=0, default=0)
bpy.types.Scene.render_rot_head_dt = bpy.props.IntProperty(min=1, default=1)
bpy.types.Scene.render_rot_head_cb = bpy.props.FloatProperty(min=0, default=1)


def draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'render_rot_head_from', text='from')
    layout.prop(context.scene, 'render_rot_head_to', text='to')
    layout.prop(context.scene, 'render_rot_head_dt', text='dt')
