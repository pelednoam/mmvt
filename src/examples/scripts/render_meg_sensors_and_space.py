import bpy
import os.path as op


def run(mmvt):
    mu = mmvt.utils
    output_fol = mu.make_dir(op.join(mu.get_user_fol(), 'figures', 'meg_sensors_and_source'))

    mmvt.appearance.show_hide_meg_sensors(True)
    mmvt.appearance.show_hide_eeg_sensors(False)
    mmvt.appearance.show_hide_electrodes(False)
    mmvt.appearance.show_hide_connections(False)
    mmvt.show_hide.show_hemis()
    mmvt.show_hide.show_subcorticals()
    mmvt.show_hide.show_coronal(show_frontal=True)
    mmvt.show_hide.set_rotate_brain(dz=0.8)
    mmvt.show_hide.show_head()

    mmvt.coloring.set_lower_threshold(2)
    mmvt.coloring.set_current_time(0)
    mmvt.colorbar.set_colorbar_min_max(-4, 4)
    mmvt.colorbar.set_colormap('BuPu-YlOrRd')

    mmvt.render.set_output_path(output_fol)
    mmvt.render.set_render_quality(60)

    mmvt.transparency.set_brain_transparency(0)
    mmvt.transparency.set_head_transparency(0.5)

    mmvt.play.render_movie(
        'meg_helmet_source', bpy.context.scene.render_meg_sas_from, bpy.context.scene.render_meg_sas_to,
        bpy.context.scene.render_meg_sas_dt, rotate_brain=True)


bpy.types.Scene.render_meg_sas_from = bpy.props.IntProperty(min=0, default=0)
bpy.types.Scene.render_meg_sas_to = bpy.props.IntProperty(min=0, default=0)
bpy.types.Scene.render_meg_sas_dt = bpy.props.IntProperty(min=1, default=1)


def draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'render_meg_sas_from', text='from')
    layout.prop(context.scene, 'render_meg_sas_to', text='to')
    layout.prop(context.scene, 'render_meg_sas_dt', text='dt')
