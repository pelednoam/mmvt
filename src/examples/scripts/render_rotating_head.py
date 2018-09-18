try:
    import bpy
except:
    from src.mmvt_addon.mmvt_utils import empty_bpy as bpy

import os.path as op
import numpy as np
import time


def run(mmvt):
    render_rot_head_to = mmvt.scripts.get_param('render_rot_head_to')
    render_rot_head_from = mmvt.scripts.get_param('render_rot_head_from')
    render_rot_head_type = mmvt.scripts.get_param('render_rot_head_type')
    render_rot_head_dt = mmvt.scripts.get_param('render_rot_head_dt')

    from_to = (render_rot_head_to - render_rot_head_from + 1)
    output_fol = mmvt.utils.make_dir(op.join(mmvt.utils.get_user_fol(), 'figures', render_rot_head_type))
    mmvt.utils.write_to_stderr('output fol: {}'.format(output_fol))

    mmvt.appearance.show_hide_meg_sensors(render_rot_head_type in ['meg_helmet', 'meg_helmet_source'])
    mmvt.appearance.show_hide_eeg_sensors(render_rot_head_type in ['eeg_helmet', 'eeg_helmet_source'])
    mmvt.appearance.show_hide_electrodes(False)
    mmvt.appearance.show_hide_connections(False)
    mmvt.show_hide.show_hemis()
    mmvt.show_hide.show_subcorticals()
    mmvt.show_hide.show_coronal(show_frontal=True)
    mmvt.show_hide.show_head()
    mmvt.coloring.clear_colors()

    mmvt.coloring.set_lower_threshold(2)
    mmvt.coloring.set_current_time(0)
    # mmvt.colorbar.set_colorbar_min_max(
    #     -bpy.context.scene.render_rot_head_cb_min, bpy.context.scene.render_rot_head_cb_max)
    # mmvt.colorbar.set_colormap(bpy.context.scene.render_rot_head_cm)

    mmvt.render.set_output_path(output_fol)
    mmvt.render.set_render_quality(60)

    mmvt.transparency.set_brain_transparency(0)
    if render_rot_head_type in ['meg']:
        mmvt.transparency.set_head_transparency(1)
    else:
        mmvt.transparency.set_head_transparency(0.5)

    if from_to > 360:
        dz = 360 / from_to * render_rot_head_dt
        mmvt.utils.write_to_stderr('Setting dz to {}'.format(dz))
        mmvt.show_hide.set_rotate_brain(dz=dz)
        mmvt.play.render_movie(
            render_rot_head_type, render_rot_head_from,
            render_rot_head_to, play_dt=render_rot_head_dt, rotate_brain=True)
    else:
        plot_chunks = np.array_split(np.arange(0, 360, 1), from_to)
        now, run = time.time(), 0
        for frame, plot_chunk in enumerate(plot_chunks):
            for ind, _ in enumerate(plot_chunk):
                mmvt.utils.time_to_go(now, run, 360, 1, do_write_to_stderr=True)
                mmvt.utils.write_to_stderr('run: {}/360, frame: {}/{}'.format(run, frame, from_to - 1))
                if ind == 0:
                    mmvt.utils.write_to_stderr('plotting something for frame {}'.format(frame))
                    mmvt.play.plot_something(play_type=render_rot_head_type, cur_frame=frame)
                mmvt.set_current_time(frame)
                mmvt.render.render_image('{}_{}.{}'.format(
                    render_rot_head_type, run, mmvt.render.get_figure_format()))
                mmvt.render.camera_mode('ORTHO')
                mmvt.show_hide.rotate_brain(0, 0, 1)
                mmvt.render.camera_mode('CAMERA')
                run += 1


def draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'render_rot_head_type', text='')
    if bpy.context.scene.render_rot_head_type in ['meg', 'meg_helmet_source']:
        layout.prop(context.scene, 'meg_files', '')
    if bpy.context.scene.render_rot_head_type in ['meg_helmet', 'meg_helmet_source']:
        layout.prop(context.scene, 'meg_sensors_files', text='')
        layout.prop(context.scene, 'meg_sensors_types', text='')
        layout.prop(context.scene, "meg_sensors_conditions", text="")
    layout.prop(context.scene, 'render_rot_head_from', text='from')
    layout.prop(context.scene, 'render_rot_head_to', text='to')
    layout.prop(context.scene, 'render_rot_head_dt', text='dt')


def init(mmvt):
    items_names = [
        ("meg", "MEG activity"), ("meg_helmet", "MEG helmet"), ("eeg_helmet", "EEG helmet"),
        ('meg_helmet_source', 'MEG helmet & source'), ('eeg_helmet_source', 'EEG helmet & source')]
    items = [(n[0], n[1], '', ind) for ind, n in enumerate(items_names)]
    bpy.types.Scene.render_rot_head_type = bpy.props.EnumProperty(items=items)


bpy.types.Scene.render_rot_head_type = bpy.props.EnumProperty(items=[])
bpy.types.Scene.render_rot_head_from = bpy.props.IntProperty(min=0, default=0)
bpy.types.Scene.render_rot_head_to = bpy.props.IntProperty(min=0, default=0)
bpy.types.Scene.render_rot_head_dt = bpy.props.IntProperty(min=1, default=1)


def set_param(param_name, val):
    bpy.context.scene[param_name] = val


def get_param(param_name):
    return bpy.context.scene[param_name]


def set_render_rot_head_type(val):
    bpy.context.scene.render_rot_head_type = val


def get_render_rot_head_type():
    return bpy.context.scene.render_rot_head_type


def set_render_rot_head_from(val):
    bpy.context.scene.render_rot_head_from = val


def get_render_rot_head_from():
    return bpy.context.scene.render_rot_head_from


def set_render_rot_head_to(val):
    bpy.context.scene.render_rot_head_to = val


def get_render_rot_head_to():
    return bpy.context.scene.render_rot_head_to


def set_render_rot_head_dt(val):
    bpy.context.scene.render_rot_head_dt = val


def get_render_rot_head_dt():
    return bpy.context.scene.render_rot_head_dt


if __name__ == '__main__':
    '''
    python -m src.mmvt_addon.scripts.run_a_script -s matt_hibert --script_name render_rotating_head --back 0 --script_params 
    'render_rot_head_type:meg_helmet_source,render_rot_head_from:0,render_rot_head_to:24,render_rot_head_dt:1,
    meg_sensors_files:meg_audvis_10_sensors_evoked_data,meg_sensors_types:mag,meg_sensors_conditions:RV,meg_files:matt_hibert_audvis_RV_dSPM_10'
    '''
    from src.mmvt_addon.scripts import run_mmvt

    subject, atlas = 'matt_hibert', 'dkt'
    mmvt = run_mmvt.run(subject, atlas, debug=False, run_blender=False)
    output_fol = mmvt.utils.make_dir(op.join(mmvt.utils.get_user_fol(), 'figures', 'xxxx'))

    mmvt.scripts.set_script('render_rotating_head')

    # Set the current script's params
    mmvt.scripts.set_param('render_rot_head_type', 'meg_helmet_source')
    mmvt.scripts.set_param('render_rot_head_from', 0)
    mmvt.scripts.set_param('render_rot_head_to', 24)
    mmvt.scripts.set_param('render_rot_head_dt', 1)

    # Setting the MEG params
    mmvt.coloring.set_meg_files('matt_hibert_audvis_RV_dSPM_10')
    mmvt.coloring.set_meg_sensors_files('meg_audvis_10_sensors_evoked_data')
    mmvt.coloring.set_meg_sensors_types('mag')
    mmvt.coloring.set_meg_sensors_conditions('RV')
    run(mmvt)