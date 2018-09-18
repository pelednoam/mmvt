import numpy as np
import time
import os.path as op

try:
    import bpy
except:
    from src.mmvt_addon.mmvt_utils import empty_bpy as bpy


def run(mmvt):
    output_fol = mmvt.utils.make_dir(op.join(mmvt.utils.get_user_fol(), 'figures', 'electrodes_and_connectivity'))
    connections_dict = mmvt.connections.get_connections_data()
    T = connections_dict.con_values.shape[1]
    debug = True

    mmvt.appearance.show_hide_meg_sensors(False)
    mmvt.appearance.show_hide_eeg_sensors(False)
    mmvt.appearance.show_hide_electrodes(True)
    mmvt.appearance.show_hide_connections(True)
    mmvt.show_hide.hide_hemis()
    mmvt.show_hide.show_subcorticals()
    mmvt.show_hide.hide_cerebellum()
    mmvt.show_hide.show_coronal(show_frontal=True)
    if debug:
        mmvt.show_hide.hide_head()
    else:
        mmvt.show_hide.show_head()
    mmvt.coloring.clear_colors()

    mmvt.render.set_output_path(output_fol)
    mmvt.render.set_render_quality(60)

    mmvt.transparency.set_brain_transparency(0)
    mmvt.transparency.set_head_transparency(1)

    plot_chunks = np.array_split(np.arange(0, 360, 1), T)
    now, run = time.time(), 0
    for frame, plot_chunk in enumerate(plot_chunks):
        for ind, _ in enumerate(plot_chunk):
            mmvt.utils.time_to_go(now, run, 360, 1, do_write_to_stderr=True)
            mmvt.utils.write_to_stderr('run: {}/360, frame: {}/{}'.format(run, frame, T - 1))
            mmvt.set_current_time(frame)
            if ind == 0:
                mmvt.utils.write_to_stderr('plotting something for frame {}'.format(frame))
                mmvt.connections.plot_connections(threshold=0, data_minmax=0.3, calc_t=False)
                mmvt.coloring.color_electrodes(threshold=0, data_minmax=0.1, condition='diff')
            img_name = 'electrodes_and_connectivity_{}'.format(run)
            if not debug:
                mmvt.render.render_image('{}.{}'.format(img_name, mmvt.render.get_figure_format()))
                mmvt.render.camera_mode('ORTHO')
                mmvt.show_hide.rotate_brain(0, 0, 1)
                mmvt.render.camera_mode('CAMERA')
            else:
                mmvt.render.save_image(img_name, add_index_to_name=False, add_colorbar=False, cb_ticks_num=False)
                mmvt.show_hide.rotate_brain(0, 0, 1)
            run += 1

