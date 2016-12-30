import bpy
import mmvt_utils as mu
import sys
import os.path as op
import time
import numpy as np
import traceback
from itertools import cycle
from datetime import datetime
from queue import Queue


def _addon():
    return StreamingPanel.addon


def change_graph_all_vals_thread(q, while_termination_func, **kargs):
    buffer = kargs['buffer']
    change_graph_all_vals(buffer)


def change_graph_reverse(mat):
    T = min(mat.shape[1], _addon().get_max_time_steps())
    for elc_ind, elc_name in enumerate(StreamingPanel.electrodes_names):
        bpy.data.objects[elc_name].select = True
        parent_obj = bpy.data.objects[elc_name]
        fcurve = parent_obj.animation_data.action.fcurves[0]
        N = len(fcurve.keyframe_points)
        for ind in range(N - 1, T - 1, -1):
            fcurve.keyframe_points[ind].co[1] = fcurve.keyframe_points[ind - T].co[1]
        for ind in range(T):
            fcurve.keyframe_points[ind].co[1] = mat[elc_ind][ind]
        fcurve.keyframe_points[N - 1].co[1] = 0
        fcurve.keyframe_points[0].co[1] = 0

#@mu.profileit()
# @mu.timeit
def change_graph_all_vals(mat, condition='interference'):
    # print('change_graph_all_vals', str(datetime.now() - StreamingPanel.time))
    # StreamingPanel.time = datetime.now()
    # print('change_graph_all_vals start: ', datetime.now())
    MAX_STEPS = StreamingPanel.max_steps
    T = min(mat.shape[1], MAX_STEPS)
    for elc_ind, elc_name in enumerate(StreamingPanel.electrodes_names):
        parent_obj = bpy.data.objects[elc_name]
        fcurve = parent_obj.animation_data.action.fcurves[0]
        curr_t = bpy.context.scene.frame_current
        for ind in range(T):
            t = curr_t + ind
            if t > MAX_STEPS - 1:
                t = ind
            fcurve.keyframe_points[t].co[1] = mat[elc_ind][ind]
        fcurve.keyframe_points[MAX_STEPS - 1].co[1] = 0
        fcurve.keyframe_points[0].co[1] = 0

    bpy.context.scene.frame_current += mat.shape[1]
    if bpy.context.scene.frame_current > MAX_STEPS - 1:
        bpy.context.scene.frame_current = bpy.context.scene.frame_current - MAX_STEPS
        time_diff = (datetime.now() - StreamingPanel.time)
        time_diff_sec = time_diff.seconds + time_diff.microseconds * 1e-6
        print('cycle! ', str(time_diff), time_diff_sec)
        if time_diff_sec < 2.5:
            print('sleep for {}'.format(2.5 - time_diff_sec))
            time.sleep(2.5 - time_diff_sec)
        StreamingPanel.time = datetime.now()


def show_electrodes_fcurves():
    for elc_ind, elc_name in enumerate(StreamingPanel.electrodes_names):
        bpy.data.objects[elc_name].select = True
        parent_obj = bpy.data.objects[elc_name]
        # curve_name = '{}_{}'.format(elc_name, condition)
        for f_ind, fcurve in enumerate(parent_obj.animation_data.action.fcurves):
            # if mu.get_fcurve_name(fcurve) != curve_name:
            fcurve.hide = f_ind != 0
            # if f_ind == 0:
            #     fcurve.hide = True
            #     continue



def change_color(obj, val, data_min, colors_ratio):
    colors_ind = calc_color_ind(val, data_min, colors_ratio)
    colors = StreamingPanel.cm[colors_ind]
    _addon().object_coloring(obj, colors)


def calc_color_ind(val, data_min, colors_ratio):
    colors_ind = int(((val - data_min) * colors_ratio))
    N = len(StreamingPanel.cm)
    if colors_ind < 0:
        colors_ind = 0
    if colors_ind > N - 1:
        colors_ind = N - 1
    return colors_ind


def reading_from_udp_while_termination_func():
    return StreamingPanel.is_streaming


def udp_reader(udp_queue, while_termination_func, **kargs):
    import socket
    buffer_size = kargs.get('buffer_size', 10)
    server = kargs.get('server', 'localhost')
    ip = kargs.get('ip', 10000)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = (server, ip)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(server_address)
    buffer = []
    prev_val = np.zeros((80, 1))

    while while_termination_func():
        # time.sleep(0.0001)
        now = time.time()
        try:
            sock.settimeout(0.0012)
            next_val = sock.recv(2048)
        except socket.timeout as e:
            if e.args[0] == 'timed out':
                # print('!!! timed out !!!')
                next_val = prev_val
            else:
                print('!!! {} !!!'.format(e))
                raise Exception(e)
        else:
            next_val = next_val.decode(sys.getfilesystemencoding(), 'ignore')
            next_val = np.array([float(f) for f in next_val.split(',')])
            next_val = next_val[..., np.newaxis]

        prev_val = next_val
        buffer = next_val if buffer == [] else np.hstack((buffer, next_val))
        if buffer.shape[1] >= buffer_size:
            # print('{} took {:.5f}s {}'.format('udp_reader', time.time() - now, buffer.shape[1]))
            # print('udp_reader: ', datetime.now())
            udp_queue.put(buffer)
            buffer = []


def color_electrodes(data):
    _addon().color_objects_homogeneously(
        np.mean(data, 1), StreamingPanel.electrodes_names, StreamingPanel.electrodes_conditions,
        StreamingPanel.data_min, StreamingPanel.electrodes_colors_ratio, threshold=0)


class StreamButton(bpy.types.Operator):
    bl_idname = "mmvt.stream_button"
    bl_label = "Stream botton"
    bl_options = {"UNDO"}

    _timer = None
    _time = time.time()
    _index = 0
    _obj = None
    _buffer = []
    _jobs = Queue()

    def invoke(self, context, event=None):
        StreamingPanel.is_streaming = not StreamingPanel.is_streaming
        _addon().set_colorbar_max_min(StreamingPanel.data_max, StreamingPanel.data_min)
        _addon().set_colorbar_title('Electordes Streaming Data')
        mu.show_only_render(True)
        if StreamingPanel.first_time:
            StreamingPanel.first_time = False
            try:
                context.window_manager.event_timer_remove(self._timer)
            except:
                pass
            context.window_manager.modal_handler_add(self)
            self._timer = context.window_manager.event_timer_add(0.01, context.window)
        if StreamingPanel.is_streaming:
            args = dict(buffer_size=bpy.context.scene.straming_buffer_size, server='localhost', ip=10000)
            StreamingPanel.udp_queue = mu.run_thread(udp_reader, reading_from_udp_while_termination_func, **args)

        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        # if time.time() - self._time > 0.1:
        #     self._time = time.time()
        #     job_data = mu.queue_get(self._jobs)
        #     if not job_data is None:
        #         # print(str(datetime.now() - StreamingPanel.time))
        #         # StreamingPanel.time = datetime.now()
        #         # print(job_data.shape)
        #         change_graph_all_vals(job_data)

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            StreamingPanel.is_streaming = False
            bpy.context.scene.update()
            self.cancel(context)
            return {'PASS_THROUGH'}

        if event.type == 'TIMER':
            if StreamingPanel.is_streaming and time.time() - self._time > bpy.context.scene.straming_buffer_size / 1000.0:
                self._time = time.time()
                data = mu.queue_get(StreamingPanel.udp_queue)
                if not data is None:
                    # self._jobs.put(data)
                    change_graph_all_vals(data)
                    color_electrodes(data)
                    # data = mu.queue_get(StreamingPanel.udp_queue)

        return {'PASS_THROUGH'}

    def cancel(self, context):
        try:
            context.window_manager.event_timer_remove(self._timer)
        except:
            pass
        self._timer = None
        return {'CANCELLED'}


def template_draw(self, context):
    layout = self.layout
    layout.prop(context.scene, "straming_buffer_size", text="buffer size:")
    # layout.operator(StreamListenerButton.bl_idname,
    #                 text="Start Listener" if not StreamingPanel.is_listening else 'Stop Listener',
    #                 icon='COLOR_GREEN' if not StreamingPanel.is_listening else 'COLOR_RED')
    layout.operator(StreamButton.bl_idname,
                    text="Stream data" if not StreamingPanel.is_streaming else 'Stop streaming data',
                    icon='COLOR_GREEN' if not StreamingPanel.is_streaming else 'COLOR_RED')


bpy.types.Scene.straming_buffer_size = bpy.props.IntProperty(default=100, min=10)


class StreamingPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "Stream"
    addon = None
    init = False
    is_streaming = False
    is_listening = False
    first_time = True
    # fixed_data = []
    udp_queue = None
    udp_viz_queue = None
    electrodes_file = None
    time = datetime.now()
    electrodes_names, electrodes_conditions = [], []
    data_max, data_min, electrodes_colors_ratio = 0, 0, 1

    def draw(self, context):
        if StreamingPanel.init:
            template_draw(self, context)


def init(addon):
    cm_fname = op.join(mu.file_fol(), 'color_maps', 'BuPu_YlOrRd.npy')
    if not op.isfile(cm_fname):
        return
    StreamingPanel.addon = addon
    StreamingPanel.is_listening = False
    StreamingPanel.is_streaming = False
    StreamingPanel.first_time = True
    register()
    StreamingPanel.cm = np.load(cm_fname)
    # StreamingPanel.fixed_data = fixed_data()
    electrodes_data, StreamingPanel.electrodes_names, StreamingPanel.electrodes_conditions = \
        _addon().load_electrodes_data()
    norm_percs = (3, 97) #todo: add to gui
    StreamingPanel.data_max, StreamingPanel.data_min = mu.get_data_max_min(
        electrodes_data, True, norm_percs=norm_percs, data_per_hemi=False, symmetric=True)
    StreamingPanel.electrodes_colors_ratio = 256 / (StreamingPanel.data_max - StreamingPanel.data_min)
    show_electrodes_fcurves()
    StreamingPanel.max_steps = _addon().get_max_time_steps()
    StreamingPanel.init = True


def register():
    try:
        unregister()
        bpy.utils.register_class(StreamingPanel)
        bpy.utils.register_class(StreamButton)
        # bpy.utils.register_class(StreamListenerButton)
    except:
        print("Can't register Stream Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(StreamingPanel)
        bpy.utils.unregister_class(StreamButton)
        # bpy.utils.unregister_class(StreamListenerButton)
    except:
        pass