import bpy
import mmvt_utils as mu
import sys
import os.path as op
import time
import numpy as np
import traceback
from itertools import cycle
from datetime import datetime


def _addon():
    return StreamingPanel.addon


def fixed_data():
    obj_name = 'LMF6'
    bpy.data.objects[obj_name].select = True
    _, y = mu.get_fcurve_values('LMF6', 'LMF6_interference')

    data_min, data_max = np.percentile(y, 3), np.percentile(y, 97)
    StreamingPanel.activity_colors_ratio = 256 / (data_max - data_min)
    StreamingPanel.activity_data_min = data_min

    y = cycle(y)
    for _y in y:
        yield _y
    # x = cycle(np.linspace(0, 10, 25000))
    # for _x in x:
    #     next = np.sin(2 * np.pi * _x)
    #     yield next


def change_graph(next_val):
    obj_name = 'LMF6'
    fcurve_name = 'LMF6_interference'
    bpy.data.objects[obj_name].select = True
    parent_obj = bpy.data.objects[obj_name]
    curves = [c for c in parent_obj.animation_data.action.fcurves if mu.get_fcurve_name(c) == fcurve_name]
    for fcurve in curves:
        N = len(fcurve.keyframe_points)
        for ind in range(N - 1, 0, -1):
            fcurve.keyframe_points[ind].co[1] = fcurve.keyframe_points[ind - 1].co[1]
        fcurve.keyframe_points[0].co[1] = next_val
    return next_val


def change_graph_vals(next_vals):
    print(str(datetime.now() - StreamingPanel.time))
    StreamingPanel.time = datetime.now()
    obj_name = 'LMF6'
    fcurve_name = 'LMF6_interference'
    bpy.data.objects[obj_name].select = True
    parent_obj = bpy.data.objects[obj_name]
    curves = [c for c in parent_obj.animation_data.action.fcurves if mu.get_fcurve_name(c) == fcurve_name]
    for fcurve in curves:
        N = len(fcurve.keyframe_points)
        for ind in range(N - 1, len(next_vals) - 1, -1):
            fcurve.keyframe_points[ind].co[1] = fcurve.keyframe_points[ind - len(next_vals)].co[1]
        for ind in range(len(next_vals)):
            fcurve.keyframe_points[ind].co[1] = next_vals[ind]
    return next_vals


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


def udp_reader(queue, while_termination_func, **kargs):
    import socket
    buffer_size = kargs.get('buffer_size', 10)
    server = kargs.get('server', 'localhost')
    ip = kargs.get('ip', 10000)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = (server, ip)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(server_address)
    buffer = []

    while while_termination_func():
        next_val = sock.recv(1024)
        next_val = next_val.decode(sys.getfilesystemencoding(), 'ignore')
        buffer.append(next_val)
        if len(buffer) >= buffer_size:
            queue.put(','.join(buffer))
            buffer = []

# class StreamListenerButton(bpy.types.Operator):
#     bl_idname = "mmvt.stream_listerner_button"
#     bl_label = "Stream Listener botton"
#     bl_options = {"UNDO"}
#
#     def invoke(self, context, event=None):
#         StreamingPanel.is_listening = not StreamingPanel.is_listening
#         if StreamingPanel.is_listening:
#             # script = 'src.udp.udp_listener'
#             # cmd = '{} -m {}'.format(bpy.context.scene.python_cmd, script)
#             # _, StreamingPanel.out_queue = mu.run_command_in_new_thread(
#             #     cmd, read_stderr=False, read_stdin=False, stdout_func=reading_from_usp_stdout_func)
#             StreamingPanel.out_queue = mu.run_thread(udp_reader, while_func=reading_from_udp_while_termination_func)
#         return {"FINISHED"}


class StreamButton(bpy.types.Operator):
    bl_idname = "mmvt.stream_button"
    bl_label = "Stream botton"
    bl_options = {"UNDO"}

    _timer = None
    _time = datetime.now()
    _index = 0
    # _time_step = 0.001
    _obj = None
    _buffer = []

    def invoke(self, context, event=None):
        self._obj = bpy.data.objects['LMF6']
        StreamingPanel.is_streaming = not StreamingPanel.is_streaming
        if StreamingPanel.first_time:
            StreamingPanel.first_time = False
            try:
                context.window_manager.event_timer_remove(self._timer)
            except:
                pass
            context.window_manager.modal_handler_add(self)
            self._timer = context.window_manager.event_timer_add(0.01, context.window)
        if StreamingPanel.is_streaming:
            args = dict(buffer_size=100, server='localhost', ip=10000)
            StreamingPanel.out_queue = mu.run_thread(udp_reader, reading_from_udp_while_termination_func, **args)

        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        # First frame initialization:

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            StreamingPanel.is_streaming = False
            bpy.context.scene.update()
            self.cancel(context)
            return {'PASS_THROUGH'}

        if event.type == 'TIMER':
            # print(str(datetime.now() - self._time))
            self._time = datetime.now()
            if not StreamingPanel.is_streaming:
                change_graph_vals(np.zeros(bpy.context.scene.straming_buffer_size))
            elif not StreamingPanel.out_queue is None:
                listener_stdout = mu.queue_get(StreamingPanel.out_queue)
                if not listener_stdout is None: # and StreamingPanel.is_streaming:
                    try:
                        if not isinstance(listener_stdout, str):
                            listener_stdout = listener_stdout.decode(sys.getfilesystemencoding(), 'ignore')
                    except:
                        print("Can't read the stdout from udp")
                        print(traceback.format_exc())
                    else:
                        try:
                            data = [float(x) for x in listener_stdout.split(',')]
                        except ValueError:
                            print('Listener: {}'.format(listener_stdout))
                        else:
                            self._buffer.extend(data)
                            if len(self._buffer) >= bpy.context.scene.straming_buffer_size:
                                change_graph_vals(self._buffer)
                                self._buffer = []
                            # print('data from listener: {}'.format(data))

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


bpy.types.Scene.straming_buffer_size = bpy.props.IntProperty(default=10, min=1)


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
    fixed_data = []
    in_queue = None
    out_queue = None
    time = datetime.now()

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
    StreamingPanel.fixed_data = fixed_data()
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