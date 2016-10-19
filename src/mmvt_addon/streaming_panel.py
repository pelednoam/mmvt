import bpy
import os.path as op
import glob
import time
import numpy as np
import traceback
import mmvt_utils as mu
from itertools import cycle
import socket
import select


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


def change_graph(index):
    obj_name = 'LMF6'
    fcurve_name = 'LMF6_interference'
    bpy.data.objects[obj_name].select = True
    parent_obj = bpy.data.objects[obj_name]
    T = 2500
    next_val = next(StreamingPanel.fixed_data)
    for fcurve in parent_obj.animation_data.action.fcurves:
        if mu.get_fcurve_name(fcurve) == fcurve_name:
            # for kp in fcurve.keyframe_points:
                # kp.co[1] = next(StreamingPanel.fixed_data)#[kp.co[0] + index * 100]
                # kp.co[1] = np.sin(2 * np.pi * (kp.co[0] / T * 4 - 0.1 * index))
            N = len(fcurve.keyframe_points)
            for ind in range(N - 1, 0, -1):
                fcurve.keyframe_points[ind].co[1] = fcurve.keyframe_points[ind - 1].co[1]
            fcurve.keyframe_points[0].co[1] = next_val
    return next_val


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


class StreamButton(bpy.types.Operator):
    bl_idname = "mmvt.template_button"
    bl_label = "Stream botton"
    bl_options = {"UNDO"}

    _timer = None
    _time = time.time()
    _index = 0
    _time_step = 0.01
    _obj = None

    def invoke(self, context, event=None):
        self._time = time.time()
        self._obj = bpy.data.objects['LMF6']
        if StreamingPanel.first_time:
            StreamingPanel.first_time = False
            context.window_manager.modal_handler_add(self)
            self._timer = context.window_manager.event_timer_add(0.01, context.window)
        StreamingPanel.is_streaming = not StreamingPanel.is_streaming
        print(__name__)
        script = 'src.mmvt_addon.streaming_panel'
        cmd = '{} -m {} -s {}'.format(bpy.context.scene.python_cmd, script)
        FreeviewPanel.freeview_in_queue, FreeviewPanel.freeview_out_queue mu.run_command_in_new_thread(cmd, queues=True)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        # First frame initialization:
        if not StreamingPanel.is_streaming:
            return {'PASS_THROUGH'}

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            StreamingPanel.is_streaming = False
            bpy.context.scene.update()
            self.cancel(context)
            return {'PASS_THROUGH'}

        if event.type == 'TIMER':
            # print(time.time() - self._time)
            # now = time.time()
            # print(now - self._time)
            # mu.print_current_time()
            # if now - self._time >= self._time_step:
            # print('In! {}, {}'.format(now - self._time, self._time_step))
            # mu.print_current_time()
            # self._time = now
            # self._index += 1

            # next_val = change_graph(self._index)
            # change_color(self._obj, next_val, StreamingPanel.activity_data_min, StreamingPanel.activity_colors_ratio)

            ready = select.select([StreamingPanel.sock], [], [], 0.1)
            if ready[0]:
                next_val = StreamingPanel.sock.recv(4096)
                change_color(self._obj, next_val, StreamingPanel.activity_data_min,
                             StreamingPanel.activity_colors_ratio)

        return {'PASS_THROUGH'}

    def cancel(self, context):
        context.window_manager.event_timer_remove(self._timer)
        self._timer = None
        return {'CANCELLED'}


def template_draw(self, context):
    layout = self.layout
    layout.operator(StreamButton.bl_idname,
                    text="Stream data" if not StreamingPanel.is_streaming else 'Stop streaming data',
                    icon='COLOR_GREEN' if not StreamingPanel.is_streaming else 'COLOR_RED')


class StreamingPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "Stream"
    addon = None
    init = False
    is_streaming = False
    first_time = True
    fixed_data = []

    def draw(self, context):
        if StreamingPanel.init:
            template_draw(self, context)


def init(addon):
    StreamingPanel.addon = addon
    register()

    StreamingPanel.cm = np.load(op.join(mu.file_fol(), 'color_maps', 'BuPu_YlOrRd.npy'))
    StreamingPanel.fixed_data = fixed_data()
    StreamingPanel.init = True


def register():
    try:
        unregister()
        bpy.utils.register_class(StreamingPanel)
        bpy.utils.register_class(StreamButton)
    except:
        print("Can't register Stream Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(StreamingPanel)
        bpy.utils.unregister_class(StreamButton)
    except:
        pass


def start_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(0)
    server_address = ('localhost', 10000)

    while True:
        ready = select.select([StreamingPanel.sock], [], [], 0.1)
        if ready[0]:
            next_val = StreamingPanel.sock.recv(4096)


if __name__ == '__main__':
    start_listener()