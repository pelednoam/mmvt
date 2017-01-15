import bpy
import mmvt_utils as mu
import sys
import os.path as op
import time
import numpy as np
import glob
# import traceback
from itertools import cycle
from datetime import datetime
from queue import Queue


def _addon():
    return StreamingPanel.addon


def change_graph_all_vals_thread(q, while_termination_func, **kargs):
    buffer = kargs['buffer']
    change_graph_all_vals(buffer)


def electrodes_sep_update(self, context):
    data = get_electrodes_data()
    data_amp = np.max(data) - np.min(data)
    T = data.shape[1] - 1
    parent_obj = bpy.data.objects['Deep_electrodes']
    C = len(parent_obj.animation_data.action.fcurves)
    for fcurve_ind, fcurve in enumerate(parent_obj.animation_data.action.fcurves):
        elc_ind = fcurve_ind
        for t in range(T):
            fcurve.keyframe_points[t].co[1] = data[elc_ind, t] + \
                                              (C / 2 - fcurve_ind) * bpy.context.scene.electrodes_sep * data_amp
    mu.view_all_in_graph_editor()


def change_graph_all_vals(mat):
    MAX_STEPS = StreamingPanel.max_steps
    T = min(mat.shape[1], MAX_STEPS)
    parent_obj = bpy.data.objects['Deep_electrodes']
    C = len(parent_obj.animation_data.action.fcurves)
    for fcurve_ind, fcurve in enumerate(parent_obj.animation_data.action.fcurves):
        if fcurve_ind == 0:
            max_steps = min([len(fcurve.keyframe_points), MAX_STEPS]) - 2
        elc_ind = fcurve_ind
        if elc_ind >= mat.shape[0]:
            continue
        curr_t = bpy.context.scene.frame_current
        for ind in range(T):
            t = curr_t + ind
            if t > max_steps:
                t = ind
            fcurve.keyframe_points[t].co[1] = mat[elc_ind][ind] + (C / 2 - fcurve_ind) * bpy.context.scene.electrodes_sep
        fcurve.keyframe_points[max_steps + 1].co[1] = 0
        fcurve.keyframe_points[0].co[1] = 0

    bpy.context.scene.frame_current += mat.shape[1]
    if bpy.context.scene.frame_current > MAX_STEPS - 1:
        bpy.context.scene.frame_current = bpy.context.scene.frame_current - MAX_STEPS
        time_diff = (datetime.now() - StreamingPanel.time)
        time_diff_sec = time_diff.seconds + time_diff.microseconds * 1e-6
        print('cycle! ', str(time_diff), time_diff_sec)
        set_electrodes_data()
        max_steps_secs = MAX_STEPS / 1000
        if time_diff_sec < max_steps_secs:
            print('sleep for {}'.format(max_steps_secs - time_diff_sec))
            time.sleep(max_steps_secs - time_diff_sec)
        StreamingPanel.time = datetime.now()


def show_electrodes_fcurves():
    bpy.context.scene.selection_type = 'diff'
    _addon().select_all_electrodes()
    mu.change_fcurves_colors([bpy.data.objects['Deep_electrodes']])


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


def offline_logs_reader(udp_queue, while_termination_func, **kargs):
    buffer_size = kargs.get('buffer_size', 10)
    bad_channels = kargs.get('bad_channels', '')
    bad_channels = list(map(mu.to_int, bad_channels.split(','))) if bad_channels != '' else []
    input_fol = op.join(mu.get_user_fol(), 'electrodes', 'streaming')
    files = sorted(glob.glob(op.join(input_fol, 'streaming_data_*.npy')))
    offline_data = []
    for log_file in files:
        data = np.load(log_file)
        offline_data = data if offline_data == [] else np.hstack((offline_data, data))
        break
    offline_data[bad_channels] = 0
    offline_data = cycle(offline_data.T)
    buffer = []
    while while_termination_func:
        next_val = next(offline_data)
        next_val = next_val[..., np.newaxis]
        buffer = next_val if buffer == [] else np.hstack((buffer, next_val))
        if buffer.shape[1] >= buffer_size:
            udp_queue.put(buffer)
            buffer = []


def udp_reader(udp_queue, while_termination_func, **kargs):
    import socket

    def bind_to_server(server='', port=45454):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_address = (server, port)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(server_address)
        return sock

    def bind_to_multicast(port=45454, multicast_group='239.255.43.21'):
        import struct
        # Create the socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Bind to the server address
        sock.bind(('', port))
        # Tell the operating system to add the socket to the multicast group
        # on all interfaces.
        mreq = struct.pack('4sl', socket.inet_aton(multicast_group), socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        return sock

    buffer_size = kargs.get('buffer_size', 10)
    server = kargs.get('server', 'localhost')
    port = kargs.get('port', 45454)
    multicast_group = kargs.get('multicast_group', '239.255.43.21')
    mat_len = kargs.get('mat_len', 1)
    multicast = kargs.get('multicast', True)
    timeout = kargs.get('timeout', 0.1)
    max_val = kargs.get('max_val', 50000)
    print('udp_reader:', server, port, multicast_group, buffer_size, multicast, timeout, mat_len)
    if multicast:
        sock = bind_to_multicast(port, multicast_group)
    else:
        sock = bind_to_server(server, port)

    buffer = []
    prev_val = None
    mat_len = 0
    first_message = True

    while while_termination_func():
        try:
            sock.settimeout(timeout)
            if multicast:
                next_val, address = sock.recvfrom(2048)
            else:
                next_val = sock.recv(2048)
        except socket.timeout as e:
            if e.args[0] == 'timed out':
                # print('!!! timed out !!!')
                if prev_val is None and mat_len > 0:
                    prev_val = np.zeros((mat_len, 1))
                else:
                    continue
                next_val = prev_val
            else:
                print('!!! {} !!!'.format(e))
                raise Exception(e)
        else:
            next_val = next_val.decode(sys.getfilesystemencoding(), 'ignore')
            next_val = np.array([mu.to_float(f, 0.0) for f in next_val.split(',')])
            big_values = next_val[next_val > max_val]
            # print(big_values)
            # next_val = next_val[next_val < max_val]
            if first_message:
                mat_len = len(next_val)
                first_message = False
            else:
                if len(next_val) != mat_len:
                    print('Wrong message len! {} ({})'.format(len(next_val), big_values))
            # next_val = next_val[:mat_len]
            next_val = next_val[..., np.newaxis]

        prev_val = next_val
        buffer = next_val if buffer == [] else np.hstack((buffer, next_val))
        # buffer.append(next_val)
        if buffer.shape[1] >= buffer_size:
        # if len(buffer) >= buffer_size:
            # print('{} took {:.5f}s {}'.format('udp_reader', time.time() - now, buffer.shape[1]))
            # print('udp_reader: ', datetime.now())
            # zeros_indices = np.where(np.all(buffer == 0, 1))[0]
            # buffer = buffer[zeros_indices]
            udp_queue.put(buffer)
            buffer = []


def color_electrodes(data):
    parent_obj = bpy.data.objects['Deep_electrodes']
    names = [elec.name for elec in parent_obj.children]
    # _addon().color_objects_homogeneously(
    #     np.mean(data, 1), names, StreamingPanel.electrodes_conditions,
    #     StreamingPanel.data_min, StreamingPanel.electrodes_colors_ratio, threshold=0)


def set_electrodes_data():
    StreamingPanel.electrodes_data = data = get_electrodes_data()
    if bpy.context.scene.save_streaming:
        output_fol = op.join(mu.get_user_fol(), 'electrodes', 'streaming')
        output_fname = 'streaming_data_{}.npy'.format(datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S'))
        mu.make_dir(output_fol)
        np.save(op.join(output_fol, output_fname), data)

    norm_percs = (3, 97) #todo: add to gui
    StreamingPanel.data_max, StreamingPanel.data_min = mu.get_data_max_min(
        StreamingPanel.electrodes_data, True, norm_percs=norm_percs, data_per_hemi=False, symmetric=True)
    StreamingPanel.electrodes_colors_ratio = 256 / (StreamingPanel.data_max - StreamingPanel.data_min)


def get_electrodes_data():
    parent_obj = bpy.data.objects['Deep_electrodes']
    fcurves = parent_obj.animation_data.action.fcurves
    for fcurve_ind, fcurve in enumerate(fcurves):
        if fcurve_ind == 0:
            max_steps = min([len(fcurve.keyframe_points), StreamingPanel.max_steps]) - 2
            data = np.zeros((len(fcurves), max_steps))
        for t in range(max_steps):
            data[fcurve_ind, t] = fcurve.keyframe_points[t].co[1]
    return data

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
        if StreamingPanel.first_time:
            StreamingPanel.first_time = False
            try:
                context.window_manager.event_timer_remove(self._timer)
            except:
                pass
            context.window_manager.modal_handler_add(self)
            self._timer = context.window_manager.event_timer_add(0.01, context.window)
        if StreamingPanel.is_streaming:
            init_electrodes_fcurves()
            show_electrodes_fcurves()
            _addon().set_colorbar_max_min(StreamingPanel.data_max, StreamingPanel.data_min)
            _addon().set_colorbar_title('Electrodes Streaming Data')
            mu.show_only_render(True)
            bpy.context.scene.frame_current = 0
            args = dict(buffer_size=bpy.context.scene.streaming_buffer_size,
                        server=bpy.context.scene.streaming_server,
                        multicast_group=bpy.context.scene.multicast_group,
                        port=bpy.context.scene.streaming_server_port,
                        timeout=bpy.context.scene.timeout,
                        mat_len=len(bpy.data.objects['Deep_electrodes'].children),
                        bad_channels=bpy.context.scene.streaming_bad_channels)
            if bpy.context.scene.stream_type == 'offline':
                StreamingPanel.udp_queue = mu.run_thread(
                    offline_logs_reader, reading_from_udp_while_termination_func, **args)
            else:
                StreamingPanel.udp_queue = mu.run_thread(
                    udp_reader, reading_from_udp_while_termination_func, **args)

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
            if StreamingPanel.is_streaming and time.time() - self._time > bpy.context.scene.streaming_buffer_size / 1000.0:
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

    layout.prop(context.scene, "stream_type", text="")
    layout.prop(context.scene, "stream_edit", text="Edit settings")
    if bpy.context.scene.stream_edit:
        box = layout.box()
        col = box.column()
        if bpy.context.scene.stream_type != 'offline':
            col.prop(context.scene, "multicast", text="multicast")
            if bpy.context.scene.multicast:
                col.prop(context.scene, "multicast_group", text="group")
            else:
                col.prop(context.scene, "streaming_server", text="server")
            col.prop(context.scene, "streaming_server_port", text="port")
        col.prop(context.scene, "streaming_buffer_size", text="buffer size")
        if bpy.context.scene.stream_type != 'offline':
            col.prop(context.scene, 'streaming_bad_channels', text='bad channels')
        # col.prop(context.scene, "streaming_electrodes_num", text="electrodes num")
    layout.operator(StreamButton.bl_idname,
                    text="Stream data" if not StreamingPanel.is_streaming else 'Stop streaming data',
                    icon='COLOR_GREEN' if not StreamingPanel.is_streaming else 'COLOR_RED')

    layout.prop(context.scene, 'save_streaming', text='save streaming data')
    layout.prop(context.scene, 'electrodes_sep', text='electrodes sep')


bpy.types.Scene.streaming_buffer_size = bpy.props.IntProperty(default=100, min=10)
bpy.types.Scene.streaming_server_port = bpy.props.IntProperty(default=45454)
bpy.types.Scene.multicast_group = bpy.props.StringProperty(name='multicast_group', default='239.255.43.21')
bpy.types.Scene.multicast = bpy.props.BoolProperty(default=True)
bpy.types.Scene.timeout = bpy.props.FloatProperty(default=0.1, min=0.001, max=1)
bpy.types.Scene.streaming_server = bpy.props.StringProperty(name='streaming_server', default='localhost')
bpy.types.Scene.electrodes_sep = bpy.props.FloatProperty(default=0, min=0, update=electrodes_sep_update)
bpy.types.Scene.streaming_electrodes_num = bpy.props.IntProperty(default=0)
bpy.types.Scene.streaming_bad_channels = bpy.props.StringProperty(name='streaming_bad_channels', default='0,3')
bpy.types.Scene.stream_type = bpy.props.EnumProperty(
    items=[("draper", "Draper", "", 1), ('offline', 'Offline recordings', '', 2)],
    description='Type of stream listener')
bpy.types.Scene.stream_edit = bpy.props.BoolProperty(default=False)
bpy.types.Scene.save_streaming = bpy.props.BoolProperty(default=False)


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
    electrodes_data = None
    time = datetime.now()
    electrodes_names, electrodes_conditions = [], []
    data_max, data_min, electrodes_colors_ratio = 0, 0, 1

    def draw(self, context):
        if StreamingPanel.init:
            template_draw(self, context)


def init(addon):
    cm_fname = op.join(mu.file_fol(), 'color_maps', 'BuPu_YlOrRd.npy')
    if not op.isfile(cm_fname):
        print("Streaming: Can't load without the cm file {}".format(cm_fname))
        return
    if not bpy.data.objects.get('Deep_electrodes'):
        print('Streaming: No electrodes')
        return
    StreamingPanel.addon = addon
    StreamingPanel.is_listening = False
    StreamingPanel.is_streaming = False
    StreamingPanel.first_time = True
    StreamingPanel.electrodes_data = None
    register()
    StreamingPanel.cm = np.load(cm_fname)
    # StreamingPanel.fixed_data = fixed_data()
    StreamingPanel.electrodes_objs_names = [l.name for l in bpy.data.objects['Deep_electrodes'].children]
    bpy.context.scene.streaming_electrodes_num = len(StreamingPanel.electrodes_objs_names)
    StreamingPanel.max_steps = _addon().get_max_time_steps()
    StreamingPanel.init = True


def init_electrodes_fcurves():
    for fcurve_ind, fcurve in enumerate( bpy.data.objects['Deep_electrodes'].animation_data.action.fcurves):
        if fcurve_ind == 0:
            # max_steps = min([len(fcurve.keyframe_points), StreamingPanel.max_steps]) - 1
            max_steps = len(fcurve.keyframe_points) - 1
        for t in range(max_steps):
            fcurve.keyframe_points[t].co[1] = 0

#
# def create_electrodes_dic():
#     parent_obj = bpy.data.objects['Deep_electrodes']
#     objs_names = [l.name for l in parent_obj.children]
#     elcs_names = [l for l in StreamingPanel.electrodes_names]
#     lookup = {}
#     for obj_name in objs_names:
#         ind = elcs_names.index(obj_name)
#         lookup[obj_name] = ind
#     return lookup


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
