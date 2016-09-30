import bpy
import os.path as op
import glob
import time
import numpy as np
import traceback
import mmvt_utils as mu


def animate(i):
    x = np.linspace(0, 2, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    return y


def change_graph(index):
    obj_name = 'Activity_in_vertex'
    fcurve_name = 'data'
    bpy.data.objects[obj_name].select = True
    parent_obj = bpy.data.objects[obj_name]
    T = 2500
    for fcurve in parent_obj.animation_data.action.fcurves:
        if mu.fcurve_name(fcurve) == fcurve_name:
            for kp in fcurve.keyframe_points:
                kp.co[1] = np.sin(2 * np.pi * (kp.co[0] / T * 4 - 0.1 * index))



class StreamButton(bpy.types.Operator):
    bl_idname = "mmvt.template_button"
    bl_label = "Stream botton"
    bl_options = {"UNDO"}

    _timer = None
    _time = time.time()
    _index = 0
    _time_step = 0.1

    def invoke(self, context, event=None):
        self._time = time.time()
        if StreamingPanel.first_time:
            StreamingPanel.first_time = False
            context.window_manager.modal_handler_add(self)
            self._timer = context.window_manager.event_timer_add(0.01, context.window)
        StreamingPanel.is_streaming = not StreamingPanel.is_streaming
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
            if time.time() - self._time > self._time_step:
                self._time = time.time()
                self._index += 1
                change_graph(self._index)


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

    def draw(self, context):
        if StreamingPanel.init:
            template_draw(self, context)


def init(addon):
    StreamingPanel.addon = addon
    register()
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
