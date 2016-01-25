import bpy
import mmvt_utils
import os.path as op
import numpy as np
import traceback
import time
import pprint as pp # pretty printing: pp.pprint(something)

bpy.types.Scene.play_from = bpy.props.IntProperty(default=0, min=0, description="When to filter from")
bpy.types.Scene.play_to = bpy.props.IntProperty(default=bpy.context.scene.frame_end, min=0,
                                                  description="When to filter to")
bpy.types.Scene.play_dt = bpy.props.IntProperty(default=50, min=1, description="play dt (ms)")
bpy.types.Scene.play_time_step = bpy.props.FloatProperty(default=0.1, min=0,
                                                  description="How much time (s) to wait between frames")
bpy.types.Scene.play_type = bpy.props.EnumProperty(
    items=[("meg", "MEG activity", "", 1), ("meg_coh", "MEG Coherence", "", 2),
           ("elecs", "Electrodes activity", "", 3), ("elecs_coh", "Electrodes coherence", "", 4),
           ("elecs_act_coh", "Electrodes activity & coherence", "", 5),
           ("meg_elecs", "Meg & Electrodes activity", "", 6), ("meg_elecs_coh", "Meg & Electrodes activity & coherence", "", 7)],
           description='Type pf data to play')


class ModalTimerOperator(bpy.types.Operator):
    """Operator which runs its self from a timer"""
    bl_idname = "wm.modal_timer_operator"
    bl_label = "Modal Timer Operator"

    limits = bpy.props.IntProperty(default=bpy.context.scene.play_from)
    _timer = None
    _time = time.time()

    def modal(self, context, event):
        # First frame initialization:
        if PlayPanel.init_play:
            PlayPanel.init_play = False
            self.limits = bpy.context.scene.frame_current

        if not PlayPanel.is_playing:
            return {'PASS_THROUGH'}

        if event.type in {'RIGHTMOUSE', 'ESC'} or self.limits > bpy.context.scene.play_to:
            self.limits = bpy.context.scene.play_from
            PlayPanel.is_playing = False
            bpy.context.scene.update()
            self.cancel(context)
            return {'PASS_THROUGH'}

        if event.type == 'TIMER':
            # print(time.time() - self._time)
            if time.time() - self._time > bpy.context.scene.play_time_step:
                bpy.context.scene.frame_current = self.limits
                print(self.limits, time.time() - self._time)
                self._time = time.time()
                plot_something(context, self.limits)
                self.limits = self.limits - bpy.context.scene.play_dt if PlayPanel.play_reverse else \
                        self.limits + bpy.context.scene.play_dt

        return {'PASS_THROUGH'}

    def execute(self, context):
        self.cancel(context)
        # ModalTimerOperator._timer = wm.event_timer_add(time_step=bpy.context.scene.play_time_step, window=context.window)

        wm = context.window_manager
        if ModalTimerOperator._timer:
            print('timer is already running!')
        else:
            ModalTimerOperator._timer = wm.event_timer_add(time_step=0.05, window=context.window)
            self._time = time.time()
            wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        # if ModalTimerOperator._timer:
        if ModalTimerOperator._timer:
            wm = context.window_manager
            wm.event_timer_remove(ModalTimerOperator._timer)


def plot_something(context, cur_frame):
    threshold = bpy.context.scene.coloring_threshold
    plot_subcorticals=True
    play_type = bpy.context.scene.play_type

    if play_type in ['meg', 'meg_elecs', 'meg_elecs_coh']:
        # if PlayPanel.loop_indices:
        #     PlayPanel.addon.default_coloring(PlayPanel.loop_indices)
        # PlayPanel.loop_indices =
        PlayPanel.addon.plot_activity('MEG', PlayPanel.faces_verts, threshold,
            PlayPanel.meg_sub_activity, plot_subcorticals)
    if play_type in ['elecs', 'meg_elecs', 'elecs_act_coh', 'meg_elecs_coh']:
        # PlayPanel.addon.set_appearance_show_electrodes_layer(bpy.context.scene, True)
        plot_electrodes(cur_frame)
    if play_type == 'meg_coh':
        pass
    if play_type in ['elecs_coh', 'elecs_act_coh', 'meg_elecs_coh']:
        p = PlayPanel.addon.connections_panel
        d = p.ConnectionsPanel.d
        connections_type = bpy.context.scene.connections_type
        threshold = bpy.context.scene.connections_threshold
        abs_threshold = bpy.context.scene.abs_threshold
        condition = bpy.context.scene.conditions
        p.plot_connections(context, d, cur_frame, connections_type, condition, threshold, abs_threshold)
    PlayPanel.addon.render_image()

def plot_electrodes(cur_frame):
    threshold = bpy.context.scene.coloring_threshold
    names, colors = PlayPanel.electrodes_names, PlayPanel.electrodes_data['colors']
    for obj_name, object_colors in zip(names, colors):
        # obj_name = obj_name.astype(str)
        new_color = object_colors[cur_frame]
        if bpy.data.objects.get(obj_name) is not None:
            PlayPanel.addon.object_coloring(bpy.data.objects[obj_name], new_color)
        else:
            print('color_object_homogeneously: {} was not loaded!'.format(obj_name))


def init_plotting():
    data_fname = op.join(mmvt_utils.get_user_fol(), 'electrodes_data.npz')
    if op.isfile(data_fname):
        PlayPanel.electrodes_data = np.load(data_fname)
        PlayPanel.electrodes_names = [elc.astype(str) for elc in PlayPanel.electrodes_data['names']]
    else:
        print('No electrodes data file!')
    PlayPanel.faces_verts = PlayPanel.addon.init_activity_map_coloring('MEG')
    PlayPanel.meg_sub_activity = PlayPanel.addon.load_meg_subcortical_activity()
    # connections_file = op.join(mmvt_utils.get_user_fol(), 'electrodes_coh.npz')
    # if not op.isfile(connections_file):
    #     print('No connections coherence file! {}'.format(connections_file))
    # else:
    #     PlayPanel.electrodes_coherence = mmvt_utils.Bag(np.load(connections_file))


class PlayPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
    bl_label = "Play"
    addon = None
    data = None
    loop_indices = None
    is_playing = False
    play_reverse = False
    first_time = True

    def draw(self, context):
        play_panel_draw(context, self.layout)


def play_panel_draw(context, layout):
    row = layout.row(align=0)
    row.prop(context.scene, "play_from", text="From")
    row.operator(GrabFromPlay.bl_idname, text="", icon='BORDERMOVE')
    row.prop(context.scene, "play_to", text="To")
    row.operator(GrabToPlay.bl_idname, text="", icon='BORDERMOVE')
    row = layout.row(align=0)
    row.prop(context.scene, "play_dt", text="dt")
    row.prop(context.scene, "play_time_step", text="time step")
    layout.prop(context.scene, "play_type", text="")
    row = layout.row(align=True)
    # row.operator(Play.bl_idname, text="", icon='PLAY' if not PlayPanel.is_playing else 'PAUSE')
    row.operator(PrevKeyFrame.bl_idname, text="", icon='PREV_KEYFRAME')
    row.operator(Reverse.bl_idname, text="", icon='PLAY_REVERSE')
    row.operator(Pause.bl_idname, text="", icon='PAUSE')
    row.operator(Play.bl_idname, text="", icon='PLAY')
    row.operator(NextKeyFrame.bl_idname, text="", icon='NEXT_KEYFRAME')



class Play(bpy.types.Operator):
    bl_idname = "ohad.play"
    bl_label = "play"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        PlayPanel.is_playing = True
        PlayPanel.play_reverse = False
        PlayPanel.init_play = True
        if PlayPanel.first_time:
            PlayPanel.first_time = False
            ModalTimerOperator.limits = bpy.context.scene.play_from
            bpy.ops.wm.modal_timer_operator()
        return {"FINISHED"}


class Reverse(bpy.types.Operator):
    bl_idname = "ohad.reverse"
    bl_label = "reverse"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        PlayPanel.is_playing = True
        PlayPanel.play_reverse = True
        if PlayPanel.first_time:
            PlayPanel.first_time = False
            PlayPanel.timer_op = bpy.ops.wm.modal_timer_operator()
        return {"FINISHED"}


class Pause(bpy.types.Operator):
    bl_idname = "ohad.pause"
    bl_label = "pause"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        PlayPanel.is_playing = False
        return {"FINISHED"}


class PrevKeyFrame(bpy.types.Operator):
    bl_idname = 'ohad.prev_key_frame'
    bl_label = 'prevKeyFrame'
    bl_options = {'UNDO'}

    def invoke(self, context, event=None):
        PlayPanel.is_playing = False
        bpy.context.scene.frame_current -= bpy.context.scene.play_from
        plot_something()
        return {'FINISHED'}


class NextKeyFrame(bpy.types.Operator):
    bl_idname = "ohad.next_key_frame"
    bl_label = "nextKeyFrame"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        PlayPanel.is_playing = False
        bpy.context.scene.frame_current += bpy.context.scene.play_dt
        plot_something()
        return {"FINISHED"}


class GrabFromPlay(bpy.types.Operator):
    bl_idname = "ohad.grab_from_play"
    bl_label = "grab from"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        context.scene.play_from = bpy.context.scene.frame_current
        bpy.data.scenes['Scene'].frame_preview_start = context.scene.frame_current
        ModalTimerOperator.limits = bpy.context.scene.frame_current
        return {"FINISHED"}


class GrabToPlay(bpy.types.Operator):
    bl_idname = "ohad.grab_to_play"
    bl_label = "grab to"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        context.scene.play_to = bpy.context.scene.frame_current
        bpy.data.scenes['Scene'].frame_preview_end = context.scene.frame_current
        return {"FINISHED"}


def init(addon):
    register()
    PlayPanel.addon = addon
    bpy.context.scene.play_to = addon.T
    init_plotting()
    print('PlayPanel initialization completed successfully!')


def register():
    try:
        unregister()
        bpy.utils.register_class(PlayPanel)
        bpy.utils.register_class(GrabFromPlay)
        bpy.utils.register_class(GrabToPlay)
        bpy.utils.register_class(Play)
        bpy.utils.register_class(Reverse)
        bpy.utils.register_class(PrevKeyFrame)
        bpy.utils.register_class(NextKeyFrame)
        bpy.utils.register_class(Pause)
        bpy.utils.register_class(ModalTimerOperator)
        print('PlayPanel was registered!')
    except:
        print("Can't register PlayPanel!")
        print(traceback.format_exc())


def unregister():
    try:
        bpy.utils.unregister_class(PlayPanel)
        bpy.utils.unregister_class(GrabFromPlay)
        bpy.utils.unregister_class(GrabToPlay)
        bpy.utils.unregister_class(Play)
        bpy.utils.unregister_class(Reverse)
        bpy.utils.unregister_class(PrevKeyFrame)
        bpy.utils.unregister_class(NextKeyFrame)
        bpy.utils.unregister_class(Pause)
        bpy.utils.unregister_class(ModalTimerOperator)
    except:
        print("Can't unregister PlayPanel!")
        print(traceback.format_exc())