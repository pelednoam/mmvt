import bpy
import mmvt_utils as mu
import os.path as op
import numpy as np
import traceback
import time
import os
import pprint as pp # pretty printing: pp.pprint(something)
from collections import OrderedDict

HEMIS = mu.HEMIS

bpy.types.Scene.play_from = bpy.props.IntProperty(default=0, min=0, description="When to filter from")
bpy.types.Scene.play_to = bpy.props.IntProperty(default=bpy.context.scene.frame_end, min=0,
                                                  description="When to filter to")
bpy.types.Scene.play_dt = bpy.props.IntProperty(default=50, min=1, description="play dt (ms)")
bpy.types.Scene.play_time_step = bpy.props.FloatProperty(default=0.1, min=0,
                                                  description="How much time (s) to wait between frames")
bpy.types.Scene.render_movie = bpy.props.BoolProperty(default=False, description="Render the movie")
bpy.types.Scene.play_type = bpy.props.EnumProperty(
    items=[("meg", "MEG activity", "", 1), ("meg_labels", 'MEG Labels', '', 2), ("meg_coh", "MEG Coherence", "", 3),
           ("elecs", "Electrodes activity", "", 4),
           ("elecs_coh", "Electrodes coherence", "",5), ("elecs_act_coh", "Electrodes activity & coherence", "", 6),
           ("stim", "Electrodes stimulation", "", 7), ("stim_sources", "Electrodes stimulation & sources", "", 8),
           ("meg_elecs", "Meg & Electrodes activity", "", 9),
           ("meg_elecs_coh", "Meg & Electrodes activity & coherence", "",10)],
           description='Type pf data to play')


class ModalTimerOperator(bpy.types.Operator):
    """Operator which runs its self from a timer"""
    bl_idname = "wm.modal_timer_operator"
    bl_label = "Modal Timer Operator"

    # limits = bpy.props.IntProperty(default=bpy.context.scene.play_from)
    limits = bpy.context.scene.play_from
    _timer = None
    _time = time.time()
    _uuid = mu.rand_letters(5)

    def modal(self, context, event):
        # First frame initialization:
        if PlayPanel.init_play:
            ModalTimerOperator._uuid = mu.rand_letters(5)
            self.limits = bpy.context.scene.frame_current

        if not PlayPanel.is_playing:
            return {'PASS_THROUGH'}

        if event.type in {'RIGHTMOUSE', 'ESC'} or self.limits > bpy.context.scene.play_to:
            plot_something(self, context, bpy.context.scene.play_to, ModalTimerOperator._uuid)
            print('Stop!')
            self.limits = bpy.context.scene.play_from
            PlayPanel.is_playing = False
            bpy.context.scene.update()
            self.cancel(context)
            return {'PASS_THROUGH'}

        if event.type == 'TIMER':
            # print(time.time() - self._time)
            if time.time() - self._time > bpy.context.scene.play_time_step:
                bpy.context.scene.frame_current = self.limits
                # print(self.limits, time.time() - self._time)
                self._time = time.time()
                try:
                    plot_something(self, context, self.limits, ModalTimerOperator._uuid)
                except:
                    print(traceback.format_exc())
                    print('Error in plotting at {}!'.format(self.limits))
                self.limits = self.limits - bpy.context.scene.play_dt if PlayPanel.play_reverse else \
                        self.limits + bpy.context.scene.play_dt
                bpy.context.scene.frame_current = self.limits

        return {'PASS_THROUGH'}

    def execute(self, context):
        # ModalTimerOperator._timer = wm.event_timer_add(time_step=bpy.context.scene.play_time_step, window=context.window)
        wm = context.window_manager
        # if ModalTimerOperator._timer:
        #     print('timer is already running!')
        #     print('last tick {}'.format(time.time() - self._time))
        # else:
        self.cancel(context)
        ModalTimerOperator._timer = wm.event_timer_add(time_step=0.05, window=context.window)
        self._time = time.time()
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        # if ModalTimerOperator._timer:
        if ModalTimerOperator._timer:
            wm = context.window_manager
            wm.event_timer_remove(ModalTimerOperator._timer)


def render_movie(play_type, play_from, play_to, play_dt=1):
    bpy.context.scene.play_type = play_type
    bpy.context.scene.render_movie = True
    print('In play movie!')
    for limits in range(play_from, play_to + 1, play_dt):
        print('limits: {}'.format(limits))
        bpy.context.scene.frame_current = limits
        try:
            plot_something(None, bpy.context, limits)
        except:
            print(traceback.format_exc())
            print('Error in plotting at {}!'.format(limits))


def plot_something(self, context, cur_frame, uuid=''):
    if bpy.context.scene.frame_current > bpy.context.scene.play_to:
        return

    threshold = bpy.context.scene.coloring_threshold
    plot_subcorticals = True
    play_type = bpy.context.scene.play_type
    # image_fol = op.join(mu.get_user_fol(), 'images', uuid)

    # todo: implement the important times
    # imp_time = False
    # for imp_time_range in PlayPanel.imp_times:
    #     if imp_time_range[0] <= cur_frame <= imp_time_range[1]:
    #         imp_time = True
    # if not imp_time:
    #     return

    #todo: need a different threshold value for each modality!
    meg_threshold = threshold
    electrodes_threshold = threshold

    #todo: Check the init_play!
    # if False: #PlayPanel.init_play:

    successful_ret = True
    if play_type in ['meg', 'meg_elecs', 'meg_elecs_coh']:
        # if PlayPanel.loop_indices:
        #     PlayPanel.addon.default_coloring(PlayPanel.loop_indices)
        # PlayPanel.loop_indices =
        successful_ret = PlayPanel.addon.plot_activity('MEG', PlayPanel.faces_verts, meg_threshold,
            PlayPanel.meg_sub_activity, plot_subcorticals)
    if play_type in ['elecs', 'meg_elecs', 'elecs_act_coh', 'meg_elecs_coh']:
        # PlayPanel.addon.set_appearance_show_electrodes_layer(bpy.context.scene, True)
        plot_electrodes(cur_frame, electrodes_threshold)
    if play_type == 'meg_labels':
        # todo: get the aparc_name
        PlayPanel.addon.meg_labels_coloring(override_current_mat=True)
    if play_type == 'meg_coh':
        pass
    if play_type in ['elecs_coh', 'elecs_act_coh', 'meg_elecs_coh']:
        p = PlayPanel.addon.connections_panel
        d = p.ConnectionsPanel.d
        connections_type = bpy.context.scene.connections_type
        threshold = bpy.context.scene.connections_threshold
        abs_threshold = bpy.context.scene.abs_threshold
        condition = bpy.context.scene.conditions
        p.plot_connections(self, context, d, cur_frame, connections_type, condition, threshold, abs_threshold)
    if play_type in ['stim', 'stim_sources']:
        # plot_electrodes(cur_frame, electrodes_threshold, stim=True)
        PlayPanel.addon.color_object_homogeneously(PlayPanel.stim_data)
    if play_type in ['stim_sources']:
        PlayPanel.addon.color_electrodes_sources()
    if bpy.context.scene.render_movie:
        if successful_ret:
            PlayPanel.addon.render_image()
        else:
            print("The image wasn't rendered due to an error in the plotting.")
    # plot_graph(context, graph_data, graph_colors, image_fol)


def capture_graph(play_type=None, output_path=None, selection_type=None):
    if play_type:
        bpy.context.scene.play_type = play_type
    if output_path:
        bpy.context.scene.output_path = output_path
    if selection_type:
        bpy.context.scene.selection_type = selection_type

    play_type = bpy.context.scene.play_type
    # image_fol = op.join(mu.get_user_fol(), 'images', ExportGraph.uuid)
    image_fol = bpy.path.abspath(bpy.context.scene.output_path)
    if not op.isdir(image_fol):
        raise Exception('You need to set first the images folder in the Render panel!')
    graph_data, graph_colors = {}, {}
    per_condition = bpy.context.scene.selection_type == 'conds'

    if play_type in ['elecs_coh', 'elecs_act_coh', 'meg_elecs_coh']:
        graph_data['coherence'], graph_colors['coherence'] = PlayPanel.addon.connections_panel.capture_graph_data(per_condition)
    if play_type in ['elecs', 'meg_elecs', 'elecs_act_coh', 'meg_elecs_coh']:
        graph_data['electrodes'], graph_colors['electrodes'] = get_electrodes_data(per_condition)
    if play_type in ['meg', 'meg_elecs', 'meg_elecs_coh']:
        graph_data['meg'], graph_colors['meg'] = get_meg_data(per_condition)
    if play_type in ['meg_labels']:
        graph_data['meg_labels'], graph_colors['meg_labels'] = get_meg_labels_data()
    if play_type in ['stim', 'stim_sources']:
        graph_data['stim'], graph_colors['stim'] = get_electrodes_data(per_condition=True)
    if play_type in ['stim_sources']:
        graph_data['stim_sources'], graph_colors['stim_sources'] = get_electrodes_sources_data()
    # should let the user set the:
    #  xticklabels (--xticklabels '-1,stim_onset,0,end_of_stim')
    #  time_range (--time_range '-1,1.5,0.01')
    #  xtick_dt (--xtick_dt 0.5)
    T = PlayPanel.addon.get_max_time_steps()
    cmd = "{} -m src.make_movie -f plot_only_graph --data_in_graph {} --time_range {} --xlabel time --images_folder '{}'".format(
        bpy.context.scene.python_cmd, play_type, T, image_fol)
    print('Running {}'.format(cmd))
    # mu.run_command_in_new_thread(cmd, False)

    # if op.isfile(op.join(image_fol, 'data.pkl')):
    #     print('The file already exists!')
    #     return

    save_graph_data(graph_data, graph_colors, image_fol)


def save_graph_data(data, graph_colors, image_fol):
    if not os.path.isdir(image_fol):
        os.makedirs(image_fol)
    mu.save((data, graph_colors), op.join(image_fol, 'data.pkl'))
    print('Saving data into {}'.format(op.join(image_fol, 'data.pkl')))


def plot_graph(context, data, graph_colors, image_fol, plot_time=False):
    if not os.path.isdir(image_fol):
        os.makedirs(image_fol)

    try:
        # http://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server/4935945#4935945
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt

        # fig = plt.figure()
        time_range = range(PlayPanel.addon.get_max_time_steps())
        fig, ax1 = plt.subplots()
        axes = [ax1]
        if len(data.keys()) > 1:
            ax2 = ax1.twinx()
            axes = [ax1, ax2]
        for (data_type, data_values), ax in zip(data.items(), axes):
            for k, values in data_values.items():
                ax.plot(time_range, values, label=k, color=tuple(graph_colors[data_type][k]))
        current_t = context.scene.frame_current
        # ax = plt.gca()
        # ymin, ymax = axes[0].get_ylim()
        # plt.xlim([0, time_range[-1]])

        # Shrink current axis by 20%
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        # ax.legend(loc='upper center', bbox_to_anchor=(1, 0.5))

        # plt.legend(loc='best')
        plt.xlabel('Time (ms)')
        # plt.title('Coherence')
        image_fname = op.join(image_fol, 'g.png')
        fig.savefig(image_fname)
        # mu.save(ax, op.join(image_fol, 'plt.pkl'))
        # if plot_time:
        #     plt.plot([current_t, current_t], [ymin, ymax], 'g-')
        #     image_fname_t = op.join(image_fol, 'g{}.png'.format(current_t))
        #     fig.savefig(image_fname_t)
        print('saving to {}'.format(image_fname))
    except:
        print('No matplotlib!')


def plot_electrodes(cur_frame, threshold, stim=False):
    # todo: need to use the threshold
    # threshold = bpy.context.scene.coloring_threshold
    # if stim:
    #     names, colors = PlayPanel.stim_names, PlayPanel.stim_colors
    # else:
    names, colors = PlayPanel.electrodes_names, PlayPanel.electrodes_colors
    for obj_name, object_colors in zip(names, colors):
        if cur_frame < len(object_colors):
            new_color = object_colors[cur_frame]
            if bpy.data.objects.get(obj_name) is not None:
                PlayPanel.addon.object_coloring(bpy.data.objects[obj_name], new_color)
            else:
                print('color_object_homogeneously: {} was not loaded!'.format(obj_name))


def get_meg_data(per_condition=True):
    time_range = range(PlayPanel.addon.get_max_time_steps())
    brain_obj = bpy.data.objects['Brain']
    if per_condition:
        meg_data, meg_colors = OrderedDict(), OrderedDict()
        rois_objs = bpy.data.objects['Cortex-lh'].children + bpy.data.objects['Cortex-rh'].children
        for roi_obj in rois_objs:
            if roi_obj.animation_data:
                meg_data_roi, meg_colors_roi = mu.evaluate_fcurves(roi_obj, time_range)
                meg_data.update(meg_data_roi)
                meg_colors.update(meg_colors_roi)
    else:
        meg_data, meg_colors = mu.evaluate_fcurves(brain_obj, time_range)
    return meg_data, meg_colors


def get_meg_labels_data():
    meg_data, meg_colors = OrderedDict(), OrderedDict()
    for hemi in HEMIS:
        labels_data = np.load(os.path.join(mu.get_user_fol(), 'meg_labels_coloring_{}.npz'.format(hemi)))
        for label_data, label_colors, label_name in zip(labels_data['data'], labels_data['colors'], labels_data['names']):
            meg_data[label_name] = label_data
            meg_colors[label_name] = label_colors
    return meg_data, meg_colors


def get_electrodes_data(per_condition=True):
    if bpy.context.scene.selection_type == 'spec_cond' and bpy.context.scene.conditions_selection == '':
        print('You must choose the condition first!')
        return None, None
    elecs_data, elecs_colors = OrderedDict(), OrderedDict()
    time_range = range(PlayPanel.addon.get_max_time_steps())
    if per_condition:
        for obj_name in PlayPanel.electrodes_names:
            if bpy.data.objects.get(obj_name) is None:
                continue
            elec_obj = bpy.data.objects[obj_name]
            if elec_obj.hide or elec_obj.animation_data is None:
                continue
            curr_cond = bpy.context.scene.conditions_selection if \
                bpy.context.scene.selection_type == 'spec_cond' else None
            data, colors = mu.evaluate_fcurves(elec_obj, time_range, curr_cond)
            elecs_data.update(data)
            elecs_colors.update(colors)
    else:
        parent_obj = bpy.data.objects['Deep_electrodes']
        elecs_data, elecs_colors = mu.evaluate_fcurves(parent_obj, time_range)
    return elecs_data, elecs_colors


def get_electrodes_sources_data():
    elecs_sources_data, elecs_sources_colors = OrderedDict(), OrderedDict()
    labels_data, subcortical_data = PlayPanel.addon.get_elecctrodes_sources()
    cond_inds = np.where(subcortical_data['conditions'] == bpy.context.scene.conditions_selection)[0]
    if len(cond_inds) == 0:
        print("!!! Can't find the current condition in the data['conditions'] !!!")
        return
    for region, color_mat, data_mat in zip(subcortical_data['names'], subcortical_data['colors'],
                                           subcortical_data['data']):
        color = color_mat[:, cond_inds[0], :]
        data = data_mat[:, cond_inds[0]]
        elecs_sources_data[region] = data
        elecs_sources_colors[region] = color
    for hemi in mu.HEMIS:
        for label, color_mat, data_mat in zip(labels_data[hemi]['names'], labels_data[hemi]['colors'],
                                              labels_data[hemi]['data']):
            color = color_mat[:, cond_inds[0], :]
            data = data_mat[:, cond_inds[0]]
            elecs_sources_data[label] = data
            elecs_sources_colors[label] = color
    return elecs_sources_data, elecs_sources_colors


def init_plotting():
    stat = 'avg' if bpy.context.scene.selection_type == 'conds' else 'diff'
    fol = op.join(mu.get_user_fol(), 'electrodes')
    data_fname = op.join(fol, 'electrodes_data_{}.npz'.format(stat))
    meta_fname = op.join(fol, 'electrodes_data_{}_meta.npz'.format(stat))
    colors_fname = op.join(fol, 'electrodes_data_{}_colors.npy'.format(stat))
    d = None
    if op.isfile(data_fname):
        d = np.load(data_fname)
        PlayPanel.electrodes_colors = d['colors']
    elif op.isfile(meta_fname):
        d = np.load(meta_fname)
        PlayPanel.electrodes_colors = np.load(colors_fname)
    else:
        print('No electrodes data file!')
    if not d is None:
        PlayPanel.electrodes_names = [elc.astype(str) for elc in d['names']]
    # Warning: Not sure why we call this line, it changes the brain to the rendered brain
    # PlayPanel.addon.init_activity_map_coloring('MEG')
    PlayPanel.faces_verts = PlayPanel.addon.get_faces_verts()
    PlayPanel.meg_sub_activity = PlayPanel.addon.load_meg_subcortical_activity()
    # connections_file = op.join(mu.get_user_fol(), 'electrodes_coh.npz')
    # if not op.isfile(connections_file):
    #     print('No connections coherence file! {}'.format(connections_file))
    # else:
    #     PlayPanel.electrodes_coherence = mu.Bag(np.load(connections_file))


def init_stim():
    stim_fname = op.join(mu.get_user_fol(), 'electrodes',
        'stim_electrodes_{}.npz'.format(bpy.context.scene.stim_files.replace(' ', '_')))
    if op.isfile(stim_fname):
        PlayPanel.stim_data = np.load(stim_fname)
        PlayPanel.electrodes_names = [elc.astype(str) for elc in PlayPanel.stim_data['names']]
        # PlayPanel.stim_colors = d['colors']
        # PlayPanel.stim_names = [elc.astype(str) for elc in d['names']]


def set_play_from(play_from):
    bpy.context.scene.frame_current = play_from
    bpy.context.scene.play_from = play_from
    bpy.data.scenes['Scene'].frame_preview_start = play_from
    ModalTimerOperator.limits = play_from


def set_play_to(play_to):
    bpy.context.scene.play_to = play_to
    bpy.data.scenes['Scene'].frame_preview_end = play_to


def set_play_dt(play_dt):
    bpy.context.scene.play_dt = play_dt


def set_play_type(play_type):
    bpy.context.scene.play_type = play_type


class PlayPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "Play"
    addon = None
    data = None
    loop_indices = None
    is_playing = False
    play_reverse = False
    first_time = True
    init_play = True
    # imp_times = [[148, 221], [247, 273], [410, 555], [903, 927]]

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
    layout.prop(context.scene, 'render_movie', text="Render to a movie")
    layout.operator(ExportGraph.bl_idname, text="Export graph", icon='SNAP_NORMAL')


class Play(bpy.types.Operator):
    bl_idname = "mmvt.play"
    bl_label = "play"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        PlayPanel.is_playing = True
        PlayPanel.play_reverse = False
        PlayPanel.init_play = True
        if PlayPanel.first_time:
            print('Starting the play timer!')
            # PlayPanel.first_time = False
            ModalTimerOperator.limits = bpy.context.scene.play_from
            bpy.ops.wm.modal_timer_operator()
        return {"FINISHED"}


class Reverse(bpy.types.Operator):
    bl_idname = "mmvt.reverse"
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
    bl_idname = "mmvt.pause"
    bl_label = "pause"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        PlayPanel.is_playing = False
        plot_something(self, context, bpy.context.scene.frame_current, ModalTimerOperator._uuid)
        print('Stop!')
        return {"FINISHED"}


class PrevKeyFrame(bpy.types.Operator):
    bl_idname = 'mmvt.prev_key_frame'
    bl_label = 'prevKeyFrame'
    bl_options = {'UNDO'}

    def invoke(self, context, event=None):
        PlayPanel.is_playing = False
        bpy.context.scene.frame_current -= bpy.context.scene.play_from
        plot_something(self, context, bpy.context.scene.frame_current, ModalTimerOperator._uuid)
        return {'FINISHED'}


class NextKeyFrame(bpy.types.Operator):
    bl_idname = "mmvt.next_key_frame"
    bl_label = "nextKeyFrame"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        PlayPanel.is_playing = False
        bpy.context.scene.frame_current += bpy.context.scene.play_dt
        plot_something(self, context, bpy.context.scene.frame_current, ModalTimerOperator._uuid)
        return {"FINISHED"}


class GrabFromPlay(bpy.types.Operator):
    bl_idname = "mmvt.grab_from_play"
    bl_label = "grab from"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        set_play_from(bpy.context.scene.frame_current)
        return {"FINISHED"}


class GrabToPlay(bpy.types.Operator):
    bl_idname = "mmvt.grab_to_play"
    bl_label = "grab to"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        set_play_to(bpy.context.scene.frame_current)
        return {"FINISHED"}


class ExportGraph(bpy.types.Operator):
    bl_idname = "mmvt.export_graph"
    bl_label = "mmvt export_graph"
    bl_options = {"UNDO"}
    uuid = mu.rand_letters(5)

    @staticmethod
    def invoke(self, context, event=None):
        # image_fol = op.join(mu.get_user_fol(), 'images', ExportGraph.uuid)
        capture_graph()
        return {"FINISHED"}


def init(addon):
    register()
    PlayPanel.addon = addon
    init_plotting()
    # print('PlayPanel initialization completed successfully!')


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
        bpy.utils.register_class(ExportGraph)
        # print('PlayPanel was registered!')
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
        bpy.utils.unregister_class(ExportGraph)
    except:
        pass
        # print("Can't unregister PlayPanel!")
        # print(traceback.format_exc())