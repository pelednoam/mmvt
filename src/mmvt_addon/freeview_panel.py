import bpy
import mmvt_utils as mu
import numpy as np
import os.path as op
import time
import glob
import sys
from sys import platform as _platform
import re

# MAC_FREEVIEW_CMD = '/Applications/freesurfer/Freeview.app/Contents/MacOS/Freeview'

bpy.types.Scene.freeview_listen_to_keyboard = bpy.props.BoolProperty(default=False)
bpy.types.Scene.freeview_listener_is_running = bpy.props.BoolProperty(default=False)


def save_cursor_position():
    root = mu.get_user_fol()
    point = bpy.context.scene.cursor_location * 10.0
    freeview_cmd = 'freeview --ras {} {} {} tkreg\n'.format(point[0], point[1], point[2]) # .encode()
    if FreeviewPanel.freeview_in_queue:
        FreeviewPanel.freeview_in_queue.put(freeview_cmd)
    freeview_fol = op.join(root, 'freeview')
    mu.make_dir(freeview_fol)
    np.savetxt(op.join(freeview_fol, 'edit.dat'), point)
    cursor_position = np.array(bpy.context.scene.cursor_location) * 10
    ret = mu.conn_to_listener.send_command(dict(cmd='slice_viewer_change_pos',data=dict(
        position=cursor_position)))
    # if not ret:
    #     mu.message(None, 'Listener was stopped! Try to restart')


def goto_cursor_position():
    root = mu.get_user_fol()
    point = np.genfromtxt(op.join(root, 'freeview', 'edit.dat'))
    bpy.context.scene.cursor_location = tuple(point / 10.0)


class FreeviewKeyboardListener(bpy.types.Operator):
    bl_idname = 'ohad.freeview_keyboard_listener'
    bl_label = 'freeview_keyboard_listener'
    bl_options = {'UNDO'}
    press_time = time.time()

    def modal(self, context, event):
        if time.time() - self.press_time > 1 and bpy.context.scene.freeview_listen_to_keyboard and \
                event.type not in ['TIMER', 'MOUSEMOVE', 'WINDOW_DEACTIVATE', 'INBETWEEN_MOUSEMOVE', 'TIMER_REPORT', 'NONE']:
            self.press_time = time.time()
            print(event.type)
            if event.type == 'LEFTMOUSE':
                save_cursor_position()
            else:
                pass
        return {'PASS_THROUGH'}

    def invoke(self, context, event=None):
        if not bpy.context.scene.freeview_listener_is_running:
            context.window_manager.modal_handler_add(self)
            bpy.context.scene.freeview_listener_is_running = True
        bpy.context.scene.freeview_listen_to_keyboard = not bpy.context.scene.freeview_listen_to_keyboard
        return {'RUNNING_MODAL'}


class FreeviewGotoCursor(bpy.types.Operator):
    bl_idname = "ohad.freeview_goto_cursor"
    bl_label = "Goto Cursor"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        goto_cursor_position()
        return {"FINISHED"}


class FreeviewSaveCursor(bpy.types.Operator):
    bl_idname = "ohad.freeview_save_cursor"
    bl_label = "Save Cursor"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        save_cursor_position()
        return {"FINISHED"}


class SliceViewerOpen(bpy.types.Operator):
    bl_idname = "ohad.slice_viewer"
    bl_label = "Slice Viewer"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        mri_fname = op.join(mu.get_user_fol(), 'freeview', 'orig.mgz')
        cursor_position = np.array(bpy.context.scene.cursor_location) * 10.0
        ret = mu.conn_to_listener.send_command(dict(cmd='open_slice_viewer',data=dict(
            mri_fname=mri_fname, position=cursor_position)))
        if not ret:
            mu.message(self, 'Listener was stopped! Try to restart')
        return {"FINISHED"}


def decode_freesurfer_output(fv_str):
    # fv_str = str(b'\xc0\x8ak\x01tkReg: -8.23399;\x0f(\xa45 0\x96o) 1.28009\n')
    fv_str = str(fv_str)
    fv_str = fv_str.replace('\\xc0\\x8ak\\x01', '')
    fv_str = fv_str.replace('\\n', '')
    from_ind = fv_str.find('\\')
    ind = from_ind
    while from_ind != -1:
        ind2 = fv_str[ind + 1:].find(' ')
        ind3 = fv_str[ind + 1:].find('\\')
        if ind2 == -1 and ind3 == -1:
            ind_to = len(fv_str) - ind
        elif ind2 == -1 or ind3 == -1:
            ind_to = ind2 if ind3 == -1 else ind3
        else:
            ind_to = min([ind2, ind3])
        str_to_remove = fv_str[ind:ind_to + ind + 1]
        fv_str = fv_str.replace(str_to_remove, '')
        from_ind = fv_str[ind:].find('\\')
        ind += from_ind
    print('stdout after cleaning: {}'.format(fv_str))
    return fv_str


class FreeviewOpen(bpy.types.Operator):
    bl_idname = "ohad.freeview_open"
    bl_label = "Open Freeview"
    bl_options = {"UNDO"}
    _updating = False
    _calcs_done = False
    _timer = None

    def modal(self, context, event):
        if event.type == 'TIMER' and not self._updating:
            self._updating = True
            # print('Listening to the queue')
            if not FreeviewPanel.freeview_out_queue is None:
                from queue import Empty
                try:
                    freeview_data = FreeviewPanel.freeview_out_queue.get(block=False)
                    try:
                        print('stdout from freeview: {}'.format(freeview_data))
                        # data_deocded = freeview_data.decode(sys.getfilesystemencoding(), 'ignore')
                        data_deocded = decode_freesurfer_output(freeview_data)
                        if 'tkReg' in data_deocded:
                            point = mu.read_numbers_rx(data_deocded)
                            print(point)
                            bpy.context.scene.cursor_location = tuple(np.array(point, dtype=np.float) / 10)
                    except:
                        print("Can't read the stdout from freeview")
                except Empty:
                    pass
            self._updating = False
        if self._calcs_done:
            self.cancel(context)
        return {'PASS_THROUGH'}

    def cancel(self, context):
        context.window_manager.event_timer_remove(self._timer)
        self._timer = None
        return {'CANCELLED'}

    def invoke(self, context, event=None):
        root = mu.get_user_fol()
        if bpy.context.scene.fMRI_files_exist and bpy.context.scene.freeview_load_fMRI:
            sig_fname = op.join(root, 'freeview', '{}.mgz'.format(bpy.context.scene.fmri_files))
            sig_cmd = '-v "{}":colormap=heat:heatscale=2,3,6'.format(sig_fname) if op.isfile(sig_fname) else ''
        else:
            sig_cmd = ''
        T1 = op.join(root, 'freeview', 'T1.mgz')  # sometimes 'orig.mgz' is better
        aseg = op.join(root, 'freeview', '{}+aseg.mgz'.format(bpy.context.scene.atlas))
        lut = op.join(root, 'freeview', '{}ColorLUT.txt'.format(bpy.context.scene.atlas))
        electrodes_cmd = self.get_electrodes_command(root)
        cmd = '{} {} "{}":opacity=0.3 "{}":opacity=0.05:colormap=lut:lut="{}"{}{}{}'.format(
            FreeviewPanel.addon_prefs.freeview_cmd, sig_cmd, T1, aseg, lut, electrodes_cmd,
            ' -verbose' if FreeviewPanel.addon_prefs.freeview_cmd_verbose else '',
            ' -stdin' if FreeviewPanel.addon_prefs.freeview_cmd_stdin else '')
        print(cmd)
        FreeviewPanel.freeview_in_queue, FreeviewPanel.freeview_out_queue = mu.run_command_in_new_thread(cmd)
        context.window_manager.modal_handler_add(self)
        self._updating = False
        self._timer = context.window_manager.event_timer_add(0.1, context.window)
        return {'RUNNING_MODAL'}
        # return {"FINISHED"}

    def get_electrodes_command(self, root):
        if bpy.data.objects.get('Deep_electrodes'):
            cmd = ' -c '
            groups = set([obj.name[:3] for obj in bpy.data.objects['Deep_electrodes'].children])
            for group in groups:
                cmd += '"{}" '.format(op.join(root, 'freeview', '{}.dat'.format(group)))
        else:
            cmd = ''
        return cmd


bpy.types.Scene.electrodes_exist = bpy.props.BoolProperty(default=True)
bpy.types.Scene.freeview_load_electrodes = bpy.props.BoolProperty(default=True, description='Load electrodes')
bpy.types.Scene.fMRI_files_exist = bpy.props.BoolProperty(default=True)
bpy.types.Scene.freeview_load_fMRI = bpy.props.BoolProperty(default=True, description='Load fMRI')


class FreeviewPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
    bl_label = "Freeview Panel"
    addon = None
    freeview_in_queue = None
    freeview_out_queue = None

    def draw(self, context):
        layout = self.layout
        # row = layout.row(align=0)
        layout.operator(FreeviewOpen.bl_idname, text="Freeview", icon='PARTICLES')
        if bpy.data.objects.get('Deep_electrodes'):
            layout.prop(context.scene, 'freeview_load_electrodes', text="Load electrodes")
        if bpy.context.scene.fMRI_files_exist and \
                op.isfile(op.join(mu.get_user_fol(), 'freeview', '{}.mgz'.format(bpy.context.scene.fmri_files))):
            layout.prop(context.scene, 'freeview_load_fMRI', text="Load fMRI")
        row = layout.row(align=0)
        row.operator(FreeviewGotoCursor.bl_idname, text="Goto Cursor", icon='HAND')
        row.operator(FreeviewSaveCursor.bl_idname, text="Save Cursor", icon='FORCE_CURVE')
        row = layout.row(align=0)
        row.operator(SliceViewerOpen.bl_idname, text="Slice Viewer", icon='PARTICLES')
        if not bpy.context.scene.freeview_listen_to_keyboard:
            layout.operator(FreeviewKeyboardListener.bl_idname, text="Listen to keyboard", icon='COLOR_GREEN')
        else:
            layout.operator(FreeviewKeyboardListener.bl_idname, text="Stop listen to keyboard", icon='COLOR_RED')


def init(addon, addon_prefs=None):
    FreeviewPanel.addon = addon
    print('freeview command: {}'.format(addon_prefs.freeview_cmd))
    print('Use -verbose? {}'.format(addon_prefs.freeview_cmd_verbose))
    print('Use -stdin? {}'.format(addon_prefs.freeview_cmd_stdin))
    FreeviewPanel.addon_prefs = addon_prefs
    bpy.context.scene.freeview_listen_to_keyboard = False
    bpy.context.scene.freeview_listener_is_running = False
    bpy.context.scene.fMRI_files_exist = len(glob.glob(op.join(mu.get_user_fol(), 'fmri', '*_lh.npy'))) > 0
        #mu.hemi_files_exists(op.join(mu.get_user_fol(), 'fmri_{hemi}.npy'))
    bpy.context.scene.electrodes_exist = not bpy.data.objects.get('Deep_electrodes', None) is None
    register()


def register():
    try:
        unregister()
        bpy.utils.register_class(FreeviewGotoCursor)
        bpy.utils.register_class(FreeviewSaveCursor)
        bpy.utils.register_class(FreeviewOpen)
        bpy.utils.register_class(FreeviewPanel)
        bpy.utils.register_class(SliceViewerOpen)
        bpy.utils.register_class(FreeviewKeyboardListener)
        print('Freeview Panel was registered!')
    except:
        print("Can't register Freeview Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(FreeviewGotoCursor)
        bpy.utils.unregister_class(FreeviewSaveCursor)
        bpy.utils.unregister_class(FreeviewOpen)
        bpy.utils.unregister_class(FreeviewPanel)
        bpy.utils.unregister_class(SliceViewerOpen)
        bpy.utils.unregister_class(FreeviewKeyboardListener)
    except:
        pass
        # print("Can't unregister Freeview Panel!")
