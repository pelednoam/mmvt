import bpy
import mmvt_utils as mu
import numpy as np
import os.path as op


def save_cursor_position():
    root = mu.get_user_fol()
    point = bpy.context.scene.cursor_location * 10.0
    freeview_app = '/Applications/freesurfer/Freeview.app/Contents/MacOS/Freeview' #if mu.is_mac() else 'freeview'
    freeview_cmd = '{} --ras {} {} {}\n'.format(freeview_app, point[0], point[1], point[2]).encode()
    if FreeviewPanel.freeview_queue:
        FreeviewPanel.freeview_queue.put(freeview_cmd)
    freeview_fol = op.join(root, 'freeview')
    mu.make_dir(freeview_fol)
    np.savetxt(op.join(freeview_fol, 'edit.dat'), point)
    cursor_position = np.array(bpy.context.scene.cursor_location) * 10
    ret = mu.conn_to_listener.send_command(dict(cmd='slice_viewer_change_pos',data=dict(
        position=cursor_position)))
    if not ret:
        mu.message(None, 'Listener was stopped! Try to restart')


def goto_cursor_position():
    root = mu.get_user_fol()
    point = np.genfromtxt(op.join(root, 'freeview', 'edit.dat'))
    bpy.context.scene.cursor_location = point / 10.0


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


class FreeviewOpen(bpy.types.Operator):
    bl_idname = "ohad.freeview_open"
    bl_label = "Open Freeview"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        root = mu.get_user_fol()
        sig = op.join(root, 'freeview', 'sig_subject.mgz')
        sig_cmd = '' #'-v {}:colormap=heat' if op.isfile(sig) else ''
        T1 = op.join(root, 'freeview', 'T1.mgz')#'orig.mgz')
        aseg = op.join(root, 'freeview', '{}+aseg.mgz'.format(bpy.context.scene.atlas))
        lut = op.join(root, 'freeview', '{}ColorLUT.txt'.format(bpy.context.scene.atlas))
        electrodes = self.get_electrodes_groups(root)
        freeview_app = '/Applications/freesurfer/Freeview.app/Contents/MacOS/Freeview' #if mu.is_mac() else 'freeview'
        cmd = r'{} {} {}:opacity=0.3 {}:opacity=0.05:colormap=lut:lut={} -c {}'.format(freeview_app, sig_cmd, T1, aseg, lut, electrodes)
        FreeviewPanel.freeview_queue = mu.run_command_in_new_thread(cmd)
        return {"FINISHED"}

    def get_electrodes_groups(self, root):
        groups = set([obj.name[:3] for obj in bpy.data.objects['Deep_electrodes'].children])
        groups_files = ''
        for group in groups:
            groups_files += op.join(root, 'freeview', '{}.dat '.format(group))
        return groups_files


class FreeviewPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
    bl_label = "Freeview Panel"
    addon = None
    freeview_queue = None

    def draw(self, context):
        layout = self.layout
        row = layout.row(align=0)
        row.operator(FreeviewOpen.bl_idname, text="Freeview", icon='PARTICLES')
        row = layout.row(align=0)
        row.operator(FreeviewGotoCursor.bl_idname, text="Goto Cursor", icon='HAND')
        row.operator(FreeviewSaveCursor.bl_idname, text="Save Cursor", icon='FORCE_CURVE')
        row = layout.row(align=0)
        row.operator(SliceViewerOpen.bl_idname, text="Slice Viewer", icon='PARTICLES')


def init(addon):
    FreeviewPanel.addon = addon
    register()


def register():
    try:
        unregister()
        bpy.utils.register_class(FreeviewGotoCursor)
        bpy.utils.register_class(FreeviewSaveCursor)
        bpy.utils.register_class(FreeviewOpen)
        bpy.utils.register_class(FreeviewPanel)
        bpy.utils.register_class(SliceViewerOpen)
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
    except:
        print("Can't unregister Freeview Panel!")
