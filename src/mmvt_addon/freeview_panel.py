import bpy
import mmvt_utils as mu
import numpy as np
import os.path as op


def save_cursor_position():
    root = mu.get_user_fol()
    point = bpy.context.scene.cursor_location * 10.0
    np.savetxt(op.join(root, 'freeview', 'edit.dat'), point)


def goto_cursor_position():
    root = mu.get_user_fol()
    point = np.genfromtxt(op.join(root, 'freeview', 'edit.dat'))
    bpy.context.scene.cursor_location = point / 10.0


class FreeviewGotoCursor(bpy.types.Operator):
    bl_idname = "ohad.freeview_goto_cursor"
    bl_label = "Goto Cursor"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        goto_cursor_position
        return {"FINISHED"}


class FreeviewSaveCursor(bpy.types.Operator):
    bl_idname = "ohad.freeview_save_cursor"
    bl_label = "Save Cursor"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        save_cursor_position()
        return {"FINISHED"}


class FreeviewOpen(bpy.types.Operator):
    bl_idname = "ohad.freeview_open"
    bl_label = "Open Freeview"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        root = mu.get_user_fol()
        sig = op.join(root, 'freeview', 'sig_subject.mgz')
        sig_cmd = '-v {}:colormap=heat' if op.isfile(sig) else ''
        T1 = op.join(root, 'freeview', 'T1.mgz')
        aseg = op.join(root, 'freeview', '{}+aseg.mgz'.format(bpy.context.scene.atlas))
        lut = op.join(root, 'freeview', '{}ColorLUT.txt'.format(bpy.context.scene.atlas))
        electrodes = self.get_electrodes_groups(root)
        cmd = 'freeview {} {}:opacity=0.3 {}:opacity=0.05:colormap=lut:lut={} -c {}'.format(sig_cmd, T1, aseg, lut, electrodes)
        mu.run_command_in_new_thread(cmd)
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

    def draw(self, context):
        layout = self.layout
        row = layout.row(align=0)
        row.operator(FreeviewOpen.bl_idname, text="Freeview", icon='PARTICLES')
        row = layout.row(align=0)
        row.operator(FreeviewGotoCursor.bl_idname, text="Goto Cursor", icon='HAND')
        row.operator(FreeviewSaveCursor.bl_idname, text="Save Cursor", icon='FORCE_CURVE')


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
        print('Freeview Panel was registered!')
    except:
        print("Can't register Freeview Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(FreeviewGotoCursor)
        bpy.utils.unregister_class(FreeviewSaveCursor)
        bpy.utils.unregister_class(FreeviewOpen)
        bpy.utils.unregister_class(FreeviewPanel)

    except:
        print("Can't unregister Freeview Panel!")
