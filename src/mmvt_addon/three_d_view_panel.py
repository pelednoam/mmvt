import bpy
import mmvt_utils as mu


def _addon():
    return ThreeDPanel.addon


def view_center_cursor():
    bpy.ops.view3d.view_center_cursor()


def view_center_camera():
    bpy.ops.view3d.view_center_camera()


def three_d_draw(self, context):
    layout = self.layout
    layout.operator(ViewCenterCamera.bl_idname, text='view center camera', icon='EDITMODE_HLT')
    layout.operator(ViewAll.bl_idname, text='view all', icon='EDITMODE_HLT')


class ViewCenterCamera(bpy.types.Operator):
    bl_idname = "mmvt.view_center_camera"
    bl_label = "ThreeDPanel"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        view_center_cursor()
        return {'PASS_THROUGH'}


class ViewAll(bpy.types.Operator):
    bl_idname = "mmvt.three_d_view_all"
    bl_label = "ThreeDPanel"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        bpy.ops.view3d.view_all()
        return {'PASS_THROUGH'}


class ThreeDPanel(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "three_d_view"
    addon = None
    init = False
    init_play = False
    is_playing = False
    first_time = True
    play_time_step = 1

    def draw(self, context):
        if ThreeDPanel.init:
            three_d_draw(self, context)


def init(addon):
    ThreeDPanel.addon = addon
    register()
    ThreeDPanel.init = True


def register():
    try:
        unregister()
        bpy.utils.register_class(ThreeDPanel)
        bpy.utils.register_class(ViewCenterCamera)
        bpy.utils.register_class(ViewAll)
    except:
        print("Can't register three_d Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(ThreeDPanel)
        bpy.utils.unregister_class(ViewCenterCamera)
        bpy.utils.unregister_class(ViewAll)
    except:
        pass
