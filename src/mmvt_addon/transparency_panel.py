import bpy


def _addon():
    return TransparencyPanel.addon


def appearance_update(self=None, context=None):
    _addon().make_brain_solid_or_transparent()
    _addon().update_layers()


def set_brain_transparency(val):
    if 0 <= val <= 1:
        bpy.context.scene.appearance_solid_slider = 1 - val
        appearance_update()
    else:
        print('transparency value must be between 0 (not transparent) and 1')


def set_light_layers_depth(val):
    if 0 <= val <= 10:
        bpy.context.scene.appearance_depth_slider = val
        appearance_update()
    else:
        print('light layers depth must be between 0 and 10')


def transparency_draw(self, context):
    if context.scene.filter_view_type == 'rendered' and bpy.context.scene.appearance_show_rois_activity == 'activity':
    # if context.scene.filter_view_type == 'RENDERED' and bpy.context.scene.appearance_show_activity_layer is True:
        layout = self.layout
        layout.prop(context.scene, 'appearance_solid_slider', text="Show solid brain")
        split2 = layout.split()
        # split2.prop(context.scene, 'appearance_depth_Bool', text="Show cortex deep layers")
        split2.prop(context.scene, 'appearance_depth_slider', text="Depth")
        # layout.operator("mmvt.appearance_update", text="Update")


class UpdateAppearance(bpy.types.Operator):
    bl_idname = "mmvt.appearance_update"
    bl_label = "filter clear"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        _addon().make_brain_solid_or_transparent()
        _addon().update_layers()
        return {"FINISHED"}


class TransparencyPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "Transparency"
    addon = None
    init = False

    def draw(self, context):
        transparency_draw(self, context)


bpy.types.Scene.appearance_solid_slider = bpy.props.FloatProperty(default=0.0, min=0, max=1, update=appearance_update)
bpy.types.Scene.appearance_depth_slider = bpy.props.IntProperty(default=0, min=0, max=10, update=appearance_update)


def init(addon):
    TransparencyPanel.addon = addon
    TransparencyPanel.init = True
    register()


def register():
    try:
        unregister()
        bpy.utils.register_class(TransparencyPanel)
        bpy.utils.register_class(UpdateAppearance)
        # print('Transparency Panel was registered!')
    except:
        print("Can't register Transparency Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(TransparencyPanel)
        bpy.utils.unregister_class(UpdateAppearance)
    except:
        pass
        # print("Can't unregister Freeview Panel!")
