import bpy
import os.path as op
import glob
import mmvt_utils as mu


def _addon():
    return TemplatePanel.addon


def update_something():
    pass


def do_somthing():
    pass


def template_files_update(self, context):
    if TemplatePanel.init:
        update_something()


def template_draw(self, context):
    layout = self.layout
    layout.operator(TemplateButton.bl_idname, text="Do something", icon='ROTATE')


class TemplateButton(bpy.types.Operator):
    bl_idname = "mmvt.template_button"
    bl_label = "Template botton"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        do_somthing()
        return {'PASS_THROUGH'}


bpy.types.Scene.template_files = bpy.props.EnumProperty(items=[], description="tempalte files")


class TemplatePanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "Template"
    addon = None
    init = False

    def draw(self, context):
        if TemplatePanel.init:
            template_draw(self, context)


def init(addon):
    TemplatePanel.addon = addon
    user_fol = mu.get_user_fol()
    template_files = glob.glob(op.join(user_fol, 'template', 'template*.npz'))
    if len(template_files) == 0:
        return None
    files_names = [mu.namebase(fname)[len('template'):].replace('_', ' ') for fname in template_files]
    template_items = [(c, c, '', ind) for ind, c in enumerate(files_names)]
    bpy.types.Scene.template_files = bpy.props.EnumProperty(
        items=template_items, description="tempalte files",update=template_files_update)
    bpy.context.scene.template_files = files_names[0]
    register()
    TemplatePanel.init = True


def register():
    try:
        unregister()
        bpy.utils.register_class(TemplatePanel)
        bpy.utils.register_class(TemplateButton)
    except:
        print("Can't register Template Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(TemplatePanel)
        bpy.utils.unregister_class(TemplateButton)
    except:
        pass
