import bpy
import os.path as op
import glob
import time
import mmvt_utils as mu


def do_somthing():
    bpy.data.objects['Activity_in_vertex'].select = True


def template_draw(self, context):
    layout = self.layout
    layout.operator(TemplateButton.bl_idname, text="Do something", icon='ROTATE')


class TemplateButton(bpy.types.Operator):
    bl_idname = "ohad.template_button"
    bl_label = "Template botton"
    bl_options = {"UNDO"}
    _time = time.time()

    def modal(self, context, event):
        if event.type == 'TIMER':
            if time.time() - self._time > bpy.context.scene.play_time_step:
                pass



    def invoke(self, context, event=None):
        do_somthing()
        return {'PASS_THROUGH'}


class TempaltePanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
    bl_label = "Template"
    addon = None
    init = False

    def draw(self, context):
        if TempaltePanel.init:
            template_draw(self, context)


def init(addon):
    TempaltePanel.addon = addon
    user_fol = mu.get_user_fol()
    register()
    TempaltePanel.init = True


def register():
    try:
        unregister()
        bpy.utils.register_class(TempaltePanel)
        bpy.utils.register_class(TemplateButton)
    except:
        print("Can't register Template Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(TempaltePanel)
        bpy.utils.unregister_class(TemplateButton)
    except:
        pass
