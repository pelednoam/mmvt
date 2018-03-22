import bpy
import os.path as op
import sys
import glob
import importlib
import inspect
import traceback
import mmvt_utils as mu


def _addon():
    return ScriptsPanel.addon


def run_script():
    try:
        script_name = bpy.context.scene.scripts_files.replace(' ', '_')
        lib = importlib.import_module(script_name)
        importlib.reload(lib)
        run_func = getattr(lib, 'run')
        func_signature = inspect.signature(run_func)
        if len(func_signature.parameters) == 2:
            run_func(_addon(), bpy.context.scene.scripts_overwrite)
        elif len(func_signature.parameters) == 1:
            run_func(_addon())
    except:
        print("run_script: Can't run {}!".format(script_name))
        print(traceback.format_exc())


def scripts_draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'scripts_files', text='')
    # row = layout.row(align=0)
    layout.operator(RunScript.bl_idname, text="Run script", icon='POSE_HLT')
    layout.prop(context.scene, 'scripts_overwrite', 'Overwrite')


class RunScript(bpy.types.Operator):
    bl_idname = "mmvt.scripts_button"
    bl_label = "Scripts botton"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        run_script()
        return {'PASS_THROUGH'}


bpy.types.Scene.scripts_files = bpy.props.EnumProperty(items=[], description="scripts files")
bpy.types.Scene.scripts_overwrite = bpy.props.BoolProperty(default=False)


class ScriptsPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "Scripts"
    addon = None
    init = False

    def draw(self, context):
        if ScriptsPanel.init:
            scripts_draw(self, context)


def init(addon):
    ScriptsPanel.addon = addon
    user_fol = mu.get_user_fol()
    scripts_files = glob.glob(op.join(mu.get_parent_fol(user_fol), 'scripts', '*.py'))
    if len(scripts_files) == 0:
        return None
    sys.path.append(op.join(mu.get_parent_fol(user_fol), 'scripts'))
    files_names = [mu.namebase(fname).replace('_', ' ') for fname in scripts_files]
    scripts_items = [(c, c, '', ind) for ind, c in enumerate(files_names)]
    bpy.types.Scene.scripts_files = bpy.props.EnumProperty(items=scripts_items, description="scripts files")
    bpy.context.scene.scripts_files = files_names[0]
    register()
    ScriptsPanel.init = True


def register():
    try:
        unregister()
        bpy.utils.register_class(ScriptsPanel)
        bpy.utils.register_class(RunScript)
    except:
        print("Can't register Scripts Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(ScriptsPanel)
        bpy.utils.unregister_class(RunScript)
    except:
        pass
