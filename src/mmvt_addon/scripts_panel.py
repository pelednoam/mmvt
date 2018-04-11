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


def check_script(script_name):
    try:
        lib = importlib.import_module(script_name)
        importlib.reload(lib)
        run_func = getattr(lib, 'run')
        func_signature = inspect.signature(run_func)
        if 1 <= len(func_signature.parameters) <= 2:
            ScriptsPanel.funcs[script_name] = (run_func, len(func_signature.parameters))
            return True
        else:
            return False
    except:
        return False


def run_script():
    try:
        script_name = bpy.context.scene.scripts_files.replace(' ', '_')
        run_func, params_num = ScriptsPanel.funcs[script_name]
        if params_num == 2:
            run_func(_addon(), bpy.context.scene.scripts_overwrite)
        elif params_num == 1:
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
    funcs = {}

    def draw(self, context):
        if ScriptsPanel.init:
            scripts_draw(self, context)


def init(addon):
    ScriptsPanel.addon = addon
    user_fol = mu.get_user_fol()
    scripts_files = glob.glob(op.join(mu.get_mmvt_code_root(), 'src', 'examples', 'scripts', '*.py'))
    scripts_files_names = [mu.namebase(f) for f in scripts_files]
    scripts_files += [f for f in glob.glob(op.join(mu.get_parent_fol(user_fol), 'scripts', '*.py'))
                      if mu.namebase(f) not in scripts_files_names]
    if len(scripts_files) == 0:
        return None
    sys.path.append(op.join(mu.get_parent_fol(user_fol), 'scripts'))
    sys.path.append(op.join(mu.get_mmvt_code_root(), 'src', 'examples', 'scripts'))
    scripts_files = [f for f in scripts_files if check_script(mu.namebase(f))]
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
