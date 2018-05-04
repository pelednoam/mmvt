import bpy
import os.path as op
import glob
import time
import traceback
import mmvt_utils as mu
import os

preview_collections = {}

def _addon():
    return HNNPanel.addon


def update_something():
    pass


def run_hnn():
    hnn_folder = op.abspath(bpy.path.abspath(bpy.context.scene.hnn_folder))
    hnn_cmd = '{} -dataf {} -paramf {}'.format(
        op.join(hnn_folder, 'hnn.sh'),
        op.join(hnn_folder, 'data',  '{}.txt'.format(bpy.context.scene.hnn_data_files)),
        op.join(hnn_folder, 'param', '{}.param'.format(bpy.context.scene.hnn_param_files)))
    mu.run_command_in_new_thread(hnn_cmd, False, cwd=hnn_folder)


def set_hnn_folder(self, context):
    hnn_folder = op.abspath(bpy.path.abspath(bpy.context.scene.hnn_folder))
    if not op.isdir(hnn_folder):
        return
    hnn_link = op.join(mu.get_links_dir(), 'hnn')
    if op.islink(hnn_link):
        os.remove(hnn_link)
    try:
        os.symlink(hnn_folder, hnn_link)
    except:
        print('set_hnn_folder: Error in creating hnn link!')
    init_hnn_files()


def init_hnn_files():
    hnn_folder = bpy.path.abspath(bpy.context.scene.hnn_folder)
    HNNPanel.data_files_names = [mu.namebase(fname) for fname in glob.glob(op.join(hnn_folder, 'data', '*.txt'))]
    if len(HNNPanel.data_files_names) > 0:
        hnn_items = [(c, c, '', ind) for ind, c in enumerate(HNNPanel.data_files_names)]
        bpy.types.Scene.hnn_data_files = bpy.props.EnumProperty(items=hnn_items, description="data files")
        bpy.context.scene.hnn_data_files = HNNPanel.data_files_names[0]
    HNNPanel.params_files_names = [mu.namebase(fname) for fname in glob.glob(op.join(hnn_folder, 'param', '*.param'))]
    if len(HNNPanel.params_files_names) > 0:
        hnn_items = [(c, c, '', ind) for ind, c in enumerate(HNNPanel.params_files_names)]
        bpy.types.Scene.hnn_param_files = bpy.props.EnumProperty(items=hnn_items, description="param files")
        bpy.context.scene.hnn_param_files = HNNPanel.params_files_names[0]


def hnn_draw(self, context):
    hnn_icon = preview_collections['main']['hnn_icon']
    layout = self.layout
    layout.prop(context.scene, 'hnn_folder')
    if op.isdir(bpy.path.abspath(bpy.context.scene.hnn_folder)):
        layout.prop(context.scene, 'hnn_data_files', text='Data file')
        layout.prop(context.scene, 'hnn_param_files', text='Param file')
        # if len(HNNPanel.params_files_names) == 0:
        #     layout.operator(LoadParamFile.bl_idname, text="Load param file", icon='ROTATE')
        # if len(HNNPanel.data_files_names) == 0:
        #     layout.operator(LoadDataFile.bl_idname, text="Load data file", icon='ROTATE')
        if len(HNNPanel.params_files_names) > 0 and len(HNNPanel.data_files_names) > 0:
            layout.operator(RunHNN.bl_idname, text="Run HNN", icon='POSE_HLT')


class LoadParamFile(bpy.types.Operator):
    bl_idname = "mmvt.load_hnn_param_file"
    bl_label = "HNN load hnn param file"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        pass #load_param_file()
        return {'PASS_THROUGH'}


class LoadDataFile(bpy.types.Operator):
    bl_idname = "mmvt.load_hnn_data_file"
    bl_label = "HNN load hnn data file"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        pass # load_data_file()
        return {'PASS_THROUGH'}


class RunHNN(bpy.types.Operator):
    bl_idname = "mmvt.run_hnn"
    bl_label = "HNN run"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        run_hnn()
        return {'PASS_THROUGH'}


bpy.types.Scene.hnn_data_files = bpy.props.EnumProperty(items=[], description="data files")
bpy.types.Scene.hnn_param_files = bpy.props.EnumProperty(items=[], description="params files")
bpy.types.Scene.hnn_folder = bpy.props.StringProperty(subtype='DIR_PATH', update=set_hnn_folder)


class HNNPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "HNN"
    addon = None
    init = False
    data_files_names = []
    params_files_names = []

    def draw(self, context):
        if HNNPanel.init:
            hnn_draw(self, context)


def init(addon):
    HNNPanel.addon = addon
    hnn_folder = mu.get_link_dir(mu.get_links_dir(), 'hnn')
    if op.isdir(hnn_folder):
        bpy.context.scene.hnn_folder = hnn_folder
        init_hnn_files()
    register()
    HNNPanel.init = True


def init_logo():
    import bpy.utils.previews
    pcoll = bpy.utils.previews.new()
    hnn_icons_dir = op.join(mu.get_parent_fol(mu.get_user_fol()), 'icons')
    # load a preview thumbnail of a file and store in the previews collection
    pcoll.load("hnn_icon", os.path.join(hnn_icons_dir, "hnn.png"), 'IMAGE')
    preview_collections["main"] = pcoll


def register():
    try:
        unregister()
        init_logo()
        bpy.utils.register_class(HNNPanel)
        bpy.utils.register_class(RunHNN)
        bpy.utils.register_class(LoadParamFile)
        bpy.utils.register_class(LoadDataFile)
    except:
        print("Can't register HNN Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(HNNPanel)
        bpy.utils.unregister_class(RunHNN)
        bpy.utils.unregister_class(LoadParamFile)
        bpy.utils.unregister_class(LoadDataFile)
    except:
        pass
