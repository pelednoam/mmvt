import bpy
import bpy_extras
import os.path as op
import shutil
import mmvt_utils as mu

try:
    import mne
    MNE_EXIST = True
except:
    MNE_EXIST = False


def _addon():
    return LoadResultsPanel.addon


class LoadSTCFile(bpy.types.Operator, bpy_extras.io_utils.ExportHelper):
    bl_idname = "mmvt.choose_stc_file" 
    bl_label = "Choose STC file"

    filename_ext = ".stc"
    filter_glob = bpy.props.StringProperty(default='*.stc', options={'HIDDEN'}, maxlen=255)

    def execute(self, context):
        stc_fname = self.filepath
        user_fol = mu.get_user_fol()
        stc_fol = mu.get_fname_folder(stc_fname)
        if stc_fol != op.join(user_fol, 'meg'):
            other_hemi_stc_fname = op.join(stc_fol, '{}.stc'.format(mu.change_hemi(mu.namebase(stc_fname))))
            shutil.copy(stc_fname, op.join(user_fol, 'meg', mu.namesbase_with_ext(stc_fname)))
            shutil.copy(other_hemi_stc_fname, op.join(user_fol, 'meg', mu.namesbase_with_ext(other_hemi_stc_fname)))
            _addon().create_stc_files_list()
        _, _, label, hemi = mu.get_hemi_delim_and_pos(mu.namebase(stc_fname))
        bpy.context.scene.meg_files = label
        return {'FINISHED'}


def template_draw(self, context):
    layout = self.layout
    if MNE_EXIST:
        layout.operator(LoadSTCFile.bl_idname, text="Load stc file", icon='LOAD_FACTORY')


class LoadResultsPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "LoadResults"
    addon = None
    init = False

    def draw(self, context):
        if LoadResultsPanel.init:
            template_draw(self, context)


def init(addon):
    LoadResultsPanel.addon = addon
    user_fol = mu.get_user_fol()
    register()
    LoadResultsPanel.init = True


def register():
    try:
        unregister()
        bpy.utils.register_class(LoadResultsPanel)
        bpy.utils.register_class(LoadSTCFile)
    except:
        print("Can't register LoadResults Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(LoadResultsPanel)
        bpy.utils.unregister_class(LoadSTCFile)
    except:
        pass
