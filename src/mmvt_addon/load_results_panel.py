import bpy
import bpy_extras
import os.path as op
import shutil
import glob
import mmvt_utils as mu

try:
    import mne
    MNE_EXIST = True
except:
    MNE_EXIST = False


def _addon():
    return LoadResultsPanel.addon


bpy.types.Scene.nii_label_prompt = bpy.props.StringProperty(description='')
bpy.types.Scene.nii_label_output = bpy.props.StringProperty(description='')


class ChooseSTCFile(bpy.types.Operator, bpy_extras.io_utils.ExportHelper):
    bl_idname = "mmvt.choose_stc_file" 
    bl_label = "Choose STC file"

    filename_ext = '.stc'
    filter_glob = bpy.props.StringProperty(default='*.stc', options={'HIDDEN'}, maxlen=255)

    def execute(self, context):
        stc_fname = self.filepath
        user_fol = mu.get_user_fol()
        stc_fol = mu.get_fname_folder(stc_fname)
        if stc_fol != op.join(user_fol, 'meg'):
            other_hemi_stc_fname = op.join(stc_fol, '{}.stc'.format(mu.get_other_hemi_label_name(mu.namebase(stc_fname))))
            shutil.copy(stc_fname, op.join(user_fol, 'meg', mu.namesbase_with_ext(stc_fname)))
            shutil.copy(other_hemi_stc_fname, op.join(user_fol, 'meg', mu.namesbase_with_ext(other_hemi_stc_fname)))
            _addon().init_meg_activity_map()
        _, _, label, hemi = mu.get_hemi_delim_and_pos(mu.namebase(stc_fname))
        bpy.context.scene.meg_files = label
        return {'FINISHED'}


def build_local_fname(nii_fname, user_fol):
    if mu.get_hemi_from_fname(mu.namesbase_with_ext(nii_fname)) == '':
        local_fname = mu.get_label_for_full_fname(nii_fname)
        # local_fname = '{}.{}.{}'.format(mu.namebase(nii_fname), hemi, mu.file_type(nii_fname))
    else:
        local_fname = mu.namesbase_with_ext(nii_fname)
    return op.join(user_fol, 'fmri', local_fname)


def load_surf_files(nii_fname):
    fmri_file_template = ''
    user_fol = mu.get_user_fol()
    nii_fol = mu.get_fname_folder(nii_fname)
    fmri_hemis = mu.get_both_hemis_files(nii_fname)
    hemi = mu.get_hemi_from_full_fname(nii_fname)
    local_fname = build_local_fname(nii_fname, user_fol)
    mu.make_dir(op.join(user_fol, 'fmri'))
    if nii_fol != op.join(user_fol, 'fmri'):
        mu.make_link(nii_fname, local_fname, True)
    other_hemi = mu.other_hemi(hemi)
    other_hemi_fname = fmri_hemis[other_hemi]
    if op.isfile(other_hemi_fname):
        local_other_hemi_fname = build_local_fname(other_hemi_fname, user_fol)
        if nii_fol != op.join(user_fol, 'fmri'):
            mu.make_link(other_hemi_fname, local_other_hemi_fname, True)
        fmri_file_template = mu.get_template_hemi_label_name(mu.namesbase_with_ext(local_fname))
        cmd = '{} -m src.preproc.fMRI -s {} -f load_surf_files --fmri_file_template "{}" --ignore_missing 1'.format(
            bpy.context.scene.python_cmd, mu.get_user(), fmri_file_template)
        mu.run_command_in_new_thread(cmd, False)
    return fmri_file_template, hemi


def clean_nii_temp_files(fmri_file_template):
    file_type = mu.file_type(fmri_file_template)
    file_temp = op.join(mu.get_user_fol(), 'fmri', '{}'.format(
        fmri_file_template.replace('{hemi}', '?h')[:-len(file_type)]))
    mu.delete_files('{}{}'.format(file_temp, file_type))
    mu.delete_files('{}{}'.format(file_temp, 'mgz'))


class ChooseNiftiiFile(bpy.types.Operator, bpy_extras.io_utils.ExportHelper):
    bl_idname = "mmvt.choose_niftii_file"
    bl_label = "Choose niftii file"
    filename_ext = '.nii.gz'
    filter_glob = bpy.props.StringProperty(default='*.nii.gz', options={'HIDDEN'}, maxlen=255)
    running = False
    fmri_file_template = ''
    _timer = None

    def execute(self, context):
        self.fmri_file_template, hemi = load_surf_files(self.filepath)
        self.fmri_npy_template_fname = op.join(mu.get_user_fol(), 'fmri', 'fmri_{}.npy'.format(
            mu.namebase(self.fmri_file_template)))
        if self.fmri_file_template != '':
            bpy.context.scene.nii_label_output = 'Loading nii file...'
            ChooseNiftiiFile.running = True
            context.window_manager.modal_handler_add(self)
            self._timer = context.window_manager.event_timer_add(0.1, context.window)
        else:
            bpy.context.scene.nii_label_prompt = 'Please select the nii file for the {} hemi'.format(
                'right' if hemi == 'lh' else 'left')
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'TIMER' and ChooseNiftiiFile.running:
            if mu.both_hemi_files_exist(self.fmri_npy_template_fname):
                _addon().plot_fmri_file(self.fmri_file_template)
                clean_nii_temp_files(self.fmri_file_template)
                ChooseNiftiiFile.running = False
                bpy.context.scene.nii_label_output = ''
                self.cancel(context)
        return {'PASS_THROUGH'}

    def cancel(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
            ChooseNiftiiFile.running = False
        return {'CANCELLED'}


def template_draw(self, context):
    layout = self.layout
    if MNE_EXIST:
        layout.operator(ChooseSTCFile.bl_idname, text="Load stc file", icon='LOAD_FACTORY').filepath=op.join(
            mu.get_user_fol(), 'meg', '*.stc')
    if bpy.context.scene.nii_label_output == '':
        layout.operator(ChooseNiftiiFile.bl_idname, text="Load surface nii file", icon='LOAD_FACTORY').filepath = op.join(
            mu.get_user_fol(), 'fmri', '*.nii*')
        if bpy.context.scene.nii_label_prompt != '':
            layout.label(text=bpy.context.scene.nii_label_prompt)
    else:
        layout.label(text=bpy.context.scene.nii_label_output)


class LoadResultsPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "Load Results"
    addon = None
    init = False

    def draw(self, context):
        if LoadResultsPanel.init:
            template_draw(self, context)


def init(addon):
    LoadResultsPanel.addon = addon
    mu.make_dir(op.join(mu.get_user_fol(), 'meg'))
    bpy.context.scene.nii_label_prompt = ''
    bpy.context.scene.nii_label_output = ''
    register()
    LoadResultsPanel.init = True


def register():
    try:
        unregister()
        bpy.utils.register_class(LoadResultsPanel)
        bpy.utils.register_class(ChooseSTCFile)
        bpy.utils.register_class(ChooseNiftiiFile)
    except:
        print("Can't register LoadResults Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(LoadResultsPanel)
        bpy.utils.unregister_class(ChooseSTCFile)
        bpy.utils.unregister_class(ChooseNiftiiFile)
    except:
        pass
