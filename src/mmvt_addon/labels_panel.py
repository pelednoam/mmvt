import bpy
import bpy_extras
import os.path as op
import glob
import time
import traceback
import shutil
import numpy as np
import mmvt_utils as mu


def _addon():
    return LabelsPanel.addon


def update_something():
    pass


def contours_coloring_update(self, context):
    _addon().set_no_plotting(True)
    LabelsPanel.labels_contours = labels_contours = load_labels_contours()
    items = [('all labels', 'all labels', '', 0)]
    for hemi_ind, hemi in enumerate(mu.HEMIS):
        LabelsPanel.labels[hemi] = labels_contours[hemi]['labels']
        extra = 0 if hemi_ind == 0 else len(labels_contours[mu.HEMIS[0]]['labels'])
        items.extend([(c, c, '', ind + extra + 1) for ind, c in enumerate(labels_contours[hemi]['labels'])])
    bpy.types.Scene.labels_contours = bpy.props.EnumProperty(items=items, update=labels_contours_update)
    bpy.context.scene.labels_contours = 'all labels' #d[hemi]['labels'][0]
    _addon().set_no_plotting(False)


def load_labels_contours(atlas=''):
    labels_contours = {}
    if atlas == '':
        atlas = bpy.context.scene.contours_coloring
    for hemi in mu.HEMIS:
        labels_contours[hemi] = np.load(op.join(mu.get_user_fol(), 'labels', '{}_contours_{}.npz'.format(atlas, hemi)))
    return labels_contours


def labels_contours_update(self, context):
    if not _addon().coloring_panel_initialized or _addon().get_no_plotting():
        return
    if bpy.context.scene.labels_contours == 'all labels':
        color_contours(cumulate=False)
    else:
        hemi = 'rh' if bpy.context.scene.labels_contours in LabelsPanel.labels['rh'] else 'lh'
        color_contours(bpy.context.scene.labels_contours, hemi, cumulate=False)


def clear_contours():
    # todo: there is a better way to do it
    labels_contours = LabelsPanel.labels_contours
    for hemi in mu.HEMIS:
        contours = labels_contours[hemi]['contours']
        selected_contours = np.zeros(contours.shape)
        mesh = mu.get_hemi_obj(hemi).data
        mesh.vertex_colors.active_index = mesh.vertex_colors.keys().index('contours')
        mesh.vertex_colors['contours'].active_render = True
        _addon().color_hemi_data(hemi, selected_contours, 0.1, 256, override_current_mat=True,
                        coloring_layer='contours', check_valid_verts=False)


def plot_labels_data():
    labels_data_fname = glob.glob(op.join(mu.get_user_fol(), 'labels', 'labels_data', '{}.*'.format(bpy.context.scene.labels_data_files.replace(' ', '_'))))[0]
    load_labels_data(labels_data_fname)


def labels_data_files_update(self, context):
    if LabelsPanel.init:
        update_something()


def new_label_r_update(self, context):
    build_new_label_name()


def build_new_label_name():
    closest_label_output = bpy.context.scene.closest_label_output
    if closest_label_output == '':
        new_label_name = 'Unknown'
    else:
        delim, pos, label, label_hemi = mu.get_hemi_delim_and_pos(closest_label_output)
        label = '{}-{}mm'.format(label, bpy.context.scene.new_label_r)
        new_label_name = mu.build_label_name(delim, pos, label, label_hemi)
    bpy.context.scene.new_label_name = new_label_name


def grow_a_label():
    closest_mesh_name, vertex_ind, _ = \
        _addon().find_vertex_index_and_mesh_closest_to_cursor(use_shape_keys=True)
    hemi = closest_mesh_name[len('infalted_'):] if _addon().is_inflated() else closest_mesh_name
    subject, atlas = mu.get_user(), bpy.context.scene.subject_annot_files
    label_name, label_r = bpy.context.scene.new_label_name, bpy.context.scene.new_label_r
    flags = '-a {} --vertice_indice {} --hemi {} --label_name {} --label_r {}'.format(
        atlas, vertex_ind, hemi, label_name, label_r)
    mu.run_mmvt_func('src.preproc.anatomy', 'grow_label', flags=flags)


def color_contours(specific_labels=[], specific_hemi='both', labels_contours=None, cumulate=False, change_colorbar=False,
                   specific_colors=None, atlas='', move_cursor=True):
    if isinstance(specific_labels, str):
        specific_labels = [specific_labels]
    if atlas != '' and atlas != bpy.context.scene.contours_coloring and atlas in LabelsPanel.existing_contoures:
        bpy.context.scene.contours_coloring = atlas
    if atlas == '' and bpy.context.scene.atlas in LabelsPanel.existing_contoures:
        bpy.context.scene.contours_coloring = bpy.context.scene.atlas
    if labels_contours is None:
        labels_contours = LabelsPanel.labels_contours
    contour_max = max([labels_contours[hemi]['max'] for hemi in mu.HEMIS])
    if not _addon().colorbar_values_are_locked() and change_colorbar:
        _addon().set_colormap('jet')
        _addon().set_colorbar_title('{} labels contours'.format(bpy.context.scene.contours_coloring))
        _addon().set_colorbar_max_min(contour_max, 1)
        _addon().set_colorbar_prec(0)
    _addon().show_activity()
    specific_label_ind = 0
    if specific_colors is not None:
        specific_colors = np.tile(specific_colors, (len(specific_labels), 1))
    for hemi in mu.HEMIS:
        contours = labels_contours[hemi]['contours']
        if specific_hemi != 'both' and hemi != specific_hemi:
            selected_contours = np.zeros(contours.shape)
        elif len(specific_labels) > 0:
            selected_contours = np.zeros(contours.shape) if specific_colors is None else np.zeros((contours.shape[0], 4))
            for specific_label in specific_labels:
                if mu.get_hemi_from_fname(specific_label) != hemi:
                    continue
                label_ind = np.where(np.array(labels_contours[hemi]['labels']) == specific_label)
                if len(label_ind) > 0 and len(label_ind[0]) > 0:
                    label_ind = label_ind[0][0]
                    selected_contours[np.where(contours == label_ind + 1)] = \
                        label_ind + 1 if specific_colors is None else [1, *specific_colors[specific_label_ind]]
                    specific_label_ind += 1
                    if move_cursor and len(specific_labels) == 1 and 'centers' in labels_contours[hemi]:
                        vert = labels_contours[hemi]['centers'][label_ind]
                        _addon().move_cursor_according_to_vert(vert, 'inflated_{}'.format(hemi))
                        _addon().set_closest_vertex_and_mesh_to_cursor(vert, 'inflated_{}'.format(hemi))
                        _addon().create_slices()
                else:
                    print("Can't find {} in the labels contours!".format(specific_label))
        else:
            selected_contours = labels_contours[hemi]['contours']
        mesh = mu.get_hemi_obj(hemi).data
        mesh.vertex_colors.active_index = mesh.vertex_colors.keys().index('contours')
        mesh.vertex_colors['contours'].active_render = True
        _addon().color_hemi_data(hemi, selected_contours, 0.1, 256 / contour_max, override_current_mat=not cumulate,
                        coloring_layer='contours', check_valid_verts=False)
    _addon().what_is_colored().add(_addon().WIC_CONTOURS)

    # if bpy.context.scene.contours_coloring in _addon().get_annot_files():
    #     bpy.context.scene.subject_annot_files = bpy.context.scene.contours_coloring


def load_labels_data(labels_data_fname):
    labels_data_type = mu.file_type(labels_data_fname)
    if labels_data_type == 'npz':
        d = mu.load_npz_to_bag(labels_data_fname)
    elif labels_data_type == 'mat':
        d = mu.load_mat_to_bag(labels_data_fname)
        d.names = mu.matlab_cell_str_to_list(d.names)
        d.atlas = d.atlas[0] if not isinstance(d.atlas, str) else d.atlas
        #d.cmap = d.cmap[0] if not isinstance(d.cmap, str) else d.cmap
    else:
        print('Currently we support only mat and npz files')
        return False
    labels, data = d.names, d.data
    if 'atlas' not in d:
        atlas = mu.check_atlas_by_labels_names(labels)
        if atlas == '':
            print('The labeling file must contains an atlas field!')
            return False
    else:
        atlas = str(d.atlas)
    labels = [l.replace('.label', '') for l in labels]
    cb_title = str(d.get('title', ''))
    labels_min = d.get('data_min', np.min(data))
    labels_max = d.get('data_max', np.max(data))
    cmap = str(d.get('cmap', None))
    _addon().color_labels_data(labels, data, atlas, cb_title, labels_max, labels_min, cmap)
    new_fname = op.join(mu.get_user_fol(), 'labels', 'labels_data', mu.namebase_with_ext(labels_data_fname))
    if 'atlas' not in d:
        npz_dict = dict(d)
        npz_dict['atlas'] = atlas
        np.savez(new_fname, **npz_dict)
    else:
        if new_fname != labels_data_fname:
            shutil.copy(labels_data_fname, new_fname)
    init_labels_data_files()
    return True


def labels_draw(self, context):
    layout = self.layout

    col = layout.box().column()
    col.label(text='Cortical labels data:')
    if len(LabelsPanel.labels_data_files) > 0:
        col.prop(context.scene, 'labels_data_files', text='')
        col.prop(context.scene, 'color_rois_homogeneously', text="Color labels homogeneously")
        col.operator(PlotLabelsData.bl_idname, text="Plot labels", icon='TPAINT_HLT')
    col.operator(ChooseLabesDataFile.bl_idname, text="Load labels file", icon='LOAD_FACTORY')

    if LabelsPanel.contours_coloring_exist:
        col = layout.box().column()
        col.label(text='Contours:')
        col.prop(context.scene, 'contours_coloring', '')
        col.operator(ColorContours.bl_idname, text="Plot Contours", icon='POTATO')
        row = col.row(align=True)
        row.operator(PrevLabelConture.bl_idname, text="", icon='PREV_KEYFRAME')
        row.prop(context.scene, 'labels_contours', '')
        row.operator(NextLabelConture.bl_idname, text="", icon='NEXT_KEYFRAME')

    col = layout.box().column()
    if not GrowLabel.running:
        col.label(text='Creating a new label:')
        col.prop(context.scene, 'new_label_name', text='')
        col.prop(context.scene, 'new_label_r', text='Radius (mm)')
        txt = 'Grow a label' if bpy.context.scene.cursor_is_snapped else 'First Snap the cursor'
        col.operator(GrowLabel.bl_idname, text=txt, icon='OUTLINER_DATA_MESH')
    else:
        col.label(text='Growing the label...')
    layout.operator(ClearContours.bl_idname, text="Clear contours", icon='PANEL_CLOSE')
    layout.operator(_addon().ClearColors.bl_idname, text="Clear", icon='PANEL_CLOSE')


class ChooseLabesDataFile(bpy.types.Operator, bpy_extras.io_utils.ExportHelper):
    bl_idname = "mmvt.choose_labels_npz_file"
    bl_label = "Choose labels data"

    filename_ext = '.*'
    filter_glob = bpy.props.StringProperty(default='*.*', options={'HIDDEN'}, maxlen=255)

    def execute(self, context):
        _addon().load_labels_data(self.filepath.replace('.*', ''))
        return {'FINISHED'}


class GrowLabel(bpy.types.Operator):
    bl_idname = "mmvt.grow_label"
    bl_label = "mmvt grow label"
    bl_options = {"UNDO"}
    running = False

    def cancel(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
            GrowLabel.running = False
        return {'CANCELLED'}

    @staticmethod
    def invoke(self, context, event=None):
        if not bpy.context.scene.cursor_is_snapped:
            _addon().snap_cursor(True)
            _addon().find_closest_label()
            build_new_label_name()
        else:
            GrowLabel.running = True
            context.window_manager.modal_handler_add(self)
            self._timer = context.window_manager.event_timer_add(0.1, context.window)
            grow_a_label()
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'TIMER' and GrowLabel.running:
            new_label_fname = op.join(
                mu.get_user_fol(), 'labels', '{}.label'.format(bpy.context.scene.new_label_name))
            if op.isfile(new_label_fname):
                _addon().plot_label(new_label_fname)
                GrowLabel.running = False
                self.cancel(context)
        return {'PASS_THROUGH'}


class PlotLabelsData(bpy.types.Operator):
    bl_idname = "mmvt.plot_labels_data"
    bl_label = "plot_labels_data"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        plot_labels_data()
        return {'PASS_THROUGH'}


class ColorContours(bpy.types.Operator):
    bl_idname = "mmvt.color_contours"
    bl_label = "mmvt color contours"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        color_contours(atlas=bpy.context.scene.contours_coloring)
        return {"FINISHED"}


class ClearContours(bpy.types.Operator):
    bl_idname = "mmvt.clear_contours"
    bl_label = "mmvt clear contours"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        clear_contours()
        return {"FINISHED"}


class PrevLabelConture(bpy.types.Operator):
    bl_idname = "mmvt.labels_contours_prev"
    bl_label = "mmvt labels contours prev"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        all_labels = np.concatenate((LabelsPanel.labels['rh'], LabelsPanel.labels['lh']))
        if bpy.context.scene.labels_contours == 'all labels':
            bpy.context.scene.labels_contours = all_labels[-1]
        else:
            label_ind = np.where(all_labels == bpy.context.scene.labels_contours)[0][0]
            bpy.context.scene.labels_contours = all_labels[label_ind - 1] if label_ind > 0 else all_labels[-1]
        return {"FINISHED"}


class NextLabelConture(bpy.types.Operator):
    bl_idname = "mmvt.labels_contours_next"
    bl_label = "mmvt labels contours next"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        all_labels = np.concatenate((LabelsPanel.labels['rh'], LabelsPanel.labels['lh']))
        if bpy.context.scene.labels_contours == 'all labels':
            bpy.context.scene.labels_contours = all_labels[0]
        else:
            label_ind = np.where(all_labels == bpy.context.scene.labels_contours)[0][0]
            bpy.context.scene.labels_contours = all_labels[label_ind + 1] \
                if label_ind < len(all_labels) else all_labels[0]
        return {"FINISHED"}


bpy.types.Scene.labels_data_files = bpy.props.EnumProperty(items=[], description="label files")
bpy.types.Scene.new_label_name = bpy.props.StringProperty()
bpy.types.Scene.new_label_r = bpy.props.IntProperty(min=1, default=5, update=new_label_r_update)
bpy.types.Scene.contours_coloring = bpy.props.EnumProperty(items=[], description="labels contours coloring")
bpy.types.Scene.labels_contours = bpy.props.EnumProperty(items=[])


class LabelsPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "Cortical Labels"
    addon = None
    init = False
    
    labels_data_files = []
    existing_contoures = []
    contours_coloring_exist = False
    labels_contours = {}
    labels = dict(rh=[], lh=[])

    def draw(self, context):
        if LabelsPanel.init:
            labels_draw(self, context)


def init(addon):
    LabelsPanel.addon = addon
    init_labels_data_files()
    init_contours_coloring()
    register()
    LabelsPanel.init = True


def init_contours_coloring():
    user_fol = mu.get_user_fol()
    contours_files = glob.glob(op.join(user_fol, 'labels', '*contours_lh.npz'))
    if len(contours_files) > 0:
        LabelsPanel.contours_coloring_exist = True
        LabelsPanel.existing_contoures = files_names = \
            [mu.namebase(fname)[:-len('_contours_lh')] for fname in contours_files]
        items = [(c, c, '', ind) for ind, c in enumerate(files_names)]
        bpy.types.Scene.contours_coloring = bpy.props.EnumProperty(items=items, update=contours_coloring_update)
        bpy.context.scene.contours_coloring = files_names[0]


def init_labels_data_files():
    user_fol = mu.get_user_fol()
    mu.make_dir(op.join(user_fol, 'labels', 'labels_data'))
    LabelsPanel.labels_data_files = labels_data_files = glob.glob(op.join(user_fol, 'labels', 'labels_data', '*.npz')) + glob.glob(op.join(user_fol, 'labels', 'labels_data', '*.mat'))
    if len(labels_data_files) > 0:
        files_names = [mu.namebase(fname).replace('_', ' ') for fname in labels_data_files]
        labels_items = [(c, c, '', ind) for ind, c in enumerate(files_names)]
        bpy.types.Scene.labels_data_files = bpy.props.EnumProperty(
            items=labels_items, description="label files",update=labels_data_files_update)
        bpy.context.scene.labels_data_files = files_names[0]


def register():
    try:
        unregister()
        bpy.utils.register_class(LabelsPanel)
        bpy.utils.register_class(GrowLabel)
        bpy.utils.register_class(ColorContours)
        bpy.utils.register_class(ClearContours)
        bpy.utils.register_class(NextLabelConture)
        bpy.utils.register_class(PrevLabelConture)
        bpy.utils.register_class(PlotLabelsData)
        bpy.utils.register_class(ChooseLabesDataFile)
    except:
        print("Can't register Labels Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(LabelsPanel)
        bpy.utils.unregister_class(GrowLabel)
        bpy.utils.unregister_class(ColorContours)
        bpy.utils.unregister_class(ClearContours)
        bpy.utils.unregister_class(NextLabelConture)
        bpy.utils.unregister_class(PrevLabelConture)
        bpy.utils.unregister_class(PlotLabelsData)
        bpy.utils.unregister_class(ChooseLabesDataFile)
    except:
        pass
