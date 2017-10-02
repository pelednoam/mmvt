import bpy
import mathutils
import numpy as np
import mmvt_utils as mu
import glob
import traceback
from collections import Counter
import os.path as op
import os
import time

def _addon():
    return WhereAmIPanel.addon


def _trans():
    return WhereAmIPanel.subject_orig_trans


def where_i_am_draw(self, context):
    layout = self.layout
    layout.label(text='tkreg RAS (surface):')
    row = layout.row(align=0)
    row.prop(context.scene, "tkreg_ras_x", text="x")
    row.prop(context.scene, "tkreg_ras_y", text="y")
    row.prop(context.scene, "tkreg_ras_z", text="z")
    if not _trans() is None:
        layout.label(text='mni305:')
        row = layout.row(align=0)
        row.prop(context.scene, "ras_x", text="x")
        row.prop(context.scene, "ras_y", text="y")
        row.prop(context.scene, "ras_z", text="z")
        layout.label(text='T1 voxel:')
        row = layout.row(align=0)
        row.prop(context.scene, "voxel_x", text="x")
        row.prop(context.scene, "voxel_y", text="y")
        row.prop(context.scene, "voxel_z", text="z")
    for atlas, name in WhereAmIPanel.atlas_ids.items():
        row = layout.row(align=0)
        row.label(text='{}: {}'.format(atlas, name))
        # row.operator(ChooseVoxelID.bl_idname, text="Select", icon='SNAP_SURFACE')
    layout.operator(WhereAmI.bl_idname, text="Find closest object", icon='SNAP_SURFACE')
    if bpy.context.scene.where_am_i_str != '':
        layout.label(text=bpy.context.scene.where_am_i_str)
    # if bpy.context.scene.subject_annot_files != '':
    #     col = layout.box().column()
    #     col.prop(context.scene, 'subject_annot_files', text='')
    #     col.operator(ClosestLabel.bl_idname, text="Find closest label", icon='SNAP_SURFACE')
    #     if bpy.context.scene.closest_label_output != '':
    #         col.label(text=bpy.context.scene.closest_label_output)

    col = layout.box().column()
    if not GrowLabel.running:
        col.label(text='Create a new label')
        col.prop(context.scene, 'new_label_name', text='')
        col.prop(context.scene, 'new_label_r', text='Radius (mm)')
        txt = 'Grow a label' if bpy.context.scene.cursor_is_snapped else 'First Snap the cursor'
        col.operator(GrowLabel.bl_idname, text=txt, icon='OUTLINER_DATA_MESH')
    else:
        col.label(text='Growing the label...')
    layout.operator(ClearWhereAmI.bl_idname, text="Clear", icon='PANEL_CLOSE')


def tkras_coo_update(self, context):
    if not WhereAmIPanel.call_update:
        return

    # print('tkras_coo_update')
    if WhereAmIPanel.move_cursor:
        bpy.context.scene.cursor_location[0] = bpy.context.scene.tkreg_ras_x / 10
        bpy.context.scene.cursor_location[1] = bpy.context.scene.tkreg_ras_y / 10
        bpy.context.scene.cursor_location[2] = bpy.context.scene.tkreg_ras_z / 10

    if not _trans() is None and WhereAmIPanel.update:
        coo = [bpy.context.scene.tkreg_ras_x, bpy.context.scene.tkreg_ras_y, bpy.context.scene.tkreg_ras_z]
        vox = apply_trans(_trans().ras_tkr2vox, np.array([coo]))
        ras = apply_trans(_trans().vox2ras, vox)
        WhereAmIPanel.update = False
        set_ras_coo(ras[0])
        set_voxel_coo(vox[0])
        WhereAmIPanel.update = True


def ras_coo_update(self, context):
    if not WhereAmIPanel.call_update:
        return

    # print('ras_coo_update')
    if not _trans() is None and WhereAmIPanel.update:
        coo = [bpy.context.scene.ras_x, bpy.context.scene.ras_y, bpy.context.scene.ras_z]
        vox = apply_trans(_trans().ras2vox, np.array([coo]))
        ras_tkr = apply_trans(_trans().vox2ras_tkr, vox)
        WhereAmIPanel.update = False
        set_tkreg_ras_coo(ras_tkr[0])
        set_voxel_coo(vox[0])
        WhereAmIPanel.update = True


def voxel_coo_update(self, context):
    if not WhereAmIPanel.call_update:
        return

    # print('voxel_coo_update')
    vox_x, vox_y, vox_z = bpy.context.scene.voxel_x, bpy.context.scene.voxel_y, bpy.context.scene.voxel_z
    if not _trans() is None and WhereAmIPanel.update:
        vox = [vox_x, vox_y, vox_z]
        ras = apply_trans(_trans().vox2ras, np.array([vox]))
        ras_tkr = apply_trans(_trans().vox2ras_tkr, [vox])
        WhereAmIPanel.update = False
        set_tkreg_ras_coo(ras_tkr[0])
        set_ras_coo(ras[0])
        WhereAmIPanel.update = True
    get_3d_atlas_name()


def get_3d_atlas_name():
    vox_x, vox_y, vox_z = bpy.context.scene.voxel_x, bpy.context.scene.voxel_y, bpy.context.scene.voxel_z
    names = {}
    for atlas in WhereAmIPanel.vol_atlas.keys():
        try:
            vol_atlas = WhereAmIPanel.vol_atlas[atlas]
            vol_atlas_lut = WhereAmIPanel.vol_atlas_lut[atlas]
            try:
                id = vol_atlas[vox_x, vox_y, vox_z]
            except:
                continue
            id_inds = np.where(vol_atlas_lut['ids'] == id)[0]
            if len(id_inds) == 0:
                continue
            id_ind = id_inds[0]
            names[atlas] = vol_atlas_lut['names'][id_ind]
            if names[atlas] == 'Unknown':
                all_vals = vol_atlas[vox_x - 1:vox_x + 2, vox_y - 1:vox_y + 2, vox_z - 1:vox_z + 2]
                vals = np.unique(all_vals)
                vals = list(set(vals) - set([0]))
                if len(vals) > 0:
                    val = vals[0]
                    if len(vals) > 1:
                        mcs = Counter(all_vals.ravel()).most_common()
                        # print(atlas, mcs)
                        val = mcs[0][0] if val != 0 else mcs[1][0]
                    id_inds = np.where(vol_atlas_lut['ids'] == val)[0]
                    if len(id_inds) > 0:
                        names[atlas] = str(vol_atlas_lut['names'][id_ind])
        except:
            print(traceback.format_exc())
            print('Error in trying to get the 3D atlas voxel value!')
        WhereAmIPanel.atlas_ids = names


def set_tkreg_ras_coo(coo, move_cursor=True):
    # print('set_tkreg_ras_coo')
    WhereAmIPanel.call_update = False
    WhereAmIPanel.move_cursor = move_cursor
    bpy.context.scene.tkreg_ras_x = coo[0]
    bpy.context.scene.tkreg_ras_y = coo[1]
    WhereAmIPanel.call_update = True
    bpy.context.scene.tkreg_ras_z = coo[2]
    WhereAmIPanel.move_cursor = True


def set_ras_coo(coo):
    # print('set_ras_coo')
    WhereAmIPanel.call_update = False
    bpy.context.scene.ras_x = coo[0]
    bpy.context.scene.ras_y = coo[1]
    WhereAmIPanel.call_update = True
    bpy.context.scene.ras_z = coo[2]


def set_voxel_coo(coo):
    # print('set_voxel_coo')
    WhereAmIPanel.call_update = False
    bpy.context.scene.voxel_x = int(np.round(coo[0]))
    bpy.context.scene.voxel_y = int(np.round(coo[1]))
    WhereAmIPanel.call_update = True
    bpy.context.scene.voxel_z = int(np.round(coo[2]))


def apply_trans(trans, points):
    return np.array([np.dot(trans, np.append(p, 1))[:3] for p in points])


def find_closest_obj(search_also_for_subcorticals=True):
    distances, names, indices = [], [], []

    parent_objects_names = ['Cortex-lh', 'Cortex-rh']
    if _addon().is_inflated():
        parent_objects_names = ['Cortex-inflated-lh', 'Cortex-inflated-rh']
    if search_also_for_subcorticals:
        parent_objects_names.append('Subcortical_structures')
    for parent_object_name in parent_objects_names:
        parent_object = bpy.data.objects.get(parent_object_name, None)
        if parent_object is None:
            continue
        # if subHierarchy == bpy.data.objects['Subcortical_structures']:
        #     cur_material = bpy.data.materials['unselected_label_Mat_subcortical']
        # else:
        #     cur_material = bpy.data.materials['unselected_label_Mat_cortex']
        for obj in parent_object.children:
            # obj.active_material = cur_material
            obj.select = False
            obj.hide = parent_object.hide

            # 3d cursor relative to the object data
            cursor = bpy.context.scene.cursor_location
            if bpy.context.object and bpy.context.object.parent == bpy.data.objects.get('Deep_electrodes', None):
                cursor = bpy.context.object.location

            co_find = cursor * obj.matrix_world.inverted()

            mesh = obj.data
            size = len(mesh.vertices)
            kd = mathutils.kdtree.KDTree(size)

            for i, v in enumerate(mesh.vertices):
                kd.insert(v.co, i)

            kd.balance()

            # Find the closest point to the 3d cursor
            for (co, index, dist) in kd.find_n(co_find, 1):
                if 'unknown' not in obj.name:
                    distances.append(dist)
                    names.append(obj.name)
                    indices.append(index)

    # print(np.argmin(np.array(distances)))
    min_index = np.argmin(np.array(distances))
    closest_area = names[np.argmin(np.array(distances))]
    # print('closest area is: '+closest_area)
    # print('dist: {}'.format(np.min(np.array(distances))))
    # print('closets vert is {}'.format(bpy.data.objects[closest_area].data.vertices[min_index].co))
    return closest_area


def find_closest_label():
    subjects_dir = mu.get_link_dir(mu.get_links_dir(), 'subjects')
    closest_mesh_name, vertex_ind, vertex_co, _ = \
        _addon().find_vertex_index_and_mesh_closest_to_cursor(use_shape_keys=True)
    hemi = closest_mesh_name[len('infalted_'):] if _addon().is_inflated() else closest_mesh_name
    annot_fname = op.join(subjects_dir, mu.get_user(), 'label', '{}.{}.annot'.format(
        hemi, bpy.context.scene.subject_annot_files))
    labels = mu.read_labels_from_annot(annot_fname)
    vert_labels = [l for l in labels if vertex_ind in l.vertices]
    if len(vert_labels) > 0:
        label = vert_labels[0]
        bpy.context.scene.closest_label_output = label.name
        return label.name, hemi
    else:
        return 'unknown', hemi


def plot_closest_label_contour(label, hemi):
    contours_files = glob.glob(op.join(mu.get_user_fol(), '*contours_lh.npz'))
    contours_names = [mu.namebase(fname)[:-len('_contours_lh')] for fname in contours_files]
    if bpy.context.scene.subject_annot_files in contours_names:
        bpy.context.scene.contours_coloring = bpy.context.scene.subject_annot_files
        _addon().color_contours(label, hemi)
    else:
        mu.create_labels_contours()


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
    closest_mesh_name, vertex_ind, _, _ = \
        _addon().find_vertex_index_and_mesh_closest_to_cursor(use_shape_keys=True)
    hemi = closest_mesh_name[len('infalted_'):] if _addon().is_inflated() else closest_mesh_name
    subject, atlas = mu.get_user(), bpy.context.scene.subject_annot_files
    label_name, label_r = bpy.context.scene.new_label_name, bpy.context.scene.new_label_r
    cmd = '{} -m src.preproc.anatomy -s {} -a {} -f grow_label '.format(
        bpy.context.scene.python_cmd, subject, atlas)
    cmd += '--vertice_indice {} --hemi {} --label_name {} --label_r {}'.format(vertex_ind, hemi, label_name, label_r)
    mu.run_command_in_new_thread(cmd, False)


# @mu.timeit
def update_slices(modality='mri'):
    screen = bpy.data.screens['Neuro']
    images_names = ['{}_{}.png'.format(modality, pres) for pres in ['sagital', 'coronal', 'axial']]
    images_fol = op.join(mu.get_user_fol(), 'figures', 'slices')
    ind = 0
    extra_images = set([img.name for img in bpy.data.images]) - set(['mri_axial.png', 'mri_coronal.png', 'mri_sagital.png', 'Render Result'])
    for img_name in extra_images:
        bpy.data.images.remove(bpy.data.images[img_name])
    for area in screen.areas:
        if area.type == 'IMAGE_EDITOR':
            override = bpy.context.copy()
            override['area'] = area
            override["screen"] = screen
            if images_names[ind] not in bpy.data.images:
                bpy.data.images.load(op.join(images_fol, images_names[ind]), check_existing=False)
            # bpy.data.images[images_names[ind]].reload()
            image = bpy.data.images[images_names[ind]]
            image.reload()
            area.spaces.active.image = image
            # bpy.ops.image.replace(override, filepath=op.join(images_fol, images_names[ind]))
            bpy.ops.image.view_zoom_ratio(override, ratio=1)
            ind += 1
    # mu.conn_to_listener.close()


def init_slices():
    extra_images = set([img.name for img in bpy.data.images]) - set(['Render Result'])
    for img_name in extra_images:
        bpy.data.images.remove(bpy.data.images[img_name])



def start_slicer_server():
    cmd = '{} -m src.listeners.slicer_listener'.format(bpy.context.scene.python_cmd)
    mu.run_command_in_new_thread(cmd, False)


def init_listener():
    if mu.conn_to_listener.handle_is_open:
        return True
    ret = False
    tries = 0
    while not ret and tries < 3:
        try:
            ret = mu.conn_to_listener.init()
            if ret:
                mu.conn_to_listener.send_command(b'Hey!\n')
            else:
                mu.message(None, 'Error initialize the listener. Try again')
        except:
            print("Can't open connection to listener")
            print(traceback.format_exc())
        tries += 1
    return ret


def create_slices(modalities='mri'):
    init_listener()
    slice_brain(bpy.context.scene.cursor_location, bpy.types.Scene.cut_type,
                op.join(mu.get_user_fol(), 'figures', 'slices'))
    print()
    pos = bpy.context.scene.cursor_location * 10

    # x, y, z = apply_trans(_trans().ras_tkr2vox, np.array([pos])).astype(np.int)[0]
    # xyz = ','.join(map(str, [x, y, z]))
    xyz = ','.join(map(str, pos))
    # output_files = glob.glob(op.join(mu.get_user_fol(), 'figures', 'slices', '{}_*.png'.format(modality)))
    # for output_file in output_files:
    #     os.remove(output_file)

    flag_fname = op.join(mu.get_user_fol(), 'figures', 'slices', '{}_slices.txt'.format(
        '_'.join(modalities.split(','))))
    mu.remove_file(flag_fname)
    # cmd = '{} -m src.preproc.anatomy -s {} -f create_slices --slice_xyz {} --slices_modalities {}'.format(
    #     bpy.context.scene.python_cmd, mu.get_user(), pos, modalities)
    # print('Running {}'.format(cmd))
    # WhereAmIPanel.tic = time.time()
    # print(WhereAmIPanel.tic)
    # mu.run_command_in_new_thread(cmd, False)

    ret = mu.conn_to_listener.send_command(dict(cmd='slice_viewer_change_pos', data=dict(
        subject=mu.get_user(), xyz=xyz, modalities=modalities, coordinates_system='tk_ras')))
    bpy.ops.mmvt.wait_for_slices()


class WaitForSlices(bpy.types.Operator):
    bl_idname = "mmvt.wait_for_slices"
    bl_label = "wait_for_slices"
    bl_options = {"UNDO"}
    running = False
    modalities = 'mri'

    def cancel(self, context):
        # print('WaitForSlices: cancel')
        try:
            if self._timer:
                context.window_manager.event_timer_remove(self._timer)
                self._timer = None
                self.running = False
        except:
            print('Error in WaitForSlices.cancel')
        return {'CANCELLED'}

    def invoke(self, context, event=None):
        # print('WaitForSlices: invoke')
        return {'RUNNING_MODAL'}

    def execute(self, context):
        # print('WaitForSlices: execute')
        if not self.running:
            self.running = True
            context.window_manager.modal_handler_add(self)
            self._timer = context.window_manager.event_timer_add(0.1, context.window)
            return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'TIMER' and self.running:
            output_file = op.join(mu.get_user_fol(), 'figures', 'slices', '{}_slices.txt'.format(
                '_'.join(self.modalities.split(','))))
            if op.isfile(output_file):
                os.remove(output_file)
                # print('took {:.5f}s'.format(time.time() - WhereAmIPanel.tic))
                update_slices()
                self.cancel(context)
        return {'PASS_THROUGH'}


class ChooseVoxelID(bpy.types.Operator):
    bl_idname = "mmvt.choose_voxel_id"
    bl_label = "mmvt choose_voxel_id"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        obj = bpy.data.objects.get(bpy.context.scene.where_am_i_atlas, None)
        if not obj is None:
            obj.select = True
        return {"FINISHED"}


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
            find_closest_label()
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


class ClosestLabel(bpy.types.Operator):
    bl_idname = "mmvt.closest_label"
    bl_label = "mmvt closest label"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        label, hemi = find_closest_label()
        if label != 'unknown':
            plot_closest_label_contour(label, hemi)
        return {"FINISHED"}


class WhereAmI(bpy.types.Operator):
    bl_idname = "mmvt.where_i_am"
    bl_label = "mmvt where i am"
    bl_options = {"UNDO"}

    where_am_I_selected_obj = None
    where_am_I_selected_obj_org_hide = True

    @staticmethod
    def setup_environment(self):
        WhereAmIPanel.addon.show_rois()

    @staticmethod
    def main_func(self):
        bpy.data.objects['Brain'].select = False
        closest_area = find_closest_obj()
        bpy.types.Scene.where_am_i_str = closest_area
        WhereAmI.where_am_I_selected_obj = bpy.data.objects[closest_area]
        WhereAmI.where_am_I_selected_obj_org_hide = bpy.data.objects[closest_area].hide
        bpy.context.scene.objects.active = bpy.data.objects[closest_area]
        closest_area_type = mu.check_obj_type(closest_area)
        if closest_area_type in [mu.OBJ_TYPE_CORTEX_LH, mu.OBJ_TYPE_CORTEX_RH, mu.OBJ_TYPE_ELECTRODE,
                                 mu.OBJ_TYPE_EEG]:

            _addon().select_roi(closest_area)
        else:
            bpy.data.objects[closest_area].select = True
        bpy.data.objects[closest_area].hide = False
        bpy.data.objects[closest_area].active_material = bpy.data.materials['selected_label_Mat']

    def invoke(self, context, event=None):
        self.setup_environment(self)
        self.main_func(self)
        return {"FINISHED"}


class ClearWhereAmI(bpy.types.Operator):
    bl_idname = "mmvt.where_am_i_clear"
    bl_label = "where am i clear"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        for subHierarchy in bpy.data.objects['Brain'].children:
            new_mat = bpy.data.materials['unselected_label_Mat_cortex']
            if subHierarchy.name == 'Subcortical_structures':
                new_mat = bpy.data.materials['unselected_label_Mat_subcortical']
            for obj in subHierarchy.children:
                obj.active_material = new_mat

        if 'Deep_electrodes' in bpy.data.objects:
            for obj in bpy.data.objects['Deep_electrodes'].children:
                obj.active_material.node_tree.nodes["Layer Weight"].inputs[0].default_value = 1
        if bpy.data.objects.get(' '):
            context.scene.objects.active = bpy.data.objects[' ']

        for obj in bpy.data.objects:
            obj.select = False

        if WhereAmI.where_am_I_selected_obj is not None:
            WhereAmI.where_am_I_selected_obj.hide = WhereAmI.where_am_I_selected_obj_org_hide
            WhereAmI.where_am_I_selected_obj = None

        if bpy.context.scene.closest_label_output != '':
            _addon().clear_cortex()

        bpy.types.Scene.where_am_i_str = ''
        bpy.context.scene.closest_label_output = ''
        bpy.context.scene.new_label_name = 'New-label'
        # where_i_am_draw(self, context)
        return {"FINISHED"}

bpy.types.Scene.where_am_i = bpy.props.StringProperty(description="Find closest curve to cursor",
                                                      update=where_i_am_draw)
bpy.types.Scene.ras_x = bpy.props.FloatProperty(update=ras_coo_update)
bpy.types.Scene.ras_y = bpy.props.FloatProperty(update=ras_coo_update)
bpy.types.Scene.ras_z = bpy.props.FloatProperty(update=ras_coo_update)
bpy.types.Scene.tkreg_ras_x = bpy.props.FloatProperty(update=tkras_coo_update)
bpy.types.Scene.tkreg_ras_y = bpy.props.FloatProperty(update=tkras_coo_update)
bpy.types.Scene.tkreg_ras_z = bpy.props.FloatProperty(update=tkras_coo_update)
bpy.types.Scene.voxel_x = bpy.props.IntProperty(update=voxel_coo_update)
bpy.types.Scene.voxel_y = bpy.props.IntProperty(update=voxel_coo_update)
bpy.types.Scene.voxel_z = bpy.props.IntProperty(update=voxel_coo_update)
bpy.types.Scene.where_am_i_str = bpy.props.StringProperty()
bpy.types.Scene.subject_annot_files = bpy.props.EnumProperty(items=[])
bpy.types.Scene.closest_label_output = bpy.props.StringProperty()
bpy.types.Scene.closest_label = bpy.props.StringProperty()
bpy.types.Scene.new_label_name = bpy.props.StringProperty()
bpy.types.Scene.new_label_r = bpy.props.IntProperty(min=1, default=5, update=new_label_r_update)
bpy.types.Scene.cut_type = 'sagital'
# bpy.types.Scene.where_am_i_atlas = bpy.props.StringProperty()


class WhereAmIPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "Where Am I"
    addon = None
    init = False
    subject_orig_trans = None
    vol_atlas = {}
    vol_atlas_lut = {}
    atlas_ids = {}
    update = True
    call_update = True
    move_cursor = True

    def draw(self, context):
        where_i_am_draw(self, context)


def init(addon):
    try:
        trans_fname = op.join(mu.get_user_fol(), 'orig_trans.npz')
        volumes = glob.glob(op.join(mu.get_user_fol(), 'freeview', '*+aseg.npy'))
        luts = glob.glob(op.join(mu.get_user_fol(), 'freeview', '*ColorLUT.npz'))
        if op.isfile(trans_fname):
            WhereAmIPanel.subject_orig_trans = mu.Bag(np.load(trans_fname))
        for atlas_vol_fname, atlas_vol_lut_fname in zip(volumes, luts):
            atlas = mu.namebase(atlas_vol_fname)[:-len('+aseg')]
            WhereAmIPanel.vol_atlas[atlas] = np.load(atlas_vol_fname)
            WhereAmIPanel.vol_atlas_lut[atlas] = np.load(atlas_vol_lut_fname)
        subjects_dir = mu.get_link_dir(mu.get_links_dir(), 'subjects')
        annot_files = glob.glob(op.join(subjects_dir, mu.get_user(), 'label', 'rh.*.annot'))
        if len(annot_files) > 0:
            files_names = [mu.namebase(fname)[3:] for fname in annot_files]
            items = [(c, c, '', ind) for ind, c in enumerate(files_names)]
            bpy.types.Scene.subject_annot_files = bpy.props.EnumProperty(items=items)
            bpy.context.scene.subject_annot_files = files_names[0]
        else:
            bpy.types.Scene.subject_annot_files = bpy.props.EnumProperty(items=[])
            # bpy.context.scene.subject_annot_files = ''

        bpy.context.scene.closest_label_output = ''
        bpy.context.scene.new_label_r = 5
        WhereAmIPanel.addon = addon
        WhereAmIPanel.init = True
        start_slicer_server()
        init_slices()
        register()
    except:
        print("Can't init where-am-I panel!")


def register():
    try:
        unregister()
        bpy.utils.register_class(WhereAmIPanel)
        bpy.utils.register_class(WhereAmI)
        bpy.utils.register_class(ClearWhereAmI)
        bpy.utils.register_class(ClosestLabel)
        bpy.utils.register_class(ChooseVoxelID)
        bpy.utils.register_class(GrowLabel)
        bpy.utils.register_class(WaitForSlices)
        # print('Where am I Panel was registered!')
    except:
        print("Can't register Where am I Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(WhereAmIPanel)
        bpy.utils.unregister_class(WhereAmI)
        bpy.utils.unregister_class(ClearWhereAmI)
        bpy.utils.unregister_class(ClosestLabel)
        bpy.utils.unregister_class(ChooseVoxelID)
        bpy.utils.unregister_class(GrowLabel)
        bpy.utils.unregister_class(WaitForSlices)
    except:
        # print("Can't unregister Where am I Panel!")
        pass
