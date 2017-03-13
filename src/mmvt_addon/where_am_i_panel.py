import bpy
import mathutils
import numpy as np
import os.path as op
import mmvt_utils as mu
import glob
import traceback
from collections import Counter

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
    layout.operator(ClearWhereAmI.bl_idname, text="Clear", icon='PANEL_CLOSE')
    layout.label(text=bpy.context.scene.where_am_i_str)


def tkras_coo_update(self, context):
    if not WhereAmIPanel.call_update:
        return

    # print('tkras_coo_update')
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
            id_ind = np.where(vol_atlas_lut['ids'] == id)[0][0]
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
                    id_ind = np.where(vol_atlas_lut['ids'] == val)[0][0]
                    names[atlas] = str(vol_atlas_lut['names'][id_ind])
                    # if atlas == bpy.context.scene.atlas:
                    #     obj_rh = bpy.data.objects.get('{}-rh'.format(names[atlas]))
                    #     obj_lh = bpy.data.objects.get('{}-lh'.format(names[atlas]))
                    #     if not obj_lh is None and not obj_rh is None:
                    #         obj_rh.select = True
                    #         obj_lh.select = True
                    #         print('select {} obj'.format(names[atlas]))
        except:
            print(traceback.format_exc())
            print('Error in trying to get the 3D atlas voxel value!')
        WhereAmIPanel.atlas_ids = names


def set_tkreg_ras_coo(coo):
    # print('set_tkreg_ras_coo')
    WhereAmIPanel.call_update = False
    bpy.context.scene.tkreg_ras_x = coo[0]
    bpy.context.scene.tkreg_ras_y = coo[1]
    WhereAmIPanel.call_update = True
    bpy.context.scene.tkreg_ras_z = coo[2]


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
    print('closest area is: '+closest_area)
    print('dist: {}'.format(np.min(np.array(distances))))
    print('closets vert is {}'.format(bpy.data.objects[closest_area].data.vertices[min_index].co))
    return closest_area


class ChooseVoxelID(bpy.types.Operator):
    bl_idname = "mmvt.choose_voxel_id"
    bl_label = "mmvt choose_voxel_id"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        obj = bpy.data.objects.get(bpy.context.scene.where_am_i_atlas, None)
        if not obj is None:
            obj.select = True
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

        bpy.types.Scene.where_am_i_str = ''
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
# bpy.types.Scene.where_am_i_atlas = bpy.props.StringProperty()


class WhereAmIPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "Where Am I"
    addon = None
    subject_orig_trans = None
    vol_atlas = {}
    vol_atlas_lut = {}
    atlas_ids = {}
    update = True
    call_update = True

    def draw(self, context):
        where_i_am_draw(self, context)


def init(addon):
    trans_fname = op.join(mu.get_user_fol(), 'orig_trans.npz')
    volumes = glob.glob(op.join(mu.get_user_fol(), 'freeview', '*+aseg.npy'))
    luts = glob.glob(op.join(mu.get_user_fol(), 'freeview', '*ColorLUT.npz'))
    if op.isfile(trans_fname):
        WhereAmIPanel.subject_orig_trans = mu.Bag(np.load(trans_fname))
    for atlas_vol_fname, atlas_vol_lut_fname in zip(volumes, luts):
        atlas = mu.namebase(atlas_vol_fname)[:-len('+aseg')]
        WhereAmIPanel.vol_atlas[atlas] = np.load(atlas_vol_fname)
        WhereAmIPanel.vol_atlas_lut[atlas] = np.load(atlas_vol_lut_fname)
    WhereAmIPanel.addon = addon
    register()


def register():
    try:
        unregister()
        bpy.utils.register_class(WhereAmIPanel)
        bpy.utils.register_class(WhereAmI)
        bpy.utils.register_class(ClearWhereAmI)
        bpy.utils.register_class(ChooseVoxelID)
        # print('Where am I Panel was registered!')
    except:
        print("Can't register Where am I Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(WhereAmIPanel)
        bpy.utils.unregister_class(WhereAmI)
        bpy.utils.unregister_class(ClearWhereAmI)
        bpy.utils.unregister_class(ChooseVoxelID)
    except:
        # print("Can't unregister Where am I Panel!")
        pass
