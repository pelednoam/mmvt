import bpy
import mathutils
import numpy as np
import os.path as op
import mmvt_utils as mu

bpy.types.Scene.where_am_i_str = ''


def _addon():
    return WhereAmIPanel.addon


def _trans():
    return WhereAmIPanel.subject_orig_trans


def where_i_am_draw(self, context):
    layout = self.layout
    layout.label(text='tkreg RAS coordinates')
    row = layout.row(align=0)
    row.prop(context.scene, "tkreg_ras_x", text="x")
    row.prop(context.scene, "tkreg_ras_y", text="y")
    row.prop(context.scene, "tkreg_ras_z", text="z")
    if not _trans() is None:
        layout.label(text='mni305 coordinates')
        row = layout.row(align=0)
        row.prop(context.scene, "ras_x", text="x")
        row.prop(context.scene, "ras_y", text="y")
        row.prop(context.scene, "ras_z", text="z")
        layout.label(text='T1 voxel indices')
        row = layout.row(align=0)
        row.prop(context.scene, "voxel_x", text="x")
        row.prop(context.scene, "voxel_y", text="y")
        row.prop(context.scene, "voxel_z", text="z")
        layout.label(text='')

    layout.operator("mmvt.where_i_am", text="Where Am I?", icon='SNAP_SURFACE')
    layout.operator("mmvt.where_am_i_clear", text="Clear", icon='PANEL_CLOSE')
    layout.label(text=bpy.types.Scene.where_am_i_str)


def tkras_coo_update(self, context):
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
    if not _trans() is None and WhereAmIPanel.update:
        coo = [bpy.context.scene.ras_x, bpy.context.scene.ras_y, bpy.context.scene.ras_z]
        vox = apply_trans(_trans().ras2vox, np.array([coo]))
        ras_tkr = apply_trans(_trans().vox2ras_tkr, vox)
        WhereAmIPanel.update = False
        set_tkreg_ras_coo(ras_tkr[0])
        set_voxel_coo(vox[0])
        WhereAmIPanel.update = True


def voxel_coo_update(self, context):
    if not _trans() is None and WhereAmIPanel.update:
        vox = [bpy.context.scene.voxel_x, bpy.context.scene.voxel_y, bpy.context.scene.voxel_z]
        ras = apply_trans(_trans().vox2ras, np.array([vox]))
        ras_tkr = apply_trans(_trans().vox2ras_tkr, [vox])
        WhereAmIPanel.update = False
        set_tkreg_ras_coo(ras_tkr[0], update_others=False)
        set_ras_coo(ras[0])
        WhereAmIPanel.update = True


def set_tkreg_ras_coo(coo, update_others=True):
    bpy.context.scene.tkreg_ras_x = coo[0]
    bpy.context.scene.tkreg_ras_y = coo[1]
    bpy.context.scene.tkreg_ras_z = coo[2]


def set_ras_coo(coo):
    bpy.context.scene.ras_x = coo[0]
    bpy.context.scene.ras_y = coo[1]
    bpy.context.scene.ras_z = coo[2]


def set_voxel_coo(coo):
    bpy.context.scene.voxel_x = int(np.round(coo[0]))
    bpy.context.scene.voxel_y = int(np.round(coo[1]))
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
            if bpy.context.object.parent == bpy.data.objects.get('Deep_electrodes', None):
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


class WhereAmIPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "Where Am I"
    addon = None
    subject_orig_trans = None
    update = True

    def draw(self, context):
        where_i_am_draw(self, context)


def init(addon):
    if op.isfile(op.join(mu.get_user_fol(), 'orig_trans.npz')):
        WhereAmIPanel.subject_orig_trans = mu.Bag(np.load(op.join(mu.get_user_fol(), 'orig_trans.npz')))
    WhereAmIPanel.addon = addon
    register()


def register():
    try:
        unregister()
        bpy.utils.register_class(WhereAmIPanel)
        bpy.utils.register_class(WhereAmI)
        bpy.utils.register_class(ClearWhereAmI)
        # print('Where am I Panel was registered!')
    except:
        print("Can't register Where am I Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(WhereAmIPanel)
        bpy.utils.unregister_class(WhereAmI)
        bpy.utils.unregister_class(ClearWhereAmI)
    except:
        # print("Can't unregister Where am I Panel!")
        pass
