import bpy
import mathutils
import os.path as op
import numpy as np
import mmvt_utils as mu


def _addon():
    return SkullPanel.addon


def import_skull():
    mu.change_layer(_addon().BRAIN_EMPTY_LAYER)
    layers_array = bpy.context.scene.layers
    emptys_name = 'Skull'
    base_path = op.join(mu.get_user_fol(), 'skull')
    _addon().create_empty_if_doesnt_exists(emptys_name, _addon().BRAIN_EMPTY_LAYER, layers_array)

    for skull_type in ['inner_skull', 'outer_skull']:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.import_mesh.ply(filepath=op.join(base_path, '{}.ply'.format(skull_type)))
        cur_obj = bpy.context.selected_objects[0]
        cur_obj.select = True
        bpy.ops.object.shade_smooth()
        cur_obj.scale = [0.1] * 3
        cur_obj.hide = False
        cur_obj.name = skull_type
        cur_obj.active_material = bpy.data.materials['Activity_map_mat']
        cur_obj.parent = bpy.data.objects[emptys_name]
        cur_obj.hide_select = True
        cur_obj.data.vertex_colors.new()


# def calc_thickness():
#     inner_skull = bpy.data.objects['inner_skull']
#     # inner_skull.data.normals_make_consistent(inside=False)
#     vert = inner_skull.data.vertices[0]
#     mw = bpy.data.objects['inner_skull'].matrix_world
#     mwi = mw.inverted()
#
#     # src and dst in local space of cb
#
#     origin = mwi * vert.co
#     direction = vert.normal
#
#     hit, loc, norm, face = inner_skull.ray_cast(vert.co + vert.normal / 10, vert.normal / 10)
#
#     print('vert loc {}'.format(vert.co))
#     if hit:
#         print("Hit at ", loc, " (local)")
#         bpy.ops.object.empty_add(location=mw * loc)
#     else:
#         print("No HIT")
#

def check_intersections():
    inner_skull = bpy.data.objects['inner_skull']
    outer_skull = bpy.data.objects['outer_skull']
    output_fname = op.join(mu.get_user_fol(), 'skull', 'intersections.npz')
    N = len(inner_skull.data.vertices)
    intersections = np.zeros((N, 2, 3))
    verts_faces = np.zeros((N, 1))
    for vert_num, vert in enumerate(inner_skull.data.vertices):
        outer_skull_hit_point, face_ind = check_vert_intersections(vert, outer_skull)
        intersections[vert_num, 0] = np.array(vert.co)
        intersections[vert_num, 1] = np.array(outer_skull_hit_point)
        verts_faces[vert_num] = face_ind
        # print(np.array(vert.co), np.array(outer_skull_hit_point), face_ind)
        if vert_num % 100 == 0:
            print('{} / {}'.format(vert_num, N))
            # np.savez(output_fname, intersections=intersections, verts_faces=verts_faces)
        #     print('Saving in {}!'.format(output_fname))
        #     np.save(output_fname, intersections)
    np.savez(output_fname, intersections=intersections, verts_faces=verts_faces)
    print('Finish!!!')


def check_intersections_fron_outer_skull():
    inner_skull = bpy.data.objects['inner_skull']
    outer_skull = bpy.data.objects['outer_skull']
    output_fname = op.join(mu.get_user_fol(), 'skull', 'intersections_from_outer_skull.npz')
    N = len(outer_skull.data.vertices)
    intersections = np.zeros((N, 2, 3))
    verts_faces = np.zeros((N, 1))
    for vert_num, vert in enumerate(outer_skull.data.vertices):
        inner_skull_hit_point, face_ind = check_vert_intersections(vert, inner_skull)
        intersections[vert_num, 0] = np.array(vert.co)
        if inner_skull_hit_point is not None:
            intersections[vert_num, 1] = np.array(inner_skull_hit_point)
            verts_faces[vert_num] = face_ind
        else:
            intersections[vert_num, 1] = np.array(vert.co)
            verts_faces[vert_num] = -1
            print('No intersection for {}'.format(vert_num))
        # print(np.array(vert.co), np.array(outer_skull_hit_point), face_ind)
        if vert_num % 100 == 0:
            print('{} / {}'.format(vert_num, N))
            # np.savez(output_fname, intersections=intersections, verts_faces=verts_faces)
        #     print('Saving in {}!'.format(output_fname))
        #     np.save(output_fname, intersections)
    np.savez(output_fname, intersections=intersections, verts_faces=verts_faces)
    print('Finish!!!')


def plot_distances():
    f = mu.Bag(np.load(op.join(mu.get_user_fol(), 'skull', 'intersections.npz')))
    distances = np.linalg.norm(f.intersections[:, 0] - f.intersections[:, 1], axis=1)
    faces_verts = np.load(op.join(mu.get_user_fol(), 'skull', 'faces_verts_inner_skull.npy'))
    inner_skull = bpy.data.objects['inner_skull']
    data_max = np.percentile(distances, 75)
    if _addon().colorbar_values_are_locked():
        data_max, data_min = _addon().get_colorbar_max_min()
    else:
        _addon().set_colorbar_max_min(data_max, 0)
    colors_ratio = 256 / data_max
    _addon().activity_map_obj_coloring(inner_skull, distances, faces_verts, 0, True, 0, colors_ratio)


def plot_distances_from_outer():
    f = mu.Bag(np.load(op.join(mu.get_user_fol(), 'skull', 'intersections_from_outer_skull.npz')))
    distances = np.linalg.norm(f.intersections[:, 0] - f.intersections[:, 1], axis=1)
    faces_verts = np.load(op.join(mu.get_user_fol(), 'skull', 'faces_verts_outer_skull.npy'))
    outer_skull = bpy.data.objects['outer_skull']
    data_max = np.percentile(distances, 75)
    if _addon().colorbar_values_are_locked():
        data_max, data_min = _addon().get_colorbar_max_min()
    else:
        _addon().set_colorbar_max_min(data_max, 0)
    colors_ratio = 256 / data_max
    _addon().activity_map_obj_coloring(outer_skull, distances, faces_verts, 0, True, 0, colors_ratio)


def fix_normals():
    bpy.ops.object.select_all(action='DESELECT')
    obj.select = True
    bpy.context.scene.objects.active = obj
    # go edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    # select al faces
    bpy.ops.mesh.select_all(action='SELECT')
    # recalculate outside normals
    bpy.ops.mesh.normals_make_consistent(inside=False)
    # go object mode again
    bpy.ops.object.editmode_toggle()


def check_vert_intersections(vert, skull):
    for face_ind, face in enumerate(skull.data.polygons):
        face_verts = [skull.data.vertices[vert].co for vert in face.vertices]
        intersection_point = mathutils.geometry.intersect_ray_tri(
            face_verts[0], face_verts[1], face_verts[2], vert.normal, vert.co, True)
        if intersection_point is not None:
            return intersection_point, face_ind
    return None, -1


def skull_draw(self, context):
    layout = self.layout
    layout.operator(ImportSkull.bl_idname, text="import skull", icon='ROTATE')
    layout.operator(CalcThickness.bl_idname, text="calc thickness", icon='ROTATE')
    layout.operator(PlotThickness.bl_idname, text="plot thickness", icon='ROTATE')


class ImportSkull(bpy.types.Operator):
    bl_idname = "mmvt.import_skull"
    bl_label = "Import skull"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        import_skull()
        return {'PASS_THROUGH'}


class CalcThickness(bpy.types.Operator):
    bl_idname = "mmvt.calc_thickness"
    bl_label = "calc_thickness"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        # calc_thickness()
        check_intersections()
        # check_intersections_fron_outer_skull()
        return {'PASS_THROUGH'}


class PlotThickness(bpy.types.Operator):
    bl_idname = "mmvt.plot_thickness"
    bl_label = "plot_thickness"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        plot_distances()
        # plot_distances_from_outer()
        return {'PASS_THROUGH'}


bpy.types.Scene.skull_files = bpy.props.EnumProperty(items=[], description="tempalte files")


class SkullPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "Skull"
    addon = None
    init = False

    def draw(self, context):
        if SkullPanel.init:
            skull_draw(self, context)


def init(addon):
    SkullPanel.addon = addon
    # user_fol = mu.get_user_fol()
    # skull_files = glob.glob(op.join(user_fol, 'skull', 'skull*.npz'))
    # if len(skull_files) == 0:
    #     return None
    # files_names = [mu.namebase(fname)[len('skull'):].replace('_', ' ') for fname in skull_files]
    # skull_items = [(c, c, '', ind) for ind, c in enumerate(files_names)]
    # bpy.types.Scene.skull_files = bpy.props.EnumProperty(
    #     items=skull_items, description="tempalte files",update=skull_files_update)
    # bpy.context.scene.skull_files = files_names[0]
    register()
    SkullPanel.init = True


def register():
    try:
        unregister()
        bpy.utils.register_class(SkullPanel)
        bpy.utils.register_class(ImportSkull)
        bpy.utils.register_class(CalcThickness)
        bpy.utils.register_class(PlotThickness)
    except:
        print("Can't register Skull Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(SkullPanel)
        bpy.utils.unregister_class(ImportSkull)
        bpy.utils.unregister_class(CalcThickness)
        bpy.utils.unregister_class(PlotThickness)
    except:
        pass
