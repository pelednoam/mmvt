import bpy
import mathutils
import os.path as op
import numpy as np
import mmvt_utils as mu


def _addon():
    return SkullPanel.addon


def thickness_arrows_update(self, context):
    mu.show_hide_hierarchy(bpy.context.scene.thickness_arrows, 'thickness_arrows', also_parent=True, select=False)


def import_skull():
    mu.change_layer(_addon().BRAIN_EMPTY_LAYER)
    base_path = op.join(mu.get_user_fol(), 'skull')
    emptys_name = 'Skull'
    layers_array = bpy.context.scene.layers
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
    # f = mu.Bag(np.load(op.join(mu.get_user_fol(), 'skull', 'intersections.npz')))
    # distances = np.linalg.norm(f.intersections[:, 0] - f.intersections[:, 1], axis=1)
    distances = np.load(op.join(mu.get_user_fol(), 'skull', 'ray_casts.npy'))
    faces_verts = np.load(op.join(mu.get_user_fol(), 'skull', 'faces_verts_inner_skull.npy'))
    inner_skull = bpy.data.objects['inner_skull']
    data_max = np.percentile(distances, 75)
    if _addon().colorbar_values_are_locked():
        data_max, data_min = _addon().get_colorbar_max_min()
    else:
        _addon().set_colorbar_max_min(data_max, 0)
    _addon().set_colorbar_title('Skull thickness (mm)')
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


def find_point_thickness():
    vertex_ind, closest_mesh_name = _addon().snap_cursor()
    vertex_co = bpy.data.objects['inner_skull'].data.vertices[vertex_ind].co
    distances = np.load(op.join(mu.get_user_fol(), 'skull', 'ray_casts.npy'))
    rays_info = mu.load(op.join(mu.get_user_fol(), 'skull', 'ray_casts_info.pkl'))
    # closest_mesh_name, vertex_ind, vertex_co, _ = _addon().find_vertex_index_and_mesh_closest_to_cursor(
    #     objects_names=['inner_skull'])
    # distance = np.linalg.norm(f.intersections[vertex_ind, 0] - f.intersections[vertex_ind, 1])
    distance = distances[vertex_ind][0]
    (hit, loc, norm, index) = rays_info[vertex_ind]

    # bpy.context.scene.cursor_location = vertex_co
    print(closest_mesh_name, vertex_ind, vertex_co, distance, loc)
    SkullPanel.vertex_skull_thickness = distance


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


def ray_cast():
    context = bpy.context
    scene = context.scene
    layers_array = bpy.context.scene.layers
    emptys_name = 'thickness_arrows'
    show_hit = bpy.data.objects.get(emptys_name, None) is None
    _addon().create_empty_if_doesnt_exists(emptys_name, _addon().BRAIN_EMPTY_LAYER, layers_array, 'Skull')

    def draw_empty_arrow(loc, dir):
        R = (-dir).to_track_quat('Z', 'X').to_matrix().to_4x4()
        mt = bpy.data.objects.new("mt", None)
        R.translation = loc + dir
        # mt.show_name = True
        mt.matrix_world = R
        mt.empty_draw_type = 'SINGLE_ARROW'
        mt.empty_draw_size = dir.length
        scene.objects.link(mt)
        mt.parent = bpy.data.objects[emptys_name]

    # check thickness by raycasting from inner object out.
    # select inner and outer obj, make inner active
    inner_obj = bpy.data.objects['inner_skull']
    outer_obj = bpy.data.objects['outer_skull']
    omwi = outer_obj.matrix_world.inverted()

    output_fname = op.join(mu.get_user_fol(), 'skull', 'ray_casts.npy')
    output_info_fname = op.join(mu.get_user_fol(), 'skull', 'ray_casts_info.pkl')
    N = len(inner_obj.data.vertices)
    vertices_thickness = np.zeros((N, 1))
    thickness_info = {}

    imw = inner_obj.matrix_world
    omw = outer_obj.matrix_world
    mat = omwi * imw
    factor = np.linalg.inv(omw)[0, 0]
    hits = []  # vectors from inner to outer
    # for face in inner_obj.data.polygons:
    for vert_ind, vert in enumerate(inner_obj.data.vertices):
        # o = mat * face.center
        # n = mat * (face.center + face.normal) - o
        o = mat * vert.co
        n = mat * (vert.co + vert.normal) - o

        hit, loc, norm, index = outer_obj.ray_cast(o, n)
        if hit:
            print('{}/{} hit outer on face {}'.format(vert_ind, N, index))
            hits.append((o, loc))
            thickness = (omw * loc - omw * o).length * factor
        else:
            print('{}/{} no hit!'.format(vert_ind, N))
            thickness = 0
        vertices_thickness[vert_ind] = thickness
        thickness_info[vert_ind] = (hit, np.array(loc), np.array(norm), index)

    np.save(output_fname, vertices_thickness)
    mu.save(thickness_info, output_info_fname)

    if hits:
        avge_thickness = sum((omw * hit - omw * o).length for o, hit in hits) / len(hits)
        print(avge_thickness)
        if show_hit:
            for hit, o in hits:
                draw_empty_arrow(omw * o, omw * hit - omw * o)


def skull_draw(self, context):
    layout = self.layout
    layout.operator(ImportSkull.bl_idname, text="Import skull", icon='MATERIAL_DATA')
    # layout.operator(CalcThickness.bl_idname, text="calc thickness", icon='MESH_ICOSPHERE')
    layout.operator(RayCast.bl_idname, text="Calc thickness", icon='MESH_ICOSPHERE')
    layout.operator(PlotThickness.bl_idname, text="Plot thickness", icon='GROUP_VCOL')
    # layout.operator(FindPointThickness.bl_idname, text="Calc point thickness", icon='MESH_DATA')
    if SkullPanel.vertex_skull_thickness > 0:
        layout.label(text='Thickness: {:.3f}'.format(SkullPanel.vertex_skull_thickness))
    layout.prop(context.scene, 'thickness_arrows', text='Thickness arrows')


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


class RayCast(bpy.types.Operator):
    bl_idname = "mmvt.ray_cast"
    bl_label = "ray_cast"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        ray_cast()
        return {'PASS_THROUGH'}


class PlotThickness(bpy.types.Operator):
    bl_idname = "mmvt.plot_thickness"
    bl_label = "plot_thickness"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        plot_distances()
        # plot_distances_from_outer()
        return {'PASS_THROUGH'}


class FindPointThickness(bpy.types.Operator):
    bl_idname = "mmvt.find_point_thickness"
    bl_label = "find_point_thickness"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        find_point_thickness()
        # plot_distances_from_outer()
        return {'PASS_THROUGH'}


bpy.types.Scene.thickness_arrows = bpy.props.BoolProperty(default=False, update=thickness_arrows_update)


class SkullPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "Skull"
    addon = None
    init = False
    vertex_skull_thickness = 0

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
    bpy.context.scene.thickness_arrows = False


def register():
    try:
        unregister()
        bpy.utils.register_class(SkullPanel)
        bpy.utils.register_class(ImportSkull)
        bpy.utils.register_class(CalcThickness)
        bpy.utils.register_class(PlotThickness)
        bpy.utils.register_class(FindPointThickness)
        bpy.utils.register_class(RayCast)
    except:
        print("Can't register Skull Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(SkullPanel)
        bpy.utils.unregister_class(ImportSkull)
        bpy.utils.unregister_class(CalcThickness)
        bpy.utils.unregister_class(PlotThickness)
        bpy.utils.unregister_class(FindPointThickness)
        bpy.utils.unregister_class(RayCast)
    except:
        pass
