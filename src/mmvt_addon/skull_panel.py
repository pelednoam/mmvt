import bpy
import mathutils
import os.path as op
import numpy as np
import mmvt_utils as mu


# https://blender.stackexchange.com/questions/7465/create-a-flat-plane-with-beveled-edges
# x = 5.59mm
# y = 4.19mm
# z rot=-90

def _addon():
    return SkullPanel.addon


def thickness_arrows_update(self, context):
    mu.show_hide_hierarchy(bpy.context.scene.thickness_arrows, 'thickness_arrows', also_parent=True, select=False)


def cast_ray_source_update(self, context):
    inner_skull = bpy.data.objects.get('inner_skull', None)
    outer_skull = bpy.data.objects.get('outer_skull', None)
    if inner_skull is None or outer_skull is None:
        return
    inner_skull.hide = bpy.context.scene.cast_ray_source != 'inner'
    outer_skull.hide = bpy.context.scene.cast_ray_source == 'inner'


def show_point_arrow_update(self, context):
    if SkullPanel.prev_vertex_arrow is not None:
        SkullPanel.prev_vertex_arrow.hide = True


def import_skull():
    mu.change_layer(_addon().BRAIN_EMPTY_LAYER)
    base_path = op.join(mu.get_user_fol(), 'skull')
    emptys_name = 'Skull'
    layers_array = bpy.context.scene.layers
    _addon().create_empty_if_doesnt_exists(emptys_name, _addon().SKULL_LAYER, layers_array)

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
#
# def check_intersections():
#     inner_skull = bpy.data.objects['inner_skull']
#     outer_skull = bpy.data.objects['outer_skull']
#     output_fname = op.join(mu.get_user_fol(), 'skull', 'intersections.npz')
#     N = len(inner_skull.data.vertices)
#     intersections = np.zeros((N, 2, 3))
#     verts_faces = np.zeros((N, 1))
#     for vert_num, vert in enumerate(inner_skull.data.vertices):
#         outer_skull_hit_point, face_ind = check_vert_intersections(vert, outer_skull)
#         intersections[vert_num, 0] = np.array(vert.co)
#         intersections[vert_num, 1] = np.array(outer_skull_hit_point)
#         verts_faces[vert_num] = face_ind
#         # print(np.array(vert.co), np.array(outer_skull_hit_point), face_ind)
#         if vert_num % 100 == 0:
#             print('{} / {}'.format(vert_num, N))
#             # np.savez(output_fname, intersections=intersections, verts_faces=verts_faces)
#         #     print('Saving in {}!'.format(output_fname))
#         #     np.save(output_fname, intersections)
#     np.savez(output_fname, intersections=intersections, verts_faces=verts_faces)
#     print('Finish!!!')

#
# def check_intersections_fron_outer_skull():
#     inner_skull = bpy.data.objects['inner_skull']
#     outer_skull = bpy.data.objects['outer_skull']
#     output_fname = op.join(mu.get_user_fol(), 'skull', 'intersections_from_outer_skull.npz')
#     N = len(outer_skull.data.vertices)
#     intersections = np.zeros((N, 2, 3))
#     verts_faces = np.zeros((N, 1))
#     for vert_num, vert in enumerate(outer_skull.data.vertices):
#         inner_skull_hit_point, face_ind = check_vert_intersections(vert, inner_skull)
#         intersections[vert_num, 0] = np.array(vert.co)
#         if inner_skull_hit_point is not None:
#             intersections[vert_num, 1] = np.array(inner_skull_hit_point)
#             verts_faces[vert_num] = face_ind
#         else:
#             intersections[vert_num, 1] = np.array(vert.co)
#             verts_faces[vert_num] = -1
#             print('No intersection for {}'.format(vert_num))
#         # print(np.array(vert.co), np.array(outer_skull_hit_point), face_ind)
#         if vert_num % 100 == 0:
#             print('{} / {}'.format(vert_num, N))
#             # np.savez(output_fname, intersections=intersections, verts_faces=verts_faces)
#         #     print('Saving in {}!'.format(output_fname))
#         #     np.save(output_fname, intersections)
#     np.savez(output_fname, intersections=intersections, verts_faces=verts_faces)
#     print('Finish!!!')
#

def plot_distances(from_inner=True):
    # f = mu.Bag(np.load(op.join(mu.get_user_fol(), 'skull', 'intersections.npz')))
    # distances = np.linalg.norm(f.intersections[:, 0] - f.intersections[:, 1], axis=1)
    source_str = 'from_inner' if from_inner else 'from_outer'
    distances = np.load(op.join(mu.get_user_fol(), 'skull', 'ray_casts_{}.npy'.format(source_str)))
    faces_verts = np.load(op.join(mu.get_user_fol(), 'skull', 'faces_verts_{}_skull.npy'.format('inner' if from_inner else 'outer')))
    skull_obj = bpy.data.objects['{}_skull'.format('inner' if from_inner else 'outer')]
    data_max = 25 #np.percentile(distances, 75)
    if _addon().colorbar_values_are_locked():
        data_max, data_min = _addon().get_colorbar_max_min()
    else:
        _addon().set_colorbar_max_min(data_max, 0)
        _addon().set_colormap('hot')
    colors_ratio = 256 / data_max
    _addon().activity_map_obj_coloring(skull_obj, distances, faces_verts, 0, True, 0, colors_ratio)


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
    source_str = 'from_inner' if bpy.context.scene.cast_ray_source == 'inner' else 'from_outer'
    distances_fname = op.join(mu.get_user_fol(), 'skull', 'ray_casts_{}.npy'.format(source_str))
    ray_info_fname = op.join(mu.get_user_fol(), 'skull', 'ray_casts_info_{}.pkl'.format(source_str))
    if not op.isfile(distances_fname) or not op.isfile(ray_info_fname):
        print("Can't find distances file! {}".format(distances_fname))
        return
    vertex_ind, closest_mesh_name = _addon().snap_cursor()
    vertex_co = bpy.data.objects['inner_skull'].data.vertices[vertex_ind].co
    distances = np.load(distances_fname)
    rays_info = mu.load(ray_info_fname)
    # closest_mesh_name, vertex_ind, vertex_co, _ = _addon().find_vertex_index_and_mesh_closest_to_cursor(
    #     objects_names=['inner_skull'])
    # distance = np.linalg.norm(f.intersections[vertex_ind, 0] - f.intersections[vertex_ind, 1])
    distance = distances[vertex_ind][0]
    (hit, loc, norm, index) = rays_info[vertex_ind]

    # bpy.context.scene.cursor_location = vertex_co
    print(closest_mesh_name, vertex_ind, vertex_co, distance, loc)
    SkullPanel.vertex_skull_thickness = distance
    if not bpy.context.scene.thickness_arrows and bpy.context.scene.show_point_arrow:
        if SkullPanel.prev_vertex_arrow is not None:
            SkullPanel.prev_vertex_arrow.hide = True
        vertex_arrow = bpy.data.objects.get('mt_{}'.format(vertex_ind), None)
        if vertex_arrow is not None:
            vertex_arrow.hide = False
            SkullPanel.prev_vertex_arrow = vertex_arrow


def fix_normals():
    # c = mu.get_view3d_context()
    bpy.ops.object.select_all(action='DESELECT')

    for skull_type in ['inner_skull', 'outer_skull']:
        obj = bpy.context.scene.objects[skull_type]
        obj.select = True
        bpy.context.scene.objects.active = obj
        # go edit mode
        # bpy.ops.object.mode_set(c, mode='OBJECT')
        # bpy.ops.object.mode_set(c, mode='EDIT')
        # select al faces
        # bpy.ops.mesh.select_all(c, action='SELECT')
        # recalculate outside normals
        # bpy.ops.mesh.normals_make_consistent(inside=skull_type == 'outer_skull')
        select_all_faces(obj.data, skull_type=='outer_skull')
        # go object mode again
        # bpy.ops.object.editmode_toggle()


def select_all_faces(mesh, reverse=False):
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    # if reverse:
    #     bmesh.ops.reverse_faces(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.clear()
    mesh.update()
    bm.free()


def align_plane_to_point(obj, point):
    dir = point - obj.location
    return dir.to_track_quat("Z", "X").to_euler()


def calc_trans_mat(obj, face, target):
    from mathutils import Matrix
    mat_world = obj.matrix_world
    #transform the face to world space
    #to take non-uniform scaling into account
    #which may change the angle of face.normal
    for index in face.vertices:
        vert = obj.data.vertices[index]
        vert.co = mat_world * vert.co

    #get the rotation difference
    track = target - face.center
    q = face.normal.rotation_difference(track)

    #compose the matrix
    #rotation around face.center in world space
    mat = Matrix.Translation(face.center) * \
          q.to_matrix().to_4x4() * \
          Matrix.Translation(-face.center)
    #transform the face back to object space afterwards
    mat_obj = mat_world.inverted() * mat
    #apply the matrix to the vertices of the face
    for index in face.vertices:
        vert = obj.data.vertices[index]
        vert.co = mat_obj * vert.co


# def rotate_plane(skull_face, plane):
#     from mathutils import Matrix, Vector
#
#     def scale_from_vector(v):
#         mat = Matrix.Identity(4)
#         for i in range(3):
#             mat[i][i] = v[i]
#         return mat
#
#     loc_dst, rot_dst, scale_dst = plane.matrix_world.decompose()
#     loc_src, rot_src, scale_src = skull_face.matrix_world.decompose()
#
#     plane.matrix_world = (
#         Matrix.Translation(loc_dst) *
#         rot_src.to_matrix().to_4x4() *
#         scale_from_vector(scale_dst)
#     )


def track_to_point(obj, point):
    import mathutils
    normal = obj.data.polygons[0].normal.xyz
    mat_obj = obj.matrix_basis
    mat_scale = mathutils.Matrix.Scale(1, 4, mat_obj.to_scale() )
    trans = mat_obj.to_translation()
    mat_trans = mathutils.Matrix.Translation(trans)
    print( "mat_scale\n" + str(mat_obj.to_scale()))
    point_trans = point -trans
    q = normal.rotation_difference( point_trans )
    mat_rot = q.to_matrix()
    mat_rot.resize_4x4()

    mat_obj = mat_trans * mat_rot * mat_scale
    obj.matrix_basis = mat_obj


def align_plane():
    # https://blender.stackexchange.com/questions/12314/in-python-rotate-a-polygon-to-face-something/12324#12324
    # https://blender.stackexchange.com/questions/32649/how-to-rotate-an-object-from-a-reference-plane
    plane = bpy.data.objects.get('skull_plane', None)
    if plane is None:
        print('plane is None!')
        return
    # vertices_face = mu.load(op.join(mu.get_user_fol(), 'skull', 'outer_skull_vertices_faces.pkl'))
    skull = bpy.data.objects['outer_skull']
    _, vertex_ind, vertex_co, _ = \
        _addon().find_vertex_index_and_mesh_closest_to_cursor(plane.location, objects_names=['outer_skull'])
    if bpy.context.scene.align_plane_to_cursor:
        vertex_co = bpy.context.scene.cursor_location + skull.data.vertices[vertex_ind].normal * 5
    if np.linalg.norm(plane.location - vertex_co) > 0.001:
        calc_trans_mat(plane, plane.data.polygons[0], vertex_co)
        plane.location = vertex_co
        get_plane_values()
    # faces = vertices_face[vertex_ind]
    # face_ind = faces[0]
    # skull_face = skull.data.polygons[face_ind]
    # track_to_point(plane, vertex_co)
    # plane.rotation_euler = align_plane_to_point(plane, vertex_co)
    # vert = skull.data.vertices[vertex_ind]
    # plane.rotation_euler = vert.normal.to_track_quat('Y', 'Z').to_euler()


def get_plane_values():
    skull_thickness = np.load(op.join(mu.get_user_fol(), 'skull', 'ray_casts_from_outer.npy'))

    inner_obj = bpy.data.objects['outer_skull']
    outer_obj = bpy.data.objects['skull_plane']
    omwi = outer_obj.matrix_world.inverted()
    imw = inner_obj.matrix_world
    mat = omwi * imw
    plane_thikness = []
    vertices = inner_obj.data.vertices
    for vert_ind, vert in enumerate(vertices):
        o = mat * vert.co
        n = mat * (vert.co + vert.normal) - o
        hit, _, _, _ = outer_obj.ray_cast(o, n)
        if hit:
            plane_thikness.append(skull_thickness[vert_ind])

    # np.save(output_fname, vertices_thickness)
    # mu.save(thickness_info, output_info_fname)

    if len(plane_thikness) > 0:
        plane_thikness = np.array(plane_thikness).squeeze()
        SkullPanel.plane_thickness = (np.min(plane_thikness), np.max(plane_thikness), np.mean(plane_thikness))

def check_vert_intersections(vert, skull):
    for face_ind, face in enumerate(skull.data.polygons):
        face_verts = [skull.data.vertices[vert].co for vert in face.vertices]
        intersection_point = mathutils.geometry.intersect_ray_tri(
            face_verts[0], face_verts[1], face_verts[2], vert.normal, vert.co, True)
        if intersection_point is not None:
            return intersection_point, face_ind
    return None, -1


def ray_cast(from_inner=True):
    context = bpy.context
    scene = context.scene
    layers_array = bpy.context.scene.layers
    from_string = 'from_{}'.format('inner' if from_inner else 'outer')
    emptys_name = 'thickness_arrows_{}'.format(from_string)
    show_hit = bpy.data.objects.get(emptys_name, None) is None and bpy.context.scene.create_thickness_arrows

    # check thickness by raycasting from inner object out.
    # select inner and outer obj, make inner active
    inner_obj = bpy.data.objects['inner_skull']
    outer_obj = bpy.data.objects['outer_skull']
    omwi = outer_obj.matrix_world.inverted() if from_inner else inner_obj.matrix_world.inverted()
    output_fname = op.join(mu.get_user_fol(), 'skull', 'ray_casts_{}.npy'.format(from_string))
    output_info_fname = op.join(mu.get_user_fol(), 'skull', 'ray_casts_info_{}.pkl'.format(from_string))
    N = len(inner_obj.data.vertices) if from_inner else len(outer_obj.data.vertices)
    vertices_thickness = np.zeros((N, 1))
    thickness_info = {}

    imw = inner_obj.matrix_world if from_inner else outer_obj.matrix_world
    omw = outer_obj.matrix_world if from_inner else inner_obj.matrix_world
    mat = omwi * imw
    factor = np.linalg.inv(omw)[0, 0]
    ray_obj = outer_obj if from_inner else inner_obj
    hits = []  # vectors from inner to outer
    # for face in inner_obj.data.polygons:
    vertices = inner_obj.data.vertices if from_inner else outer_obj.data.vertices
    for vert_ind, vert in enumerate(vertices):
        # o = mat * face.center
        # n = mat * (face.center + face.normal) - o
        o = mat * vert.co
        n = mat * (vert.co + vert.normal) - o
        if not from_inner:
            n *= -1
        hit, loc, norm, index = ray_obj.ray_cast(o, n)
        if hit:
            print('{}/{} hit {} on face {}'.format(vert_ind, N, 'outer' if from_inner else 'innner', index))
            hits.append((vert_ind, o, loc))
            thickness = (omw * loc - omw * o).length * factor
        else:
            print('{}/{} no hit!'.format(vert_ind, N))
            thickness = 0
        vertices_thickness[vert_ind] = thickness
        thickness_info[vert_ind] = (hit, np.array(loc), np.array(norm), index)

    np.save(output_fname, vertices_thickness)
    mu.save(thickness_info, output_info_fname)

    if hits:
        avge_thickness = sum((omw * hit - omw * o).length for vert_ind, o, hit in hits) / len(hits)
        print(avge_thickness)
        if show_hit:
            _addon().create_empty_if_doesnt_exists(emptys_name, _addon().BRAIN_EMPTY_LAYER, layers_array, 'Skull')
            for vert_ind, hit, o in hits:
                draw_empty_arrow(scene, emptys_name, vert_ind, omw * o, omw * hit - omw * o)


def draw_empty_arrow(scene, empty_name, vert_ind, loc, dir):
    R = (-dir).to_track_quat('Z', 'X').to_matrix().to_4x4()
    mt = bpy.data.objects.new('mt_{}'.format(vert_ind), None)
    mt.name = 'mt_{}'.format(vert_ind)
    R.translation = loc + dir
    # mt.show_name = True
    mt.matrix_world = R
    mt.empty_draw_type = 'SINGLE_ARROW'
    mt.empty_draw_size = dir.length
    scene.objects.link(mt)
    mt.parent = bpy.data.objects[empty_name]


def skull_draw(self, context):
    layout = self.layout
    layout.operator(ImportSkull.bl_idname, text="Import skull", icon='MATERIAL_DATA')
    # layout.operator(CalcThickness.bl_idname, text="calc thickness", icon='MESH_ICOSPHERE')
    layout.operator(CalcThickness.bl_idname, text="Calc thickness", icon='MESH_ICOSPHERE')
    layout.prop(context.scene, 'create_thickness_arrows', text='Create thickness arrows')
    layout.operator(PlotThickness.bl_idname, text="Plot thickness", icon='GROUP_VCOL')
    layout.operator(AlignPlane.bl_idname, text="Align plane", icon='LATTICE_DATA')
    layout.prop(context.scene, 'align_plane_to_cursor', text='Align to cursor')
    if SkullPanel.plane_thickness is not None:
        box = layout.box()
        col = box.column()
        mu.add_box_line(col, 'min', '{:.2f}'.format(SkullPanel.plane_thickness[0]), 0.8)
        mu.add_box_line(col, 'max', '{:.2f}'.format(SkullPanel.plane_thickness[1]), 0.8)
        mu.add_box_line(col, 'mean', '{:.2f}'.format(SkullPanel.plane_thickness[2]), 0.8)

    layout.prop(context.scene, 'cast_ray_source', expand=True)
    # layout.operator(FindPointThickness.bl_idname, text="Calc point thickness", icon='MESH_DATA')
    if SkullPanel.vertex_skull_thickness > 0:
        layout.label(text='Thickness: {:.3f}'.format(SkullPanel.vertex_skull_thickness))

    from_string = 'from_{}'.format('inner' if bpy.context.scene.cast_ray_source else 'outer')
    arrows_empty_name = 'thickness_arrows_{}'.format(from_string)
    if bpy.data.objects.get(arrows_empty_name):
        layout.prop(context.scene, 'show_point_arrow', text='Show point thickness vector')
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
        fix_normals()
        for inner in [True, False]:
            ray_cast(inner) #bpy.context.scene.cast_ray_source == 'inner')
        return {'PASS_THROUGH'}


class PlotThickness(bpy.types.Operator):
    bl_idname = "mmvt.plot_thickness"
    bl_label = "plot_thickness"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        _addon().set_colorbar_title('Skull thickness (mm)')
        for inner in [True, False]:
            plot_distances(inner) #bpy.context.scene.cast_ray_source == 'inner')
        return {'PASS_THROUGH'}


class FindPointThickness(bpy.types.Operator):
    bl_idname = "mmvt.find_point_thickness"
    bl_label = "find_point_thickness"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        find_point_thickness()
        return {'PASS_THROUGH'}


class AlignPlane(bpy.types.Operator):
    bl_idname = "mmvt.align_plane"
    bl_label = "align_plane"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        align_plane()
        return {'PASS_THROUGH'}


bpy.types.Scene.thickness_arrows = bpy.props.BoolProperty(default=False, update=thickness_arrows_update)
bpy.types.Scene.show_point_arrow = bpy.props.BoolProperty(default=False, update=show_point_arrow_update)
bpy.types.Scene.cast_ray_source = bpy.props.EnumProperty(items=[('inner', 'inner', '', 0), ('outer', 'outer', '', 1)],
                                                         update=cast_ray_source_update)
bpy.types.Scene.create_thickness_arrows = bpy.props.BoolProperty(default=False)
bpy.types.Scene.align_plane_to_cursor = bpy.props.BoolProperty(default=False)


class SkullPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "Skull"
    addon = None
    init = False
    vertex_skull_thickness = 0
    prev_vertex_arrow = None
    plane_thickness = None

    def draw(self, context):
        if SkullPanel.init:
            skull_draw(self, context)


def init(addon):
    import math
    SkullPanel.addon = addon
    user_fol = mu.get_user_fol()
    skull_ply_files_exist = op.isfile(op.join(user_fol, 'skull', 'inner_skull.ply')) and \
                            op.isfile(op.join(user_fol, 'skull', 'outer_skull.ply'))
    skull_objs_exist = bpy.data.objects.get('inner_skull', None) is not None and \
                       bpy.data.objects.get('inner_skull', None) is not None
    if not skull_ply_files_exist and not skull_objs_exist:
        return
    for layer_ind in range(len(bpy.context.scene.layers)):
        bpy.context.scene.layers[layer_ind] = layer_ind == _addon().SKULL_LAYER
    plane = bpy.data.objects.get('skull_plane', None)
    if plane is not None:
        # plane.dimensions[0] = 5.59
        # plane.dimensions[1] = 4.19
        plane.rotation_euler[0] = plane.rotation_euler[1] = 0
        plane.rotation_euler[2] = -math.pi
        plane.location[0] = plane.location[1] = 0
        plane.location[2] = 10


    register()
    SkullPanel.init = True
    bpy.context.scene.thickness_arrows = False
    bpy.context.scene.show_point_arrow = False


def register():
    try:
        unregister()
        bpy.utils.register_class(SkullPanel)
        bpy.utils.register_class(ImportSkull)
        bpy.utils.register_class(CalcThickness)
        bpy.utils.register_class(PlotThickness)
        bpy.utils.register_class(FindPointThickness)
        bpy.utils.register_class(AlignPlane)
    except:
        print("Can't register Skull Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(SkullPanel)
        bpy.utils.unregister_class(ImportSkull)
        bpy.utils.unregister_class(CalcThickness)
        bpy.utils.unregister_class(PlotThickness)
        bpy.utils.unregister_class(FindPointThickness)
        bpy.utils.unregister_class(AlignPlane)
    except:
        pass
