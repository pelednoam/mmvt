import bpy
import mmvt_utils as mu
import mathutils
import numpy as np
import os.path as op
import glob


def _addon():
    return DataInVertMakerPanel.addon


def find_vertex_index_and_mesh_closest_to_cursor(cursor=None, hemis=None, use_shape_keys=False, objects_names=None):
    # 3d cursor relative to the object data
    # print('cursor at:' + str(bpy.context.scene.cursor_location))
    # co_find = context.scene.cursor_location * obj.matrix_world.inverted()
    distances, names, vertices_idx, vertices_co = [], [], [], []

    # base_obj = bpy.data.objects['Functional maps']
    # meshes = HEMIS
    #        for obj in base_obj.children:
    if objects_names is not None:
        hemis = objects_names
    elif hemis is None:
        # hemis = mu.HEMIS if _addon().is_pial() else mu.INF_HEMIS
        hemis = mu.INF_HEMIS
    if cursor is None:
        cursor = bpy.context.scene.cursor_location
    else:
        cursor = mathutils.Vector(cursor)
    for obj_name in hemis:
        obj = bpy.data.objects[obj_name]
        co_find = cursor * obj.matrix_world.inverted()
        mesh = obj.data
        size = len(mesh.vertices)
        kd = mathutils.kdtree.KDTree(size)

        if use_shape_keys:
            me = obj.to_mesh(bpy.context.scene, True, 'PREVIEW')
            for i, v in enumerate(mesh.vertices):
                kd.insert(me.vertices[i].co, i)
            bpy.data.meshes.remove(me)
        else:
            for i, v in enumerate(mesh.vertices):
                kd.insert(v.co, i)

        kd.balance()
        # print(obj.name)
        for (co, index, dist) in kd.find_n(co_find, 1):
            # print('cursor at {} ,vertex {}, index {}, dist {}'.format(str(co_find), str(co), str(index), str(dist)))
            distances.append(dist)
            names.append(obj.name)
            vertices_idx.append(index)
            vertices_co.append(co)

    distances = np.array(distances)
    closest_mesh_name = names[np.argmin(distances)]
    # print('closest_mesh =' + str(closest_mesh_name))
    vertex_ind = vertices_idx[np.argmin(distances)]
    # print('vertex_ind = ' + str(vertex_ind))
    vertex_co = vertices_co[np.argmin(distances)] * obj.matrix_world
    distance = np.min(distances)
    # print('vertex_co', vertex_co)
    # print(closest_mesh_name, bpy.data.objects[closest_mesh_name].data.vertices[vertex_ind].co)
    # print(closest_mesh_name.replace('inflated_', ''), bpy.data.objects[closest_mesh_name.replace('inflated_', '')].data.vertices[vertex_ind].co)
    return closest_mesh_name, vertex_ind, vertex_co, distance


class ClearVertexData(bpy.types.Operator):
    bl_idname = "mmvt.vertex_data_clear"
    bl_label = "vertex data clear"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        for obj in bpy.data.objects:
            if obj.name.startswith('Activity_in_vertex'):
                obj.select = True
                # bpy.context.scene.objects.unlink(obj)
                bpy.data.objects.remove(obj)

        return {"FINISHED"}


class CreateVertexData(bpy.types.Operator):
    bl_idname = "mmvt.vertex_data_create"
    bl_label = "vertex data create"
    bl_options = {"UNDO"}

    @staticmethod
    def create_empty_in_vertex_location(vertex_location):
        mu.create_empty_in_vertex(vertex_location, 'Activity_in_vertex', DataInVertMakerPanel.addon.ACTIVITY_LAYER)


    @staticmethod
    def keyframe_empty(self, empty_name, closest_mesh_name, vertex_ind, data_path):
        obj = bpy.data.objects[empty_name]
        number_of_time_points = len(glob.glob(op.join(data_path, 'activity_map_' + closest_mesh_name + '2', '', ) + '*.npy'))
        mu.insert_keyframe_to_custom_prop(obj, 'data', 0, 0)
        mu.insert_keyframe_to_custom_prop(obj, 'data', 0, number_of_time_points + 1)
        for ii in range(number_of_time_points):
            # print(ii)
            frame_str = str(ii)
            f = np.load(op.join(data_path, 'activity_map_' + closest_mesh_name + '2', 't' + frame_str + '.npy'))
            mu.insert_keyframe_to_custom_prop(obj, 'data', float(f[vertex_ind, 0]), ii + 1)

        fcurves = bpy.data.objects[empty_name].animation_data.action.fcurves[0]
        mod = fcurves.modifiers.new(type='LIMITS')

    def keyframe_empty_test(self, empty_name, closest_mesh_name, vertex_ind, data_path):
        obj = bpy.data.objects[empty_name]
        lookup = np.load(op.join(data_path, 'activity_map_' + closest_mesh_name + '_verts_lookup.npy'))
        file_num_str = str(int(lookup[vertex_ind, 0]))
        line_num = int(lookup[vertex_ind, 1])
        data_file = np.load(
            op.join(data_path, 'activity_map_' + closest_mesh_name + '_verts', file_num_str + '.npy'))
        data = data_file[line_num, :].squeeze()

        number_of_time_points = len(data)
        mu.insert_keyframe_to_custom_prop(obj, 'data', 0, 0)
        mu.insert_keyframe_to_custom_prop(obj, 'data', 0, number_of_time_points + 1)
        for ii in range(number_of_time_points):
            # print(ii)
            frame_str = str(ii)
            mu.insert_keyframe_to_custom_prop(obj, 'data', float(data[ii]), ii + 1)
            # insert_keyframe_to_custom_prop(obj,'data',0,ii+1)
        fcurves = bpy.data.objects[empty_name].animation_data.action.fcurves[0]
        mod = fcurves.modifiers.new(type='LIMITS')

    def invoke(self, context, event=None):
        closest_mesh_name, vertex_ind, vertex_co, _ = find_vertex_index_and_mesh_closest_to_cursor()
        print(vertex_co)
        self.create_empty_in_vertex_location(vertex_co)
        data_path = mu.get_user_fol()
        self.keyframe_empty_test('Activity_in_vertex', closest_mesh_name, vertex_ind, data_path)
        return {"FINISHED"}


class DataInVertMakerPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "Data in vertex"
    addon = None
    init = False

    def draw(self, context):
        layout = self.layout
        layout.operator(CreateVertexData.bl_idname, text="Get data in vertex", icon='ROTATE')
        layout.operator(ClearVertexData.bl_idname, text="Clear", icon='PANEL_CLOSE')
        # layout.operator(, text="Get data in vertex", icon='ROTATE')


def init(addon):
    DataInVertMakerPanel.addon = addon
    lookup_files = glob.glob(op.join(mu.get_user_fol(), 'activity_map_*_verts_lookup.npy'))
    if len(lookup_files) == 0:
        print('No lookup files for vertex_data_panel')
        DataInVertMakerPanel.init = False
    DataInVertMakerPanel.init = True
    register()


def register():
    try:
        unregister()
        bpy.utils.register_class(DataInVertMakerPanel)
        bpy.utils.register_class(CreateVertexData)
        bpy.utils.register_class(ClearVertexData)
        # print('Vertex Panel was registered!')
    except:
        print("Can't register Vertex Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(DataInVertMakerPanel)
        bpy.utils.unregister_class(CreateVertexData)
        bpy.utils.unregister_class(ClearVertexData)
    except:
        pass