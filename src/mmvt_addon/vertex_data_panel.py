import bpy
import mmvt_utils as mu
import mathutils
import numpy as np
import os.path as op
import glob


class ClearVertexData(bpy.types.Operator):
    bl_idname = "ohad.vertex_data_clear"
    bl_label = "vertex data clear"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        for obj in bpy.data.objects:
            if obj.name.startswith('Activity_in_vertex'):
                obj.select = True
                bpy.context.scene.objects.unlink(obj)
                bpy.data.objects.remove(obj)

        return {"FINISHED"}


class CreateVertexData(bpy.types.Operator):
    bl_idname = "ohad.vertex_data_create"
    bl_label = "vertex data create"
    bl_options = {"UNDO"}

    @staticmethod
    def find_vertex_index_and_mesh_closest_to_cursor(self):
        # 3d cursor relative to the object data
        print('cursor at:' + str(bpy.context.scene.cursor_location))
        # co_find = context.scene.cursor_location * obj.matrix_world.inverted()
        distances = []
        names = []
        vertices_idx = []
        vertices_co = []

        # base_obj = bpy.data.objects['Functional maps']
        # meshes = HEMIS
        #        for obj in base_obj.children:
        for cur_obj in mu.HEMIS:
            obj = bpy.data.objects[cur_obj]
            co_find = bpy.context.scene.cursor_location * obj.matrix_world.inverted()
            mesh = obj.data
            size = len(mesh.vertices)
            kd = mathutils.kdtree.KDTree(size)

            for i, v in enumerate(mesh.vertices):
                kd.insert(v.co, i)

            kd.balance()
            print(obj.name)
            for (co, index, dist) in kd.find_n(co_find, 1):
                print('cursor at {} ,vertex {}, index {}, dist {}'.format(str(co_find), str(co), str(index),str(dist)))
                distances.append(dist)
                names.append(obj.name)
                vertices_idx.append(index)
                vertices_co.append(co)

        closest_mesh_name = names[np.argmin(np.array(distances))]
        print('closest_mesh =' + str(closest_mesh_name))
        vertex_ind = vertices_idx[np.argmin(np.array(distances))]
        print('vertex_ind =' + str(vertex_ind))
        vertex_co = vertices_co[np.argmin(np.array(distances))] * obj.matrix_world
        return closest_mesh_name, vertex_ind, vertex_co

    @staticmethod
    def create_empty_in_vertex_location(self, vertex_location):
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
            print(ii)
            frame_str = str(ii)
            self.insert_keyframe_to_custom_prop(self, obj, 'data', float(data[ii]), ii + 1)
            # insert_keyframe_to_custom_prop(obj,'data',0,ii+1)
        fcurves = bpy.data.objects[empty_name].animation_data.action.fcurves[0]
        mod = fcurves.modifiers.new(type='LIMITS')

    def invoke(self, context, event=None):
        # Noam: is was self.find_vertex_index_and_mesh_closest_to_cursor(self) before, are you sure we need to send the self?
        closest_mesh_name, vertex_ind, vertex_co = self.find_vertex_index_and_mesh_closest_to_cursor(self)
        print(vertex_co)
        self.create_empty_in_vertex_location(self, vertex_co)
        # data_path = '/homes/5/npeled/space3/MEG/ECR/mg79'
        data_path = mu.get_user_fol() # bpy.path.abspath(bpy.context.scene.conf_path)
        # keyframe_empty('Activity_in_vertex',closest_mesh_name,vertex_ind,data_path)
        self.keyframe_empty_test('Activity_in_vertex', closest_mesh_name, vertex_ind, data_path)
        return {"FINISHED"}


class DataInVertMakerPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
    bl_label = "Data in vertex"
    addon = None

    def draw(self, context):
        layout = self.layout
        layout.operator("ohad.vertex_data_create", text="Get data in vertex", icon='ROTATE')
        layout.operator("ohad.vertex_data_clear", text="Clear", icon='PANEL_CLOSE')


def init(addon):
    DataInVertMakerPanel.addon = addon
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