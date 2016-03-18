import bpy
import mathutils
import numpy as np

bpy.types.Scene.where_am_i_str = ''


def where_i_am_draw(self, context):
    layout = self.layout
    layout.operator("ohad.where_i_am", text="Where Am I?", icon='SNAP_SURFACE')
    layout.operator("ohad.where_am_i_clear", text="Clear", icon='PANEL_CLOSE')
    layout.label(text=bpy.types.Scene.where_am_i_str)


class WhereAmI(bpy.types.Operator):
    bl_idname = "ohad.where_i_am"
    bl_label = "ohad where i am"
    bl_options = {"UNDO"}

    where_am_I_selected_obj = None
    where_am_I_selected_obj_org_hide = True

    @staticmethod
    def setup_environment(self):
        WhereAmIMakerPanel.addon.set_appearance_show_rois_layer(bpy.context.scene, True)

    @staticmethod
    def main_func(self):
        distances = []
        names = []

        bpy.data.objects['Brain'].select = False
        for subHierarchy in bpy.data.objects['Brain'].children:
            if subHierarchy == bpy.data.objects['Subcortical_structures']:
                cur_material = bpy.data.materials['unselected_label_Mat_subcortical']
            else:
                cur_material = bpy.data.materials['unselected_label_Mat_cortex']
            for obj in subHierarchy.children:
                obj.active_material = cur_material
                obj.select = False
                obj.hide = subHierarchy.hide

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

                # Find the closest 10 points to the 3d cursor
                # print("Close 1 points")
                for (co, index, dist) in kd.find_n(co_find, 1):
                    # print("    ", obj.name,co, index, dist)
                    if 'unknown' not in obj.name:
                        distances.append(dist)
                        names.append(obj.name)

        # print(np.argmin(np.array(distances)))
        min_index = np.argmin(np.array(distances))
        closest_area = names[np.argmin(np.array(distances))]
        bpy.types.Scene.where_am_i_str = closest_area

        print('closest area is: '+closest_area)
        print('dist: {}'.format(np.min(np.array(distances))))
        print('closets vert is {}'.format(bpy.data.objects[closest_area].data.vertices[min_index].co))
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
    bl_idname = "ohad.where_am_i_clear"
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
        where_i_am_draw(self, context)
        return {"FINISHED"}


bpy.types.Scene.where_am_i = bpy.props.StringProperty(description="Find closest curve to cursor",
                                                      update=where_i_am_draw)


class WhereAmIMakerPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
    bl_label = "Where Am I"
    addon = None

    def draw(self, context):
        where_i_am_draw(self, context)


def init(addon):
    WhereAmIMakerPanel.addon = addon
    register()


def register():
    try:
        unregister()
        bpy.utils.register_class(WhereAmIMakerPanel)
        bpy.utils.register_class(WhereAmI)
        bpy.utils.register_class(ClearWhereAmI)
        print('Where am I Panel was registered!')
    except:
        print("Can't register Where am I Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(WhereAmIMakerPanel)
        bpy.utils.unregister_class(WhereAmI)
        bpy.utils.unregister_class(ClearWhereAmI)
    except:
        print("Can't unregister Where am I Panel!")
