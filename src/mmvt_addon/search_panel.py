import bpy
import os.path as op
import mmvt_utils as mu

def _addon():
    return SearchPanel.addon


class SearchFilter(bpy.types.Operator):
    bl_idname = "mmvt.selection_filter"
    bl_label = "selection filter"
    bl_description = 'Search'
    bl_options = {"UNDO"}
    marked_objects_select = {}
    marked_objects = []

    def invoke(self, context, event=None):
        label_name = context.scene.labels_regex
        SearchMark.marked_objects_select = {}
        objects = mu.get_non_functional_objects()
        SearchPanel.marked_objects = []
        for obj in objects:
            SearchFilter.marked_objects_select[obj.name] = obj.select
            obj.select = label_name in obj.name
            try:
                import fnmatch
                if fnmatch.fnmatch(obj.name, '*{}*'.format(label_name)):
                    SearchPanel.marked_objects.append(obj.name)
            except:
                if label_name in obj.name:
                    SearchPanel.marked_objects.append(obj.name)
        # todo: show rois only if the object is an ROI. Also, move the cursor
        SearchPanel.addon.show_rois()
        return {"FINISHED"}


class SearchClear(bpy.types.Operator):
    bl_idname = "mmvt.selection_clear"
    bl_label = "selection clear"
    bl_description = 'Clear'
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        # Copy from where am I clear
        _addon.selection.clear_selection()
        for obj_name, h in SearchMark.marked_objects_hide.items():
            bpy.data.objects[obj_name].hide = bool(h)
        for obj_name, h in SearchFilter.marked_objects_select.items():
            # print('bpy.data.objects[{}].select = {}'.format(obj_name, bool(h)))
            bpy.data.objects[obj_name].select = bool(h)

        SearchPanel.marked_objects = []
        return {"FINISHED"}


class SearchMark(bpy.types.Operator):
    bl_idname = "mmvt.selection_mark"
    bl_label = "selection mark"
    bl_options = {"UNDO"}
    marked_objects_hide = {}
    marked_objects = []

    def invoke(self, context, event=None):
        label_name = context.scene.labels_regex
        SearchMark.marked_objects_hide = {}
        objects = mu.get_non_functional_objects()
        SearchPanel.marked_objects = []
        for obj in objects:
            is_valid = False
            try:
                import fnmatch
                if fnmatch.fnmatch(obj.name, '*{}*'.format(label_name)):
                    is_valid = True
            except:
                if label_name in obj.name:
                    is_valid = True
            if is_valid:
                bpy.context.scene.objects.active = bpy.data.objects[obj.name]
                bpy.data.objects[obj.name].select = True
                SearchMark.marked_objects_hide[obj.name] = bpy.data.objects[obj.name].hide
                bpy.data.objects[obj.name].hide = False
                # todo: mark the object in a different way
                # bpy.data.objects[obj.name].active_material = bpy.data.materials['selected_label_Mat']
                SearchPanel.marked_objects.append(obj.name)
                # bpy.data.objects['inflated_'+obj.name].select = True
        SearchPanel.addon.show_rois()
        return {"FINISHED"}


class SearchExport(bpy.types.Operator):
    bl_idname = "mmvt.search_export"
    bl_label = "selection export"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        with open(op.join(mu.get_user_fol(), 'search_panel.txt'), 'w') as text_file:
            for obj_name in SearchPanel.marked_objects:
                print(obj_name, file=text_file)
        print('List was saved to {}'.format(op.join(mu.get_user_fol(), 'search_panel.txt')))
        return {"FINISHED"}


bpy.types.Scene.labels_regex = bpy.props.StringProperty(default= '', description='Searches all objects types in MMVT')


class SearchPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "Search Panel"
    addon = None
    init = False
    marked_objects = []

    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene, "labels_regex", text="Keyword")
        # row = layout.row(align=0)
        layout.operator(SearchFilter.bl_idname, text="Search", icon = 'BORDERMOVE')
        # row.operator(SearchMark.bl_idname, text="Mark", icon = 'GREASEPENCIL')
        if len(SearchPanel.marked_objects) > 0:
            box = layout.box()
            col = box.column()
            for obj_name in SearchPanel.marked_objects:
                mu.add_box_line(col, obj_name, percentage=1)
        layout.operator(SearchClear.bl_idname, text="Clear", icon='PANEL_CLOSE')
            # row = layout.row(align=0)
            # row.operator(SearchExport.bl_idname, text="Export list", icon = 'FORCE_LENNARDJONES')


def init(addon):
    SearchPanel.addon = addon
    bpy.context.scene.labels_regex = ''
    SearchPanel.init = True
    register()


def register():
    try:
        unregister()
        bpy.utils.register_class(SearchPanel)
        bpy.utils.register_class(SearchMark)
        bpy.utils.register_class(SearchClear)
        bpy.utils.register_class(SearchFilter)
        bpy.utils.register_class(SearchExport)
        # print('Search Panel was registered!')
    except:
        print("Can't register Search Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(SearchPanel)
        bpy.utils.unregister_class(SearchMark)
        bpy.utils.unregister_class(SearchClear)
        bpy.utils.unregister_class(SearchFilter)
        bpy.utils.unregister_class(SearchExport)
    except:
        pass
        # print("Can't unregister Search Panel!")

