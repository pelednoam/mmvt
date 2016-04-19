import bpy

def show_hide_hierarchy(do_hide, obj):
    if bpy.data.objects.get(obj) is not None:
        bpy.data.objects[obj].hide = do_hide
        for child in bpy.data.objects[obj].children:
            child.hide = do_hide
            child.hide_render = do_hide


def show_hide_hemi(val, obj_func_name, obj_brain_name):
    if bpy.data.objects.get(obj_func_name) is not None:
        bpy.data.objects[obj_func_name].hide = val
        bpy.data.objects[obj_func_name].hide_render = val
    show_hide_hierarchy(val, obj_brain_name)


def show_hide_rh(self, context):
    show_hide_hemi(bpy.context.scene.objects_show_hide_rh, "rh", "Cortex-rh")


def show_hide_lh(self, context):
    show_hide_hemi(bpy.context.scene.objects_show_hide_lh, "lh", "Cortex-lh")


def show_hide_sub_cortical_update(self, context):
    show_hide_sub_corticals(bpy.context.scene.objects_show_hide_sub_cortical)


def show_hide_sub_corticals(do_hide):
    show_hide_hierarchy(do_hide, "Subcortical_structures")
    # show_hide_hierarchy(bpy.context.scene.objects_show_hide_sub_cortical, "Subcortical_activity_map")
    # We split the activity map into two types: meg for the same activation for the each structure, and fmri
    # for a better resolution, like on the cortex.
    # todo: might cause some problems in the future
    if not do_hide:
        show_hide_hierarchy(True, "Subcortical_fmri_activity_map")
        show_hide_hierarchy(True if not do_hide else False, "Subcortical_meg_activity_map")
    else:
        show_hide_hierarchy(True, "Subcortical_fmri_activity_map")
        show_hide_hierarchy(True, "Subcortical_meg_activity_map")

bpy.types.Scene.objects_show_hide_lh = bpy.props.BoolProperty(default=True, description="Show left hemisphere",
                                                              update=show_hide_lh)
bpy.types.Scene.objects_show_hide_rh = bpy.props.BoolProperty(default=True, description="Show right hemisphere",
                                                              update=show_hide_rh)
bpy.types.Scene.objects_show_hide_sub_cortical = bpy.props.BoolProperty(default=True, description="Show sub cortical",
                                                                        update=show_hide_sub_cortical_update)

class ShowHideObjectsPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
    bl_label = "Show Hide Objects"
    addon = None

    def draw(self, context):
        col1 = self.layout.column(align=True)
        col1.prop(context.scene, 'objects_show_hide_lh', text="Left Hemisphere", icon='RESTRICT_VIEW_OFF')
        col1.prop(context.scene, 'objects_show_hide_rh', text="Right Hemisphere", icon='RESTRICT_VIEW_OFF')
        col1.prop(context.scene, 'objects_show_hide_sub_cortical', text="Sub Cortical", icon='RESTRICT_VIEW_OFF')


def init(addon):
    ShowHideObjectsPanel.addon = addon
    register()


def register():
    try:
        unregister()
        bpy.utils.register_class(ShowHideObjectsPanel)
        print('Show Hide Panel was registered!')
    except:
        print("Can't register Show Hide Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(ShowHideObjectsPanel)
    except:
        pass
        # print("Can't unregister Freeview Panel!")
