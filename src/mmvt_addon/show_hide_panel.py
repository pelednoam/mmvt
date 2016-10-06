import bpy
import mmvt_utils as mu

def show_hide_hierarchy(do_hide, obj_name):
    if bpy.data.objects.get(obj_name) is not None:
        obj = bpy.data.objects[obj_name]
        hide_obj(obj, do_hide)
        # bpy.data.objects[obj].hide = do_hide
        for child in obj.children:
            hide_obj(child, do_hide)


def show_hide_hemi(val, hemi):
    show_hide_hierarchy(val, 'Cortex-{}'.format(hemi))
    show_hide_hierarchy(val, 'Cortex-inflated-{}'.format(hemi))
    for obj_name in [hemi, 'inflated_{}'.format(hemi)]:
        if bpy.data.objects.get(obj_name) is not None:
            hide_obj(bpy.data.objects[obj_name], val)


def show_hemis():
    for obj_name in ['rh', 'lh', 'Cortex-rh', 'Cortex-lh']:
        show_hide_hierarchy(False, obj_name)


def hide_obj(obj, val=True):
    obj.hide = val
    obj.hide_render = val


def show_hide_sub_cortical_update(self, context):
    show_hide_sub_corticals(bpy.context.scene.objects_show_hide_sub_cortical)


def show_hide_sub_corticals(do_hide):
    show_hide_hierarchy(do_hide, "Subcortical_structures")
    # show_hide_hierarchy(bpy.context.scene.objects_show_hide_sub_cortical, "Subcortical_activity_map")
    # We split the activity map into two types: meg for the same activation for the each structure, and fmri
    # for a better resolution, like on the cortex.
    if not do_hide:
        fmri_show = bpy.context.scene.subcortical_layer == 'fmri'
        meg_show = bpy.context.scene.subcortical_layer == 'meg'
        show_hide_hierarchy(not fmri_show, "Subcortical_fmri_activity_map")
        show_hide_hierarchy(not meg_show, "Subcortical_meg_activity_map")
    else:
        show_hide_hierarchy(True, "Subcortical_fmri_activity_map")
        show_hide_hierarchy(True, "Subcortical_meg_activity_map")


class ShowHideLH(bpy.types.Operator):
    bl_idname = "mmvt.show_hide_lh"
    bl_label = "mmvt show_hide_lh"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        bpy.context.scene.objects_show_hide_lh = not bpy.context.scene.objects_show_hide_lh
        show_hide_hemi(bpy.context.scene.objects_show_hide_lh, 'lh')
        return {"FINISHED"}


class ShowHideRH(bpy.types.Operator):
    bl_idname = "mmvt.show_hide_rh"
    bl_label = "mmvt show_hide_rh"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        bpy.context.scene.objects_show_hide_rh = not bpy.context.scene.objects_show_hide_rh
        show_hide_hemi(bpy.context.scene.objects_show_hide_rh, 'rh')
        return {"FINISHED"}


class ShowHideSubCorticals(bpy.types.Operator):
    bl_idname = "mmvt.show_hide_sub"
    bl_label = "mmvt show_hide_sub"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        bpy.context.scene.objects_show_hide_sub_cortical = not bpy.context.scene.objects_show_hide_sub_cortical
        show_hide_sub_corticals(bpy.context.scene.objects_show_hide_sub_cortical)
        return {"FINISHED"}


class ShowHideObjectsPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "Show Hide Brain"
    addon = None
    init = False

    def draw(self, context):
        layout = self.layout
        vis = dict(Right = not bpy.context.scene.objects_show_hide_rh, Left = not bpy.context.scene.objects_show_hide_lh)
        show_hide_icon = dict(show='RESTRICT_VIEW_OFF', hide='RESTRICT_VIEW_ON')
        row = layout.row(align=True)
        for hemi in ['Left', 'Right']:
            action = 'show' if vis[hemi] else 'hide'
            show_text = '{} {}'.format('Hide' if vis[hemi] else 'Show', hemi)
            show_icon = show_hide_icon[action]
            bl_idname = ShowHideLH.bl_idname if hemi == 'Left' else ShowHideRH.bl_idname
            row.operator(bl_idname, text=show_text, icon=show_icon)
        sub_vis = not bpy.context.scene.objects_show_hide_sub_cortical
        sub_show_text = '{} Subcortical'.format('Hide' if sub_vis else 'Show')
        sub_icon = show_hide_icon['show' if sub_vis else 'hide']
        layout.operator(ShowHideSubCorticals.bl_idname, text=sub_show_text, icon=sub_icon)


bpy.types.Scene.objects_show_hide_lh = bpy.props.BoolProperty(
    default=True, description="Show left hemisphere")#,update=show_hide_lh)
bpy.types.Scene.objects_show_hide_rh = bpy.props.BoolProperty(
    default=True, description="Show right hemisphere")#, update=show_hide_rh)
bpy.types.Scene.objects_show_hide_sub_cortical = bpy.props.BoolProperty(
    default=True, description="Show sub cortical")#, update=show_hide_sub_cortical_update)



def init(addon):
    ShowHideObjectsPanel.addon = addon
    bpy.context.scene.objects_show_hide_rh = False
    bpy.context.scene.objects_show_hide_lh = False
    bpy.context.scene.objects_show_hide_sub_cortical = False
    show_hide_sub_corticals(False)
    show_hemis()


    # show_hide_hemi(False, 'rh')
    # show_hide_hemi(False, 'lh')
    # hide_obj(bpy.data.objects[obj_func_name], val)

    register()
    ShowHideObjectsPanel.init = True


def register():
    try:
        unregister()
        bpy.utils.register_class(ShowHideObjectsPanel)
        bpy.utils.register_class(ShowHideLH)
        bpy.utils.register_class(ShowHideRH)
        bpy.utils.register_class(ShowHideSubCorticals)
        # print('Show Hide Panel was registered!')
    except:
        print("Can't register Show Hide Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(ShowHideObjectsPanel)
        bpy.utils.unregister_class(ShowHideLH)
        bpy.utils.unregister_class(ShowHideRH)
        bpy.utils.unregister_class(ShowHideSubCorticals)
    except:
        pass
        # print("Can't unregister Freeview Panel!")
