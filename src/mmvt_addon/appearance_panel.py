import bpy
import connections_panel
import electrodes_panel
import mmvt_utils as mu
# from MMVT_Addon import (CONNECTIONS_LAYER, ELECTRODES_LAYER, ROIS_LAYER, ACTIVITY_LAYER, LIGHTS_LAYER,
#         BRAIN_EMPTY_LAYER, EMPTY_LAYER)

(CONNECTIONS_LAYER, ELECTRODES_LAYER, ROIS_LAYER, ACTIVITY_LAYER, LIGHTS_LAYER,
    BRAIN_EMPTY_LAYER, EMPTY_LAYER) = 3, 1, 10, 11, 12, 5, 14


def setup_layers():
    for layer_ind in range(len(bpy.context.scene.layers)):
        bpy.context.scene.layers[layer_ind] = layer_ind == EMPTY_LAYER

    bpy.context.scene.layers[ELECTRODES_LAYER] = bpy.context.scene.appearance_show_electrodes_layer
    bpy.context.scene.layers[ROIS_LAYER] = bpy.context.scene.appearance_show_rois_activity == 'rois'
    bpy.context.scene.layers[ACTIVITY_LAYER] = bpy.context.scene.appearance_show_rois_activity == 'activity'
    bpy.context.scene.layers[CONNECTIONS_LAYER] = bpy.context.scene.appearance_show_connections_layer


def change_view3d():
    viewport_shade = bpy.context.scene.filter_view_type
    if viewport_shade == 'rendered':
        bpy.context.scene.layers[LIGHTS_LAYER] = True
        viewport_shade_str = 'RENDERED'
    else:
        bpy.context.scene.layers[LIGHTS_LAYER] = False
        viewport_shade_str = 'SOLID'

    for ii in range(len(bpy.context.screen.areas)):
        if bpy.context.screen.areas[ii].type == 'VIEW_3D':
            bpy.context.screen.areas[ii].spaces[0].viewport_shade = viewport_shade_str
            break


def show_hide_electrodes(value):
    bpy.context.scene.layers[ELECTRODES_LAYER] = value


def show_rois():
    bpy.context.scene.appearance_show_rois_activity = 'rois'


def show_activity():
    bpy.context.scene.appearance_show_rois_activity = 'activity'


def show_electrodes(value=True):
    if not bpy.data.objects.get('Deep_electrodes', None) is None:
        # bpy.context.scene.appearance_show_electrodes_layer = value
        show_hide_electrodes(value)
        for elec_obj in bpy.data.objects.get('Deep_electrodes').children:
            elec_obj.hide = value


def appearance_show_rois_activity_update(self, context):
    show_activity = bpy.context.scene.appearance_show_rois_activity == 'activity'
    show_rois = bpy.context.scene.appearance_show_rois_activity == 'rois'
    bpy.context.scene.layers[ROIS_LAYER] = show_rois
    bpy.context.scene.layers[ACTIVITY_LAYER] = show_activity
    if not AppearanceMakerPanel.addon is None and show_activity:
        fmri_hide = not show_activity if bpy.context.scene.subcortical_layer == 'fmri' else show_activity
        meg_hide = not show_activity if bpy.context.scene.subcortical_layer == 'meg' else show_activity
        if not bpy.context.scene.objects_show_hide_sub_cortical:
            AppearanceMakerPanel.addon.show_hide_hierarchy(do_hide=fmri_hide, obj="Subcortical_fmri_activity_map")
            AppearanceMakerPanel.addon.show_hide_hierarchy(do_hide=meg_hide, obj="Subcortical_meg_activity_map")


def show_hide_connections(value):
    bpy.context.scene.layers[CONNECTIONS_LAYER] = value
    # if bpy.data.objects.get(connections_panel.PARENT_OBJ):
    #     bpy.data.objects.get(connections_panel.PARENT_OBJ).select = \
    #         bpy.context.scene.layers[CONNECTIONS_LAYER] == value


# def show_connections(value=True):
#     bpy.context.scene.appearance_show_connections_layer = value


def connections_visible():
    return bpy.data.objects.get(connections_panel.PARENT_OBJ) and bpy.context.scene.layers[CONNECTIONS_LAYER]


def filter_view_type_update(self, context):
    change_view3d()


def change_to_rendered_brain():
    bpy.context.scene.filter_view_type = 'rendered'


def change_to_solid_brain():
    bpy.context.scene.filter_view_type = 'solid'


def make_brain_solid_or_transparent():
    bpy.data.materials['Activity_map_mat'].node_tree.nodes['transparency_node'].inputs[
        'Fac'].default_value = bpy.context.scene.appearance_solid_slider
    if 'subcortical_activity_mat' in bpy.data.materials:
        subcortical_mat = bpy.data.materials['subcortical_activity_mat']
        subcortical_mat.node_tree.nodes['transparency_node'].inputs['Fac'].default_value = \
            bpy.context.scene.appearance_solid_slider


def update_layers():
    # depth = bpy.context.scene.appearance_depth_slider if bpy.context.scene.appearance_depth_Bool else 0
    depth = bpy.context.scene.appearance_depth_slider
    bpy.data.materials['Activity_map_mat'].node_tree.nodes["layers_depth"].inputs[1].default_value = depth


def appearance_draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'appearance_show_rois_activity', expand=True)
    layout.prop(context.scene, "filter_view_type", expand=True)
    if bpy.data.objects.get(electrodes_panel.PARENT_OBJ):
        show_hide_icon(layout, ShowHideElectrodes.bl_idname, bpy.context.scene.show_hide_electrodes, 'Electrodes')
        # layout.prop(context.scene, 'appearance_show_electrodes_layer', text="Show electrodes", icon='RESTRICT_VIEW_OFF')
    if bpy.data.objects.get(connections_panel.PARENT_OBJ):
        show_hide_icon(layout, ShowHideConnections.bl_idname, bpy.context.scene.show_hide_connections, 'Connections')
        # layout.prop(context.scene, 'appearance_show_connections_layer', text="Show connections", icon='RESTRICT_VIEW_OFF')


def show_hide_icon(layout, bl_idname, show_hide_var, var_name):
    vis = not show_hide_var
    show_text = '{} {}'.format('Show' if vis else 'Hide', var_name)
    icon = mu.show_hide_icon['show' if vis else 'hide']
    layout.operator(bl_idname, text=show_text, icon=icon)


def update_solidity(self, context):
    make_brain_solid_or_transparent()
    update_layers()


bpy.types.Scene.appearance_show_rois_activity = bpy.props.EnumProperty(
    items=[("activity", "Activity maps", "", 0), ("rois", "ROIs", "", 1)],description="",
    update=appearance_show_rois_activity_update)
# bpy.types.Scene.appearance_show_connections_layer = bpy.props.BoolProperty(
#     default=False, description="Show connectivity", update=appearance_show_connections_layer_update)
# bpy.types.Scene.appearance_show_electrodes_layer = bpy.props.BoolProperty(
#     default=False, description="Show electrodes", update=appearance_show_electrodes_layer_update)
bpy.types.Scene.subcortical_layer = bpy.props.StringProperty(description="subcortical layer")

bpy.types.Scene.filter_view_type = bpy.props.EnumProperty(
    items=[("rendered", "Rendered Brain", "", 1), ("solid", "Solid Brain", "", 2)],description="Brain appearance",
    update = filter_view_type_update)
bpy.types.Scene.show_hide_electrodes = bpy.props.BoolProperty(
    default=False, description="Show electrodes")

bpy.types.Scene.show_hide_connections = bpy.props.BoolProperty(
    default=False, description="Show connections")

class ShowHideElectrodes(bpy.types.Operator):
    bl_idname = "ohad.show_hide_elctrodes"
    bl_label = "ohad show_hide_electrodes"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        bpy.context.scene.show_hide_electrodes = not bpy.context.scene.show_hide_electrodes
        show_hide_electrodes(bpy.context.scene.show_hide_electrodes)
        return {"FINISHED"}


class ShowHideConnections(bpy.types.Operator):
    bl_idname = "ohad.show_hide_connections"
    bl_label = "ohad show_hide_connections"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        bpy.context.scene.show_hide_connections = not bpy.context.scene.show_hide_connections
        show_hide_connections(bpy.context.scene.show_hide_connections)
        return {"FINISHED"}


class AppearanceMakerPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
    bl_label = "Appearance"
    addon = None
    init = False

    def draw(self, context):
        if AppearanceMakerPanel.init:
            appearance_draw(self, context)


def init(addon):
    AppearanceMakerPanel.addon = addon
    register()
    AppearanceMakerPanel.init = True
    bpy.context.scene.subcortical_layer = 'fmri'
    # bpy.context.scene.filter_view_type = 'solid' # 'rendered'
    change_to_solid_brain()
    # bpy.context.scene.appearance_show_rois_activity = 'rois' # 'activity'
    show_rois()


def register():
    try:
        unregister()
        bpy.utils.register_class(AppearanceMakerPanel)
        bpy.utils.register_class(ShowHideElectrodes)
        bpy.utils.register_class(ShowHideConnections)
    except:
        print("Can't register Appearance Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(AppearanceMakerPanel)
        bpy.utils.unregister_class(ShowHideElectrodes)
        bpy.utils.unregister_class(ShowHideConnections)
    except:
        pass

