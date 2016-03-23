import bpy
import connections_panel
# from MMVT_Addon import (CONNECTIONS_LAYER, ELECTRODES_LAYER, ROIS_LAYER, ACTIVITY_LAYER, LIGHTS_LAYER,
#         BRAIN_EMPTY_LAYER, EMPTY_LAYER)
(CONNECTIONS_LAYER, ELECTRODES_LAYER, ROIS_LAYER, ACTIVITY_LAYER, LIGHTS_LAYER,
    BRAIN_EMPTY_LAYER, EMPTY_LAYER) = 3, 1, 10, 11, 12, 5, 14


def setup_layers():
    empty_layer = EMPTY_LAYER

    for layer_ind in range(len(bpy.context.scene.layers)):
        bpy.context.scene.layers[layer_ind] = layer_ind == empty_layer

    bpy.context.scene.layers[ELECTRODES_LAYER] = bpy.context.scene.appearance_show_electrodes_layer
    bpy.context.scene.layers[ROIS_LAYER] = bpy.context.scene.appearance_show_ROIs_layer
    bpy.context.scene.layers[ACTIVITY_LAYER] = bpy.context.scene.appearance_show_activity_layer
    bpy.context.scene.layers[CONNECTIONS_LAYER] = bpy.context.scene.appearance_show_connections_layer


def change_view3d():
    viewport_shade = bpy.context.scene.filter_view_type
    # if viewport_shade == 'RENDERED':
    if viewport_shade == '1':
        bpy.context.scene.layers[LIGHTS_LAYER] = True
        viewport_shade_str = 'RENDERED'
    else:
        bpy.context.scene.layers[LIGHTS_LAYER] = False
        viewport_shade_str = 'SOLID'

    for ii in range(len(bpy.context.screen.areas)):
        if bpy.context.screen.areas[ii].type == 'VIEW_3D':
            bpy.context.screen.areas[ii].spaces[0].viewport_shade = viewport_shade_str


def get_appearance_show_electrodes_layer(self):
    return self['appearance_show_electrodes_layer']


def set_appearance_show_electrodes_layer(self, value):
    self['appearance_show_electrodes_layer'] = value
    bpy.context.scene.layers[ELECTRODES_LAYER] = value


def get_appearance_show_rois_layer(self):
    return self['appearance_show_ROIs_layer']


def set_appearance_show_rois_layer(self, value):
    self['appearance_show_ROIs_layer'] = value
    bpy.context.scene.layers[ROIS_LAYER] = value
    if value:
        set_appearance_show_activity_layer(self, False)
        # bpy.context.scene.layers[LIGHTS_LAYER] = False


def show_rois():
    if not get_appearance_show_rois_layer(bpy.context.scene):
        set_appearance_show_rois_layer(bpy.context.scene, True)


def show_activity():
    if not get_appearance_show_electrodes_layer(bpy.context.scene):
        set_appearance_show_activity_layer(bpy.context.scene, True)


def show_electrodes():
    if not get_appearance_show_electrodes_layer(bpy.context.scene):
        set_appearance_show_electrodes_layer(bpy.context.scene, True)


def get_appearance_show_activity_layer(self):
    return self['appearance_show_activity_layer']


def set_appearance_show_activity_layer(self, value):
    self['appearance_show_activity_layer'] = value
    bpy.context.scene.layers[ACTIVITY_LAYER] = value
    if value:
        set_appearance_show_rois_layer(self, False)
        # todo: decide which one to show
        if not AppearanceMakerPanel.addon is None:
            AppearanceMakerPanel.addon.show_hide_hierarchy(value, "Subcortical_fmri_activity_map")
            AppearanceMakerPanel.addon.show_hide_hierarchy(not value, "Subcortical_meg_activity_map")


def get_appearance_show_connections_layer(self):
    return self['appearance_show_connections_layer']


def set_appearance_show_connections_layer(self, value):
    if bpy.data.objects.get(connections_panel.PARENT_OBJ):
        self['appearance_show_connections_layer'] = value
        bpy.data.objects.get(connections_panel.PARENT_OBJ).select = value
        bpy.context.scene.layers[CONNECTIONS_LAYER] = value


def get_filter_view_type(self):
    # print('in get_filter_view_type')
    # print(self['filter_view_type'])
    # print(type(self['filter_view_type']))
    if self['filter_view_type'] == 'RENDERED':
        return 1
    elif self['filter_view_type'] == 'SOLID':
        return 2
    elif type(self['filter_view_type']) == int:
        return self['filter_view_type']
    return 3


def set_filter_view_type(self, value):
    # self['filter_view_type'] = value
    bpy.data.scenes['Scene']['filter_view_type'] = value
    change_view3d()


def change_to_rendered_brain():
    set_filter_view_type(None, 1)


def change_to_solid_brain():
    set_filter_view_type(None, 2)


def make_brain_solid_or_transparent():
    bpy.data.materials['Activity_map_mat'].node_tree.nodes['transparency_node'].inputs[
        'Fac'].default_value = bpy.context.scene.appearance_solid_slider
    if 'subcortical_activity_mat' in bpy.data.materials:
        subcortical_mat = bpy.data.materials['subcortical_activity_mat']
        subcortical_mat.node_tree.nodes['transparency_node'].inputs['Fac'].default_value = \
            bpy.context.scene.appearance_solid_slider


def update_layers():
    if bpy.context.scene.appearance_depth_Bool:
        bpy.data.materials['Activity_map_mat'].node_tree.nodes["layers_depth"].inputs[
            1].default_value = bpy.context.scene.appearance_depth_slider
    else:
        bpy.data.materials['Activity_map_mat'].node_tree.nodes["layers_depth"].inputs[1].default_value = 0


def appearance_draw(self, context):
    layout = self.layout
    col1 = self.layout.column(align=True)
    col1.prop(context.scene, 'appearance_show_ROIs_layer', text="Show ROIs", icon='RESTRICT_VIEW_OFF')
    col1.prop(context.scene, 'appearance_show_activity_layer', text="Show activity maps", icon='RESTRICT_VIEW_OFF')
    col1.prop(context.scene, 'appearance_show_electrodes_layer', text="Show electrodes", icon='RESTRICT_VIEW_OFF')
    if bpy.data.objects.get(connections_panel.PARENT_OBJ):
        col1.prop(context.scene, 'appearance_show_connections_layer', text="Show connections", icon='RESTRICT_VIEW_OFF')
    split = layout.split()
    split.prop(context.scene, "filter_view_type", text="")


def update_solidity(self, context):
    make_brain_solid_or_transparent()
    update_layers()
    AppearanceMakerPanel.draw()


bpy.types.Scene.appearance_show_electrodes_layer = bpy.props.BoolProperty(default=False, description="Show electrodes",
                                                                          get=get_appearance_show_electrodes_layer,
                                                                          set=set_appearance_show_electrodes_layer)
bpy.types.Scene.appearance_show_ROIs_layer = bpy.props.BoolProperty(default=True, description="Show ROIs",
                                                                    get=get_appearance_show_rois_layer,
                                                                    set=set_appearance_show_rois_layer)
bpy.types.Scene.appearance_show_activity_layer = bpy.props.BoolProperty(default=False, description="Show activity maps",
                                                                        get=get_appearance_show_activity_layer,
                                                                        set=set_appearance_show_activity_layer)
bpy.types.Scene.appearance_show_connections_layer = bpy.props.BoolProperty(default=False, description="Show connectivity",
                                                                        get=get_appearance_show_connections_layer,
                                                                        set=set_appearance_show_connections_layer)

bpy.types.Scene.filter_view_type = bpy.props.EnumProperty(
    items=[("1", "Rendered Brain", "", 1), ("2", " Solid Brain", "", 2)],description="Brain appearance",
    get=get_filter_view_type, set=set_filter_view_type, default='2')


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


def register():
    try:
        unregister()
        bpy.utils.register_class(AppearanceMakerPanel)
        print('Appearance Panel was registered!')
    except:
        print("Can't register Appearance Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(AppearanceMakerPanel)
    except:
        pass
        # print("Can't unregister Appearance Panel!")

