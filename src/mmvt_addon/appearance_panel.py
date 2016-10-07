import bpy
import connections_panel
import electrodes_panel
import mmvt_utils as mu
import time


def _addon():
    return AppearanceMakerPanel.addon


def setup_layers():
    for layer_ind in range(len(bpy.context.scene.layers)):
        bpy.context.scene.layers[layer_ind] = layer_ind == _addon().EMPTY_LAYER

    bpy.context.scene.layers[_addon().ELECTRODES_LAYER] = bpy.context.scene.appearance_show_electrodes_layer
    bpy.context.scene.layers[_addon().ROIS_LAYER] = is_rois()
    bpy.context.scene.layers[_addon().ACTIVITY_LAYER] = is_activity()
    bpy.context.scene.layers[_addon().CONNECTIONS_LAYER] = bpy.context.scene.appearance_show_connections_layer


def change_view3d():
    viewport_shade = bpy.context.scene.filter_view_type
    if viewport_shade == 'rendered':
        bpy.context.scene.layers[_addon().LIGHTS_LAYER] = True
        viewport_shade_str = 'RENDERED'
        bpy.context.scene.render.engine = 'CYCLES'
    else:
        bpy.context.scene.layers[_addon().LIGHTS_LAYER] = False
        viewport_shade_str = 'SOLID'
        bpy.context.scene.render.engine = 'BLENDER_RENDER'

    for ii in range(len(bpy.context.screen.areas)):
        if bpy.context.screen.areas[ii].type == 'VIEW_3D':
            bpy.context.screen.areas[ii].spaces[0].viewport_shade = viewport_shade_str
            break


def show_hide_electrodes(do_show):
    bpy.context.scene.layers[_addon().ELECTRODES_LAYER] = do_show
    if do_show:
        bpy.context.scene.show_only_lead = False


def show_rois():
    bpy.context.scene.appearance_show_rois_activity = 'rois'


def show_activity():
    bpy.context.scene.appearance_show_rois_activity = 'activity'


def show_pial():
    bpy.context.scene.surface_type = 'pial'


def show_inflated():
    bpy.context.scene.surface_type == 'inflated'


def show_electrodes(value=True):
    show_hide_electrodes(value)


def is_pial():
    return bpy.context.scene.surface_type == 'pial'


def is_inflated():
    return bpy.context.scene.surface_type == 'inflated'


def is_activity():
    return bpy.context.scene.appearance_show_rois_activity == 'activity'


def is_rois():
    return bpy.context.scene.appearance_show_rois_activity == 'rois'


def appearance_show_rois_activity_update(self, context):
    # todo: Figure out why the hell
    for _ in range(2):
        if bpy.context.scene.surface_type == 'pial':
            bpy.context.scene.layers[_addon().ROIS_LAYER] = is_rois()
            bpy.context.scene.layers[_addon().ACTIVITY_LAYER] = is_activity()
        elif bpy.context.scene.surface_type == 'inflated':
            bpy.context.scene.layers[_addon().INFLATED_ROIS_LAYER] = is_rois()
            bpy.context.scene.layers[_addon().INFLATED_ACTIVITY_LAYER] = is_activity()
    # print('roi: {}, activity: {}'.format(bpy.context.scene.layers[ROIS_LAYER], bpy.context.scene.layers[ACTIVITY_LAYER]))
    # print('should be {}, {}'.format(is_rois(), is_activity()))
    if bpy.context.scene.layers[_addon().ROIS_LAYER] != is_rois() or \
                    bpy.context.scene.layers[_addon().ACTIVITY_LAYER] != is_activity():
        print('Error in displaying the layers!')
    if not AppearanceMakerPanel.addon is None and is_activity():
        fmri_hide = not is_activity() if bpy.context.scene.subcortical_layer == 'fmri' else is_activity()
        meg_hide = not is_activity() if bpy.context.scene.subcortical_layer == 'meg' else is_activity()
        if not bpy.context.scene.objects_show_hide_sub_cortical:
            AppearanceMakerPanel.addon.show_hide_hierarchy(do_hide=fmri_hide, obj_name="Subcortical_fmri_activity_map")
            AppearanceMakerPanel.addon.show_hide_hierarchy(do_hide=meg_hide, obj_name="Subcortical_meg_activity_map")


def show_hide_connections(value):
    bpy.context.scene.layers[_addon().CONNECTIONS_LAYER] = value
    # if bpy.data.objects.get(connections_panel.PARENT_OBJ):
    #     bpy.data.objects.get(connections_panel.PARENT_OBJ).select = \
    #         bpy.context.scene.layers[CONNECTIONS_LAYER] == value


# def show_connections(value=True):
#     bpy.context.scene.appearance_show_connections_layer = value


def connections_visible():
    return bpy.data.objects.get(connections_panel.PARENT_OBJ) and bpy.context.scene.layers[_addon().CONNECTIONS_LAYER]


def filter_view_type_update(self, context):
    change_view3d()


def surface_type_update(self, context):
    inflated = bpy.context.scene.surface_type == 'inflated'
    for _ in range(2):
        if is_rois():
            bpy.context.scene.layers[_addon().INFLATED_ROIS_LAYER] = inflated
            bpy.context.scene.layers[_addon().ROIS_LAYER] = not inflated
        elif is_activity():
            bpy.context.scene.layers[_addon().INFLATED_ACTIVITY_LAYER] = inflated
            bpy.context.scene.layers[_addon().ACTIVITY_LAYER] = not inflated


def show_pial():
    bpy.context.scene.surface_type = 'pial'


def show_inflated():
    bpy.context.scene.surface_type = 'inflated'


def change_to_rendered_brain():
    bpy.context.scene.filter_view_type = 'rendered'
    bpy.context.scene.render.engine = 'CYCLES'


def change_to_solid_brain():
    bpy.context.scene.filter_view_type = 'solid'
    bpy.context.scene.render.engine = 'BLENDER_RENDER'


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
    layout.prop(context.scene, "surface_type", expand=True)
    # layout.operator(SelectionListener.bl_idname, text="", icon='PREV_KEYFRAME')
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


class SelectionListener(bpy.types.Operator):
    bl_idname = 'mmvt.selection_listener'
    bl_label = 'selection_listener'
    bl_options = {'UNDO'}
    press_time = time.time()
    running = False
    right_clicked = False

    def modal(self, context, event):
        def show_fcurves(obj):
            mu.change_fcurves_colors(obj)
            mu.view_all_in_graph_editor()

        if self.right_clicked:
            if len(bpy.context.selected_objects):
                selected_obj_name = bpy.context.selected_objects[0].name
                selected_obj_type = mu.check_obj_type(selected_obj_name)
                if selected_obj_type in [mu.OBJ_TYPE_CORTEX_LH, mu.OBJ_TYPE_CORTEX_RH]:
                    pial_obj = bpy.data.objects.get(selected_obj_name)
                    show_fcurves(pial_obj)
                if selected_obj_type in [mu.OBJ_TYPE_CORTEX_INFLATED_LH, mu.OBJ_TYPE_CORTEX_INFLATED_RH]:
                    pial_obj_name = selected_obj_name[len('inflated_'):]
                    pial_obj = bpy.data.objects.get(pial_obj_name)
                    if not pial_obj is None:
                        pial_obj.select = True
                        mu.change_fcurves_colors(pial_obj)
                        mu.view_all_in_graph_editor()
            self.right_clicked = False

        if time.time() - self.press_time > 1 and event.type == 'RIGHTMOUSE':
            self.press_time = time.time()
            self.right_clicked = True
        return {'PASS_THROUGH'}

    def invoke(self, context, event=None):
        return {'RUNNING_MODAL'}

    def execute(self, context):
        if not self.running:
            context.window_manager.modal_handler_add(self)
            self.running = True
        return {'RUNNING_MODAL'}


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

bpy.types.Scene.surface_type = bpy.props.EnumProperty(
    items=[("pial", "Pial", "", 1), ("inflated", "Inflated", "", 2)],description="Surface type",
    update = surface_type_update)

bpy.types.Scene.show_hide_electrodes = bpy.props.BoolProperty(
    default=False, description="Show electrodes")

bpy.types.Scene.show_hide_connections = bpy.props.BoolProperty(
    default=False, description="Show connections")


class ShowHideElectrodes(bpy.types.Operator):
    bl_idname = "mmvt.show_hide_elctrodes"
    bl_label = "mmvt show_hide_electrodes"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        bpy.context.scene.show_hide_electrodes = not bpy.context.scene.show_hide_electrodes
        show_hide_electrodes(bpy.context.scene.show_hide_electrodes)
        return {"FINISHED"}


class ShowHideConnections(bpy.types.Operator):
    bl_idname = "mmvt.show_hide_connections"
    bl_label = "mmvt show_hide_connections"
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
    bl_category = "mmvt"
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
    bpy.ops.mmvt.selection_listener()



def register():
    try:
        unregister()
        bpy.utils.register_class(AppearanceMakerPanel)
        bpy.utils.register_class(ShowHideElectrodes)
        bpy.utils.register_class(ShowHideConnections)
        bpy.utils.register_class(SelectionListener)
    except:
        print("Can't register Appearance Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(AppearanceMakerPanel)
        bpy.utils.unregister_class(ShowHideElectrodes)
        bpy.utils.unregister_class(ShowHideConnections)
        bpy.utils.unregister_class(SelectionListener)
    except:
        pass

