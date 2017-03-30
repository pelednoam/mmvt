import bpy
import connections_panel
import electrodes_panel
import mmvt_utils as mu
import time
import sys


def _addon():
    return AppearanceMakerPanel.addon


def setup_layers():
    for layer_ind in range(len(bpy.context.scene.layers)):
        bpy.context.scene.layers[layer_ind] = layer_ind == _addon().EMPTY_LAYER

    bpy.context.scene.layers[_addon().ELECTRODES_LAYER] = bpy.context.scene.show_hide_electrodes
    bpy.context.scene.layers[_addon().EEG_LAYER] = bpy.context.scene.show_hide_eeg
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


def show_hide_meg_sensors(do_show=True):
    bpy.context.scene.layers[_addon().MEG_LAYER] = do_show


def show_hide_eeg(do_show=True):
    bpy.context.scene.layers[_addon().EEG_LAYER] = do_show


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


def hemis_inf_distance_update(self, context):
    if AppearanceMakerPanel.init:
        if bpy.data.objects.get('Cortex-inflated-rh') and bpy.data.objects.get('inflated_rh'):
            bpy.data.objects['Cortex-inflated-rh'].location[0] = bpy.data.objects['inflated_rh'].location[0] = \
                AppearanceMakerPanel.cortex_inflated_rh + bpy.context.scene.hemis_inf_distance
            bpy.data.objects['Cortex-inflated-lh'].location[0] = bpy.data.objects['inflated_lh'].location[0] = \
                AppearanceMakerPanel.cortex_inflated_lh - bpy.context.scene.hemis_inf_distance


def hemis_distance_update(self, context):
    if AppearanceMakerPanel.init:
        if bpy.data.objects.get('Cortex-rh') and bpy.data.objects.get('rh'):
            bpy.data.objects['Cortex-rh'].location[0] = bpy.data.objects['rh'].location[0] = \
                AppearanceMakerPanel.cortex_rh + bpy.context.scene.hemis_distance
            bpy.data.objects['Cortex-lh'].location[0] = bpy.data.objects['lh'].location[0] = \
                AppearanceMakerPanel.cortex_lh - bpy.context.scene.hemis_distance


def inflating_update(self, context):
    try:
        bpy.data.shape_keys['Key'].key_blocks["inflated"].value = bpy.context.scene.inflating
        bpy.data.shape_keys['Key.001'].key_blocks["inflated"].value = bpy.context.scene.inflating
        bpy.context.scene.hemis_inf_distance = - (1 - bpy.context.scene.inflating) * 5
    except:
        print('Error in inflating update!')


def set_inflated_ratio(ratio):
    bpy.context.scene.inflating = ratio


def get_inflated_ratio():
    return bpy.context.scene.inflating


def appearance_show_rois_activity_update(self=None, context=None):
    # todo: Figure out why the hell
    for _ in range(2):
        if bpy.context.scene.surface_type == 'pial':
            bpy.context.scene.layers[_addon().ROIS_LAYER] = is_rois()
            bpy.context.scene.layers[_addon().ACTIVITY_LAYER] = is_activity()
        elif bpy.context.scene.surface_type == 'inflated':
            bpy.context.scene.layers[_addon().INFLATED_ROIS_LAYER] = is_rois()
            bpy.context.scene.layers[_addon().INFLATED_ACTIVITY_LAYER] = is_activity()
            # if is_activity():
            #     # bpy.context.scene.inflating = 0
            #     # bpy.context.scene.hemis_inf_distance = -5
            #     pass
            # else:
            #     bpy.context.scene.hemis_inf_distance = 0
            #     pass
    # print('roi: {}, activity: {}'.format(bpy.context.scene.layers[ROIS_LAYER], bpy.context.scene.layers[ACTIVITY_LAYER]))
    # print('should be {}, {}'.format(is_rois(), is_activity()))
    # if bpy.context.scene.layers[_addon().ROIS_LAYER] != is_rois() or \
    #                 bpy.context.scene.layers[_addon().ACTIVITY_LAYER] != is_activity():
    #     print('Error in displaying the layers!')
    if not _addon() is None and is_activity():
        fmri_hide = not is_activity() if bpy.context.scene.subcortical_layer == 'fmri' else is_activity()
        meg_hide = not is_activity() if bpy.context.scene.subcortical_layer == 'meg' else is_activity()
        if not bpy.context.scene.objects_show_hide_sub_cortical:
            _addon().show_hide_hierarchy(do_hide=fmri_hide, obj_name="Subcortical_fmri_activity_map")
            _addon().show_hide_hierarchy(do_hide=meg_hide, obj_name="Subcortical_meg_activity_map")


def show_hide_connections(value=True):
    bpy.context.scene.layers[_addon().CONNECTIONS_LAYER] = value
    if value and bpy.data.objects.get(_addon().get_connections_parent_name()):
        _addon().show_hide_hierarchy(False, _addon().get_connections_parent_name())
        # bpy.data.objects.get(_addon().get_connections_parent_name()).hide = False
        # bpy.data.objects.get(_addon().get_connections_parent_name()).hide_render = False


# def show_connections(value=True):
#     bpy.context.scene.appearance_show_connections_layer = value


def connections_visible():
    return bpy.data.objects.get(_addon().get_connections_parent_name()) and bpy.context.scene.layers[_addon().CONNECTIONS_LAYER]


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
    _addon().update_camera_files()


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


def is_solid():
    return bpy.context.scene.filter_view_type == 'solid'


def is_rendered():
    return bpy.context.scene.filter_view_type == 'rendered'


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
    if 'Key' in bpy.data.shape_keys and is_inflated() and is_activity():
        layout.prop(context.scene, 'inflating')
    if bpy.context.scene.surface_type == 'pial':
        layout.prop(context.scene, 'hemis_distance', text='hemis dist')
    else:
        if bpy.data.objects.get('Cortex-inflated-rh') and bpy.data.objects.get('Cortex-inflated-lh'):
            layout.prop(context.scene, 'hemis_inf_distance', text='hemis dist')
    # layout.operator(SelectionListener.bl_idname, text="", icon='PREV_KEYFRAME')
    if bpy.data.objects.get(electrodes_panel.PARENT_OBJ):
        show_hide_icon(layout, ShowHideElectrodes.bl_idname, bpy.context.scene.show_hide_electrodes, 'Electrodes')
    if bpy.data.objects.get('MEG_sensors'):
        show_hide_icon(layout, ShowHideMEGSensors.bl_idname, bpy.context.scene.show_hide_meg_sensors, 'MEG sensors')
    if bpy.data.objects.get('EEG_sensors'):
        show_hide_icon(layout, ShowHideEEG.bl_idname, bpy.context.scene.show_hide_eeg, 'EEG sensors')
    if bpy.data.objects.get(_addon().get_connections_parent_name()):
        show_hide_icon(layout, ShowHideConnections.bl_idname, bpy.context.scene.show_hide_connections, 'Connections')


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
        # def show_fcurves(obj):
        #     mu.change_fcurves_colors(obj)
            # mu.view_all_in_graph_editor()

        if self.right_clicked:
            if len(bpy.context.selected_objects):
                mu.unfilter_graph_editor()
                selected_obj_name = bpy.context.selected_objects[0].name
                selected_obj_type = mu.check_obj_type(selected_obj_name)
                if selected_obj_type in [mu.OBJ_TYPE_CORTEX_LH, mu.OBJ_TYPE_CORTEX_RH]:
                    _addon().select_roi(selected_obj_name)
                elif selected_obj_type in [mu.OBJ_TYPE_ELECTRODE, mu.OBJ_TYPE_EEG]:
                    bpy.data.objects[selected_obj_name].select = True
                elif selected_obj_type in [mu.OBJ_TYPE_CORTEX_INFLATED_LH, mu.OBJ_TYPE_CORTEX_INFLATED_RH]:
                    pial_obj_name = selected_obj_name[len('inflated_'):]
                    pial_obj = bpy.data.objects.get(pial_obj_name)
                    if not pial_obj is None:
                        # pial_obj.select = True
                        _addon().select_roi(pial_obj_name)
                        mu.change_fcurves_colors(pial_obj)
                elif selected_obj_type == mu.OBJ_TYPE_CON:
                    _addon().select_connection(selected_obj_name)
                elif selected_obj_type == mu.OBJ_TYPE_CON_VERTICE:
                    _addon().vertices_selected(selected_obj_name)
            self.right_clicked = False

        if time.time() - self.press_time > 1:
            if event.type == 'RIGHTMOUSE':
                self.press_time = time.time()
                self.right_clicked = True
            if event.type == 'LEFTMOUSE':
                if _addon().fMRI_clusters_files_exist() and bpy.context.scene.plot_fmri_cluster_per_click:
                    _addon().find_closest_cluster(only_within=True)
                _addon().set_tkreg_ras_coo(bpy.context.scene.cursor_location * 10)
                _addon().save_cursor_position()

        if _addon() and _addon().render_in_queue():
            rendering_data = mu.queue_get(_addon().render_in_queue())
            if not rendering_data is None:
                try:
                    rendering_data = rendering_data.decode(sys.getfilesystemencoding(), 'ignore')
                    if '*** finish rendering! ***' in rendering_data.lower():
                        print('Finish rendering!')
                        _addon().finish_rendering()
                except:
                    print("Can't read the stdout from the rendering")

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
bpy.types.Scene.subcortical_layer = bpy.props.StringProperty(description="subcortical layer")
bpy.types.Scene.filter_view_type = bpy.props.EnumProperty(
    items=[("rendered", "Rendered Brain", "", 1), ("solid", "Solid Brain", "", 2)],description="Brain appearance",
    update = filter_view_type_update)
bpy.types.Scene.surface_type = bpy.props.EnumProperty(
    items=[("pial", "Pial", "", 1), ("inflated", "Inflated", "", 2)],description="Surface type",
    update = surface_type_update)
bpy.types.Scene.show_hide_electrodes = bpy.props.BoolProperty(default=False)
bpy.types.Scene.show_hide_eeg = bpy.props.BoolProperty(default=False)
bpy.types.Scene.show_hide_meg_sensors = bpy.props.BoolProperty(default=False)
bpy.types.Scene.show_hide_connections = bpy.props.BoolProperty(default=False)
bpy.types.Scene.inflating = bpy.props.FloatProperty(min=0, max=1, default=0, update=inflating_update)
bpy.types.Scene.hemis_inf_distance = bpy.props.FloatProperty(min=-5, max=5, default=0, update=hemis_inf_distance_update)
bpy.types.Scene.hemis_distance = bpy.props.FloatProperty(min=-5, max=5, default=0, update=hemis_distance_update)


class ShowHideMEGSensors(bpy.types.Operator):
    bl_idname = "mmvt.show_hide_meg_sensors"
    bl_label = "mmvt show_hide_meg_sensors"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        bpy.context.scene.show_hide_meg_sensors = not bpy.context.scene.show_hide_meg_sensors
        show_hide_meg_sensors(bpy.context.scene.show_hide_meg_sensors)
        return {"FINISHED"}


class ShowHideEEG(bpy.types.Operator):
    bl_idname = "mmvt.show_hide_eeg"
    bl_label = "mmvt show_hide_eeg"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        bpy.context.scene.show_hide_eeg = not bpy.context.scene.show_hide_eeg
        show_hide_eeg(bpy.context.scene.show_hide_eeg)
        return {"FINISHED"}


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
    cortex_inflated_rh = 0
    cortex_inflated_lh = 0
    cortex_rh = 0
    cortex_lh = 0

    def draw(self, context):
        if AppearanceMakerPanel.init:
            appearance_draw(self, context)


def init(addon):
    AppearanceMakerPanel.addon = addon
    register()
    AppearanceMakerPanel.init = True
    bpy.context.scene.subcortical_layer = 'fmri'
    change_to_solid_brain()
    # show_rois()
    loc_val = 5
    if bpy.data.objects.get('Cortex-inflated-rh') and bpy.data.objects.get('inflated_rh'):
        AppearanceMakerPanel.cortex_inflated_rh = bpy.data.objects['Cortex-inflated-rh'].location[0] = \
            bpy.data.objects['inflated_rh'].location[0] = loc_val
        AppearanceMakerPanel.cortex_inflated_lh = bpy.data.objects['Cortex-inflated-lh'].location[0] = \
            bpy.data.objects['inflated_lh'].location[0] = -1*loc_val
    if bpy.data.objects.get('Cortex-rh') and bpy.data.objects.get('lh'):
        AppearanceMakerPanel.cortex_rh = bpy.data.objects['Cortex-rh'].location[0] = \
            bpy.data.objects['rh'].location[0] = 0
        AppearanceMakerPanel.cortex_lh = bpy.data.objects['Cortex-lh'].location[0] = \
            bpy.data.objects['rh'].location[0] = 0
    bpy.context.scene.hemis_distance = 0
    bpy.context.scene.hemis_inf_distance = 0 #-5
    set_inflated_ratio(1)
    appearance_show_rois_activity_update()
    bpy.ops.mmvt.selection_listener()


def register():
    try:
        unregister()
        bpy.utils.register_class(AppearanceMakerPanel)
        bpy.utils.register_class(ShowHideMEGSensors)
        bpy.utils.register_class(ShowHideElectrodes)
        bpy.utils.register_class(ShowHideEEG)
        bpy.utils.register_class(ShowHideConnections)
        bpy.utils.register_class(SelectionListener)
    except:
        print("Can't register Appearance Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(AppearanceMakerPanel)
        bpy.utils.unregister_class(ShowHideMEGSensors)
        bpy.utils.unregister_class(ShowHideElectrodes)
        bpy.utils.unregister_class(ShowHideEEG)
        bpy.utils.unregister_class(ShowHideConnections)
        bpy.utils.unregister_class(SelectionListener)
    except:
        pass

