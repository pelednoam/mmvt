import bpy
import mmvt_utils as mu


def _addon():
    return ShowHideObjectsPanel.addon


def show_only_redner_update(self, context):
    mu.show_only_render(bpy.context.scene.show_only_render)


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


def show_hide_sub_corticals(do_hide=True):
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


def flip_camera_ortho_view():
    options = ['ORTHO', 'CAMERA']
    bpy.types.Scene.in_camera_view = not bpy.types.Scene.in_camera_view
    bpy.data.screens['Neuro'].areas[1].spaces[0].region_3d.view_perspective = options[int(bpy.types.Scene.in_camera_view)]


class ShowSaggital(bpy.types.Operator):
    bl_idname = "mmvt.show_saggital"
    bl_label = "mmvt show_saggital"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        if bpy.types.Scene.in_camera_view and bpy.data.objects.get("Camera_empty") is not None:
            if mu.get_time_from_event(mu.get_time_obj()) > 2 or bpy.types.Scene.current_view != 'saggital':
                bpy.data.objects["Camera_empty"].rotation_euler = [0.0, 0.0, 1.5707963705062866]
                bpy.types.Scene.current_view = 'saggital'
                bpy.types.Scene.current_view_direction = 0
            else:
                if bpy.types.Scene.current_view_direction == 1:
                    bpy.data.objects["Camera_empty"].rotation_euler = [0.0, 0.0, 1.5707963705062866]
                else:
                    # print('in ShowSaggital else')
                    bpy.data.objects["Camera_empty"].rotation_euler = [0.0, 0.0, -1.5707963705062866]
                bpy.types.Scene.current_view_direction = not bpy.types.Scene.current_view_direction
            # bpy.types.Scene.time_of_view_selection = mu.get_time_obj()
        else:
            bpy.data.screens['Neuro'].areas[1].spaces[0].region_3d.view_perspective = 'ORTHO'
            if mu.get_time_from_event(mu.get_time_obj()) > 2 or bpy.types.Scene.current_view != 'saggital':
                bpy.data.screens['Neuro'].areas[1].spaces[0].region_3d.view_rotation = [0.5, 0.5, -0.5, -0.5]
                bpy.types.Scene.current_view = 'saggital'
                bpy.types.Scene.current_view_direction = 0
            else:
                if bpy.types.Scene.current_view_direction == 1:
                    bpy.data.screens['Neuro'].areas[1].spaces[0].region_3d.view_rotation = [0.5, 0.5, -0.5, -0.5]
                else:
                    bpy.data.screens['Neuro'].areas[1].spaces[0].region_3d.view_rotation = [0.5, 0.5, 0.5, 0.5]
                bpy.types.Scene.current_view_direction = not bpy.types.Scene.current_view_direction

        bpy.types.Scene.time_of_view_selection = mu.get_time_obj()
        # print(bpy.ops.view3d.viewnumpad())
        return {"FINISHED"}


class ShowCoronal(bpy.types.Operator):
    bl_idname = "mmvt.show_coronal"
    bl_label = "mmvt show_coronal"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        if bpy.types.Scene.in_camera_view and bpy.data.objects.get("Camera_empty") is not None:
            if mu.get_time_from_event(mu.get_time_obj()) > 2 or bpy.types.Scene.current_view != 'coronal':
                bpy.data.objects["Camera_empty"].rotation_euler = [0.0, 0.0, 3.1415927410125732]
                bpy.types.Scene.current_view = 'coronal'
                bpy.types.Scene.current_view_direction = 0
            else:
                if bpy.types.Scene.current_view_direction == 1:
                    bpy.data.objects["Camera_empty"].rotation_euler = [0.0, 0.0, 3.1415927410125732]
                else:
                    # print('in ShowCoronal else')
                    bpy.data.objects["Camera_empty"].rotation_euler = [0.0, 0.0, 0.0]
                bpy.types.Scene.current_view_direction = not bpy.types.Scene.current_view_direction
            bpy.types.Scene.time_of_view_selection = mu.get_time_obj()
        else:
            bpy.data.screens['Neuro'].areas[1].spaces[0].region_3d.view_perspective = 'ORTHO'
            if mu.get_time_from_event(mu.get_time_obj()) > 2 or bpy.types.Scene.current_view != 'coronal':
                bpy.data.screens['Neuro'].areas[1].spaces[0].region_3d.view_rotation = [0.7071068286895752, 0.7071068286895752, -0.0, -0.0]
                bpy.types.Scene.current_view = 'coronal'
                bpy.types.Scene.current_view_direction = 0
            else:
                if bpy.types.Scene.current_view_direction == 1:
                    bpy.data.screens['Neuro'].areas[1].spaces[0].region_3d.view_rotation = [0.7071068286895752, 0.7071068286895752, -0.0, -0.0]
                else:
                    bpy.data.screens['Neuro'].areas[1].spaces[0].region_3d.view_rotation = [0, 0, 0.7071068286895752, 0.7071068286895752]
                bpy.types.Scene.current_view_direction = not bpy.types.Scene.current_view_direction
            bpy.types.Scene.time_of_view_selection = mu.get_time_obj()
        # print(bpy.ops.view3d.viewnumpad())
        return {"FINISHED"}


class ShowAxial(bpy.types.Operator):
    bl_idname = "mmvt.show_axial"
    bl_label = "mmvt show_axial"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        if bpy.types.Scene.in_camera_view and bpy.data.objects.get("Camera_empty") is not None:
            if mu.get_time_from_event(mu.get_time_obj()) > 2 or bpy.types.Scene.current_view != 'axial':
                bpy.data.objects["Camera_empty"].rotation_euler = [1.5707963705062866, 0.0, 3.1415927410125732]
                bpy.types.Scene.current_view = 'axial'
                bpy.types.Scene.current_view_direction = 0
            else:
                if bpy.types.Scene.current_view_direction == 1:
                    bpy.data.objects["Camera_empty"].rotation_euler = [1.5707963705062866, 0.0, 3.1415927410125732]
                else:
                    # print('in ShowAxial else')
                    bpy.data.objects["Camera_empty"].rotation_euler = [-1.5707963705062866, 0.0, 3.1415927410125732]
                bpy.types.Scene.current_view_direction = not bpy.types.Scene.current_view_direction
            bpy.types.Scene.time_of_view_selection = mu.get_time_obj()
        else:
            bpy.data.screens['Neuro'].areas[1].spaces[0].region_3d.view_perspective = 'ORTHO'
            if mu.get_time_from_event(mu.get_time_obj()) > 2 or bpy.types.Scene.current_view != 'axial':
                bpy.data.screens['Neuro'].areas[1].spaces[0].region_3d.view_rotation = [1, 0, 0, 0]
                bpy.types.Scene.current_view = 'axial'
                bpy.types.Scene.current_view_direction = 0
            else:
                if bpy.types.Scene.current_view_direction == 1:
                    bpy.data.screens['Neuro'].areas[1].spaces[0].region_3d.view_rotation = [1, 0, 0, 0]
                else:
                    bpy.data.screens['Neuro'].areas[1].spaces[0].region_3d.view_rotation = [0, 1, 0, 0]
                bpy.types.Scene.current_view_direction = not bpy.types.Scene.current_view_direction
            bpy.types.Scene.time_of_view_selection = mu.get_time_obj()
        # print(bpy.ops.view3d.viewnumpad())
        return {"FINISHED"}


class FlipCameraView(bpy.types.Operator):
    bl_idname = "mmvt.flip_camera_view"
    bl_label = "mmvt flip camera view"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        flip_camera_ortho_view()
        return {"FINISHED"}


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


class ShowHideSubCerebellum(bpy.types.Operator):
    bl_idname = "mmvt.show_hide_cerebellum"
    bl_label = "mmvt show_hide_cerebellum"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        bpy.context.scene.objects_show_hide_cerebellum = not bpy.context.scene.objects_show_hide_cerebellum
        show_hide_hierarchy(bpy.context.scene.objects_show_hide_cerebellum, 'Cerebellum')
        show_hide_hierarchy(bpy.context.scene.objects_show_hide_cerebellum, 'Cerebellum_fmri_activity_map')
        show_hide_hierarchy(bpy.context.scene.objects_show_hide_cerebellum, 'Cerebellum_meg_activity_map')
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
        if _addon().is_pial():
            sub_vis = not bpy.context.scene.objects_show_hide_sub_cortical
            sub_show_text = '{} Subcortical'.format('Hide' if sub_vis else 'Show')
            sub_icon = show_hide_icon['show' if sub_vis else 'hide']
            layout.operator(ShowHideSubCorticals.bl_idname, text=sub_show_text, icon=sub_icon)
        # if bpy.data.objects.get('Cerebellum'):
        #     sub_vis = not bpy.context.scene.objects_show_hide_cerebellum
        #     sub_show_text = '{} Cerebellum'.format('Hide' if sub_vis else 'Show')
        #     sub_icon = show_hide_icon['show' if sub_vis else 'hide']
        #     layout.operator(ShowHideSubCerebellum.bl_idname, text=sub_show_text, icon=sub_icon)
        row = layout.row(align=True)
        row.operator(ShowAxial.bl_idname, text='Axial', icon='AXIS_TOP')
        row.operator(ShowCoronal.bl_idname, text='Coronal', icon='AXIS_FRONT')
        row.operator(ShowSaggital.bl_idname, text='Saggital', icon='AXIS_SIDE')
        views_options = ['Camera', 'Ortho']
        next_view = views_options[int(bpy.context.scene.in_camera_view)]
        icons = ['SCENE', 'MANIPUL']
        next_icon = icons[int(bpy.context.scene.in_camera_view)]
        # row = layout.row(align=True)
        # row.operator(FlipCameraView.bl_idname, text='Change to {} view'.format(next_view), icon=next_icon)
        layout.prop(context.scene, 'show_only_render', text="Show only rendered objects")



bpy.types.Scene.objects_show_hide_lh = bpy.props.BoolProperty(
    default=True, description="Show left hemisphere")#,update=show_hide_lh)
bpy.types.Scene.objects_show_hide_rh = bpy.props.BoolProperty(
    default=True, description="Show right hemisphere")#, update=show_hide_rh)
bpy.types.Scene.objects_show_hide_sub_cortical = bpy.props.BoolProperty(
    default=True, description="Show sub cortical")#, update=show_hide_sub_cortical_update)
bpy.types.Scene.objects_show_hide_cerebellum = bpy.props.BoolProperty(
    default=True, description="Show Cerebellum")
bpy.types.Scene.show_only_render = bpy.props.BoolProperty(
    default=True, description="Show only rendered objects", update=show_only_redner_update)
bpy.types.Scene.current_view = 'free'
bpy.types.Scene.time_of_view_selection = mu.get_time_obj
bpy.types.Scene.current_view_direction = 0
bpy.types.Scene.in_camera_view = 0


def init(addon):
    ShowHideObjectsPanel.addon = addon
    bpy.context.scene.objects_show_hide_rh = False
    bpy.context.scene.objects_show_hide_lh = False
    bpy.context.scene.objects_show_hide_sub_cortical = False
    show_hide_sub_corticals(False)
    show_hemis()
    bpy.context.scene.show_only_render = False
    for fol in ['Cerebellum', 'Cerebellum_fmri_activity_map', 'Cerebellum_meg_activity_map']:
        show_hide_hierarchy(True, fol)

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
        bpy.utils.register_class(ShowSaggital)
        bpy.utils.register_class(ShowCoronal)
        bpy.utils.register_class(ShowAxial)
        bpy.utils.register_class(FlipCameraView)
        bpy.utils.register_class(ShowHideSubCorticals)
        bpy.utils.register_class(ShowHideSubCerebellum)
        # print('Show Hide Panel was registered!')
    except:
        print("Can't register Show Hide Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(ShowHideObjectsPanel)
        bpy.utils.unregister_class(ShowHideLH)
        bpy.utils.unregister_class(ShowHideRH)
        bpy.utils.unregister_class(ShowSaggital)
        bpy.utils.unregister_class(ShowCoronal)
        bpy.utils.unregister_class(ShowAxial)
        bpy.utils.unregister_class(FlipCameraView)
        bpy.utils.unregister_class(ShowHideSubCorticals)
        bpy.utils.unregister_class(ShowHideSubCerebellum)
    except:
        pass
        # print("Can't unregister Freeview Panel!")
