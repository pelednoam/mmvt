import bpy
import math
import os.path as op
import glob
import numpy as np
import time
import mmvt_utils as mu

bpy.types.Scene.output_path = bpy.props.StringProperty(
    name="", default="", description="Define the path for the output files", subtype='DIR_PATH')


def _addon():
    return RenderingMakerPanel.addon


def camera_files_update(self, context):
    load_camera()

def set_render_quality(quality):
    bpy.context.scene.quality = quality
    

def set_render_output_path(output_path):
    bpy.context.scene.output_path = output_path
    

def set_render_smooth_figure(smooth_figure):
    bpy.context.scene.smooth_figure = smooth_figure


def load_camera(camera_fname=''):
    if camera_fname == '':
        camera_fname = op.join(mu.get_user_fol(), 'camera', '{}.pkl'.format(bpy.context.scene.camera_files))
    if op.isfile(camera_fname):
        camera_name = mu.namebase(camera_fname)
        for hemi in mu.HEMIS:
            if hemi in camera_name:
                _addon().show_hide_hemi(False, hemi)
                _addon().show_hide_hemi(True, mu.other_hemi(hemi))
        X_rotation, Y_rotation, Z_rotation, X_location, Y_location, Z_location = mu.load(camera_fname)
        RenderFigure.update_camera = False
        bpy.context.scene.X_rotation = X_rotation
        bpy.context.scene.Y_rotation = Y_rotation
        bpy.context.scene.Z_rotation = Z_rotation
        bpy.context.scene.X_location = X_location
        bpy.context.scene.Y_location = Y_location
        bpy.context.scene.Z_location = Z_location
        RenderFigure.update_camera = True
        update_camera()
    else:
        print('No camera file was found in {}!'.format(camera_fname))


def camera_mode():
    # for obj in bpy.data.objects:
    #     obj.select = False
    # for obj in bpy.context.visible_objects:
    #     if not (obj.hide or obj.hide_render):
    #         obj.select = True
    ret = bpy.ops.view3d.camera_to_view_selected()
    ret = bpy.ops.view3d.viewnumpad(type='CAMERA')
    print(ret)


def grab_camera(self=None, do_save=True):
    RenderFigure.update_camera = False
    bpy.context.scene.X_rotation = X_rotation = math.degrees(bpy.data.objects['Camera'].rotation_euler.x)
    bpy.context.scene.Y_rotation = Y_rotation = math.degrees(bpy.data.objects['Camera'].rotation_euler.y)
    bpy.context.scene.Z_rotation = Z_rotation = math.degrees(bpy.data.objects['Camera'].rotation_euler.z)
    bpy.context.scene.X_location = X_location = bpy.data.objects['Camera'].location.x
    bpy.context.scene.Y_location = Y_location = bpy.data.objects['Camera'].location.y
    bpy.context.scene.Z_location = Z_location = bpy.data.objects['Camera'].location.z
    if do_save:
        if op.isdir(op.join(mu.get_user_fol(), 'camera')):
            camera_fname = op.join(mu.get_user_fol(), 'camera', 'camera.pkl')
            mu.save((X_rotation, Y_rotation, Z_rotation, X_location, Y_location, Z_location), camera_fname)
            print('Camera location was saved to {}'.format(camera_fname))
        else:
            mu.message(self, "Can't find the folder {}".format(mu.get_user_fol(), 'camera'))
    RenderFigure.update_camera = True


def render_draw(self, context):
    layout = self.layout
    layout.label(text='Output Path:')
    layout.prop(context.scene, 'output_path')
    layout.prop(context.scene, "quality", text='Quality')

    # layout.operator(CameraMode.bl_idname, text="Camera Mode", icon='CAMERA_DATA')
    # layout.operator("view3d.viewnumpad", text="View Camera", icon='CAMERA_DATA').type = 'CAMERA'
    layout.operator(RenderFigure.bl_idname, text="Render", icon='SCENE')
    camera_files = glob.glob(op.join(mu.get_user_fol(), 'camera', 'camera_*.pkl')) + \
                   glob.glob(op.join(mu.get_user_fol(), 'camera', 'camera_*.pkl'))
    if len(camera_files) > 1:
        layout.operator(RenderAllFigures.bl_idname, text="Render All", icon='SCENE')
    perspectives_files_exist = op.isdir(
        op.join(mu.get_user_fol(), 'camera')) and \
        np.all([op.isfile(op.join(mu.get_user_fol(), 'camera', '{}.pkl'.format(pers_name))) for pers_name in
        ['camera_lateral_lh', 'camera_lateral_rh', 'camera_medial_lh', 'camera_medial_rh']])
    if perspectives_files_exist:
        layout.operator(RenderPerspectives.bl_idname, text="Render Perspectives", icon='SCENE')
    layout.prop(context.scene, 'render_background')
    layout.prop(context.scene, 'smooth_figure')
    # layout.operator(CameraMode.bl_idname, text="Camera view", icon='CAMERA_DATA')
    layout.operator(GrabCamera.bl_idname, text="Grab Camera", icon='BORDER_RECT')
    if len(bpy.context.scene.camera_files) > 0:
        layout.prop(context.scene, 'camera_files', text='')
        layout.operator(LoadCamera.bl_idname, text="Load Camera", icon='RENDER_REGION')
    layout.operator(MirrorCamera.bl_idname, text="Mirror Camera", icon='RENDER_REGION')
    # layout.prop(context.scene, "lighting", text='Lighting')
    layout.prop(context.scene, "show_camera_props", text='Show camera props')
    if bpy.context.scene.show_camera_props:
        col = layout.column(align=True)
        col.prop(context.scene, "X_rotation", text='X rotation')
        col.prop(context.scene, "Y_rotation", text='Y rotation')
        col.prop(context.scene, "Z_rotation", text='Z rotation')
        col = layout.column(align=True)
        col.prop(context.scene, "X_location", text='X location')
        col.prop(context.scene, "Y_location", text='Y location')
        col.prop(context.scene, "Z_location", text='Z location')


def update_camera(self=None, context=None):
    if RenderFigure.update_camera:
        bpy.data.objects['Camera'].rotation_euler.x = math.radians(bpy.context.scene.X_rotation)
        bpy.data.objects['Camera'].rotation_euler.y = math.radians(bpy.context.scene.Y_rotation)
        bpy.data.objects['Camera'].rotation_euler.z = math.radians(bpy.context.scene.Z_rotation)
        bpy.data.objects['Camera'].location.x = bpy.context.scene.X_location
        bpy.data.objects['Camera'].location.y = bpy.context.scene.Y_location
        bpy.data.objects['Camera'].location.z = bpy.context.scene.Z_location


def lighting_update(self, context):
    bpy.data.materials['light'].node_tree.nodes["Emission"].inputs[1].default_value = bpy.context.scene.lighting
    pass


def mirror():
    camera_rotation_z = bpy.context.scene.Z_rotation
    # target_rotation_z = math.degrees(bpy.data.objects['Camera'].rotation_euler.z)
    bpy.data.objects['Target'].rotation_euler.z += math.radians(180 - camera_rotation_z)
    print(bpy.data.objects['Target'].rotation_euler.z)


bpy.types.Scene.X_rotation = bpy.props.FloatProperty(
    default=0, min=-360, max=360, description="Camera rotation around x axis", update=update_camera)
bpy.types.Scene.Y_rotation = bpy.props.FloatProperty(
    default=0, min=-360, max=360, description="Camera rotation around y axis", update=update_camera)
bpy.types.Scene.Z_rotation = bpy.props.FloatProperty(
    default=0, min=-360, max=360, description="Camera rotation around z axis", update=update_camera)
bpy.types.Scene.X_location = bpy.props.FloatProperty(description="Camera x location", update=update_camera)
bpy.types.Scene.Y_location = bpy.props.FloatProperty(description="Camera y lovation", update=update_camera)
bpy.types.Scene.Z_location = bpy.props.FloatProperty(description="Camera z locationo", update=update_camera)
bpy.types.Scene.quality = bpy.props.FloatProperty(
    default=20, min=1, max=100,description="quality of figure in parentage")
bpy.types.Scene.smooth_figure = bpy.props.BoolProperty(
    name='Smooth image', description="This significantly affect rendering speed")
bpy.types.Scene.render_background = bpy.props.BoolProperty(
    name='Background rendering', description="Render in the background")
bpy.types.Scene.lighting = bpy.props.FloatProperty(
    default=1, min=0, max=2,description="lighting", update=lighting_update)
bpy.types.Scene.camera_files = bpy.props.EnumProperty(items=[], update=camera_files_update)
bpy.types.Scene.show_camera_props = bpy.props.BoolProperty(default=False)



class MirrorCamera(bpy.types.Operator):
    bl_idname = "mmvt.mirror_camera"
    bl_label = "Mirror Camera"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        mirror()
        return {"FINISHED"}


class CameraMode(bpy.types.Operator):
    bl_idname = "mmvt.camera_mode"
    bl_label = "Camera Mode"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        camera_mode()
        return {"FINISHED"}


class GrabCamera(bpy.types.Operator):
    bl_idname = "mmvt.grab_camera"
    bl_label = "Grab Camera"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        grab_camera(self)
        return {"FINISHED"}


class LoadCamera(bpy.types.Operator):
    bl_idname = "mmvt.load_camera"
    bl_label = "Load Camera"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        load_camera()
        return {"FINISHED"}


class RenderFigure(bpy.types.Operator):
    bl_idname = "mmvt.rendering"
    bl_label = "Render figure"
    bl_options = {"UNDO"}
    update_camera = True

    def invoke(self, context, event=None):
        render_image()
        return {"FINISHED"}


class RenderPerspectives(bpy.types.Operator):
    bl_idname = "mmvt.render_all_perspectives"
    bl_label = "Render all perspectives"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        camera_files = [op.join(mu.get_user_fol(), 'camera', '{}.pkl'.format(pers_name)) for pers_name in
            ['camera_lateral_lh', 'camera_lateral_rh', 'camera_medial_lh', 'camera_medial_rh']]
        render_all_images(camera_files, hide_subcorticals=True)
        cmd = '{} -m src.utils.figures_utils --fol {}'.format(
            bpy.context.scene.python_cmd, mu.get_user_fol(), 'camera')
        print('Running {}'.format(cmd))
        mu.run_command_in_new_thread(cmd, False)
        return {"FINISHED"}


class RenderAllFigures(bpy.types.Operator):
    bl_idname = "mmvt.render_all_figures"
    bl_label = "Render all figures"
    bl_options = {"UNDO"}
    update_camera = True

    def invoke(self, context, event=None):
        render_all_images()
        return {"FINISHED"}


def render_all_images(camera_files=None, hide_subcorticals=False):
    if camera_files is None:
        camera_files = glob.glob(op.join(mu.get_user_fol(), 'camera', 'camera_*.pkl'))
    for camera_file in camera_files:
        # load_camera(camera_file)
        render_image(camera_fname=camera_file, hide_subcorticals=hide_subcorticals)


def render_image(image_name='', image_fol='', quality=0, use_square_samples=None, render_background=None,
                 camera_fname='', hide_subcorticals=False):
    bpy.context.scene.render.resolution_percentage = bpy.context.scene.quality if quality == 0 else quality
    bpy.context.scene.cycles.use_square_samples = bpy.context.scene.smooth_figure if use_square_samples is None \
        else use_square_samples
    if not render_background is None:
        bpy.context.scene.render_background = render_background
    if camera_fname == '':
        camera_fname = op.join(mu.get_user_fol(), 'camera', '{}.pkl'.format(bpy.context.scene.camera_files))
    if image_name == '':
        cur_frame = bpy.context.scene.frame_current
        camera_name = mu.namebase(camera_fname)
        image_name = '{}_{}'.format(camera_name[len('camera') + 1:], cur_frame)
    image_fol = bpy.path.abspath(bpy.context.scene.output_path) if image_fol == '' else image_fol
    print('file name: {}'.format(op.join(image_fol, image_name)))
    bpy.context.scene.render.filepath = op.join(image_fol, image_name)
    # Render and save the rendered scene to file. ------------------------------
    print('Image quality:')
    print(bpy.context.scene.render.resolution_percentage)
    print("Rendering...")
    if not bpy.context.scene.render_background:
        _addon().change_to_rendered_brain()
        if hide_subcorticals:
            _addon().show_hide_sub_corticals()
        bpy.ops.render.render(write_still=True)
    else:
        hide_subs_in_background = True if hide_subcorticals else bpy.context.scene.objects_show_hide_sub_cortical
        mu.change_fol_to_mmvt_root()
        electrode_marked = _addon().is_current_electrode_marked()
        script = 'src.mmvt_addon.scripts.render_image'
        cmd = '{} -m {} -s {} -a {} -i {} -o {} -q {} -b {} -c "{}" '.format(
            bpy.context.scene.python_cmd, script, mu.get_user(), bpy.context.scene.atlas,
            image_name, image_fol, bpy.context.scene.render.resolution_percentage,
            bpy.context.scene.bipolar, camera_fname) + \
            '--hide_lh {} --hide_rh {} --hide_subs {} --show_elecs {} --curr_elec {} --show_only_lead {} '.format(
            bpy.context.scene.objects_show_hide_lh, bpy.context.scene.objects_show_hide_rh,
            hide_subs_in_background, bpy.context.scene.show_hide_electrodes,
            bpy.context.scene.electrodes if electrode_marked else None,
            bpy.context.scene.show_only_lead if electrode_marked else None) + \
            '--show_connections {}'.format(_addon().connections_visible())
        print('Running {}'.format(cmd))
        mu.save_blender_file()
        mu.run_command_in_new_thread(cmd, queues=False)

    print("Finished")


class RenderingListener(bpy.types.Operator):
    bl_idname = 'mmvt.rendering_listener'
    bl_label = 'rendering_listener'
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
                selected_obj_name = bpy.context.selected_objects[0].name
                selected_obj_type = mu.check_obj_type(selected_obj_name)
                if selected_obj_type in [mu.OBJ_TYPE_CORTEX_LH, mu.OBJ_TYPE_CORTEX_RH, mu.OBJ_TYPE_ELECTRODE,
                                         mu.OBJ_TYPE_EEG]:
                    obj = bpy.data.objects.get(selected_obj_name)
                    if obj:
                        mu.change_fcurves_colors(obj)
                if selected_obj_type in [mu.OBJ_TYPE_CORTEX_INFLATED_LH, mu.OBJ_TYPE_CORTEX_INFLATED_RH]:
                    pial_obj_name = selected_obj_name[len('inflated_'):]
                    pial_obj = bpy.data.objects.get(pial_obj_name)
                    if not pial_obj is None:
                        pial_obj.select = True
                        mu.change_fcurves_colors(pial_obj)
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


class RenderingMakerPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "Render"
    addon = None

    def draw(self, context):
        render_draw(self, context)


def init(addon):
    RenderingMakerPanel.addon = addon
    bpy.data.objects['Target'].rotation_euler.x = 0
    bpy.data.objects['Target'].rotation_euler.y = 0
    bpy.data.objects['Target'].rotation_euler.z = 0
    bpy.data.objects['Target'].location.x = 0
    bpy.data.objects['Target'].location.y = 0
    bpy.data.objects['Target'].location.z = 0
    mu.make_dir(op.join(mu.get_user_fol(), 'camera'))
    grab_camera()
    caera_files = glob.glob(op.join(mu.get_user_fol(), 'camera', '*camera*.pkl'))
    if len(caera_files) > 0:
        files_names = [mu.namebase(fname) for fname in caera_files]
        items = [(c, c, '', ind) for ind, c in enumerate(files_names)]
        bpy.types.Scene.camera_files = bpy.props.EnumProperty(
            items=items, description="electrodes sources", update=camera_files_update)
        bpy.context.scene.camera_files = 'camera'
    register()


def register():
    try:
        unregister()
        bpy.utils.register_class(RenderingMakerPanel)
        bpy.utils.register_class(RenderAllFigures)
        bpy.utils.register_class(CameraMode)
        bpy.utils.register_class(GrabCamera)
        bpy.utils.register_class(LoadCamera)
        bpy.utils.register_class(MirrorCamera)
        bpy.utils.register_class(RenderFigure)
        bpy.utils.register_class(RenderPerspectives)
        bpy.utils.register_class(RenderingListener)
        # print('Render Panel was registered!')
    except:
        print("Can't register Render Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(RenderingMakerPanel)
        bpy.utils.unregister_class(RenderAllFigures)
        bpy.utils.unregister_class(CameraMode)
        bpy.utils.unregister_class(GrabCamera)
        bpy.utils.unregister_class(LoadCamera)
        bpy.utils.unregister_class(MirrorCamera)
        bpy.utils.unregister_class(RenderFigure)
        bpy.utils.unregister_class(RenderPerspectives)
        bpy.utils.unregister_class(RenderingListener)
    except:
        pass
        # print("Can't unregister Render Panel!")
