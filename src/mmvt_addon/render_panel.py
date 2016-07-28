import bpy
import math
import os.path as op
import glob
import mmvt_utils as mu

bpy.types.Scene.output_path = bpy.props.StringProperty(
    name="", default="", description="Define the path for the output files", subtype='DIR_PATH')


def set_render_quality(quality):
    bpy.context.scene.quality = quality
    

def set_render_output_path(output_path):
    bpy.context.scene.output_path = output_path
    

def set_render_smooth_figure(smooth_figure):
    bpy.context.scene.smooth_figure = smooth_figure


def load_camera(camera_fname=''):
    if camera_fname == '':
        camera_fname = op.join(bpy.path.abspath(bpy.context.scene.output_path), 'camera.pkl')
    if op.isfile(camera_fname):
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
    for obj in bpy.data.objects:
        obj.select = False
    for obj in bpy.context.visible_objects:
        if not (obj.hide or obj.hide_render):
            obj.select = True
    ret = bpy.ops.view3d.camera_to_view_selected()
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
        if op.isdir(bpy.path.abspath(bpy.context.scene.output_path)):
            camera_fname = op.join(bpy.path.abspath(bpy.context.scene.output_path), 'camera.pkl')
            mu.save((X_rotation, Y_rotation, Z_rotation, X_location, Y_location, Z_location), camera_fname)
            print('Camera location was saved to {}'.format(camera_fname))
        else:
            mu.message(self, "Can't find the folder {}".format(bpy.path.abspath(bpy.context.scene.output_path)))
    RenderFigure.update_camera = True


def render_draw(self, context):
    layout = self.layout
    layout.label(text='Output Path:')
    layout.prop(context.scene, 'output_path')
    layout.prop(context.scene, "quality", text='Quality')

    # layout.operator(CameraMode.bl_idname, text="Camera Mode", icon='CAMERA_DATA')
    # layout.operator("view3d.viewnumpad", text="View Camera", icon='CAMERA_DATA').type = 'CAMERA'
    layout.operator(RenderFigure.bl_idname, text="Render", icon='SCENE')
    if len(glob.glob(op.join(bpy.path.abspath(bpy.context.scene.output_path), 'camera*.pkl'))) > 1:
        layout.operator(RenderAllFigures.bl_idname, text="Render All", icon='SCENE')
    layout.prop(context.scene, 'smooth_figure')
    layout.operator(GrabCamera.bl_idname, text="Grab Camera", icon='BORDER_RECT')
    layout.operator(LoadCamera.bl_idname, text="Load Camera", icon='RENDER_REGION')
    layout.operator(MirrorCamera.bl_idname, text="Mirror Camera", icon='RENDER_REGION')

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
    name='smooth image', description="This significantly affect rendering speed")


class MirrorCamera(bpy.types.Operator):
    bl_idname = "ohad.mirror_camera"
    bl_label = "Mirror Camera"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        mirror()
        return {"FINISHED"}


class CameraMode(bpy.types.Operator):
    bl_idname = "ohad.camera_mode"
    bl_label = "Camera Mode"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        camera_mode()
        return {"FINISHED"}


class GrabCamera(bpy.types.Operator):
    bl_idname = "ohad.grab_camera"
    bl_label = "Grab Camera"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        grab_camera(self)
        return {"FINISHED"}


class LoadCamera(bpy.types.Operator):
    bl_idname = "ohad.load_camera"
    bl_label = "Load Camera"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        load_camera()
        return {"FINISHED"}


class RenderFigure(bpy.types.Operator):
    bl_idname = "ohad.rendering"
    bl_label = "Render figure"
    bl_options = {"UNDO"}
    update_camera = True

    def invoke(self, context, event=None):
        render_image()
        return {"FINISHED"}


class RenderAllFigures(bpy.types.Operator):
    bl_idname = "ohad.render_all_figures"
    bl_label = "Render all figures"
    bl_options = {"UNDO"}
    update_camera = True

    def invoke(self, context, event=None):
        render_all_images()
        return {"FINISHED"}


def render_all_images():
    camera_files = glob.glob(op.join(bpy.path.abspath(bpy.context.scene.output_path), 'camera_*.pkl'))
    for camera_file in camera_files:
        load_camera(camera_file)
        camera_name = mu.namebase(camera_file)
        for hemi in mu.HEMIS:
            if hemi in camera_name:
                RenderingMakerPanel.addon.show_hide_hemi(False, hemi)
                RenderingMakerPanel.addon.show_hide_hemi(True, mu.other_hemi(hemi))
        render_image('{}_fig'.format(camera_name[len('camera') + 1:]))


def render_image(image_name=''):
    quality = bpy.context.scene.quality
    use_square_samples = bpy.context.scene.smooth_figure

    bpy.context.scene.render.resolution_percentage = quality
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('use_square_samples: {}'.format(use_square_samples))
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    bpy.context.scene.cycles.use_square_samples = use_square_samples

    cur_frame = bpy.context.scene.frame_current
    if image_name == '':
        image_name = 't{}'.format(cur_frame)
    file_name = op.join(bpy.path.abspath(bpy.context.scene.output_path), image_name)
    print('file name: {}'.format(file_name))
    bpy.context.scene.render.filepath = file_name
    # Render and save the rendered scene to file. ------------------------------
    print('Image quality:')
    print(bpy.context.scene.render.resolution_percentage)
    print("Rendering...")
    bpy.ops.render.render(write_still=True)
    print("Finished")


class RenderingMakerPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
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
    grab_camera(None, False)
    register()


def register():
    try:
        unregister()
        bpy.utils.register_class(RenderingMakerPanel)
        bpy.utils.register_class(RenderAllFigures)

        # bpy.utils.register_class(CameraMode)
        bpy.utils.register_class(GrabCamera)
        bpy.utils.register_class(LoadCamera)
        bpy.utils.register_class(MirrorCamera)
        bpy.utils.register_class(RenderFigure)
        # print('Render Panel was registered!')
    except:
        print("Can't register Render Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(RenderingMakerPanel)
        bpy.utils.unregister_class(RenderAllFigures)
        # bpy.utils.unregister_class(CameraMode)
        bpy.utils.unregister_class(GrabCamera)
        bpy.utils.unregister_class(LoadCamera)
        bpy.utils.unregister_class(MirrorCamera)
        bpy.utils.unregister_class(RenderFigure)
    except:
        pass
        # print("Can't unregister Render Panel!")
