import bpy
import math
import os.path as op
import mmvt_utils as mu

bpy.types.Scene.output_path = bpy.props.StringProperty(
    name="Output Path", default="", description="Define the path for the output files", subtype='DIR_PATH')

def render_draw(self, context):
    layout = self.layout
    col = layout.column(align=True)
    col.prop(context.scene, "X_rotation", text='X rotation')
    col.prop(context.scene, "Y_rotation", text='Y rotation')
    col.prop(context.scene, "Z_rotation", text='Z rotation')
    layout.prop(context.scene, "quality", text='Quality')
    layout.prop(context.scene, 'output_path')
    layout.prop(context.scene, 'smooth_figure')
    layout.operator("ohad.rendering", text="Render", icon='SCENE')


def update_rotation(self, context):
    bpy.data.objects['Target'].rotation_euler.x = math.radians(bpy.context.scene.X_rotation)
    bpy.data.objects['Target'].rotation_euler.y = math.radians(bpy.context.scene.Y_rotation)
    bpy.data.objects['Target'].rotation_euler.z = math.radians(bpy.context.scene.Z_rotation)


def update_quality(self, context):
    print(bpy.context.scene.quality)
    # bpy.context.scene.quality = bpy.context.scene.quality


bpy.types.Scene.X_rotation = bpy.props.FloatProperty(default=0, min=-360, max=360,
                                                     description="Camera rotation around x axis",
                                                     update=update_rotation)
bpy.types.Scene.Y_rotation = bpy.props.FloatProperty(default=0, min=-360, max=360,
                                                     description="Camera rotation around y axis",
                                                     update=update_rotation)
bpy.types.Scene.Z_rotation = bpy.props.FloatProperty(default=0, min=-360, max=360,
                                                     description="Camera rotation around z axis",
                                                     update=update_rotation)
bpy.types.Scene.quality = bpy.props.FloatProperty(default=20, min=1, max=100,
                                                  description="quality of figure in parentage", update=update_quality)
bpy.types.Scene.smooth_figure = bpy.props.BoolProperty(name='smooth image',
                                                       description="This significantly affect rendering speed")


class RenderFigure(bpy.types.Operator):
    bl_idname = "ohad.rendering"
    bl_label = "Render figure"
    bl_options = {"UNDO"}
    current_output_path = bpy.path.abspath(bpy.context.scene.output_path)
    x_rotation = bpy.context.scene.X_rotation
    y_rotation = bpy.context.scene.Y_rotation
    z_rotation = bpy.context.scene.Z_rotation
    quality = bpy.context.scene.quality

    def invoke(self, context, event=None):
        render_image()
        return {"FINISHED"}


def render_image():
    x_rotation = bpy.context.scene.X_rotation
    y_rotation = bpy.context.scene.Y_rotation
    z_rotation = bpy.context.scene.Z_rotation
    quality = bpy.context.scene.quality
    use_square_samples = bpy.context.scene.smooth_figure

    # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$In Render$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    bpy.data.objects['Target'].rotation_euler.x = math.radians(x_rotation)
    bpy.data.objects['Target'].rotation_euler.y = math.radians(y_rotation)
    bpy.data.objects['Target'].rotation_euler.z = math.radians(z_rotation)
    bpy.context.scene.render.resolution_percentage = quality
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('use_square_samples: {}'.format(use_square_samples))
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    bpy.context.scene.cycles.use_square_samples = use_square_samples

    cur_frame = bpy.context.scene.frame_current
    file_name = op.join(bpy.path.abspath(bpy.context.scene.output_path), 'f{}'.format(cur_frame))
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
    register()


def register():
    try:
        unregister()
        bpy.utils.register_class(RenderingMakerPanel)
        bpy.utils.register_class(RenderFigure)
        # print('Render Panel was registered!')
    except:
        print("Can't register Render Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(RenderingMakerPanel)
        bpy.utils.unregister_class(RenderFigure)
    except:
        pass
        # print("Can't unregister Render Panel!")
