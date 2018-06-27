import bpy


def run(mmvt):
    for elc_obj in bpy.data.objects['Deep_electrodes'].children:
        if bpy.context.scene.show_electrodes_with_no_data:
            elc_obj.hide = False
        else:
            elc_obj.hide = elc_obj.animation_data is None

bpy.types.Scene.show_electrodes_with_no_data = bpy.props.BoolProperty(default=False)

def draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'show_electrodes_with_no_data', text='Show')
