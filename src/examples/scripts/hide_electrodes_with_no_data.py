import bpy


def run(mmvt):
    for elc_obj in bpy.data.objects['Deep_electrodes'].children:
        if bpy.context.scene.show_electrodes_with_no_data:
            mmvt.utils.show_hide_obj(elc_obj, True)
        else:
            mmvt.utils.show_hide_obj(elc_obj, elc_obj.animation_data is not None)


bpy.types.Scene.show_electrodes_with_no_data = bpy.props.BoolProperty(default=False)


def draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'show_electrodes_with_no_data', text='Show')
