import bpy


def run(mmvt):
    mmvt.show_hide.show_hemis()
    mmvt.show_hide.hide_subcorticals()
    mmvt.coloring
    mmvt.set_render_quality(60)
    mmvt.set_render_output_path(op.join())



bpy.types.Scene.plot_theta_power_from = bpy.props.IntProperty(default=0, min=0)
bpy.types.Scene.plot_theta_power_to = bpy.props.IntProperty(default=0, min=0)


def draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'plot_theta_power_from', text='From')
    layout.prop(context.scene, 'plot_theta_power_to', text='To')
