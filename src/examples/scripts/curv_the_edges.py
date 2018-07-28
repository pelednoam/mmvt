import bpy


def run(mmvt):
    set_curve(bpy.context.scene.curv_the_edges_val)


def set_curve(val):
    for obj in bpy.data.objects['connections_{}'.format(bpy.context.scene.connectivity_files)].children:
        if obj == bpy.data.objects['connections_vertices']:
            continue
        c1 = obj.data.splines[0].bezier_points[0].co
        obj.data.splines[0].bezier_points[0].handle_right = c1
        obj.data.splines[0].bezier_points[0].handle_left = c1
        obj.data.splines[0].bezier_points[0].handle_right[2] = c1[2] + val
        obj.data.splines[0].bezier_points[0].handle_left[2] = c1[2] - val
        c2 = obj.data.splines[0].bezier_points[1].co
        obj.data.splines[0].bezier_points[1].handle_right = c2
        obj.data.splines[0].bezier_points[1].handle_left = c2
        obj.data.splines[0].bezier_points[1].handle_right[2] = c2[2] - val
        obj.data.splines[0].bezier_points[1].handle_left[2] = c2[2] + val
        obj.data.resolution_u = 64


def curv_the_edges_val_update(self, context):
    set_curve(bpy.context.scene.curv_the_edges_val)


bpy.types.Scene.curv_the_edges_val = bpy.props.FloatProperty(default=2, update=curv_the_edges_val_update)


def draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'curv_the_edges_val', text='Curv')
