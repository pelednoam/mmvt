import bpy
import os.path as op
import glob
import numpy as np
import mmvt_utils as mu


PERC_FORMATS = {0:'{:.0f}', 1:'{:.1f}', 2:'{:.2f}', 3:'{:.3f}', 4:'{:.4f}', 5:'{:.5f}'}


def load_colormap():
    colormap_fname = op.join(mu.file_fol(), 'color_maps', '{}.npy'.format(
        bpy.context.scene.colorbar_files.replace('-', '_')))
    colormap = np.load(colormap_fname)
    for ind in range(colormap.shape[0]):
        cb_obj_name = 'cb.{0:0>3}'.format(ind)
        cb_obj = bpy.data.objects[cb_obj_name]
        cur_mat = cb_obj.active_material
        cur_mat.diffuse_color = colormap[ind]
        # print('Changing {} to {}'.format(cb_obj_name, colormap[ind]))


def colormap_update(self, context):
    if ColorbarPanel.init:
        load_colormap()


def colorbar_update(self, context):
    if ColorbarPanel.init:
        prec = PERC_FORMATS[bpy.context.scene.colorbar_prec]
        prev_max = float(bpy.data.objects['colorbar_max'].data.body)
        prev_min = float(bpy.data.objects['colorbar_min'].data.body)
        if bpy.context.scene.colorbar_max > bpy.context.scene.colorbar_min:
            bpy.data.objects['colorbar_max'].data.body = bpy.data.objects['colorbar_max_camera'].data.body = \
                prec.format(bpy.context.scene.colorbar_max)
            bpy.data.objects['colorbar_min'].data.body = bpy.data.objects['colorbar_min_camera'].data.body = \
                prec.format(bpy.context.scene.colorbar_min)
        else:
            ColorbarPanel.init = False
            bpy.context.scene.colorbar_max = prev_max
            bpy.context.scene.colorbar_min = prev_min
            ColorbarPanel.init = True
        bpy.data.objects['colorbar_title'].data.body = bpy.data.objects['colorbar_title_camera'].data.body = \
            bpy.context.scene.colorbar_title


def show_cb_in_render_update(self, context):
    show_cb_in_render(bpy.context.scene.show_cb_in_render)


def show_cb_in_render(val=True):
    mu.show_hide_hierarchy(val, 'colorbar_camera', True, False)
    mu.show_hide_hierarchy(val, 'cCB_camera', True, False)


def colorbar_draw(self, context):
    layout = self.layout
    layout.prop(context.scene, "colorbar_files", text="")
    layout.prop(context.scene, "colorbar_title", text="Title:")
    row = layout.row(align=0)
    row.prop(context.scene, "colorbar_min", text="min:")
    row.prop(context.scene, "colorbar_max", text="max:")
    layout.prop(context.scene, 'colorbar_prec', text='precision')
    layout.prop(context.scene, 'show_cb_in_render', text='Show in rendering')
    # layout.operator(ColorbarButton.bl_idname, text="Do something", icon='ROTATE')


class ColorbarButton(bpy.types.Operator):
    bl_idname = "mmvt.colorbar_button"
    bl_label = "Colorbar botton"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        load_colormap()
        return {'PASS_THROUGH'}


bpy.types.Scene.colorbar_files = bpy.props.EnumProperty(items=[], description="colormap files", update=colormap_update)
bpy.types.Scene.colorbar_max = bpy.props.FloatProperty(description="", update=colorbar_update)
bpy.types.Scene.colorbar_min = bpy.props.FloatProperty(description="", update=colorbar_update)
bpy.types.Scene.colorbar_title = bpy.props.StringProperty(description="", update=colorbar_update)
bpy.types.Scene.colorbar_prec = bpy.props.IntProperty(min=0, default=2, max=5, description="", update=colorbar_update)
bpy.types.Scene.show_cb_in_render = bpy.props.BoolProperty(
    default=True, description="show_cb_in_render", update=show_cb_in_render_update)



class ColorbarPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "Colorbar"
    addon = None
    init = False

    def draw(self, context):
        if ColorbarPanel.init:
            colorbar_draw(self, context)


def init(addon):
    ColorbarPanel.addon = addon
    colorbar_files = glob.glob(op.join(mu.file_fol(), 'color_maps', '*.npy'))
    if len(colorbar_files) == 0:
        return None
    files_names = [mu.namebase(fname).replace('_', '-') for fname in colorbar_files]
    colorbar_items = [(c, c, '', ind) for ind, c in enumerate(files_names)]
    bpy.types.Scene.colorbar_files = bpy.props.EnumProperty(
        items=colorbar_items, description="colormaps files",update=colormap_update)
    bpy.context.scene.colorbar_files = files_names[0]
    for space in mu.get_3d_spaces():
        if space.lock_object and space.lock_object.name == 'full_colorbar':
            space.show_only_render = True
    register()
    ColorbarPanel.init = True
    bpy.context.scene.show_cb_in_render = False
    mu.select_hierarchy('colorbar_camera', False, False)
    bpy.context.scene.colorbar_min = -1
    bpy.context.scene.colorbar_max = 1
    bpy.context.scene.colorbar_title = 'MEG'


def register():
    try:
        unregister()
        bpy.utils.register_class(ColorbarPanel)
        bpy.utils.register_class(ColorbarButton)
    except:
        print("Can't register Colorbar Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(ColorbarPanel)
        bpy.utils.unregister_class(ColorbarButton)
    except:
        pass
