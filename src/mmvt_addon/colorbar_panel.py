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


def set_colorbar_title(val):
    bpy.data.objects['colorbar_title'].data.body = bpy.data.objects['colorbar_title_camera'].data.body = val


def set_colorbar_max_min(max_val, min_val, prec=None):
    if max_val > min_val:
        bpy.context.scene.colorbar_max = max_val
        bpy.context.scene.colorbar_min = min_val


def set_colorbar_max(val, prec=None, check_minmax=True):
    if not check_minmax or bpy.context.scene.colorbar_max > bpy.context.scene.colorbar_min:
        _set_colorbar_min_max('max', val, prec)
    else:
        prev_max = float(bpy.data.objects['colorbar_max'].data.body)
        ColorbarPanel.init = False
        bpy.context.scene.colorbar_max = prev_max
        ColorbarPanel.init = True


def set_colorbar_min(val, prec=None, check_minmax=True):
    if not check_minmax or bpy.context.scene.colorbar_max > bpy.context.scene.colorbar_min:
        _set_colorbar_min_max('min', val, prec)
    else:
        prev_min = float(bpy.data.objects['colorbar_min'].data.body)
        ColorbarPanel.init = False
        bpy.context.scene.colorbar_min = prev_min
        ColorbarPanel.init = True


def _set_colorbar_min_max(field, val, prec):
    if prec is None or prec not in PERC_FORMATS:
        prec = bpy.context.scene.colorbar_prec
        if prec not in PERC_FORMATS:
            print('Wrong value for prec, should be in {}'.format(PERC_FORMATS.keys()))
    prec_str = PERC_FORMATS[prec]
    cb_obj = bpy.data.objects.get('colorbar_{}'.format(field))
    cb_camera_obj = bpy.data.objects.get('colorbar_{}_camera'.format(field))
    if not cb_obj is None and not cb_camera_obj is None:
        cb_obj.data.body = cb_camera_obj.data.body = prec_str.format(val)
    else:
        print('_set_colorbar_min_max: field error ({})! must be max / min!'.format(field))


def colormap_update(self, context):
    if ColorbarPanel.init:
        load_colormap()


def colorbar_update(self, context):
    if ColorbarPanel.init:
        set_colorbar_title(bpy.context.scene.colorbar_title)
        set_colorbar_max(bpy.context.scene.colorbar_max)
        set_colorbar_min(bpy.context.scene.colorbar_min)


def show_cb_in_render_update(self, context):
    show_cb_in_render(bpy.context.scene.show_cb_in_render)


def show_cb_in_render(val=True):
    mu.show_hide_hierarchy(val, 'colorbar_camera', True, False)
    mu.show_hide_hierarchy(val, 'cCB_camera', True, False)


def colorbar_y_update(self, context):
    bpy.data.objects['cCB'].location[0] = -bpy.context.scene.colorbar_y


def colorbar_text_y_update(self, context):
    bpy.data.objects['colorbar_max'].location[0] = -bpy.context.scene.colorbar_text_y
    bpy.data.objects['colorbar_min'].location[0] = -bpy.context.scene.colorbar_text_y
    bpy.data.objects['colorbar_title'].location[0] = -bpy.context.scene.colorbar_text_y

def colorbar_draw(self, context):
    layout = self.layout
    layout.prop(context.scene, "colorbar_files", text="")
    layout.prop(context.scene, "colorbar_title", text="Title:")
    row = layout.row(align=0)
    row.prop(context.scene, "colorbar_min", text="min:")
    row.prop(context.scene, "colorbar_max", text="max:")
    layout.prop(context.scene, 'colorbar_prec', text='precision')
    layout.prop(context.scene, 'show_cb_in_render', text='Show in rendering')
    layout.prop(context.scene, 'update_cb_location', text='Update location')
    if bpy.context.scene.update_cb_location:
        layout.prop(context.scene, "colorbar_y", text="y axis")
        layout.prop(context.scene, "colorbar_text_y", text="text y axis")
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
bpy.types.Scene.update_cb_location = bpy.props.BoolProperty(default=False)
bpy.types.Scene.colorbar_y = bpy.props.FloatProperty(min=-2, max=2, default=0, update=colorbar_y_update)
bpy.types.Scene.colorbar_text_y = bpy.props.FloatProperty(min=-2, max=2, default=0, update=colorbar_text_y_update)



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
    if not bpy.data.objects.get('full_colorbar', None):
        return
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
    bpy.context.scene.colorbar_files = 'BuPu-YlOrRd'
    bpy.context.scene.colorbar_y = 0.18
    bpy.context.scene.colorbar_text_y = -1.53


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
