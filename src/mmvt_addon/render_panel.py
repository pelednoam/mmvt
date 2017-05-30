import bpy
import math
import os.path as op
import glob
import numpy as np
from queue import PriorityQueue
from functools import partial
import traceback
import logging
import mmvt_utils as mu
import re


bpy.types.Scene.output_path = bpy.props.StringProperty(
    name="", default="", description="Define the path for the output files", subtype='DIR_PATH')


def _addon():
    return RenderingMakerPanel.addon


def render_in_queue():
    return RenderingMakerPanel.render_in_queue


def finish_rendering():
    logging.info('render panel: finish rendering!')
    RenderingMakerPanel.background_rendering = False
    RenderingMakerPanel.render_in_queue = None
    pop_from_queue()
    if queue_len() > 0:
        logging.info('render panel: run another rendering job')
        run_func_in_queue()
    # logging.handlers[0].flush()


def reading_from_rendering_stdout_func():
    return RenderingMakerPanel.background_rendering


def camera_files_update(self, context):
    load_camera()


def background_color_update(self, context):
    if bpy.context.scene.background_color == 'white':
        bpy.data.worlds['World'].horizon_color = [1.0, 1.0, 1.0]
    else:
        bpy.data.worlds['World'].horizon_color = [.0, .0, .0]


def set_background_color(color):
    if color in ['white', 'black']:
        bpy.context.scene.background_color = color
    else:
        print('Background color can be only white/black!')


def set_render_quality(quality):
    bpy.context.scene.quality = quality
    

def set_render_output_path(output_path):
    bpy.context.scene.output_path = output_path
    

def set_render_smooth_figure(smooth_figure):
    bpy.context.scene.smooth_figure = smooth_figure


def get_rendering_in_the_background():
    return bpy.context.scene.render_background


def set_rendering_in_the_background(val):
    bpy.context.scene.render_background = val


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
        print('Camera loaded: rotation: {},{},{} locatioin: {},{},{}'.format(
            X_rotation, Y_rotation, Z_rotation, X_location, Y_location, Z_location))
        print('Camera loaded: {}'.format(camera_fname))
        RenderFigure.update_camera = True
        update_camera()
    else:
        print('No camera file was found in {}!'.format(camera_fname))


def get_view_mode():
    area = bpy.data.screens['Neuro'].areas[1]
    view = area.spaces[0].region_3d.view_perspective
    return view


def is_camera_view():
    return get_view_mode() == 'CAMERA'


def set_to_camera_view():
    camera_mode('CAMERA')


def exit_from_camera_view():
    camera_mode('ORTHO')


def camera_mode(view=None):
    area = bpy.data.screens['Neuro'].areas[1]
    if view is None:
        view = area.spaces[0].region_3d.view_perspective
    else:
        view = 'ORTHO' if view == 'CAMERA' else 'CAMERA'
    if view == 'CAMERA':
        area.spaces[0].region_3d.view_perspective = 'ORTHO'
        mu.select_all_brain(False)
        bpy.data.cameras['Camera'].lens = 35
    else:
        bpy.data.objects['Camera'].select = True
        bpy.context.scene.objects.active = bpy.data.objects['Camera']
        ob = bpy.context.object
        bpy.context.scene.camera = ob
        for region in area.regions:
            if region.type == 'WINDOW':
                override = bpy.context.copy()
                override['area'] = area
                override["region"] = region
                mu.select_all_brain(True)
                bpy.ops.view3d.camera_to_view(override)
                bpy.ops.view3d.camera_to_view_selected(override)
                bpy.data.cameras['Camera'].lens = 32
                grab_camera()
                break


def grab_camera(self=None, do_save=True, overwrite=True):
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
            if not op.isfile(camera_fname) or overwrite:
                mu.save((X_rotation, Y_rotation, Z_rotation, X_location, Y_location, Z_location), camera_fname)
                print('Camera location was saved to {}'.format(camera_fname))
                print((X_rotation, Y_rotation, Z_rotation, X_location, Y_location, Z_location))
        else:
            mu.message(self, "Can't find the folder {}".format(mu.get_user_fol(), 'camera'))
    RenderFigure.update_camera = True


def render_draw(self, context):
    layout = self.layout

    layout.operator(SaveImage.bl_idname, text='Save image', icon='ROTATE')
    layout.prop(context.scene, 'save_selected_view')

    view = bpy.data.screens['Neuro'].areas[1].spaces[0].region_3d.view_perspective
    icon = 'SCENE' if view == 'ORTHO' else 'MESH_MONKEY'
    text = 'Camera' if view == 'ORTHO' else 'Object'
    layout.operator(CameraMode.bl_idname, text='{} view'.format(text), icon=icon)

    if view == 'CAMERA':
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
            layout.operator(CombinePerspectives.bl_idname, text="Combine Perspectives", icon='OUTLINER_OB_LATTICE')

        if RenderingMakerPanel.background_rendering:
            layout.label(text='Rendering in the background...')
        layout.prop(context.scene, 'render_background')
        layout.prop(context.scene, 'smooth_figure')
        # layout.operator(CameraMode.bl_idname, text="Camera view", icon='CAMERA_DATA')
        layout.operator(GrabCamera.bl_idname, text="Grab Camera", icon='BORDER_RECT')
        if len(bpy.context.scene.camera_files) > 0:
            layout.prop(context.scene, 'camera_files', text='')
            layout.operator(LoadCamera.bl_idname, text="Load Camera", icon='RENDER_REGION')
        layout.operator(MirrorCamera.bl_idname, text="Mirror Camera", icon='RENDER_REGION')
        layout.prop(context.scene, "lighting", text='Lighting')
        layout.prop(context.scene, "background_color", expand=True)
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


def set_lighting(val):
    bpy.context.scene.lighting = val


def mirror():
    camera_rotation_z = bpy.context.scene.Z_rotation
    # target_rotation_z = math.degrees(bpy.data.objects['Camera'].rotation_euler.z)
    bpy.data.objects['Target'].rotation_euler.z += math.radians(180 - camera_rotation_z)
    print(bpy.data.objects['Target'].rotation_euler.z)


bpy.types.Scene.X_rotation = bpy.props.FloatProperty(
    default=0, description="Camera rotation around x axis", update=update_camera) # min=-360, max=360
bpy.types.Scene.Y_rotation = bpy.props.FloatProperty(
    default=0, description="Camera rotation around y axis", update=update_camera) # min=-360, max=360,
bpy.types.Scene.Z_rotation = bpy.props.FloatProperty(
    default=0, description="Camera rotation around z axis", update=update_camera) # min=-360, max=360,
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
bpy.types.Scene.camera_files = bpy.props.EnumProperty(items=[('default', 'camera.pkl', '', 0)], update=camera_files_update)
bpy.types.Scene.show_camera_props = bpy.props.BoolProperty(default=False)
bpy.types.Scene.background_color = bpy.props.EnumProperty(
    items=[('black', 'Black', '', 1), ("white", 'White', '', 2)], update=background_color_update)
bpy.types.Scene.in_camera_view = bpy.props.BoolProperty(default=False)
bpy.types.Scene.save_selected_view = bpy.props.BoolProperty(default=True, name='Fit image into view')


class MirrorCamera(bpy.types.Operator):
    bl_idname = "mmvt.mirror_camera"
    bl_label = "Mirror Camera"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        mirror()
        return {"FINISHED"}


class SaveImage(bpy.types.Operator):
    bl_idname = "mmvt.save_image"
    bl_label = "Save Image"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        save_image('image', bpy.context.scene.save_selected_view)
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
        camera_files = [op.join(mu.get_user_fol(), 'camera', '{}{}.pkl'.format(
            pers_name, '_inf' if _addon().is_inflated() else '')) for pers_name in
            ['camera_lateral_lh', 'camera_lateral_rh', 'camera_medial_lh', 'camera_medial_rh']]
        render_all_images(camera_files, hide_subcorticals=True)
        if bpy.context.scene.render_background:
            put_func_in_queue(combine_four_brain_perspectives, pop_immediately=True)
        else:
            combine_four_brain_perspectives()
        return {"FINISHED"}


class CombinePerspectives(bpy.types.Operator):
    bl_idname = "mmvt.combine_all_perspectives"
    bl_label = "Combine all perspectives"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        combine_four_brain_perspectives()
        return {"FINISHED"}


# def combine_four_brain_perspectives():
#     cmd = '{} -m src.utils.figures_utils --fol {} -f combine_four_brain_perspectives '.format(
#         bpy.context.scene.python_cmd, op.join(mu.get_user_fol(), 'figures')) + \
#         '--inflated {} --facecolor {}'.format(int(_addon().is_inflated()), bpy.context.scene.background_color)
#     mu.run_command_in_new_thread(cmd, False)


def combine_four_brain_perspectives():
    data_min, data_max = _addon().get_colorbar_max_min()
    background = bpy.context.scene.background_color
    figure_name = 'splitted_lateral_medial_{}_{}.png'.format(
        'inflated' if _addon().is_inflated() else 'pial', background)
    figure_fname = op.join(mu.get_user_fol(), 'figures', figure_name)
    colors_map = _addon().get_colormap_name().replace('-', '_')
    x_left_crop, x_right_crop, y_top_crop, y_buttom_crop = (300, 300, 0, 0)
    w_fac, h_fac = (1.5, 1)
    cmd = '{} -m src.utils.figures_utils '.format(bpy.context.scene.python_cmd) + \
        '-f combine_four_brain_perspectives,combine_brain_with_color_bar --fol {} --data_max {} --data_min {} '.format(
        op.join(mu.get_user_fol(), 'figures'), data_max, data_min) + \
        '--figure_fname {} --colors_map {} --x_left_crop {} --x_right_crop {} --y_top_crop {} --y_buttom_crop {} '.format(
        figure_fname, colors_map, x_left_crop, x_right_crop, y_top_crop, y_buttom_crop) + \
        '--w_fac {} --h_fac {} --facecolor {}'.format(w_fac, h_fac, background)
    mu.run_command_in_new_thread(cmd, False)


class RenderAllFigures(bpy.types.Operator):
    bl_idname = "mmvt.render_all_figures"
    bl_label = "Render all figures"
    bl_options = {"UNDO"}
    update_camera = True

    def invoke(self, context, event=None):
        render_all_images()
        return {"FINISHED"}


def init_rendering(inflated, inflated_ratio, transparency, light_layers_depth, lighting=1, background_color='black',
                   rendering_in_the_background=False):
    _addon().clear_colors()
    _addon().set_brain_transparency(transparency)
    _addon().set_light_layers_depth(light_layers_depth)
    set_rendering_in_the_background(rendering_in_the_background)
    if inflated:
        _addon().show_inflated()
        _addon().set_inflated_ratio(inflated_ratio)
    else:
        _addon().show_pial()
    set_background_color(background_color)
    set_lighting(lighting)


def render_all_images(camera_files=None, hide_subcorticals=False):
    if camera_files is None:
        camera_files = glob.glob(op.join(mu.get_user_fol(), 'camera', 'camera_*.pkl'))
    render_image(camera_fname=camera_files, hide_subcorticals=hide_subcorticals)


def render_lateral_medial_split_brain(data_type='', quality=20, overwrite=True):
    image_name = ['lateral_lh', 'lateral_rh', 'medial_lh', 'medial_rh']
    camera = [op.join(mu.get_user_fol(), 'camera', 'camera_{}{}.pkl'.format(
        camera_name, '_inf' if _addon().is_inflated() else '')) for camera_name in image_name]
    image_name = ['{}{}_{}_{}'.format('{}_'.format(data_type) if data_type != '' else '', name, 'inflated_{}'.format(
        _addon().get_inflated_ratio()) if _addon().is_inflated() else 'pial', bpy.context.scene.background_color)
                  for name in image_name]
    render_image(image_name, quality=quality, camera_fname=camera, hide_subcorticals=True, overwrite=overwrite)


@mu.timeit
def render_image(image_name='', image_fol='', quality=0, use_square_samples=None, render_background=None,
                 camera_fname='', hide_subcorticals=False, overwrite=True, set_to_camera_mode=True):
    if not is_camera_view() and set_to_camera_mode:
        set_to_camera_view()
    bpy.context.scene.render.resolution_percentage = bpy.context.scene.quality if quality == 0 else quality
    bpy.context.scene.cycles.use_square_samples = bpy.context.scene.smooth_figure if use_square_samples is None \
        else use_square_samples
    if not render_background is None:
        bpy.context.scene.render_background = render_background
    if camera_fname == '':
        camera_fname = op.join(mu.get_user_fol(), 'camera', '{}.pkl'.format(bpy.context.scene.camera_files))
    current_frame = bpy.context.scene.frame_current
    camera_fnames = [camera_fname] if isinstance(camera_fname, str) else camera_fname
    if image_name == '':
        images_names = ['{}_{}.png'.format(mu.namebase(camera_fname).replace('camera', ''), current_frame)
                        for camera_fname in  camera_fnames]
        images_names = [n[1:] if n.startswith('_') else n for n in images_names]
    else:
        images_names = [image_name] if isinstance(image_name, str) else image_name
    image_fol = bpy.path.abspath(bpy.context.scene.output_path) if image_fol == '' else image_fol
    image_fname = op.join(image_fol, images_names[0])
    print('Image quality: {}'.format(bpy.context.scene.render.resolution_percentage))
    print("Rendering...")
    if not bpy.context.scene.render_background:
        for image_name, camera_fname in zip(images_names, camera_fnames):
            print('file name: {}'.format(image_fname))
            bpy.context.scene.render.filepath = image_fname
            if overwrite or len(glob.glob('{}.*'.format(bpy.context.scene.render.filepath))) == 0:
                _addon().load_camera(camera_fname)
                _addon().change_to_rendered_brain()
                if hide_subcorticals:
                    _addon().show_hide_sub_corticals()
                bpy.ops.render.render(write_still=True)
        print("Finished")
        return image_fname
    else:
        camera_fnames = ','.join(camera_fnames)
        images_names = ','.join(images_names)
        render_func = partial(render_in_background, image_name=images_names, image_fol=image_fol,
                              camera_fname=camera_fnames, hide_subcorticals=hide_subcorticals, overwrite=overwrite)
        put_func_in_queue(render_func)
        if queue_len() == 1:
            run_func_in_queue()


def render_in_background(image_name, image_fol, camera_fname, hide_subcorticals, overwrite=True):
    hide_subs_in_background = True if hide_subcorticals else bpy.context.scene.objects_show_hide_sub_cortical
    mu.change_fol_to_mmvt_root()
    electrode_marked = _addon().is_current_electrode_marked()
    script = 'src.mmvt_addon.scripts.render_image'
    cmd = '{} -m {} -s {} -a {} -i {} -o {} -q {} -b {} -c "{}"'.format(
        bpy.context.scene.python_cmd, script, mu.get_user(), bpy.context.scene.atlas,
        image_name, image_fol, bpy.context.scene.render.resolution_percentage,
        bpy.context.scene.bipolar, camera_fname) + \
        ' --hide_lh {} --hide_rh {} --hide_subs {} --show_elecs {} --curr_elec {} --show_only_lead {}'.format(
        bpy.context.scene.objects_show_hide_lh, bpy.context.scene.objects_show_hide_rh,
        hide_subs_in_background, bpy.context.scene.show_hide_electrodes,
        bpy.context.scene.electrodes if electrode_marked else None,
        bpy.context.scene.show_only_lead if electrode_marked else None) + \
        ' --show_connections {}  --interactive 0  --overwrite {}'.format(
            _addon().connections_visible(), overwrite)
    print('Running {}'.format(cmd))
    RenderingMakerPanel.background_rendering = True
    mu.save_blender_file()
    _, RenderingMakerPanel.render_in_queue = mu.run_command_in_new_thread(
        cmd, read_stderr=False, read_stdin=False, stdout_func=reading_from_rendering_stdout_func)
    # mu.run_command_in_new_thread(cmd, queues=False)


def save_image(image_type='image', view_selected=True, index=-1):
    if index == -1:
        fol = bpy.path.abspath(bpy.context.scene.output_path)
        files = [mu.namebase(f) for f in glob.glob(op.join(fol, '{}*.png'.format(image_type)))]
        index = max([int(re.findall('\d+', f)[0]) for f in files]) + 1 if len(files) > 0 else 0

    if not _addon().is_solid():
        _addon().change_to_solid_brain()
    exit_from_camera_view()
    index = bpy.context.scene.frame_current if index == -1 else index
    mu.show_only_render(True)
    fol = bpy.path.abspath(bpy.context.scene.output_path)
    image_name = op.join(fol, '{}_{}.png'.format(image_type, index))
    print('Image saved in {}'.format(image_name))
    bpy.context.scene.render.filepath = image_name
    view3d_context = mu.get_view3d_context()
    if view_selected:
        _addon().view_all()
        # mu.select_all_brain(True)
        # bpy.ops.view3d.camera_to_view_selected(view3d_context)
        # mu.view_selected()
    bpy.ops.render.opengl(view3d_context, write_still=True)
    # if view_selected:
    #     _addon().zoom(1)
    return image_name
    # image_context = mu.get_image_context()
    # bpy.ops.image.save_as({'area': image_context},  # emulate an imageEditor
    #                       'INVOKE_DEFAULT',  # invoke the operator
    #                       copy=True,
    #                       filepath=image_name)  # export it to this location


def queue_len():
    return len(RenderingMakerPanel.queue.queue)


def put_func_in_queue(func, pop_immediately=False):
    try:
        logging.info('in put_func_in_queue')
        RenderingMakerPanel.queue.put((RenderingMakerPanel.item_id, func, pop_immediately))
        RenderingMakerPanel.item_id += 1
    except:
        print(traceback.format_exc())
        logging.error('Error in put_func_in_queue!')
        logging.error(traceback.format_exc())


def run_func_in_queue():
    try:
        logging.info('run_func_in_queue')
        (_, func, pop_immediately) = RenderingMakerPanel.queue.queue[0]
        func()
        if pop_immediately:
            pop_from_queue()
    except:
        print(traceback.format_exc())
        logging.error('Error in run_func_in_queue!')
        logging.error(traceback.format_exc())


def pop_from_queue():
    try:
        return RenderingMakerPanel.queue.get()
    except:
        print(traceback.format_exc())
        logging.error('Error in pop_from_queue!')
        logging.error(traceback.format_exc())


def update_camera_files():
    camera_files = glob.glob(op.join(mu.get_user_fol(), 'camera', '*camera*.pkl'))
    if len(camera_files) > 0:
        files_names = [mu.namebase(fname) for fname in camera_files]
        if _addon().is_inflated():
            files_names = [name for name in files_names if 'inf' in name]
            files_names.append('camera')
        else:
            files_names = [name for name in files_names if 'inf' not in name]
        items = [(c, c, '', ind) for ind, c in enumerate(files_names)]
        bpy.types.Scene.camera_files = bpy.props.EnumProperty(
            items=items, description="electrodes sources", update=camera_files_update)
        bpy.context.scene.camera_files = 'camera'


class RenderingMakerPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "Render"
    addon = None
    render_in_queue = None
    background_rendering = False
    queue = None
    item_id = 0

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
    grab_camera(overwrite=False)
    update_camera_files()
    bpy.context.scene.in_camera_view = False
    bpy.context.scene.save_selected_view = True
    # bpy.context.scene.lighting = 1.0
    RenderingMakerPanel.queue = PriorityQueue()
    mu.make_dir(op.join(mu.get_user_fol(), 'logs'))
    logging.basicConfig(
        filename=op.join(mu.get_user_fol(), 'logs', 'reander_panel.log'),
        level=logging.DEBUG, format='%(asctime)-15s %(levelname)8s %(name)s %(message)s')
    register()


def register():
    try:
        unregister()
        bpy.utils.register_class(RenderingMakerPanel)
        bpy.utils.register_class(RenderAllFigures)
        bpy.utils.register_class(SaveImage)
        bpy.utils.register_class(CameraMode)
        bpy.utils.register_class(GrabCamera)
        bpy.utils.register_class(LoadCamera)
        bpy.utils.register_class(MirrorCamera)
        bpy.utils.register_class(RenderFigure)
        bpy.utils.register_class(RenderPerspectives)
        bpy.utils.register_class(CombinePerspectives)
        # print('Render Panel was registered!')
    except:
        print("Can't register Render Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(RenderingMakerPanel)
        bpy.utils.unregister_class(RenderAllFigures)
        bpy.utils.unregister_class(SaveImage)
        bpy.utils.unregister_class(CameraMode)
        bpy.utils.unregister_class(GrabCamera)
        bpy.utils.unregister_class(LoadCamera)
        bpy.utils.unregister_class(MirrorCamera)
        bpy.utils.unregister_class(RenderFigure)
        bpy.utils.unregister_class(RenderPerspectives)
        bpy.utils.unregister_class(CombinePerspectives)
    except:
        pass
        # print("Can't unregister Render Panel!")
