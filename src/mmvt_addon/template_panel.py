import bpy
import os.path as op
import glob
import time
import traceback
import mmvt_utils as mu


def _addon():
    return TemplatePanel.addon


def update_something():
    pass


def do_somthing():
    pass


def template_files_update(self, context):
    if TemplatePanel.init:
        update_something()


def template_draw(self, context):
    layout = self.layout
    layout.operator(TemplateButton.bl_idname, text="Do something", icon='ROTATE')


class TemplateButton(bpy.types.Operator):
    bl_idname = "mmvt.template_button"
    bl_label = "Template botton"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        do_somthing()
        return {'PASS_THROUGH'}


def start_timer():
    TemplatePanel.is_playing = True
    TemplatePanel.init_play = True
    if TemplatePanel.first_time:
        print('Starting the timer!')
        TemplatePanel.first_time = False
        bpy.ops.wm.modal_timer_operator()


class ModalTemplateTimerOperator(bpy.types.Operator):
    """Operator which runs its self from a timer"""
    bl_idname = "wm.modal_template_timer_operator"
    bl_label = "Modal Timer Operator"

    _timer = None
    _time = time.time()

    def modal(self, context, event):
        # First frame initialization:
        if TemplatePanel.init_play:
            # Do some timer init
            pass

        if not TemplatePanel.is_playing:
            return {'PASS_THROUGH'}

        if event.type in {'ESC'}:
            print('Stop!')
            self.cancel(context)
            return {'PASS_THROUGH'}

        if event.type == 'TIMER':
            if time.time() - self._time > TemplatePanel.play_time_step:
                self._time = time.time()
                try:
                    # do something
                    pass
                except:
                    print(traceback.format_exc())
                    print('Error in plotting at {}!'.format(self.limits))

        return {'PASS_THROUGH'}

    def execute(self, context):
        wm = context.window_manager
        self.cancel(context)
        self._timer = wm.event_timer_add(time_step=0.05, window=context.window)
        self._time = time.time()
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        TemplatePanel.is_playing = False
        bpy.context.scene.update()
        if self._timer:
            try:
                wm = context.window_manager
                wm.event_timer_remove(self._timer)
            except:
                pass


bpy.types.Scene.template_files = bpy.props.EnumProperty(items=[], description="template files")


class TemplatePanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "Template"
    addon = None
    init = False
    init_play = False
    is_playing = False
    first_time = True
    play_time_step = 1

    def draw(self, context):
        if TemplatePanel.init:
            template_draw(self, context)


def init(addon):
    TemplatePanel.addon = addon
    user_fol = mu.get_user_fol()
    template_files = glob.glob(op.join(user_fol, 'template', 'template*.npz'))
    if len(template_files) == 0:
        return None
    files_names = [mu.namebase(fname)[len('template'):].replace('_', ' ') for fname in template_files]
    template_items = [(c, c, '', ind) for ind, c in enumerate(files_names)]
    bpy.types.Scene.template_files = bpy.props.EnumProperty(
        items=template_items, description="tempalte files",update=template_files_update)
    bpy.context.scene.template_files = files_names[0]
    register()
    TemplatePanel.init = True


def register():
    try:
        unregister()
        bpy.utils.register_class(TemplatePanel)
        bpy.utils.register_class(TemplateButton)
        bpy.utils.register_class(ModalTemplateTimerOperator)
    except:
        print("Can't register Template Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(TemplatePanel)
        bpy.utils.unregister_class(TemplateButton)
        bpy.utils.unregister_class(ModalTemplateTimerOperator)
    except:
        pass
