import bpy
import os.path as op
import glob
import time
import traceback
import mmvt_utils as mu


def _addon():
    return MoshePanel.addon


def sizing_update(self, context):
    try:
        if bpy.data.objects.get('512pnts', None) is None:
            return
        for obj in bpy.data.objects['512pnts'].children:
            for ii in range(3):
                obj.scale[ii] = bpy.context.scene.lois_size
    except:
        print('Error in sizing update!')
        print(traceback.format_exc())


def moshe_group_ind_update(self, context):
    try:
        do_somthing()
    except:
        print('Error in sizing update!')
        print(traceback.format_exc())


def set_moshe_group_ind(self, value):
    print("setting value", value)


def update_something():
    pass

bpy.types.Scene.moshes_valid_lois_path = bpy.props.StringProperty(
    name="", default="", description="Define the path for the output files", subtype='FILE_PATH')
bpy.types.Scene.valid_lois_color = bpy.props.FloatVectorProperty(
    name="labels_color", subtype='COLOR', default=(0, 0.5, 0), min=0.0, max=1.0, description="color picker")
bpy.types.Scene.invalid_lois_color = bpy.props.FloatVectorProperty(
    name="labels_color", subtype='COLOR', default=(0, 0.5, 0), min=0.0, max=1.0, description="color picker")
bpy.types.Scene.lois_size = bpy.props.FloatProperty(min=0.1, max=2, default=0.75, step=0.1, update=sizing_update)
bpy.types.Scene.moshe_group_ind = bpy.props.IntProperty(default=0, min=0, description="Show group with index K", update=moshe_group_ind_update)


def do_somthing():
    import csv
    with open(bpy.context.scene.moshes_valid_lois_path, 'r') as f:
        reader = csv.reader(f)
        valid_lois_list = list(reader)

    materials_names = ['valid_lois_mat', 'invalid_lois_mat']
    materials_colors = [bpy.context.scene.valid_lois_color, bpy.context.scene.invalid_lois_color]
    for ind, mat_name in enumerate(materials_names):
        # Get material
        mat = bpy.data.materials.get(mat_name)
        if mat is None:
            # create material
            mat = bpy.data.materials.new(name=mat_name)
        mat.diffuse_color = materials_colors[ind]

    cur_valid_list_ind = min(bpy.context.scene.moshe_group_ind, len(valid_lois_list)-1)
    # set_moshe_group_ind(cur_valid_list_ind)
    cur_valid_list = valid_lois_list[cur_valid_list_ind]
    for obj in bpy.data.objects['512pnts'].children:
        cur_mat = bpy.data.materials['invalid_lois_mat']
        if any(obj.name in s for s in cur_valid_list):
            cur_mat = bpy.data.materials['valid_lois_mat']
        obj.active_material = cur_mat


def moshe_files_update(self, context):
    if MoshePanel.init:
        update_something()


def moshe_draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'lois_size', text='LOIs size')
    layout.prop(context.scene, 'moshes_valid_lois_path', text='input file')
    layout.prop(context.scene, 'valid_lois_color', text='Valid LOIs')
    layout.prop(context.scene, 'invalid_lois_color', text='Invalid LOIs')
    layout.operator(MosheButton.bl_idname, text="Paint LOIs", icon='COLOR')
    layout.prop(context.scene, 'moshe_group_ind', text='Group index:')


class MosheButton(bpy.types.Operator):
    bl_idname = "mmvt.moshe_button"
    bl_label = "Moshe button"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        do_somthing()
        return {'PASS_THROUGH'}


# class MosheNextButton(bpy.types.Operator):
#     bl_idname = "mmvt.moshe_next_button"
#     bl_label = "Moshe next button"
#     bl_options = {"UNDO"}
#
#     def invoke(self, context, event=None):
#         do_somthing()
#         return {'PASS_THROUGH'}


def start_timer():
    MoshePanel.is_playing = True
    MoshePanel.init_play = True
    if MoshePanel.first_time:
        print('Starting the timer!')
        MoshePanel.first_time = False
        bpy.ops.wm.modal_timer_operator()


class ModalTemplateTimerOperator(bpy.types.Operator):
    """Operator which runs its self from a timer"""
    bl_idname = "wm.modal_moshe_timer_operator"
    bl_label = "Modal Timer Operator"

    _timer = None
    _time = time.time()

    def modal(self, context, event):
        # First frame initialization:
        if MoshePanel.init_play:
            # Do some timer init
            pass

        if not MoshePanel.is_playing:
            return {'PASS_THROUGH'}

        if event.type in {'ESC'}:
            print('Stop!')
            self.cancel(context)
            return {'PASS_THROUGH'}

        if event.type == 'TIMER':
            if time.time() - self._time > MoshePanel.play_time_step:
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
        MoshePanel.is_playing = False
        bpy.context.scene.update()
        if self._timer:
            try:
                wm = context.window_manager
                wm.event_timer_remove(self._timer)
            except:
                pass


bpy.types.Scene.moshe_files = bpy.props.EnumProperty(items=[], description="tempalte files")


class MoshePanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "Moshe"
    addon = None
    init = False
    init_play = False
    is_playing = False
    first_time = True
    play_time_step = 1

    def draw(self, context):
        if MoshePanel.init:
            moshe_draw(self, context)


def init(addon):
    MoshePanel.addon = addon
    user_fol = mu.get_user_fol()
    # moshe_files = glob.glob(op.join(user_fol, 'moshe', 'moshe*.npz'))
    # if len(moshe_files) == 0:
    #     return None
    # files_names = [mu.namebase(fname)[len('moshe'):].replace('_', ' ') for fname in moshe_files]
    # moshe_items = [(c, c, '', ind) for ind, c in enumerate(files_names)]
    # bpy.types.Scene.moshe_files = bpy.props.EnumProperty(
    #     items=moshe_items, description="tempalte files",update=moshe_files_update)
    # bpy.context.scene.moshe_files = files_names[0]
    register()
    MoshePanel.init = True


def register():
    try:
        unregister()
        if bpy.data.objects.get('512pnts') is not None:
            bpy.utils.register_class(MoshePanel)
            bpy.utils.register_class(MosheButton)
            print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&Registering Moshes panel&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&7')
            # bpy.utils.register_class(ModalTemplateTimerOperator)
    except:
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&Can't register Moshes Panel!&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")


def unregister():
    try:
        bpy.utils.unregister_class(MoshePanel)
        bpy.utils.unregister_class(MosheButton)
        # bpy.utils.unregister_class(ModalTemplateTimerOperator)
    except:
        pass
