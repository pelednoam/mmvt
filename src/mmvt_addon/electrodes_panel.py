import bpy
import mmvt_utils as mu
import colors_utils as cu
import os.path as op
import glob
import time
import numpy as np
from collections import defaultdict

PARENT_OBJ = 'Deep_electrodes'


def _addon():
    return ElecsPanel.addon


def get_electrodes_names():
    parent = bpy.data.objects.get('Deep_electrodes')
    electrodes_names = [o.name for o in parent.children] if parent is not None else []
    return electrodes_names


def elc_size_update(self, context):
    try:
        elc = bpy.context.selected_objects[0]
        elc.scale[0] = bpy.context.scene.elc_size
        elc.scale[1] = bpy.context.scene.elc_size
        elc.scale[2] = bpy.context.scene.elc_size
    except:
        pass


def get_leads():
    return ElecsPanel.leads


def get_lead_electrodes(lead):
    electrodes = []
    for electrode in ElecsPanel.groups_electrodes[lead]:
        if not ElecsPanel.lookup.get(electrode, None) is None:
            electrodes.append(electrode)
    return electrodes


def get_electrode_lead(electrode):
    return ElecsPanel.groups.get(electrode, '')


def what_to_color_update(self, context):
    if ElecsPanel.init:
        _what_to_color_update()


def _what_to_color_update():
    pass


def leads_update(self, context):
    if ElecsPanel.init:
        _leads_update()


# @mu.profileit(op.join(mu.get_user_fol(), 'leads_update_profile'), 'cumtime')
def _leads_update():
    if _addon() is None or not ElecsPanel.init:
        return
    ElecsPanel.current_lead = bpy.context.scene.leads
    init_electrodes_list()
    _show_only_current_lead_update()


def set_current_electrode(electrode, lead=''):
    if lead == '':
        lead = get_electrode_lead(electrode)
    print('set_current_electrode: {}'.format(electrode))
    bpy.context.scene.leads = lead
    bpy.context.scene.electrodes = electrode


def electrodes_update(self, context):
    if ElecsPanel.init:
        _electrodes_update()


def _electrodes_update():
    if _addon() is None or not ElecsPanel.init:
        return
    # mu.print_traceback()
    prev_elect = ElecsPanel.current_electrode
    ElecsPanel.current_electrode = current_electrode = bpy.context.scene.electrodes
    bpy.context.scene.current_lead = ElecsPanel.groups[current_electrode]
    update_cursor()
    color_electrodes(current_electrode, prev_elect)
    # bpy.context.scene.objects.active = bpy.data.objects[current_electrode]
    selected_electrodes = [obj.name for obj in bpy.context.selected_objects if
                           mu.check_obj_type(obj.name) == mu.OBJ_TYPE_ELECTRODE]
    # Check if it's a new selection:
    # print(len(selected_electrodes), current_electrode, ElecsPanel.prev_electrodes)
    if len(selected_electrodes) == 1 and current_electrode not in ElecsPanel.prev_electrodes:
        # Clear and init prev_electrodes
        unselect_prev_electrode(ElecsPanel.prev_electrodes)
        ElecsPanel.prev_electrodes = set([current_electrode])
    # Check if this is a new electrodes where the shift is pressed
    elif len(selected_electrodes) > 1 and current_electrode not in ElecsPanel.prev_electrodes:
        # Add current electrode to prev_electrodes
        ElecsPanel.prev_electrodes.add(current_electrode)
    # Check if the user unselect one of the selected electrodes
    elif len(selected_electrodes) > 1 and current_electrode in ElecsPanel.prev_electrodes:
        bpy.data.objects[current_electrode].select = False
        unselect_prev_electrode([current_electrode])
        ElecsPanel.prev_electrodes.remove(current_electrode)
    else:
        clear_electrodes_selection()

    if not ElecsPanel.lookup is None:
        loc = ElecsPanel.lookup.get(current_electrode, None)
        if loc is None:
            print("Can't find {} in ElecsPanel.lookup!".format(current_electrode))
        else:
            print_electrode_loc(loc)
            if bpy.context.scene.color_lables:
                plot_labels_probs(loc)
    else:
        print('lookup table is None!')
    # mu.change_fcurves_colors(bpy.data.objects[current_electrode])
    mu.change_selected_fcurves_colors(mu.OBJ_TYPE_ELECTRODE)


def clear_electrodes_selection():
    unselect_prev_electrode(ElecsPanel.prev_electrodes)
    ElecsPanel.prev_electrodes = set()


def select_electrode(current_electrode):
    # group, elc1, elc2 = mu.elec_group_number(current_electrode, True)
    for elec in ElecsPanel.all_electrodes:
        bpy.data.objects[elec].select = elec == current_electrode
    # _addon().filter_electrode_func(bpy.context.scene.electrodes)


def electode_was_manually_selected(selected_electrode_name):
    if not ElecsPanel.init:
        return
    # print(selected_electrode_name, bpy.context.active_object, bpy.context.selected_objects)
    group = ElecsPanel.groups[selected_electrode_name]
    # It's enough to update the lead to update also the elecctrode, according to bpy.context.active_object
    bpy.context.scene.leads = group


def color_electrodes(current_electrode, prev_electrode):
    # bpy.context.scene.bipolar = '-' in current_electrode
    current_electrode_group = mu.elec_group(current_electrode, bpy.context.scene.bipolar)
    current_electrode_hemi = ElecsPanel.groups_hemi[current_electrode_group]
    prev_electrode_group = mu.elec_group(prev_electrode, bpy.context.scene.bipolar)
    prev_electrode_hemi = ElecsPanel.groups_hemi[prev_electrode_group]
    if current_electrode_hemi != prev_electrode_hemi:
        print('flip hemis! clear {}'.format(prev_electrode_hemi))
        _addon().clear_cortex([prev_electrode_hemi])
    color = bpy.context.scene.electrodes_color
    _addon().object_coloring(bpy.data.objects[current_electrode], tuple(color)) #cu.name_to_rgb('green'))
    if prev_electrode != current_electrode:
        _addon().object_coloring(bpy.data.objects[prev_electrode], (1, 1, 1, 1))


def is_current_electrode_marked():
    current_electrode_marked = False
    current_electrode_obj = bpy.data.objects.get(bpy.context.scene.electrodes, None)
    if not current_electrode_obj is None:
        current_electrode_marked = _addon().get_obj_color(current_electrode_obj)[:3] == tuple(
            bpy.context.scene.electrodes_color)
    return current_electrode_marked


def print_electrode_loc(loc):
    # print('{}:'.format(ElecsPanel.current_electrode))
    # for subcortical_name, subcortical_prob in zip(loc['subcortical_rois'], loc['subcortical_probs']):
    #     print('{}: {}'.format(subcortical_name, subcortical_prob))
    # for cortical_name, cortical_prob in zip(loc['cortical_rois'], loc['cortical_probs']):
    #     print('{}: {}'.format(cortical_name, cortical_prob))
    ElecsPanel.subcortical_rois = loc['subcortical_rois']
    ElecsPanel.subcortical_probs = loc['subcortical_probs']
    ElecsPanel.cortical_rois = loc['cortical_rois']
    ElecsPanel.cortical_probs = loc['cortical_probs']


def update_cursor():
    current_electrode_obj = bpy.data.objects[ElecsPanel.current_electrode]
    bpy.context.scene.cursor_location = current_electrode_obj.location
    _addon().freeview_panel.save_cursor_position()


def show_lh_update(self, context):
    if ElecsPanel.init:
        show_hide_hemi_electrodes('lh', bpy.context.scene.show_lh_electrodes)
        updade_lead_hemis()
        init_electrodes_list()


def show_rh_update(self, context):
    if ElecsPanel.init:
        show_hide_hemi_electrodes('rh', bpy.context.scene.show_rh_electrodes)
        updade_lead_hemis()
        init_electrodes_list()


def show_hide_hemi_electrodes(hemi, val):
    for elec_obj in ElecsPanel.parent.children:
        elec_hemi = get_elec_hemi(elec_obj.name)
        if elec_hemi == hemi:
            elec_obj.hide = not val
            elec_obj.hide_render = not val


def show_hide_electrodes(val):
    for elec_obj in ElecsPanel.parent.children:
        elec_obj.hide = not val
        elec_obj.hide_render = not val


def get_elec_hemi(elec_name):
    return ElecsPanel.groups_hemi[ElecsPanel.groups[elec_name]]


def updade_lead_hemis():
    if bpy.context.scene.show_lh_electrodes and bpy.context.scene.show_rh_electrodes:
        leads = ElecsPanel.sorted_groups['lh'] + ElecsPanel.sorted_groups['rh']
    elif bpy.context.scene.show_lh_electrodes and not bpy.context.scene.show_rh_electrodes:
        leads = ElecsPanel.sorted_groups['lh']
    elif not bpy.context.scene.show_lh_electrodes and bpy.context.scene.show_rh_electrodes:
        leads = ElecsPanel.sorted_groups['rh']
    else:
        leads = []
    init_leads_list(leads)


def color_the_relevant_lables(val):
    bpy.context.scene.color_lables = val


def show_only_the_current_lead(val):
    bpy.context.scene.show_only_lead = val


def set_electrodes_labeling_file(fname):
    bpy.context.scene.electrodes_labeling_files = mu.namebase(fname)


def color_labels_update(self, context):
    if ElecsPanel.init:
        if bpy.context.scene.color_lables:
            _electrodes_update()
        else:
            _addon().clear_cortex()
            _addon().clear_subcortical_regions()


def electrodes_labeling_files_update(self, context):
    # if ElecsPanel.init:
    # todo: How to get the other file names?
    # list(bpy.types.Scene.electrodes_labeling_files[1].items())[3][1]
    if ElecsPanel.init:
        _electrodes_labeling_files_update()


def _electrodes_labeling_files_update():
    labeling_fname = op.join(mu.get_user_fol(), 'electrodes', '{}.pkl'.format(
        bpy.context.scene.electrodes_labeling_files))
    ElecsPanel.electrodes_locs = mu.load(labeling_fname)
    ElecsPanel.lookup = create_lookup_table(ElecsPanel.electrodes_locs, ElecsPanel.all_electrodes)


def show_only_current_lead_update(self, context):
    if ElecsPanel.init:
        _show_only_current_lead_update()


def _show_only_current_lead_update():
    if bpy.context.scene.show_only_lead:
        bpy.context.scene.current_lead = ElecsPanel.groups[ElecsPanel.current_electrode]
        for elec_obj in ElecsPanel.parent.children:
            elec_obj.hide = elec_obj.hide_render = ElecsPanel.groups[elec_obj.name] != bpy.context.scene.current_lead
    else:
        show_hide_hemi_electrodes('lh', bpy.context.scene.show_lh_electrodes)
        show_hide_hemi_electrodes('rh', bpy.context.scene.show_rh_electrodes)
        # updade_lead_hemis()


def set_show_only_lead(val):
    bpy.context.scene.show_only_lead = val


# def show_elecs_hemi_update():
#     if ElecsPanel.init:
#         show_hide_hemi_electrodes('lh', bpy.context.scene.show_lh_electrodes)
#         show_hide_hemi_electrodes('rh', bpy.context.scene.show_rh_electrodes)
#         updade_lead_hemis()


def plot_labels_probs(elc):
    _addon().init_activity_map_coloring('FMRI')
    if bpy.context.scene.electrodes_what_to_color == 'probs':
        if len(elc['cortical_rois']) > 0:
            hemi = mu.get_obj_hemi(elc['cortical_rois'][0])
            if not hemi is None:
                labels_data = dict(data=elc['cortical_probs'], names=elc['cortical_rois'])
                _addon().set_colorbar_title('Electrodes probabilities')
                _addon().set_colormap('YlOrRd')
                _addon().labels_coloring_hemi(labels_data, ElecsPanel.faces_verts, hemi, 0,
                                              colors_min=0, colors_max=1)
            else:
                print("Can't get the rois hemi!")
        else:
            _addon().clear_cortex()
        _addon().clear_subcortical_regions()
        if len(elc['subcortical_rois']) > 0:
            for region, color in zip(elc['subcortical_rois'], elc['subcortical_colors'][:, :3]):
                _addon().color_subcortical_region(region, color)
    elif bpy.context.scene.electrodes_what_to_color == 'verts':
        if len(elc['cortical_indices']) > 0:
            hemi = bpy.data.objects[elc['hemi']]
            if not hemi is None:
                _addon().init_activity_map_coloring('FMRI', subcorticals=True)
                vertices_num = np.max(elc['cortical_indices']) + 1
                activity = np.ones((vertices_num, 4))
                activity[:, 0] = 0
                activity[elc['cortical_indices'], 0] = 1
                activity[elc['cortical_indices'], 1:] = np.tile(
                    cu.name_to_rgb('blue'), (len(elc['cortical_indices']), 1))
                print('Plot {} vertices with blue'.format(len(elc['cortical_indices'])))
                _addon().activity_map_obj_coloring(hemi, activity, ElecsPanel.faces_verts[elc['hemi']], 0, True)
            else:
                print("Can't get the elec's hemi!")
        else:
            _addon().clear_cortex()
            print('No cortical vertices for {}'.format(elc['name']))


def unselect_prev_electrode(prev_electrodes):
    for prev_electrode in prev_electrodes:
        prev_elc = bpy.data.objects.get(prev_electrode)
        if not prev_elc is None:
            _addon().de_select_electrode_and_sensor(prev_elc, False)
        # prev_elc.select = False


def elecs_draw(self, context):
    layout = self.layout
    layout.prop(context.scene, "electrodes_labeling_files", text="")
    row = layout.row(align=True)
    row.operator(PrevLead.bl_idname, text="", icon='PREV_KEYFRAME')
    row.prop(context.scene, "leads", text="")
    row.operator(NextLead.bl_idname, text="", icon='NEXT_KEYFRAME')
    row = layout.row(align=True)
    row.operator(PrevElectrode.bl_idname, text="", icon='PREV_KEYFRAME')
    row.prop(context.scene, "electrodes", text="")
    row.operator(NextElectrode.bl_idname, text="", icon='NEXT_KEYFRAME')
    layout.prop(context.scene, 'show_only_lead', text="Show only the current lead")
    layout.prop(context.scene, 'color_lables', text="Color the relevant lables")
    # layout.label(text='What to color: ')
    if bpy.context.scene.color_lables:
        layout.prop(context.scene, 'electrodes_what_to_color', text='What to color', expand=True)
    row = layout.row(align=True)
    row.prop(context.scene, "show_lh_electrodes", text="Left hemi")
    row.prop(context.scene, "show_rh_electrodes", text="Right hemi")
    row = layout.row(align=True)
    row.operator(ColorElectrodes.bl_idname, text="Color electrodes")
    row.prop(context.scene, 'electrodes_color', text='')

    if not bpy.context.scene.listen_to_keyboard:
        layout.operator(KeyboardListener.bl_idname, text="Listen to keyboard", icon='COLOR_GREEN')
    else:
        layout.operator(KeyboardListener.bl_idname, text="Stop listen to keyboard", icon='COLOR_RED')
        box = layout.box()
        col = box.column()
        mu.add_box_line(col, 'Left', 'Previous electrodes')
        mu.add_box_line(col, 'Right', 'Next electrodes')
        mu.add_box_line(col, 'Down', 'Previous lead')
        mu.add_box_line(col, 'Up', 'Next lead')
    if len(ElecsPanel.subcortical_rois) > 0 or len(ElecsPanel.cortical_rois) > 0:
        box = layout.box()
        col = box.column()
        for subcortical_name, subcortical_prob in zip(ElecsPanel.subcortical_rois, ElecsPanel.subcortical_probs):
            mu.add_box_line(col, subcortical_name, '{:.2f}'.format(subcortical_prob), 0.8)
        for cortical_name, cortical_prob in zip(ElecsPanel.cortical_rois, ElecsPanel.cortical_probs):
            mu.add_box_line(col, cortical_name, '{:.2f}'.format(cortical_prob), 0.8)
    # layout.operator(ElectrodesViewOne.bl_idname, text="View1", icon='COLOR_GREEN')
    # layout.operator(ElectrodesViewTwo.bl_idname, text="View2", icon='COLOR_GREEN')
    layout.prop(context.scene, "elc_size", text="")
    layout.operator(ClearElectrodes.bl_idname, text="Clear", icon='PANEL_CLOSE')

    # Color picker:
    # row = layout.row(align=True)
    # row.label(text='Selected electrode color:')
    # row = layout.row(align=True)
    # row.label(text='             ')
    # row.prop(context.scene, 'electrodes_color', text='')
    # row.label(text='             ')



class ElectrodesViewOne(bpy.types.Operator):
    bl_idname = 'mmvt.electrodes_view_onw'
    bl_label = 'elecetrodes view one'
    bl_options = {'UNDO'}

    def invoke(self, context, event=None):
        for elc in bpy.data.objects['Deep_electrodes'].children:
            elc.hide =  elc.name not in ['RPI1', 'RPI2', 'RAT6', 'RAT7', 'RAT8', 'RAT8']
        return {'FINISHED'}

class ElectrodesViewTwo(bpy.types.Operator):
    bl_idname = 'mmvt.electrodes_view_two'
    bl_label = 'elecetrodes view two'
    bl_options = {'UNDO'}

    def invoke(self, context, event=None):
        for elc in bpy.data.objects['Deep_electrodes'].children:
            elc.hide = elc.name[:3] not in ['RPI', 'RAT', 'RAI', 'RMT', 'RPT', 'LAI', 'LPI', 'LAT', 'LMT', 'LPT']
        return {'FINISHED'}


class ClearElectrodes(bpy.types.Operator):
    bl_idname = 'mmvt.electrodes_clear'
    bl_label = 'clearElectrodes'
    bl_options = {'UNDO'}

    def invoke(self, context, event=None):
        _addon().clear_colors()
        return {'FINISHED'}


class NextElectrode(bpy.types.Operator):
    bl_idname = 'mmvt.next_electrode'
    bl_label = 'nextElectrodes'
    bl_options = {'UNDO'}

    def invoke(self, context, event=None):
        next_electrode()
        return {'FINISHED'}


def next_electrode():
    # index = ElecsPanel.electrodes.index(bpy.context.scene.electrodes)
    electrode_lead = mu.elec_group(bpy.context.scene.electrodes, bpy.context.scene.bipolar)
    lead_electrodes = ElecsPanel.groups_electrodes[electrode_lead]
    index = lead_electrodes.index(bpy.context.scene.electrodes)
    if index < len(lead_electrodes) - 1:
        next_elc = lead_electrodes[index + 1]
    else:
        next_elc = lead_electrodes[0]
    bpy.context.scene.electrodes = next_elc


class PrevElectrode(bpy.types.Operator):
    bl_idname = 'mmvt.prev_electrode'
    bl_label = 'prevElectrodes'
    bl_options = {'UNDO'}

    def invoke(self, context, event=None):
        prev_electrode()
        return {'FINISHED'}


def prev_electrode():
    electrode_lead = mu.elec_group(bpy.context.scene.electrodes, bpy.context.scene.bipolar)
    lead_electrodes = ElecsPanel.groups_electrodes[electrode_lead]
    index = lead_electrodes.index(bpy.context.scene.electrodes)
    if index > 0:
        prev_elc = lead_electrodes[index - 1]
    else:
        prev_elc = lead_electrodes[-1]
    bpy.context.scene.electrodes = prev_elc


class NextLead(bpy.types.Operator):
    bl_idname = 'mmvt.next_lead'
    bl_label = 'nextLead'
    bl_options = {'UNDO'}

    def invoke(self, context, event=None):
        next_lead()
        return {'FINISHED'}


def next_lead():
    index = ElecsPanel.leads.index(bpy.context.scene.leads)
    next_lead = ElecsPanel.leads[index + 1] if index < len(ElecsPanel.leads) - 1 else ElecsPanel.leads[0]
    bpy.context.scene.leads = next_lead


class PrevLead(bpy.types.Operator):
    bl_idname = 'mmvt.prev_lead'
    bl_label = 'prevLead'
    bl_options = {'UNDO'}

    def invoke(self, context, event=None):
        prev_lead()
        return {'FINISHED'}


def prev_lead():
    index = ElecsPanel.leads.index(bpy.context.scene.leads)
    prev_lead = ElecsPanel.leads[index - 1] if index > 0 else ElecsPanel.leads[-1]
    bpy.context.scene.leads = prev_lead


class ColorElectrodes(bpy.types.Operator):
    bl_idname = 'mmvt.color_electrodes'
    bl_label = 'color_electrodes'
    bl_options = {'UNDO'}

    def invoke(self, context, event=None):
        for elc_obj in ElecsPanel.parent.children:
            if not elc_obj.hide:
                _addon().object_coloring(elc_obj, bpy.context.scene.electrodes_color)
        return {'FINISHED'}


class KeyboardListener(bpy.types.Operator):
    bl_idname = 'mmvt.keyboard_listener'
    bl_label = 'keyboard_listener'
    bl_options = {'UNDO'}
    press_time = time.time()
    funcs = {'LEFT_ARROW':prev_electrode, 'RIGHT_ARROW':next_electrode, 'UP_ARROW':prev_lead, 'DOWN_ARROW':next_lead}

    def modal(self, context, event):
        if time.time() - self.press_time > 1 and bpy.context.scene.listen_to_keyboard and \
                event.type not in ['TIMER', 'MOUSEMOVE', 'WINDOW_DEACTIVATE', 'INBETWEEN_MOUSEMOVE', 'TIMER_REPORT', 'NONE']:
            self.press_time = time.time()
            # print(event.type)
            if event.type in KeyboardListener.funcs.keys():
                KeyboardListener.funcs[event.type]()
            else:
                pass
        return {'PASS_THROUGH'}

    def invoke(self, context, event=None):
        if not bpy.context.scene.listener_is_running:
            context.window_manager.modal_handler_add(self)
            bpy.context.scene.listener_is_running = True
            _addon().show_electrodes()
            _leads_update()
        bpy.context.scene.listen_to_keyboard = not bpy.context.scene.listen_to_keyboard
        return {'RUNNING_MODAL'}


bpy.types.Scene.show_only_lead = bpy.props.BoolProperty(
    default=False, description="Show only the current lead", update=show_only_current_lead_update)
bpy.types.Scene.color_lables = bpy.props.BoolProperty(
    default=False, description="Color the relevant lables", update=color_labels_update)
bpy.types.Scene.show_lh_electrodes = bpy.props.BoolProperty(
    default=True, description="Left Hemi", update=show_lh_update)
bpy.types.Scene.show_rh_electrodes = bpy.props.BoolProperty(
    default=True, description="Right Hemi", update=show_rh_update)
bpy.types.Scene.listen_to_keyboard = bpy.props.BoolProperty(default=False)
bpy.types.Scene.listener_is_running = bpy.props.BoolProperty(default=False)
bpy.types.Scene.current_lead = bpy.props.StringProperty()
bpy.types.Scene.electrodes_color = bpy.props.FloatVectorProperty(
    name="object_color", subtype='COLOR', default=(0, 0.5, 0), min=0.0, max=1.0, description="color picker")
    # size=2, subtype='COLOR_GAMMA', min=0, max=1)
bpy.types.Scene.electrodes_labeling_files = bpy.props.EnumProperty(
    items=[], description='Labeling files', update=electrodes_labeling_files_update)
bpy.types.Scene.electrodes = bpy.props.EnumProperty(
    items=[], description="electrodes", update=electrodes_update)
bpy.types.Scene.leads = bpy.props.EnumProperty(
    items=[], description="leads", update=leads_update)
bpy.types.Scene.electrodes_what_to_color = bpy.props.EnumProperty(
    items=[('probs', 'probabilities', '', 1), ('verts', 'vertices', '', 2)], description="what to color",
    update=what_to_color_update)

bpy.types.Scene.elc_size = bpy.props.FloatProperty(description="", update=elc_size_update)




class ElecsPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "Electrodes"
    addon, parent = None, None
    init = False
    electrodes, leads = [], []
    current_electrode = ''
    prev_electrodes = set()
    electrodes_locs, lookup = None, None
    subcortical_rois, subcortical_probs = [], []
    cortical_rois, cortical_probs = [], []
    groups_electrodes, groups, groups_hemi = [], {}, {}
    sorted_groups = {'rh':[], 'lh':[]}
    bpy.context.scene.elc_size = 1

    def draw(self, context):
        elecs_draw(self, context)


def init(addon):
    import shutil
    ElecsPanel.addon  = addon
    ElecsPanel.parent = bpy.data.objects.get('Deep_electrodes')
    if ElecsPanel.parent is None or len(ElecsPanel.parent.children) == 0:
        print("Can't register electrodes panel, no Deep_electrodes object")
        return
    mu.make_dir(op.join(mu.get_user_fol(), 'electrodes'))
    sorted_groups_fname = op.join(mu.get_user_fol(), 'electrodes', 'sorted_groups.pkl')
    if not op.isfile(sorted_groups_fname):
        # Try to get the file from the subject's root folder
        if op.isfile(op.join(mu.get_user_fol(), 'sorted_groups.pkl')):
            shutil.move(op.join(mu.get_user_fol(), 'sorted_groups.pkl'), sorted_groups_fname)
        else:
            print("Can't register electrodes panel, no sorted groups file")
            return
    # show_hide_electrodes(True)
    ElecsPanel.sorted_groups = mu.load(sorted_groups_fname)
    ElecsPanel.groups_hemi = create_groups_hemi_lookup(ElecsPanel.sorted_groups)
    ElecsPanel.all_electrodes = [el.name for el in ElecsPanel.parent.children]
    ElecsPanel.groups = create_groups_lookup_table(ElecsPanel.all_electrodes)
    ElecsPanel.groups_first_electrode = find_first_electrode_per_group(ElecsPanel.all_electrodes)
    ElecsPanel.groups_electrodes = create_groups_electrodes_lookup(ElecsPanel.all_electrodes)
    init_leads_list()
    init_electrodes_list()
    ret = init_electrodes_labeling(addon)
    if not ret:
        print('No electrodes labeling files.')
        return
    # addon.clear_colors_from_parent_childrens('Deep_electrodes')
    # addon.clear_cortex()
    bpy.context.scene.show_only_lead = False
    bpy.context.scene.listen_to_keyboard = False
    bpy.context.scene.listener_is_running = False
    bpy.context.scene.show_lh_electrodes = True
    bpy.context.scene.show_rh_electrodes = True
    _electrodes_labeling_files_update()
    if not ElecsPanel.electrodes_locs or not ElecsPanel.lookup:
        if not ElecsPanel.electrodes_locs:
            print("!!! Can't find electrodes labeling files in user/electrdes!")
        if not ElecsPanel.lookup:
            print('No electrodes lookup table!')
        print("!!! Can't plot electrodes' probabilties !!!")
    if not ElecsPanel.groups or not ElecsPanel.groups_first_electrode or not ElecsPanel.sorted_groups or \
        not ElecsPanel.groups_hemi or not ElecsPanel.groups_electrodes:
            print('Error in electrodes panel init!')
    else:
        register()
        ElecsPanel.init = True
        # print('Electrodes panel initialization completed successfully!')


def init_leads_list(leads=None):
    # ElecsPanel.leads = sorted(list(set([mu.elec_group(elc, bipolar) for elc in ElecsPanel.electrodes])))
    if leads is None:
        ElecsPanel.leads = ElecsPanel.sorted_groups['lh'] + ElecsPanel.sorted_groups['rh']
    else:
        ElecsPanel.leads = leads
    leads_items = [(lead, lead, '', ind) for ind, lead in enumerate(ElecsPanel.leads)]
    bpy.types.Scene.leads = bpy.props.EnumProperty(
        items=leads_items, description="leads", update=leads_update)
    bpy.context.scene.leads = ElecsPanel.current_lead = ElecsPanel.leads[0]


def init_electrodes_list():
    ElecsPanel.electrodes = ElecsPanel.groups_electrodes[ElecsPanel.current_lead]
    ElecsPanel.electrodes.sort(key=mu.natural_keys)
    electrodes_items = [(elec, elec, '', ind) for ind, elec in enumerate(ElecsPanel.electrodes)]
    bpy.types.Scene.electrodes = bpy.props.EnumProperty(
        items=electrodes_items, description="electrodes", update=electrodes_update)
    lead = ElecsPanel.groups[ElecsPanel.electrodes[0]]
    last_obj_name = bpy.context.active_object.name if bpy.context.active_object is not None else ''
    # mu.print_traceback()
    if last_obj_name != '' and last_obj_name in ElecsPanel.groups and ElecsPanel.groups[last_obj_name] == lead:
        bpy.context.scene.electrodes = ElecsPanel.current_electrode = last_obj_name
    else:
        bpy.context.scene.electrodes = ElecsPanel.current_electrode = ElecsPanel.groups_first_electrode[lead]


def init_electrodes_labeling(addon):
    #todo: this panel should work also without labeling file
    labling_files = find_elecrode_labeling_files()
    if len(labling_files) > 0:
        files_names = [mu.namebase(fname) for fname in labling_files if mu.load(fname)]
        labeling_items = [(c, c, '', ind) for ind, c in enumerate(files_names)]
        bpy.types.Scene.electrodes_labeling_files = bpy.props.EnumProperty(
            items=labeling_items, description='Labeling files', update=electrodes_labeling_files_update)
        bpy.context.scene.electrodes_labeling_files = files_names[0]
        # ElecsPanel.electrodes_locs = mu.load(labling_files[0])
        # ElecsPanel.lookup = create_lookup_table(ElecsPanel.electrodes_locs, ElecsPanel.electrodes)
    ElecsPanel.faces_verts = addon.get_faces_verts()
    return len(labling_files) > 0


def find_elecrode_labeling_files():
    files = []
    blender_electrodes_names = set([o.name for o in bpy.data.objects['Deep_electrodes'].children])
    labeling_template = '{}_{}_electrodes_cigar_r_*_l_*.pkl'.format(mu.get_user(), bpy.context.scene.atlas)
    labling_files = glob.glob(op.join(mu.get_user_fol(), 'electrodes', labeling_template))
    for labling_file in labling_files:
        try:
            d = mu.load(labling_file)
            electrodes_names = set([e['name'] for e in d])
            if len(electrodes_names - blender_electrodes_names) == 0:
                files.append(labling_file)
        except:
            print('Error reading {}'.format(labling_file))
            continue
    if len(files) == 0:
        print("Can't find any labeling file that match the electrodes names in{}!".format(
            op.join(mu.get_user_fol(), 'electrodes')))
    return files



def create_lookup_table(electrodes_locs, electrodes):
    lookup = {}
    if electrodes_locs is None:
        print('electrodes_locs is None!!!')
        return None
    for elc in electrodes:
        for electrode_loc in electrodes_locs:
            if electrode_loc['name'] == elc:
                lookup[elc] = electrode_loc
                break
    return lookup


def find_first_electrode_per_group(electrodes):
    groups = defaultdict(list)
    first_electrodes = {}
    bipolar = '-' in electrodes[0]
    for elc in electrodes:
        groups[mu.elec_group(elc, bipolar)].append(elc)
    for group, group_electrodes in groups.items():
        first_electrode = sorted(group_electrodes)[0]
        first_electrodes[group] = first_electrode
    return first_electrodes


def create_groups_lookup_table(electrodes):
    groups = {}
    bipolar = '-' in electrodes[0]
    for elc in electrodes:
        group = mu.elec_group(elc, bipolar)
        groups[elc] = group
    return groups


def create_groups_electrodes_lookup(electrodes):
    groups = defaultdict(list)
    bipolar = '-' in electrodes[0]
    for elc in electrodes:
        group = mu.elec_group(elc, bipolar)
        groups[group].append(elc)
    return groups


def create_groups_hemi_lookup(sorted_groups):
    groups_hemi = {}
    for hemi, groups in sorted_groups.items():
        for group in groups:
            groups_hemi[group] = hemi
    return groups_hemi


def register():
    try:
        unregister()
        bpy.utils.register_class(ElecsPanel)
        bpy.utils.register_class(NextElectrode)
        bpy.utils.register_class(PrevElectrode)
        bpy.utils.register_class(NextLead)
        bpy.utils.register_class(PrevLead)
        bpy.utils.register_class(ColorElectrodes)
        bpy.utils.register_class(KeyboardListener)
        bpy.utils.register_class(ClearElectrodes)
        bpy.utils.register_class(ElectrodesViewOne)
        bpy.utils.register_class(ElectrodesViewTwo)


        # print('Electrodes Panel was registered!')
    except:
        print("Can't register Electrodes Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(ElecsPanel)
        bpy.utils.unregister_class(NextElectrode)
        bpy.utils.unregister_class(PrevElectrode)
        bpy.utils.unregister_class(NextLead)
        bpy.utils.unregister_class(PrevLead)
        bpy.utils.unregister_class(ColorElectrodes)
        bpy.utils.unregister_class(KeyboardListener)
        bpy.utils.unregister_class(ClearElectrodes)
        bpy.utils.unregister_class(ElectrodesViewOne)
        bpy.utils.unregister_class(ElectrodesViewTwo)

    except:
        pass
        # print("Can't unregister Electrodes Panel!")


