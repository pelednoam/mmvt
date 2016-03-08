import bpy
import mmvt_utils as mu
import os.path as op
import glob


def electrodes_update(self, context):
    if ElecsPanel.addon is None:
        return
    prev_electrode = ElecsPanel.current_electrode
    ElecsPanel.current_electrode = bpy.context.scene.electrodes
    unselect_current_electrode(prev_electrode)
    ElecsPanel.addon.filter_electrode_func(bpy.context.scene.electrodes)
    update_cursor()
    if ElecsPanel.groups[prev_electrode] != ElecsPanel.groups[ElecsPanel.current_electrode]:
        show_only_current_lead(self, context)
    if not ElecsPanel.lookup is None:
        loc = ElecsPanel.lookup[ElecsPanel.current_electrode]
        if bpy.context.scene.color_lables:
            plot_labels_probs(loc)
        print('{}:'.format(ElecsPanel.current_electrode))
        for subcortical_name, subcortical_prob in zip(loc['subcortical_rois'], loc['subcortical_probs']):
            print('{}: {}'.format(subcortical_name, subcortical_prob))
        for cortical_name, cortical_prob in zip(loc['cortical_rois'], loc['cortical_probs']):
            print('{}: {}'.format(cortical_name, cortical_prob))


def update_cursor():
    current_electrode_obj = bpy.data.objects[ElecsPanel.current_electrode]
    bpy.context.scene.cursor_location = current_electrode_obj.location
    ElecsPanel.addon.freeview_panel.save_cursor_position()


def show_only_current_lead(self, context):
    if bpy.context.scene.show_only_lead:
        for elec_obj in bpy.data.objects['Deep_electrodes'].children:
            elec_obj.hide = ElecsPanel.groups[elec_obj.name] != ElecsPanel.groups[ElecsPanel.current_electrode]
    else:
        for elec_obj in bpy.data.objects['Deep_electrodes'].children:
            elec_obj.hide = False


def plot_labels_probs(elc):
    ElecsPanel.addon.show_hide_hierarchy(do_hide=False, obj='Subcortical_meg_activity_map')
    ElecsPanel.addon.show_hide_hierarchy(do_hide=True, obj='Subcortical_fmri_activity_map')
    if len(elc['cortical_rois']) > 0:
        hemi = mu.get_obj_hemi(elc['cortical_rois'][0])
        if not hemi is None:
            # if no matplotlib should calculate the colors offline :(
            labels_data = dict(data=elc['cortical_probs'], colors=elc['cortical_colors'][:, :3], names=elc['cortical_rois'])
            ElecsPanel.addon.meg_labels_coloring_hemi(
                ElecsPanel.labels_names, ElecsPanel.labels_vertices, labels_data, ElecsPanel.faces_verts, hemi, 0)
        else:
            print("Can't get the rois hemi!")
    else:
        ElecsPanel.addon.clear_cortex()
    ElecsPanel.addon.clear_subcortical_regions()
    if len(elc['subcortical_rois']) > 0:
        for region, color in zip(elc['subcortical_rois'], elc['subcortical_colors'][:, :3]):
            ElecsPanel.addon.color_subcortical_region(region, color)


def unselect_current_electrode(cur_elc_name):
    cur_elc = bpy.data.objects.get(cur_elc_name)
    if not cur_elc is None:
        ElecsPanel.addon.de_select_electrode(cur_elc, False)


def elecs_draw(self, context):
    layout = self.layout
    row = layout.row(align=True)
    row.operator(PrevElectrode.bl_idname, text="", icon='PREV_KEYFRAME')
    row.prop(context.scene, "electrodes", text="")
    row.operator(NextElectrode.bl_idname, text="", icon='NEXT_KEYFRAME')
    layout.prop(context.scene, 'show_only_lead', text="Show only the current lead")
    layout.prop(context.scene, 'color_lables', text="Color the relevant lables")


class NextElectrode(bpy.types.Operator):
    bl_idname = 'ohad.next_electrode'
    bl_label = 'nextElectrodes'
    bl_options = {'UNDO'}

    def invoke(self, context, event=None):
        index = ElecsPanel.electrodes.index(bpy.context.scene.electrodes)
        if index < len(ElecsPanel.electrodes) - 1:
            next_elc = ElecsPanel.electrodes[index + 1]
            bpy.context.scene.electrodes = next_elc
        return {'FINISHED'}


class PrevElectrode(bpy.types.Operator):
    bl_idname = 'ohad.prev_electrode'
    bl_label = 'prevElectrodes'
    bl_options = {'UNDO'}

    def invoke(self, context, event=None):
        index = ElecsPanel.electrodes.index(bpy.context.scene.electrodes)
        if index > 0:
            prev_elc = ElecsPanel.electrodes[index - 1]
            bpy.context.scene.electrodes = prev_elc
        return {'FINISHED'}

bpy.types.Scene.show_only_lead = bpy.props.BoolProperty(
    default=False, description="Show only the current lead", update=show_only_current_lead)
bpy.types.Scene.color_lables = bpy.props.BoolProperty(
    default=False, description="Color the relevant lables")


class ElecsPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
    bl_label = "Electrodes localizator"
    addon = None
    electrodes = []
    current_electrode = ''
    electrodes_locs = None
    lookup = None
    groups = {}

    def draw(self, context):
        elecs_draw(self, context)


def init(addon):
    ElecsPanel.addon = addon
    parent = bpy.data.objects.get('Deep_electrodes')
    if parent is None or len(parent.children) == 0:
        print("Can't register electrodes panel, no Deep_electrodes object!")
        return
    ElecsPanel.electrodes = [] if parent is None else [el.name for el in parent.children]
    ElecsPanel.electrodes.sort(key=mu.natural_keys)
    items = [(elec, elec, '', ind) for ind, elec in enumerate(ElecsPanel.electrodes)]
    bpy.types.Scene.electrodes = bpy.props.EnumProperty(
        items=items, description="electrodes", update=electrodes_update)
    bpy.context.scene.electrodes = ElecsPanel.electrodes[0]
    ElecsPanel.current_electrode = ElecsPanel.electrodes[0]
    loc_files = glob.glob(op.join(mu.get_user_fol(), '{}_{}_electrodes*.pkl'.format(mu.get_user(), bpy.context.scene.atlas)))
    if len(loc_files) > 0:
        # todo: there could be 2 files, one for bipolar and one for non bipolar
        ElecsPanel.electrodes_locs = mu.load(loc_files[0])
        ElecsPanel.lookup = create_lookup_table(ElecsPanel.electrodes_locs, ElecsPanel.electrodes)
        ElecsPanel.labels_names, ElecsPanel.labels_vertices = mu.load(
            op.join(mu.get_user_fol(), 'labels_vertices_{}.pkl'.format(bpy.context.scene.atlas)))
        # todo: Should be done only once in the main addon
        ElecsPanel.faces_verts = addon.load_faces_verts()
        ElecsPanel.groups = create_groups_lookup_table(ElecsPanel.electrodes)
    addon.clear_filtering()
    register()
    print('Electrodes panel initialization completed successfully!')


def create_lookup_table(electrodes_locs, electrodes):
    lookup = {}
    for elc in electrodes:
        for electrode_loc in electrodes_locs:
            if electrode_loc['name'] == elc:
                lookup[elc] = electrode_loc
                break
    return lookup


def create_groups_lookup_table(electrodes):
    groups = {}
    for elc in electrodes:
        group, num = mu.elec_group_number(elc)
        groups[elc] = group
    return groups


def register():
    try:
        unregister()
        bpy.utils.register_class(ElecsPanel)
        bpy.utils.register_class(NextElectrode)
        bpy.utils.register_class(PrevElectrode)
        print('Electrodes Panel was registered!')
    except:
        print("Can't register Electrodes Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(ElecsPanel)
        bpy.utils.unregister_class(NextElectrode)
        bpy.utils.unregister_class(PrevElectrode)

    except:
        print("Can't unregister Electrodes Panel!")

