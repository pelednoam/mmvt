import bpy
import mmvt_utils as mu
import os.path as op
import glob


def electrodes_update(self, context):
    unselect_current_electrode(ElecsPanel.current_electrode)
    ElecsPanel.addon.filter_electrode_func(bpy.context.scene.electrodes)
    ElecsPanel.current_electrode = bpy.context.scene.electrodes
    if not ElecsPanel.lookup is None:
        loc = ElecsPanel.lookup[ElecsPanel.current_electrode]
        print('{}:'.format(ElecsPanel.current_electrode))
        for subcortical_name, subcortical_prob in zip(loc['subcortical_rois'], loc['subcortical_probs']):
            print('{}: {}'.format(subcortical_name, subcortical_prob))
        for cortical_name, cortical_prob in zip(loc['cortical_rois'], loc['cortical_probs']):
            print('{}: {}'.format(cortical_name, cortical_prob))


def plot_labels_probs(elc, loc):
    if len(loc['cortical_rois']) > 0:
        # todo: not sure the hemi will always be in the end of the labels' names
        hemi = loc['cortical_rois'][0][-2:]
        # if no matplotlib should calculate the colors offline :(

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

    def draw(self, context):
        elecs_draw(self, context)


def init(addon):
    ElecsPanel.addon = addon
    parent = bpy.data.objects.get('Deep_electrodes')
    ElecsPanel.electrodes = [] if parent is None else [el.name for el in parent.children]
    ElecsPanel.electrodes.sort(key=mu.natural_keys)
    items = [(elec, elec, '', ind) for ind, elec in enumerate(ElecsPanel.electrodes)]
    bpy.types.Scene.electrodes = bpy.props.EnumProperty(
        items=items, description="electrodes", update=electrodes_update)
    bpy.context.scene.electrodes = ElecsPanel.electrodes[0]
    ElecsPanel.current_electrode = ElecsPanel.electrodes[0]
    loc_files = glob.glob(op.join(mu.get_user_fol(), '{}_{}_electrodes*.pkl'.format(mu.get_user(), addon.atlas_name)))
    if len(loc_files) > 0:
        # todo: there could be 2 files, one for bipolar and one for non bipolar
        ElecsPanel.electrodes_locs = mu.load(loc_files[0])
        ElecsPanel.lookup = create_lookup_table(ElecsPanel.electrodes_locs, ElecsPanel.electrodes)
        ElecsPanel.labels_names, ElecsPanel.labels_vertices = mu.load(
            op.join(mu.get_user_fol(), 'labels_vertices_{}.pkl'.format(addon.atlas_name)))
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

