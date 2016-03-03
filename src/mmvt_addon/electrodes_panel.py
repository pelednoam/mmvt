import bpy
import mmvt_utils as mu


def electrodes_update(self, context):
    unselect_current_electrode(ElecsPanel.current_electrode)
    ElecsPanel.addon.filter_electrode_func(bpy.context.scene.electrodes)
    ElecsPanel.current_electrode = bpy.context.scene.electrodes


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

    def draw(self, context):
        elecs_draw(self, context)


def init(addon):
    ElecsPanel.addon = addon
    parent = bpy.data.objects.get('Deep_electrodes')
    ElecsPanel.electrodes = [] if parent is None else [el.name for el in parent.children]
    ElecsPanel.electrodes.sort(key=mu.natural_keys)
    items = [(elec, elec, '', ind) for ind, elec in enumerate(ElecsPanel.electrodes)]
    bpy.types.Scene.electrodes = bpy.props.EnumProperty(
        items=items, description="electrodes", update=electrodes_update)#, get=get_electrodes_enum, set=set_electrodes_enum)
    bpy.context.scene.electrodes = ElecsPanel.electrodes[0]
    ElecsPanel.current_electrode = ElecsPanel.electrodes[0]
    register()
    print('Electrodes panel initialization completed successfully!')


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

