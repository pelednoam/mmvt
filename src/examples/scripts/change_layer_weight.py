import bpy

_mmvt = None
_default = None


def init(mmvt):
    global _mmvt, _default
    _mmvt = mmvt
    obj = mmvt.utils.get_hemis_objs()[0]
    bpy.context.scene.layer_weight = _default = \
        obj.active_material.node_tree.nodes['Layer Weight'].inputs['Blend'].default_value
    register()


def run(mmvt):
    set_layer_weight(mmvt)


def set_layer_weight(mmvt):
    for obj in mmvt.utils.get_hemis_objs():
        obj.active_material.node_tree.nodes['Layer Weight'].inputs['Blend'].default_value = \
            bpy.context.scene.layer_weight


def layer_weight_update(self, context):
    set_layer_weight(_mmvt)


class LayerWeightLoadDefault(bpy.types.Operator):
    bl_idname = "mmvt.layer_weight_load_default"
    bl_label = "layer_weight_load_default"
    bl_options = {"UNDO"}
    update_camera = True

    def invoke(self, context, event=None):
        bpy.context.scene.layer_weight = _default
        return {"FINISHED"}


bpy.types.Scene.layer_weight = bpy.props.FloatProperty(default=0.3, update=layer_weight_update)


def draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'layer_weight', text='layer weight')
    layout.operator(LayerWeightLoadDefault.bl_idname, text='Set Default', icon='SETTINGS')


def register():
    try:
        unregister()
        bpy.utils.register_class(LayerWeightLoadDefault)
    except:
        pass


def unregister():
    try:
        bpy.utils.unregister_class(LayerWeightLoadDefault)
    except:
        pass