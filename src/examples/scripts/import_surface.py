import bpy


def run(mmvt):
    mu = mmvt.utils
    mu.change_layer(mmvt.BRAIN_EMPTY_LAYER)
    layers_array = bpy.context.scene.layers
    for name in ['Functional maps', 'Subcortical_fmri_activity_map']:
        mmvt.data.create_empty_if_doesnt_exists(name, mmvt.BRAIN_EMPTY_LAYER, layers_array, 'Functional maps')

    ply_fname = mu.get_real_fname('import_surface_ply_fname')
    try:
        bpy.ops.object.select_all(action='DESELECT')
        obj_name = mu.namebase(ply_fname).split(sep='.')[0]
        surf_name = mu.namebase(ply_fname).split(sep='.')[1]
        obj_name = '{}_{}'.format(surf_name, obj_name)
        layer = mmvt.INFLATED_ACTIVITY_LAYER
        if bpy.data.objects.get(obj_name) is None:
            mmvt.data.load_ply(ply_fname, obj_name, layer=layer)
    except:
        print('Error in importing {}'.format(ply_fname))

    bpy.ops.object.select_all(action='DESELECT')


bpy.types.Scene.import_surface_ply_fname = bpy.props.StringProperty(subtype='FILE_PATH')


def draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'import_surface_ply_fname', text='ply')
