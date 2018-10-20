import bpy


def run(mmvt):
    plys_fol = mmvt.utils.get_real_fname('import_plys_fol')
    mmvt.data.import_rois(plys_fol, {'Subcortical_structures': plys_fol})
    mmvt.data.import_subcorticals(plys_fol, 'subcortical')
    mmvt.show_rois()
    mmvt.utils.deselect_all_objects()


bpy.types.Scene.import_plys_fol = bpy.props.StringProperty(subtype='DIR_PATH')


def init(mmvt):
    bpy.context.scene.import_plys_fol = mmvt.utils.get_user_fol()


def draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'import_plys_fol', text='plys folder')

