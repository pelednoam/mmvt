import bpy
import os.path as op
import glob


def run(mmvt):
    parent_name = 'Subcortical_test'
    mmvt.data.create_empty_if_doesnt_exists(parent_name,  mmvt.BRAIN_EMPTY_LAYER)
    plys_fol = mmvt.utils.get_real_fname('import_plys_fol')
    for ply_fname in glob.glob(op.join(plys_fol, '*.ply')):
        obj_name = mmvt.utils.namebase(ply_fname)
        bpy.ops.object.select_all(action='DESELECT')
        print(ply_fname)
        bpy.ops.import_mesh.ply(filepath=ply_fname)
        cur_obj = bpy.context.selected_objects[0]
        cur_obj.select = True
        bpy.ops.object.shade_smooth()
        cur_obj.scale = [0.1] * 3
        cur_obj.hide = False
        cur_obj.name = obj_name
        curMat = bpy.data.materials.get('{}_mat'.format(cur_obj.name))
        if curMat is None:
            curMat = bpy.data.materials['succortical_activity_Mat'].copy()
            curMat.name = '{}_mat'.format(cur_obj.name)
        cur_obj.active_material = bpy.data.materials[curMat.name]
        cur_obj.parent = bpy.data.objects[parent_name]


bpy.types.Scene.import_plys_fol = bpy.props.StringProperty(subtype='DIR_PATH')


def draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'import_plys_fol', text='plys folder')
