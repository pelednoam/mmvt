import bpy
import os.path as op


def run(mmvt):
    remove_materials(mmvt)
    new_mats_list = ['Helmet_map_mat', 'unselected_label_Mat_cortex', 'unselected_label_Mat_subcortical']
    materials_names = [m.name for m in bpy.data.materials]
    print([mat not in materials_names for mat in new_mats_list])
    if any([mat not in materials_names for mat in new_mats_list]):
        print('Import new materials!')
        import_new_materials(mmvt)

    labels = bpy.data.objects['Cortex-lh'].children + bpy.data.objects['Cortex-rh'].children + \
             bpy.data.objects['Cortex-inflated-lh'].children + bpy.data.objects['Cortex-inflated-rh'].children
    subcorticals = bpy.data.objects['Subcortical_structures'].children
    ret = fix_objects_material(labels, 'unselected_label_Mat_cortex') and \
          fix_objects_material(subcorticals, 'unselected_label_Mat_subcortical')
    eeg_helmet = bpy.data.objects.get('eeg_helmet', None)
    if eeg_helmet is not None:
        eeg_helmet.active_material = bpy.data.materials['Helmet_map_mat']

    if not ret:
        remove_materials(mmvt)
        bpy.ops.wm.save_mainfile()
        print('!!!!! Restart Blender !!!!!')
        # bpy.ops.wm.quit_blender()


def fix_objects_material(objects, material_name):
    materials_names = [m.name for m in bpy.data.materials]
    ret = True
    for obj in objects:
        if obj.name + '_Mat' in materials_names:
            # print(obj.name + '_Mat')
            cur_mat = bpy.data.materials[obj.name + '_Mat']
            obj.active_material = cur_mat
        else:
            if material_name in materials_names:
                obj.active_material = bpy.data.materials[material_name].copy()
                obj.active_material.name = obj.name + '_Mat'
                cur_mat = obj.active_material
        try:
            cur_mat.node_tree.nodes["RGB"].outputs[0].default_value = (1, 1, 1, 1)
            # obj.active_material = cur_mat
        except:
            ret = False
            # pass
            # remove_materials()
            # print('Quit!')
            # bpy.ops.wm.quit_blender()
    return ret


def remove_materials(mmvt):
    # objs = bpy.data.objects['Cortex-lh'].children + bpy.data.objects['Cortex-rh'].children + \
    #        bpy.data.objects['Cortex-inflated-lh'].children + bpy.data.objects['Cortex-inflated-rh'].children + \
    #        bpy.data.objects['Subcortical_structures'].children
    # for obj in objs:
    #     if bpy.data.materials.get(obj.name + '_Mat') is not None:
    #         bpy.data.materials[obj.name + '_Mat'].use_fake_user = False
    #         bpy.data.materials[obj.name + '_Mat'].user_clear()

    # get new materials from empty brain
    new_mats_list = ['Helmet_map_mat', 'unselected_label_Mat_cortex', 'unselected_label_Mat_subcortical']
    for cur_mat in new_mats_list:
        if bpy.data.materials.get(cur_mat) is not None:
            bpy.data.materials[cur_mat].use_fake_user = False
            # bpy.data.materials[cur_mat].user_clear()
            bpy.data.materials.remove(bpy.data.materials[cur_mat], do_unlink=True)


def import_new_materials(mmvt):
    # empty_brain_path = op.join(mmvt_utils.get_parent_fol(mmvt_utils.get_user_fol()), 'empty_subject.blend')
    empty_brain_path = op.join(mmvt.utils.get_resources_dir(), 'empty_subject.blend')

    new_mats_list = ['Helmet_map_mat', 'unselected_label_Mat_cortex', 'unselected_label_Mat_subcortical']
    for cur_mat in new_mats_list:
        bpy.ops.wm.append(filepath=empty_brain_path, directory=empty_brain_path + '\\Material\\', filename=cur_mat)
        bpy.data.materials[cur_mat].use_fake_user = True
