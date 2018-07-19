import bpy
import os.path as op


def run(mmvt):
    mu = mmvt.utils
    if len(bpy.context.selected_objects) != 1:
        return
    obj = bpy.context.selected_objects[0]
    if obj.animation_data is None:
        return
    curr_cond = bpy.context.scene.conditions_selection if \
        bpy.context.scene.selection_type == 'spec_cond' else None
    time_range = range(mmvt.get_max_time_steps())
    data, colors = mu.evaluate_fcurves(obj, time_range, curr_cond)
    output_fol = mu.make_dir(op.join(mu.get_user_fol(), 'export'))
    mu.save(data, op.join(output_fol, '{}_data.pkl'.format(obj.name)))