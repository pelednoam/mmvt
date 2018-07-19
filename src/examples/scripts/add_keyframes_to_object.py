import bpy


def run(mmvt):
    mu = mmvt.utils
    if len(bpy.context.selected_objects) != 1:
        return
    cur_obj = bpy.context.selected_objects[0]
    obj_name = cur_obj.name
    input_fname = mu.get_real_fname('add_keyframes_to_object_data_fname')
    data = mu.load(input_fname)
    for cond_ind, (fcruve_name, cond_data) in enumerate(data.items()):
        cond_name = fcruve_name.split('_')[-1]
        # Set the values to zeros in the first and last frame for current object(current label)
        mu.insert_keyframe_to_custom_prop(cur_obj, obj_name + '_' + cond_name, 0, 1)
        mu.insert_keyframe_to_custom_prop(cur_obj, obj_name + '_' + cond_name, 0, len(cond_data) + 2)

        print('keyframing ' + obj_name + ' object in condition ' + cond_name)
        # For every time point insert keyframe to current object
        for ind, t in enumerate(cond_data):
            mu.insert_keyframe_to_custom_prop(cur_obj, obj_name + '_' + str(cond_name), t, ind + 2)
        # remove the orange keyframe sign in the fcurves window
        fcurves = bpy.data.objects[obj_name].animation_data.action.fcurves[cond_ind]
        mod = fcurves.modifiers.new(type='LIMITS')


bpy.types.Scene.add_keyframes_to_object_data_fname = bpy.props.StringProperty(subtype='FILE_PATH')


def draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'add_keyframes_to_object_data_fname', text='Data file')
