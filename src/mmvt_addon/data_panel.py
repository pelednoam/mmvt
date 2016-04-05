import bpy
import os.path as op
import glob
import numpy as np
import time
import mmvt_utils as mu


STAT_AVG, STAT_DIFF = range(2)

bpy.types.Scene.brain_imported = False
bpy.types.Scene.electrodes_imported = False
bpy.types.Scene.brain_data_exist = False
bpy.types.Scene.electrodes_data_exist = False


def import_brain():
    brain_layer = DataMakerPanel.addon.BRAIN_EMPTY_LAYER
    bpy.context.scene.layers = [ind == brain_layer for ind in range(len(bpy.context.scene.layers))]
    layers_array = bpy.context.scene.layers
    emptys_names = ['Functional maps', 'Subcortical_meg_activity_map', 'Subcortical_fmri_activity_map']
    for name in emptys_names:
        create_empty_if_doesnt_exists(name, brain_layer, layers_array, 'Functional maps')

    brain_layer = DataMakerPanel.addon.ACTIVITY_LAYER
    bpy.context.scene.layers = [ind == brain_layer for ind in range(len(bpy.context.scene.layers))]
    # for ii in range(len(bpy.context.scene.layers)):
    #     bpy.context.scene.layers[ii] = (ii == brain_layer)

    print("importing Hemispheres")
    # # for cur_val in bpy.context.scene.layers:
    # #     print(cur_val)
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    base_path = mu.get_user_fol()
    for ply_fname in glob.glob(op.join(base_path, '*.ply')):
        bpy.ops.object.select_all(action='DESELECT')
        print(ply_fname)
        obj_name = mu.namebase(ply_fname).split(sep='.')[0]
        if bpy.data.objects.get(obj_name) is None:
            bpy.ops.import_mesh.ply(filepath=op.join(base_path, ply_fname))
            cur_obj = bpy.context.selected_objects[0]
            cur_obj.select = True
            bpy.ops.object.shade_smooth()
            cur_obj.scale = [0.1] * 3
            cur_obj.hide = False
            cur_obj.name = obj_name
            cur_obj.active_material = bpy.data.materials['Activity_map_mat']
            cur_obj.parent = bpy.data.objects["Functional maps"]
            cur_obj.hide_select = True
            cur_obj.data.vertex_colors.new()
            print('did hide_select')

    bpy.ops.object.select_all(action='DESELECT')


def create_subcortical_activity_mat(name):
    cur_mat = bpy.data.materials['subcortical_activity_mat'].copy()
    cur_mat.name = name + '_Mat'


def import_subcorticals(base_path):
    empty_layer = DataMakerPanel.addon.BRAIN_EMPTY_LAYER
    brain_layer = DataMakerPanel.addon.ACTIVITY_LAYER

    bpy.context.scene.layers = [ind == empty_layer for ind in range(len(bpy.context.scene.layers))]
    layers_array = bpy.context.scene.layers
    emptys_names = ['Functional maps', 'Subcortical_meg_activity_map', 'Subcortical_fmri_activity_map']
    for name in emptys_names:
        create_empty_if_doesnt_exists(name, empty_layer, layers_array, 'Functional maps')
    bpy.context.scene.layers = [ind == brain_layer for ind in range(len(bpy.context.scene.layers))]

    print("importing Subcortical structures")
    # for cur_val in bpy.context.scene.layers:
    #     print(cur_val)
    #  print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    base_paths = [base_path] * 2 # Read the bast_path twice, for meg and fmri
    PATH_TYPE_SUB_MEG, PATH_TYPE_SUB_FMRI = range(2)
    for path_type, base_path in enumerate(base_paths):
        for ply_fname in glob.glob(op.join(base_path, '*.ply')):
            obj_name = mu.namebase(ply_fname)
            if path_type==PATH_TYPE_SUB_MEG and not bpy.data.objects.get('{}_meg_activity'.format(obj_name)) is None:
                continue
            if path_type==PATH_TYPE_SUB_FMRI and not bpy.data.objects.get('{}_fmri_activity'.format(obj_name)) is None:
                continue
            bpy.ops.object.select_all(action='DESELECT')
            print(ply_fname)
            bpy.ops.import_mesh.ply(filepath=op.join(base_path, ply_fname))
            cur_obj = bpy.context.selected_objects[0]
            cur_obj.select = True
            bpy.ops.object.shade_smooth()
            cur_obj.scale = [0.1] * 3
            cur_obj.hide = False
            cur_obj.name = obj_name

            if path_type == PATH_TYPE_SUB_MEG:
                cur_obj.name = '{}_meg_activity'.format(obj_name)
                curMat = bpy.data.materials.get('{}_mat'.format(cur_obj.name))
                if curMat is None:
                    # todo: Fix the succortical_activity_Mat to succortical_activity_mat
                    curMat = bpy.data.materials['succortical_activity_Mat'].copy()
                    curMat.name = '{}_mat'.format(cur_obj.name)
                cur_obj.active_material = bpy.data.materials[curMat.name]
                cur_obj.parent = bpy.data.objects['Subcortical_meg_activity_map']
            elif path_type == PATH_TYPE_SUB_FMRI:
                cur_obj.name = '{}_fmri_activity'.format(obj_name)
                if 'cerebellum' in cur_obj.name.lower():
                    cur_obj.active_material = bpy.data.materials['Activity_map_mat']
                else:
                    cur_obj.active_material = bpy.data.materials['subcortical_activity_mat']
                cur_obj.parent = bpy.data.objects['Subcortical_fmri_activity_map']
            else:
                print('import_subcorticals: Wrong path_type! Nothing to do...')
            cur_obj.hide_select = True
    bpy.ops.object.select_all(action='DESELECT')


class ImportBrain(bpy.types.Operator):
    bl_idname = "ohad.brain_importing"
    bl_label = "import2 brain"
    bl_options = {"UNDO"}
    current_root_path = ''
    brain_layer = DataMakerPanel.addon.BRAIN_EMPTY_LAYER

    def invoke(self, context, event=None):
        self.current_root_path = mu.get_user_fol() #bpy.path.abspath(bpy.context.scene.conf_path)
        print("importing ROIs")
        import_rois(self.current_root_path)
        import_brain()
        import_subcorticals(op.join(self.current_root_path, 'subcortical'))
        last_obj = context.active_object.name
        print('last obj is -' + last_obj)

        if bpy.data.objects.get(' '):
            bpy.data.objects[' '].select = True
            context.scene.objects.active = bpy.data.objects[' ']
        bpy.data.objects[last_obj].select = False
        DataMakerPanel.addon.set_appearance_show_rois_layer(bpy.context.scene, True)
        bpy.types.Scene.brain_imported = True
        print('cleaning up')
        for obj in bpy.data.objects['Subcortical_structures'].children:
            # print(obj.name)
            if obj.name[-1] == '1':
                obj.name = obj.name[0:-4]
        print('Brain importing is Finished ')

        return {"FINISHED"}


def create_empty_if_doesnt_exists(name, brain_layer, layers_array, parent_obj_name='Brain'):
    if bpy.data.objects.get(name) is None:
        layers_array[brain_layer] = True
        bpy.ops.object.empty_add(type='PLAIN_AXES', radius=1, view_align=False, location=(0, 0, 0), layers=layers_array)
        bpy.data.objects['Empty'].name = name
        if name != parent_obj_name:
            bpy.data.objects[name].parent = bpy.data.objects[parent_obj_name]


def import_rois(base_path):
    anatomy_inputs = {'Cortex-rh': op.join(base_path, '{}.pial.rh'.format(bpy.context.scene.atlas)),
                      'Cortex-lh': op.join(base_path, '{}.pial.lh'.format(bpy.context.scene.atlas)),
                      'Subcortical_structures': op.join(base_path, 'subcortical')}
    brain_layer = DataMakerPanel.addon.BRAIN_EMPTY_LAYER

    bpy.context.scene.layers = [ind == brain_layer for ind in range(len(bpy.context.scene.layers))]
    layers_array = bpy.context.scene.layers
    emptys_names = ["Brain", "Subcortical_structures", "Cortex-lh", "Cortex-rh"]
    for name in emptys_names:
        create_empty_if_doesnt_exists(name, brain_layer, layers_array)
    bpy.context.scene.layers = [ind == DataMakerPanel.addon.ROIS_LAYER for ind in range(len(bpy.context.scene.layers))]

    for anatomy_name, base_path in anatomy_inputs.items():
        current_mat = bpy.data.materials['unselected_label_Mat_cortex']
        if anatomy_name == 'Subcortical_structures':
            current_mat = bpy.data.materials['unselected_label_Mat_subcortical']
        for ply_fname in glob.glob(op.join(base_path, '*.ply')):
            new_obj_name = mu.namebase(ply_fname)
            if not bpy.data.objects.get(new_obj_name) is None:
                continue
            bpy.ops.object.select_all(action='DESELECT')
            print(ply_fname)
            bpy.ops.import_mesh.ply(filepath=op.join(base_path, ply_fname))
            cur_obj = bpy.context.selected_objects[0]
            cur_obj.select = True
            bpy.ops.object.shade_smooth()
            cur_obj.parent = bpy.data.objects[anatomy_name]
            cur_obj.scale = [0.1] * 3
            cur_obj.active_material = current_mat
            cur_obj.hide = False
            cur_obj.name = new_obj_name
            # time.sleep(0.3)
    bpy.ops.object.select_all(action='DESELECT')


class ImportRois(bpy.types.Operator):
    bl_idname = "ohad.roi_importing"
    bl_label = "import2 ROIs"
    bl_options = {"UNDO"}
    current_root_path = ''

    def invoke(self, context, event=None):
        self.current_root_path = mu.get_user_fol() #bpy.path.abspath(bpy.context.scene.conf_path)
        import_brain(self.current_root_path)
        return {"FINISHED"}


def create_and_set_material(obj):
    # curMat = bpy.data.materials['OrigPatchesMat'].copy()
    if obj.active_material is None or obj.active_material.name != obj.name + '_Mat':
        if obj.name + '_Mat' in bpy.data.materials:
            cur_mat = bpy.data.materials[obj.name + '_Mat']
        else:
            cur_mat = bpy.data.materials['Deep_electrode_mat'].copy()
            cur_mat.name = obj.name + '_Mat'
        # Wasn't it originally (0, 0, 1, 1)?
        cur_mat.node_tree.nodes["RGB"].outputs[0].default_value = (0, 0, 1, 1) # (0, 1, 0, 1)
        obj.active_material = cur_mat


def import_electrodes():
    # input_file = op.join(base_path, "electrodes.npz")
    bipolar = bpy.context.scene.bipolar
    input_file = op.join(mu.get_user_fol(), 'electrodes_{}positions.npz'.format('bipolar_' if bipolar else ''))

    print('Adding deep electrodes')
    f = np.load(input_file)
    print('loaded')

    deep_electrodes_layer = 1
    electrode_size = bpy.context.scene.electrode_radius
    layers_array = [False] * 20
    create_empty_if_doesnt_exists('Deep_electrodes', DataMakerPanel.addon.BRAIN_EMPTY_LAYER, layers_array, 'Deep_electrodes')

    # if bpy.data.objects.get("Deep_electrodes") is None:
    #     layers_array[BRAIN_EMPTY_LAYER] = True
    #     bpy.ops.object.empty_add(type='PLAIN_AXES', radius=1, view_align=False, location=(0, 0, 0), layers=layers_array)
    #     bpy.data.objects['Empty'].name = 'Deep_electrodes'

    layers_array = [False] * 20
    layers_array[deep_electrodes_layer] = True

    for (x, y, z), name in zip(f['pos'], f['names']):
        elc_name = name.astype(str)
        if not bpy.data.objects.get(elc_name) is None:
            continue
        print('creating {}: {}'.format(elc_name, (x, y, z)))
        mu.create_sphere((x * 0.1, y * 0.1, z * 0.1), electrode_size, layers_array, elc_name)
        cur_obj = bpy.data.objects[elc_name]
        cur_obj.select = True
        cur_obj.parent = bpy.data.objects['Deep_electrodes']
        # cur_obj.active_material = bpy.data.materials['Deep_electrode_mat']
        create_and_set_material(cur_obj)


class ImportElectrodes(bpy.types.Operator):
    bl_idname = "ohad.electrodes_importing"
    bl_label = "import2 electrodes"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        import_electrodes()
        bpy.types.Scene.electrodes_imported = True
        print('Electrodes importing is Finished ')
        return {"FINISHED"}


def insert_keyframe_to_custom_prop(obj, prop_name, value, keyframe):
    bpy.context.scene.objects.active = obj
    obj.select = True
    obj[prop_name] = value
    obj.keyframe_insert(data_path='[' + '"' + prop_name + '"' + ']', frame=keyframe)


def add_data_to_brain():
    base_path = mu.get_user_fol()
    source_files = [op.join(base_path, 'labels_data_lh.npz'), op.join(base_path, 'labels_data_rh.npz'),
                    op.join(base_path, 'sub_cortical_activity.npz')]
    print('Adding data to Brain')
    number_of_maximal_time_steps = -1
    obj_counter = 0
    for input_file in source_files:
        if not op.isfile(input_file):
            mu.message(None, '{} does not exist!'.format(input_file))
            continue
        f = np.load(input_file)
        print('{} loaded'.format(input_file))
        number_of_maximal_time_steps = max(number_of_maximal_time_steps, len(f['data'][0]))
        for obj_name, data in zip(f['names'], f['data']):
            # print('in label loop')
            obj_name = obj_name.astype(str)
            print(obj_name)
            cur_obj = bpy.data.objects[obj_name]
            # print('cur_obj name = '+cur_obj.name)

            for cond_ind, cond_str in enumerate(f['conditions']):
                # cond_str = str(cond_str)
                # if cond_str[1] == "'":
                #     cond_str = cond_str[2:-1]
                cond_str = cond_str.astype(str)
                # Set the values to zeros in the first and last frame for current object(current label)
                insert_keyframe_to_custom_prop(cur_obj, obj_name + '_' + cond_str, 0, 1)
                insert_keyframe_to_custom_prop(cur_obj, obj_name + '_' + cond_str, 0, len(f['data'][0]) + 2)

                print('keyframing ' + obj_name + ' object')
                # For every time point insert keyframe to current object
                for ind, timepoint in enumerate(data[:, cond_ind]):
                    # print('keyframing '+obj_name+' object')
                    insert_keyframe_to_custom_prop(cur_obj, obj_name + '_' + cond_str, timepoint, ind + 2)

                # remove the orange keyframe sign in the fcurves window
                fcurves = bpy.data.objects[obj_name].animation_data.action.fcurves[cond_ind]
                mod = fcurves.modifiers.new(type='LIMITS')
    try:
        bpy.ops.graph.previewrange_set()
    except:
        pass

    bpy.types.Scene.maximal_time_steps = number_of_maximal_time_steps
    # print(bpy.types.Scene.maximal_time_steps)

    # for obj in bpy.data.objects:
    #     try:
    #         if (obj.parent is 'Cortex-lh') or ((obj.parent is 'Cortex-rh') or (obj.parent is 'Subcortical_structures')):
    #             obj.select = True
    #         else:
    #             obj.select = False
    #     except:
    #         obj.select = False
    for obj in bpy.data.objects:
        obj.select = False
    if bpy.data.objects.get(' '):
        bpy.context.scene.objects.active = bpy.data.objects[' ']
    print('Finished keyframing!!')


def add_data_to_parent_brain_obj(self, stat=STAT_DIFF):
    base_path = mu.get_user_fol()
    brain_obj = bpy.data.objects['Brain']
    labels_data_file = 'labels_data_{hemi}.npz' if stat else 'labels_data_no_conds_{hemi}.npz'
    brain_sources = [op.join(base_path, labels_data_file.format(hemi=hemi)) for hemi in mu.HEMIS]
    subcorticals_obj = bpy.data.objects['Subcortical_structures']
    subcorticals_sources = [op.join(base_path, 'subcortical_meg_activity.npz')]
    add_data_to_parent_obj(self, brain_obj, brain_sources, stat)
    add_data_to_parent_obj(self, subcorticals_obj, subcorticals_sources, stat)


def add_data_to_parent_obj(self, parent_obj, source_files, stat):
    sources = {}
    parent_obj.animation_data_clear()
    for input_file in source_files:
        if not op.isfile(input_file):
            mu.message(self, "Can't load file {}!".format(input_file))
            continue
        print('loading {}'.format(input_file))
        f = np.load(input_file)
        for obj_name, data in zip(f['names'], f['data']):
            obj_name = obj_name.astype(str)
            if bpy.data.objects.get(obj_name) is None:
                continue
            if stat == STAT_AVG:
                data_stat = np.squeeze(np.mean(data, axis=1))
            elif stat == STAT_DIFF:
                data_stat = np.squeeze(np.diff(data, axis=1))
            else:
                data_stat = data
            sources[obj_name] = data_stat
    if len(sources) == 0:
        print('No sources in {}'.format(source_files))
    sources_names = sorted(list(sources.keys()))
    N = len(sources_names)
    T = len(sources[sources_names[0]]) + 2
    now = time.time()
    for obj_counter, source_name in enumerate(sources_names):
        mu.time_to_go(now, obj_counter, N, runs_num_to_print=10)
        data = sources[source_name]
        # Set the values to zeros in the first and last frame for Brain object
        insert_keyframe_to_custom_prop(parent_obj, source_name, 0, 1)
        insert_keyframe_to_custom_prop(parent_obj, source_name, 0, T)

        # For every time point insert keyframe to the main Brain object
        # If you want to delete prints make sure no sleep is needed
        # print('keyframing Brain object {}'.format(obj_name))
        for ind in range(data.shape[0]):
            # if len(data[ind]) == 2:
            # print('keyframing Brain object')
            insert_keyframe_to_custom_prop(parent_obj, source_name, data[ind], ind + 2)
            # print('keyframed')

        # remove the orange keyframe sign in the fcurves window
        fcurves = parent_obj.animation_data.action.fcurves[obj_counter]
        mod = fcurves.modifiers.new(type='LIMITS')

    if bpy.data.objects.get(' '):
        bpy.context.scene.objects.active = bpy.data.objects[' ']
    print('Finished keyframing the brain parent obj!!')


class AddDataToBrain(bpy.types.Operator):
    bl_idname = "ohad.brain_add_data"
    bl_label = "add_data2 brain"
    bl_options = {"UNDO"}
    current_root_path = ''

    def invoke(self, context, event=None):
        # self.current_root_path = bpy.path.abspath(bpy.context.scene.conf_path)
        add_data_to_brain()
        add_data_to_parent_brain_obj(self)
        bpy.types.Scene.brain_data_exist = True
        return {"FINISHED"}


class AddDataNoCondsToBrain(bpy.types.Operator):
    bl_idname = "ohad.brain_add_data_no_conds"
    bl_label = "add_data no conds brain"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        add_data_to_parent_brain_obj(self, None)
        bpy.types.Scene.brain_data_exist = True
        return {"FINISHED"}


def insert_keyframe_to_custom_prop(obj, prop_name, value, keyframe):
    bpy.context.scene.objects.active = obj
    obj.select = True
    obj[prop_name] = value
    obj.keyframe_insert(data_path='[' + '"' + prop_name + '"' + ']', frame=keyframe)


def add_data_to_electrodes(self, source_files):
    print('Adding data to Electrodes')
    for input_file in source_files:
        # todo: we don't need to load this twice (also in add_data_to_electrodes_obj
        f = np.load(input_file)
        print('{} loaded'.format(input_file))
        now = time.time()
        N = len(f['names'])
        for obj_counter, (obj_name, data) in enumerate(zip(f['names'], f['data'])):
            mu.time_to_go(now, obj_counter, N, runs_num_to_print=10)
            obj_name = obj_name.astype(str)
            # print(obj_name)
            if bpy.data.objects.get(obj_name, None) is None:
                print("{} doesn't exist!".format(obj_name))
                continue
            cur_obj = bpy.data.objects[obj_name]
            for cond_ind, cond_str in enumerate(f['conditions']):
                cond_str = cond_str.astype(str)
                # Set the values to zeros in the first and last frame for current object(current label)
                insert_keyframe_to_custom_prop(cur_obj, obj_name + '_' + cond_str, 0, 1)
                insert_keyframe_to_custom_prop(cur_obj, obj_name + '_' + cond_str, 0, len(f['data'][0]) + 2)

                print('keyframing ' + obj_name + ' object in condition ' + cond_str)
                # For every time point insert keyframe to current object
                for ind, timepoint in enumerate(data[:, cond_ind]):
                    insert_keyframe_to_custom_prop(cur_obj, obj_name + '_' + str(cond_str), timepoint, ind + 2)
                # remove the orange keyframe sign in the fcurves window
                fcurves = bpy.data.objects[obj_name].animation_data.action.fcurves[cond_ind]
                mod = fcurves.modifiers.new(type='LIMITS')
    print('Finished keyframing!!')


def add_data_to_electrodes_parent_obj(self, parent_obj, source_files, stat):
    # todo: merge with add_data_to_brain_parent_obj, same code
    parent_obj.animation_data_clear()
    sources = {}
    for input_file in source_files:
        if not op.isfile(input_file):
            self.report({'ERROR'}, "Can't load file {}!".format(input_file))
            continue
        print('loading {}'.format(input_file))
        f = np.load(input_file)
        # for obj_name, data in zip(f['names'], f['data']):
        all_data_stat = f['stat'] if 'stat' in f else [None] * len(f['names'])
        for obj_name, data, data_stat in zip(f['names'], f['data'], all_data_stat):
            obj_name = obj_name.astype(str)
            if data_stat is None:
                if stat == STAT_AVG:
                    data_stat = np.squeeze(np.mean(data, axis=1))
                elif stat == STAT_DIFF:
                    data_stat = np.squeeze(np.diff(data, axis=1))
            sources[obj_name] = data_stat

    sources_names = sorted(list(sources.keys()))
    N = len(sources_names)
    T = DataMakerPanel.addon.get_max_time_steps() # len(sources[sources_names[0]]) + 2
    now = time.time()
    for obj_counter, source_name in enumerate(sources_names):
        mu.time_to_go(now, obj_counter, N, runs_num_to_print=10)
        data = sources[source_name]
        insert_keyframe_to_custom_prop(parent_obj, source_name, 0, 1)
        insert_keyframe_to_custom_prop(parent_obj, source_name, 0, T + 2)

        for ind in range(data.shape[0]):
            insert_keyframe_to_custom_prop(parent_obj, source_name, data[ind], ind + 2)

        fcurves = parent_obj.animation_data.action.fcurves[obj_counter]
        mod = fcurves.modifiers.new(type='LIMITS')

    print('Finished keyframing {}!!'.format(parent_obj.name))


class AddDataToElectrodes(bpy.types.Operator):
    bl_idname = "ohad.electrodes_add_data"
    bl_label = "add_data2 electrodes"
    bl_options = {"UNDO"}
    current_root_path = ''

    def invoke(self, context, event=None):
        # self.current_root_path = bpy.path.abspath(bpy.context.scene.conf_path)
        parent_obj = bpy.data.objects['Deep_electrodes']
        base_path = mu.get_user_fol()
        source_file = op.join(base_path, 'electrodes_data_{}.npz'.format(
            'avg' if bpy.context.scene.selection_type == 'conds' else 'diff'))
        if not op.isfile(source_file):
            source_file = op.join(base_path, 'electrodes_data.npz')
        if not op.isfile(source_file):
            print('No electrodes data file!')
        else:
            print('Loading electordes data from {}'.format(source_file))
            # add_data_to_electrodes(self, [source_file])
            add_data_to_electrodes_parent_obj(self, parent_obj, [source_file], STAT_DIFF)
            bpy.types.Scene.electrodes_data_exist = True
            if bpy.data.objects.get(' '):
                bpy.context.scene.objects.active = bpy.data.objects[' ']
        return {"FINISHED"}


class DataMakerPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
    bl_label = "Data Panel"
    addon = None

    def draw(self, context):
        layout = self.layout
        # layout.prop(context.scene, 'conf_path')
        col1 = self.layout.column(align=True)
        col1.prop(context.scene, 'atlas', text="Atlas")
        # if not bpy.types.Scene.brain_imported:
        col1.operator("ohad.brain_importing", text="Import Brain", icon='MATERIAL_DATA')
        # if not bpy.types.Scene.electrodes_imported:
        col1.prop(context.scene, 'bipolar', text="Bipolar")
        col1.prop(context.scene, 'electrode_radius', text="Electrodes' radius")
        col1.operator("ohad.electrodes_importing", text="Import Electrodes", icon='COLOR_GREEN')

        # if bpy.types.Scene.brain_imported and (not bpy.types.Scene.brain_data_exist):
        col2 = self.layout.column(align=True)
        col2.operator(AddDataToBrain.bl_idname, text="Add data to Brain", icon='FCURVE')
        col2.operator(AddDataNoCondsToBrain.bl_idname, text="Add no conds data to Brain", icon='FCURVE')
        # if bpy.types.Scene.electrodes_imported and (not bpy.types.Scene.electrodes_data_exist):
        col2.operator("ohad.electrodes_add_data", text="Add data to Electrodes", icon='FCURVE')


def init(addon):
    DataMakerPanel.addon = addon
    register()


def register():
    try:
        unregister()
        bpy.utils.register_class(DataMakerPanel)
        bpy.utils.register_class(AddDataToElectrodes)
        bpy.utils.register_class(AddDataNoCondsToBrain)
        bpy.utils.register_class(AddDataToBrain)
        bpy.utils.register_class(ImportElectrodes)
        bpy.utils.register_class(ImportRois)
        bpy.utils.register_class(ImportBrain)
        print('Data Panel was registered!')
    except:
        print("Can't register Data Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(DataMakerPanel)
        bpy.utils.unregister_class(AddDataToElectrodes)
        bpy.utils.unregister_class(AddDataNoCondsToBrain)
        bpy.utils.unregister_class(AddDataToBrain)
        bpy.utils.unregister_class(ImportElectrodes)
        bpy.utils.unregister_class(ImportRois)
        bpy.utils.unregister_class(ImportBrain)
    except:
        pass

