import bpy
import os.path as op
import glob
import numpy as np
import time
import mmvt_utils as mu
import selection_panel
import logging
from collections import Iterable

STAT_AVG, STAT_DIFF = range(2)

bpy.types.Scene.brain_imported = False
bpy.types.Scene.electrodes_imported = False
bpy.types.Scene.eeg_imported = False
bpy.types.Scene.brain_data_exist = False
bpy.types.Scene.electrodes_data_exist = False
bpy.types.Scene.eeg_data_exist = False


def bipolar_update(self, context):
    try:
        import electrodes_panel
        electrodes_panel.init_electrodes_labeling(DataMakerPanel.addon)
    except:
        pass


bpy.types.Scene.atlas = bpy.props.StringProperty(name='atlas', default='laus250')
bpy.types.Scene.bipolar = bpy.props.BoolProperty(default=False, description="Bipolar electrodes", update=bipolar_update)
bpy.types.Scene.electrodes_radius = bpy.props.FloatProperty(default=0.15, description="Electrodes radius", min=0.01, max=1)
bpy.types.Scene.import_unknown = bpy.props.BoolProperty(default=False, description="Import unknown labels")
bpy.types.Scene.inflated_morphing = bpy.props.BoolProperty(default=True, description="inflated_morphing")
bpy.types.Scene.labels_data_files = bpy.props.EnumProperty(items=[], description="labels data files")
bpy.types.Scene.add_meg_labels_data = bpy.props.BoolProperty(default=True, description="")
bpy.types.Scene.add_meg_subcorticals_data = bpy.props.BoolProperty(default=False, description="")
bpy.types.Scene.meg_evoked_files = bpy.props.EnumProperty(items=[], description="meg_evoked_files")
bpy.types.Scene.evoked_objects = bpy.props.EnumProperty(items=[], description="meg_evoked_types")
bpy.types.Scene.electrodes_positions_files = bpy.props.EnumProperty(items=[], description="electrodes_positions")
bpy.types.Scene.fMRI_dynamic_files = bpy.props.EnumProperty(items=[], description="fMRI_dynamic")

bpy.types.Scene.brain_no_conds_stat = bpy.props.EnumProperty(items=[('diff', 'conditions difference', '', 0), ('mean', 'conditions average', '', 1)])


def _addon():
    return DataMakerPanel.addon


def eeg_data_and_meta():
    if DataMakerPanel.eeg_data is None:
        data_fname = op.join(mu.get_user_fol(), 'eeg', 'eeg_data.npy')
        meta_fname = op.join(mu.get_user_fol(), 'eeg', 'eeg_data_meta.npz')
        if op.isfile(data_fname) and op.isfile(meta_fname):
            DataMakerPanel.eeg_data = np.load(data_fname, mmap_mode='r')
            DataMakerPanel.eeg_meta = np.load(meta_fname)
        else:
            DataMakerPanel.eeg_data = DataMakerPanel.eeg_meta = None
    return DataMakerPanel.eeg_data, DataMakerPanel.eeg_meta


@mu.tryit
def import_hemis_for_functional_maps(base_path):
    mu.change_layer(_addon().BRAIN_EMPTY_LAYER)
    layers_array = bpy.context.scene.layers
    emptys_names = ['Functional maps', 'Subcortical_meg_activity_map', 'Subcortical_fmri_activity_map']
    for name in emptys_names:
        create_empty_if_doesnt_exists(name, _addon().BRAIN_EMPTY_LAYER, layers_array, 'Functional maps')

    # for ii in range(len(bpy.context.scene.layers)):
    #     bpy.context.scene.layers[ii] = (ii == brain_layer)

    print("importing Hemispheres")
    # # for cur_val in bpy.context.scene.layers:
    # #     print(cur_val)
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    now = time.time()
    ply_files = glob.glob(op.join(base_path, 'surf', '*.ply'))
    N = len(ply_files)
    for ind, ply_fname in enumerate(ply_files):
        try:
            mu.time_to_go(now, ind, N, 10)
            bpy.ops.object.select_all(action='DESELECT')
            obj_name = mu.namebase(ply_fname).split(sep='.')[0]
            surf_name = mu.namebase(ply_fname).split(sep='.')[1]
            if surf_name == 'inflated':
                obj_name = '{}_{}'.format(surf_name, obj_name)
                mu.change_layer(_addon().INFLATED_ACTIVITY_LAYER)
            elif surf_name == 'pial':
                mu.change_layer(_addon().ACTIVITY_LAYER)
            else:
                raise Exception('The surface type {} is not supported!'.format(surf_name))
            if bpy.data.objects.get(obj_name) is None:
                bpy.ops.import_mesh.ply(filepath=op.join(base_path, 'surf', ply_fname))
                cur_obj = bpy.context.selected_objects[0]
                cur_obj.select = True
                bpy.ops.object.shade_smooth()
                cur_obj.scale = [0.1] * 3
                cur_obj.hide = False
                cur_obj.name = obj_name
                if surf_name == 'inflated':
                    cur_obj.active_material = bpy.data.materials['Inflated_Activity_map_mat']
                    cur_obj.location[0] += 5.5 if obj_name == 'inflated_rh' else -5.5
                else:
                    cur_obj.active_material = bpy.data.materials['Activity_map_mat']
                cur_obj.parent = bpy.data.objects["Functional maps"]
                cur_obj.hide_select = True
                cur_obj.data.vertex_colors.new()
            # else:
            #     print('{} already exists'.format(ply_fname))
        except:
            mu.log_err('Error in importing {}'.format(ply_fname), logging)

    _addon().create_inflated_curv_coloring()
    bpy.ops.object.select_all(action='DESELECT')


def create_subcortical_activity_mat(name):
    cur_mat = bpy.data.materials['subcortical_activity_mat'].copy()
    cur_mat.name = name + '_Mat'


@mu.tryit
def import_subcorticals(base_path, parent_name='Subcortical'):
    empty_layer = DataMakerPanel.addon.BRAIN_EMPTY_LAYER
    brain_layer = DataMakerPanel.addon.ACTIVITY_LAYER

    bpy.context.scene.layers = [ind == empty_layer for ind in range(len(bpy.context.scene.layers))]
    layers_array = bpy.context.scene.layers
    emptys_names = ['Functional maps', '{}_meg_activity_map'.format(parent_name), '{}_fmri_activity_map'.format(parent_name)]
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
            try:
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
                    cur_obj.parent = bpy.data.objects['{}_meg_activity_map'.format(parent_name)]
                elif path_type == PATH_TYPE_SUB_FMRI:
                    cur_obj.name = '{}_fmri_activity'.format(obj_name)
                    if 'cerebellum' in cur_obj.name.lower():
                        cur_obj.active_material = bpy.data.materials['Activity_map_mat']
                    else:
                        cur_obj.active_material = bpy.data.materials['subcortical_activity_mat']
                    cur_obj.parent = bpy.data.objects['{}_fmri_activity_map'.format(parent_name)]
                else:
                    mu.log_err('import_subcorticals: Wrong path_type! Nothing to do...', logging)
                cur_obj.hide_select = True
            except:
                mu.log_err('Error in importing {}!'.format(ply_fname), logging)
    bpy.ops.object.select_all(action='DESELECT')


class AnatomyPreproc(bpy.types.Operator):
    bl_idname = "mmvt.anatomy_preproc"
    bl_label = "anatomy_preproc"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        cmd = '{} -m src.preproc.anatomy_preproc -s {} -a {}'.format(
            bpy.context.scene.python_cmd, mu.get_user(), bpy.context.scene.atlas)
        print('Running {}'.format(cmd))
        mu.run_command_in_new_thread(cmd, False)
        return {"FINISHED"}


def import_brain(context=None):
    # self.brain_layer = DataMakerPanel.addon.BRAIN_EMPTY_LAYER
    # self.current_root_path = mu.get_user_fol()  # bpy.path.abspath(bpy.context.scene.conf_path)
    if _addon() is None:
        mu.log_err('import_brain: addon is None!', logging)
        return
    user_fol = mu.get_user_fol()
    print("importing ROIs")
    import_rois(user_fol)
    import_hemis_for_functional_maps(user_fol)
    import_subcorticals(op.join(user_fol, 'subcortical'))
    # if op.isdir(op.join(user_fol, 'cerebellum')):
    #     import_subcorticals(op.join(user_fol, 'cerebellum'), 'Cerebellum')
    if context:
        last_obj = context.active_object.name
        print('last obj is -' + last_obj)
    if bpy.context.scene.inflated_morphing:
        create_inflating_morphing()
    if bpy.data.objects.get(' '):
        bpy.data.objects[' '].select = True
        if context:
            context.scene.objects.active = bpy.data.objects[' ']
    if context:
        bpy.data.objects[last_obj].select = False
    DataMakerPanel.addon.show_rois()
    bpy.types.Scene.brain_imported = True
    print('cleaning up')
    for obj in bpy.data.objects['Subcortical_structures'].children:
        # print(obj.name)
        if obj.name[-1] == '1':
            obj.name = obj.name[0:-4]
    bpy.ops.object.select_all(action='DESELECT')
    print('Brain importing is Finished ')


class ImportBrain(bpy.types.Operator):
    bl_idname = "mmvt.brain_importing"
    bl_label = "import2 brain"
    bl_options = {"UNDO"}
    current_root_path = ''
    brain_layer = -1

    def invoke(self, context, event=None):
        import_brain()
        return {"FINISHED"}


def create_empty_if_doesnt_exists(name, brain_layer=None, layers_array=None, parent_obj_name='Brain'):
    if brain_layer is None:
        brain_layer = _addon().BRAIN_EMPTY_LAYER
    if layers_array is None:
        layers_array = bpy.context.scene.layers
    if bpy.data.objects.get(name) is None:
        layers_array[brain_layer] = True
        bpy.ops.object.empty_add(type='PLAIN_AXES', radius=1, view_align=False, location=(0, 0, 0), layers=layers_array)
        bpy.data.objects['Empty'].name = name
        if name != parent_obj_name:
            bpy.data.objects[name].parent = bpy.data.objects[parent_obj_name]
    return bpy.data.objects[name]


@mu.tryit
def import_rois(base_path):
    anatomy_inputs = {'Cortex-rh': op.join(base_path, '{}.pial.rh'.format(bpy.context.scene.atlas)),
                      'Cortex-lh': op.join(base_path, '{}.pial.lh'.format(bpy.context.scene.atlas)),
                      'Cortex-inflated-rh': op.join(base_path, '{}.inflated.rh'.format(bpy.context.scene.atlas)),
                      'Cortex-inflated-lh': op.join(base_path, '{}.inflated.lh'.format(bpy.context.scene.atlas)),
                      'Subcortical_structures': op.join(base_path, 'subcortical'),
                      'Cerebellum': op.join(base_path, 'cerebellum')}
    brain_layer = _addon().BRAIN_EMPTY_LAYER

    bpy.context.scene.layers = [ind == brain_layer for ind in range(len(bpy.context.scene.layers))]
    layers_array = bpy.context.scene.layers
    emptys_names = ['Brain'] + list(anatomy_inputs.keys()) # ["Brain", "Subcortical_structures", "Cortex-lh", "Cortex-rh", 'Cortex-inflated-rh', 'Cortex-inflated-rh']
    for name in emptys_names:
        create_empty_if_doesnt_exists(name, brain_layer, layers_array)
    bpy.context.scene.layers = [ind == DataMakerPanel.addon.ROIS_LAYER for ind in range(len(bpy.context.scene.layers))]

    # todo: check each hemi
    inflated_imported = False
    for anatomy_name, anatomy_input_base_path in anatomy_inputs.items():
        if not op.isdir(anatomy_input_base_path):
            mu.log_err('import_rois: The anatomy folder {} does not exist'.format(anatomy_input_base_path), logging)
            continue
        current_mat = bpy.data.materials['unselected_label_Mat_cortex']
        if anatomy_name in ['Subcortical_structures', 'Cerebellum']:
            current_mat = bpy.data.materials['unselected_label_Mat_subcortical']
        print('importing from {}'.format(anatomy_input_base_path))
        for ply_fname in glob.glob(op.join(anatomy_input_base_path, '*.ply')):
            try:
                new_obj_name = mu.namebase(ply_fname)
                fol_name = anatomy_input_base_path.split(op.sep)[-1]
                surf_name = 'pial' if fol_name == 'subcortical' or len(fol_name.split('.')) == 1 else fol_name.split('.')[-2]
                if surf_name == 'inflated':
                    new_obj_name = '{}_{}'.format(surf_name, new_obj_name)
                    mu.change_layer(_addon().INFLATED_ROIS_LAYER)
                elif surf_name == 'pial':
                    mu.change_layer(_addon().ROIS_LAYER)
                if not bpy.data.objects.get(new_obj_name) is None:
                    # print('{} was already imported'.format(new_obj_name))
                    continue
                if 'inflated' in new_obj_name:
                    inflated_imported = True
                bpy.ops.object.select_all(action='DESELECT')
                # print(ply_fname)
                bpy.ops.import_mesh.ply(filepath=op.join(anatomy_input_base_path, ply_fname))
                cur_obj = bpy.context.selected_objects[0]
                cur_obj.select = True
                bpy.ops.object.shade_smooth()
                cur_obj.parent = bpy.data.objects[anatomy_name]
                cur_obj.scale = [0.1] * 3
                cur_obj.active_material = current_mat
                cur_obj.hide = False
                cur_obj.name = new_obj_name
            except:
                mu.log_err('import_rois: Error in importing {}'.format(ply_fname), logging)
            # cur_obj.location[0] += 5.5 if 'rh' in anatomy_name else -5.5
            # time.sleep(0.3)
    if inflated_imported:
        bpy.data.objects['Cortex-inflated-rh'].location[0] += 5.5
        bpy.data.objects['Cortex-inflated-lh'].location[0] -= 5.5
    bpy.ops.object.select_all(action='DESELECT')


def create_eeg_mesh():
    mu.change_layer(_addon().BRAIN_EMPTY_LAYER)
    create_empty_if_doesnt_exists('Helmets', _addon().BRAIN_EMPTY_LAYER, bpy.context.scene.layers, 'Functional maps')
    mu.change_layer(_addon().EEG_LAYER)
    current_mat = bpy.data.materials['unselected_label_Mat_cortex']
    bpy.ops.import_mesh.ply(filepath=op.join(mu.get_user_fol(), 'eeg', 'eeg_helmet.ply'))
    mesh_obj = bpy.context.selected_objects[0]
    mesh_obj.name = 'eeg_helmet'
    mesh_obj.select = True
    bpy.ops.object.shade_smooth()
    mesh_obj.parent = bpy.data.objects['Helmets']
    mesh_obj.scale = [0.1] * 3
    mesh_obj.active_material = current_mat
    mesh_obj.hide = False
    return mesh_obj


class ImportRois(bpy.types.Operator):
    bl_idname = "mmvt.roi_importing"
    bl_label = "import2 ROIs"
    bl_options = {"UNDO"}
    current_root_path = ''

    def invoke(self, context, event=None):
        self.current_root_path = mu.get_user_fol() #bpy.path.abspath(bpy.context.scene.conf_path)
        import_hemis_for_functional_maps(self.current_root_path)
        return {"FINISHED"}


def import_meg_sensors():
    input_file = op.join(mu.get_user_fol(), 'meg', 'meg_sensors_positions.npz')
    import_electrodes(input_file, _addon().MEG_LAYER, bipolar=False, parnet_name='MEG_sensors')
    bpy.types.Scene.meg_sensors_imported = True
    print('MEG sensors importing is Finished ')


def import_electrodes(input_file, electrodes_layer=None, bipolar='', electrode_size=None, parnet_name='Deep_electrodes'):
    if electrodes_layer is None:
        electrodes_layer = _addon().ELECTRODES_LAYER
    if not electrode_size is None:
        bpy.context.scene.electrodes_radius = electrode_size
    if bipolar != '':
        bpy.context.scene.bipolar = bool(bipolar)
    mu.delete_hierarchy(parnet_name)
    f = np.load(input_file)

    electrode_size = bpy.context.scene.electrodes_radius
    layers_array = [False] * 20
    create_empty_if_doesnt_exists(parnet_name, _addon().BRAIN_EMPTY_LAYER, layers_array, parnet_name)

    layers_array = [False] * 20
    layers_array[electrodes_layer] = True

    for (x, y, z), name in zip(f['pos'], f['names']):
        elc_name = name.astype(str)
        if not bpy.data.objects.get(elc_name) is None:
            elc_obj = bpy.data.objects[elc_name]
            elc_obj.location = [x * 0.1, y * 0.1, z * 0.1]
        else:
            print('creating {}: {}'.format(elc_name, (x, y, z)))
            mu.create_sphere((x * 0.1, y * 0.1, z * 0.1), electrode_size, layers_array, elc_name)
            cur_obj = bpy.data.objects[elc_name]
            cur_obj.select = True
            cur_obj.parent = bpy.data.objects[parnet_name]
            mu.create_and_set_material(cur_obj)


@mu.tryit
def create_inflating_morphing():
    print('Creating inflation morphing')
    for hemi in mu.HEMIS:
        pial = bpy.data.objects[hemi]
        inflated = bpy.data.objects['inflated_{}'.format(hemi)]
        # if inflated.active_shape_key_index >= 0:
        #     print('{} already has a shape key'.format(hemi))
        #     continue
        inflated.shape_key_add(name='pial')
        inflated.shape_key_add(name='inflated')
        for vert_ind in range(len(inflated.data.vertices)):
            for ii in range(3):
                inflated.data.shape_keys.key_blocks['pial'].data[vert_ind].co[ii] = pial.data.vertices[vert_ind].co[ii]


class ImportElectrodes(bpy.types.Operator):
    bl_idname = "mmvt.electrodes_importing"
    bl_label = "import2 electrodes"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        input_file = op.join(mu.get_user_fol(), 'electrodes',
                             '{}.npz'.format(bpy.context.scene.electrodes_positions_files))
        import_electrodes(input_file, _addon().ELECTRODES_LAYER)
        bpy.types.Scene.electrodes_imported = True
        print('Electrodes importing is Finished ')
        return {"FINISHED"}


class ImportMEGSensors(bpy.types.Operator):
    bl_idname = "mmvt.import_meg_sensors"
    bl_label = "import meg sensors"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        import_meg_sensors()
        return {"FINISHED"}


class ImportEEG(bpy.types.Operator):
    bl_idname = "mmvt.import_eeg_sensors"
    bl_label = "import eeg sensors"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        input_file = op.join(mu.get_user_fol(), 'eeg', 'eeg_positions.npz')
        import_electrodes(input_file, _addon().EEG_LAYER, bipolar=False, parnet_name='EEG_sensors')
        bpy.types.Scene.eeg_imported = True
        print('EEG sensors importing is Finished ')
        return {"FINISHED"}


class CreateEEGMesh(bpy.types.Operator):
    bl_idname = "mmvt.eeg_mesh"
    bl_label = "eeg mesh"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        create_eeg_mesh()
        return {"FINISHED"}


def add_data_to_brain(source_files):
    print('Adding data to Brain')
    conditions = []
    for f in source_files:
        T = len(f['data'][0])
        for obj_name, data in zip(f['names'], f['data']):
            obj_name = obj_name.astype(str)
            if not bpy.context.scene.import_unknown and 'unknown' in obj_name:
                continue
            cur_obj = bpy.data.objects[obj_name]
            fcurves_num = mu.count_fcurves(cur_obj)
            if fcurves_num < len(f['conditions']):
                print('keyframing {}'.format(obj_name))
                for cond_ind, cond_str in enumerate(f['conditions']):
                    cond_str = cond_str.astype(str)
                    # Set the values to zeros in the first and last frame for current object(current label)
                    mu.insert_keyframe_to_custom_prop(cur_obj, obj_name + '_' + cond_str, 0, 1)
                    mu.insert_keyframe_to_custom_prop(cur_obj, obj_name + '_' + cond_str, 0, len(f['data'][0]) + 2)
    
                    # For every time point insert keyframe to current object
                    for ind, t in enumerate(data[:, cond_ind]):
                        mu.insert_keyframe_to_custom_prop(cur_obj, obj_name + '_' + cond_str, t, ind + 2)
    
                    # remove the orange keyframe sign in the fcurves window
                    fcurves = bpy.data.objects[obj_name].animation_data.action.fcurves[cond_ind]
                    mod = fcurves.modifiers.new(type='LIMITS')
            else:
                for fcurve_ind, fcurve in enumerate(cur_obj.animation_data.action.fcurves):
                    fcurve.keyframe_points[0].co[1] = 0
                    fcurve.keyframe_points[-1].co[1] = 0
                    for t in range(T):
                        fcurve.keyframe_points[t + 1].co[1] = data[t, fcurve_ind]

        conditions.extend(f['conditions'])
    try:
        bpy.ops.graph.previewrange_set()
    except:
        pass

    bpy.types.Scene.maximal_time_steps = T
    for obj in bpy.data.objects:
        obj.select = False
    if bpy.data.objects.get(' '):
        bpy.context.scene.objects.active = bpy.data.objects[' ']
    selection_panel.set_conditions_enum(conditions)
    print('Finished keyframing!!')


# def add_data_to_parent_brain_obj(brain_sources, subcorticals_sources, stat=STAT_DIFF):
#     if bpy.context.scene.add_meg_labels_data:
#         brain_obj = bpy.data.objects['Brain']
#         add_data_to_parent_obj(brain_obj, brain_sources, stat)
#     if bpy.context.scene.add_meg_subcorticals_data:
#         subcorticals_obj = bpy.data.objects['Subcortical_structures']
#         add_data_to_parent_obj(subcorticals_obj, subcorticals_sources, stat)
#     mu.view_all_in_graph_editor()


def add_fmri_dynamics_to_parent_obj():
    brain_obj = create_empty_if_doesnt_exists('fMRI')
    measure = bpy.context.scene.fMRI_dynamic_files.split(' ')[-1]
    source = [np.load(op.join(mu.get_user_fol(), 'fmri', 'labels_data_{}_{}_{}.npz'.format(
        bpy.context.scene.atlas, measure, hemi))) for hemi in mu.HEMIS]
    add_data_to_parent_obj(brain_obj, source, STAT_AVG)
    mu.view_all_in_graph_editor()


def add_data_to_meg_sensors(stat=STAT_DIFF):
    parnet_name = 'MEG_sensors'
    parent_obj = bpy.data.objects.get(parnet_name)
    if parent_obj is None:
        layers_array = [False] * 20
        create_empty_if_doesnt_exists(parnet_name, _addon().BRAIN_EMPTY_LAYER, layers_array, parnet_name)
    data_fname = op.join(mu.get_user_fol(), 'meg', 'meg_sensors_evoked_data.npy')
    meta_fname = op.join(mu.get_user_fol(), 'meg', 'meg_sensors_evoked_data_meta.npz')
    if not op.isfile(data_fname) or not op.isfile(meta_fname):
        mu.log_err('MEG data should be here {} (data) and here {} (meta data)'.format(data_fname, meta_fname), logging)
    else:
        data = DataMakerPanel.meg_data = np.load(data_fname, mmap_mode='r')
        meta = DataMakerPanel.meg_meta = np.load(meta_fname)
        add_data_to_electrodes(data, meta)
        add_data_to_electrodes_parent_obj(parent_obj, data, meta, stat)
        bpy.types.Scene.eeg_data_exist = True
    if bpy.data.objects.get(' '):
        bpy.context.scene.objects.active = bpy.data.objects[' ']


def add_meg_data_to_parent_obj():
    base_path = mu.get_user_fol()
    atlas = bpy.context.scene.atlas
    labels_extract_method = bpy.context.scene.labels_data_files
    brain_obj = bpy.data.objects['Brain']
    brain_sources = [np.load(op.join(base_path, 'meg', 'labels_data_{}_{}_{}.npz'.format(
        atlas, labels_extract_method, hemi))) for hemi in mu.HEMIS]
    add_data_to_parent_obj(brain_obj, brain_sources, STAT_DIFF)


def add_data_to_parent_obj(parent_obj, source_files, stat):
    sources = {}
    if not isinstance(source_files, Iterable):
        source_files = [source_files]
    for f in source_files:
        for obj_name, data in zip(f['names'], f['data']):
            obj_name = obj_name.astype(str)
            # Check if there is only one condition
            if data.ndim == 1 or data.shape[1] == 1:
                stat = STAT_AVG
            if bpy.data.objects.get(obj_name) is None:
                if obj_name.startswith('rh') or obj_name.startswith('lh'):
                    obj_name = obj_name[3:]
                if bpy.data.objects.get(obj_name) is None:
                    continue
            if not bpy.context.scene.import_unknown and 'unkown' in obj_name:
                continue
            if stat == STAT_AVG and data.ndim > 1:
                data_stat = np.squeeze(np.mean(data, axis=1))
            elif stat == STAT_DIFF and data.ndim > 1:
                data_stat = np.squeeze(np.diff(data, axis=1))
            else:
                data_stat = data
            sources[obj_name] = data_stat
    if len(sources) == 0:
        mu.log_err('No sources in {}'.format(source_files), logging)
        return
    sources_names = sorted(list(sources.keys()))
    N = len(sources_names)
    T = len(sources[sources_names[0]]) + 2
    fcurves_num = mu.count_fcurves(parent_obj)
    if fcurves_num < len(sources_names):
        parent_obj.animation_data_clear()
        now = time.time()
        for obj_counter, source_name in enumerate(sources_names):
            mu.time_to_go(now, obj_counter, N, runs_num_to_print=10)
            data = sources[source_name]
            # Set the values to zeros in the first and last frame for Brain object
            mu.insert_keyframe_to_custom_prop(parent_obj, source_name, 0, 1)
            mu.insert_keyframe_to_custom_prop(parent_obj, source_name, 0, T)

            # For every time point insert keyframe to the main Brain object
            for ind in range(data.shape[0]):
                mu.insert_keyframe_to_custom_prop(parent_obj, source_name, data[ind], ind + 2)

            # remove the orange keyframe sign in the fcurves window
            fcurves = parent_obj.animation_data.action.fcurves[obj_counter]
            mod = fcurves.modifiers.new(type='LIMITS')
    else:
        for fcurve_ind, fcurve in enumerate(parent_obj.animation_data.action.fcurves):
            fcurve_name = mu.get_fcurve_name(fcurve)
            fcurve.keyframe_points[0].co[1] = 0
            fcurve.keyframe_points[-1].co[1] = 0
            T = min([len(fcurve.keyframe_points) - 1, len(sources[fcurve_name])])
            for t in range(T):
                fcurve.keyframe_points[t + 1].co[1] = sources[fcurve_name][t]

    if bpy.data.objects.get(' '):
        bpy.context.scene.objects.active = bpy.data.objects[' ']
    print('Finished keyframing the brain parent obj!!')


class AddDataToBrain(bpy.types.Operator):
    bl_idname = "mmvt.brain_add_data"
    bl_label = "add_data brain"
    bl_options = {"UNDO"}
    current_root_path = ''

    def invoke(self, context, event=None):
        base_path = mu.get_user_fol()
        atlas = bpy.context.scene.atlas
        labels_extract_method = bpy.context.scene.labels_data_files
        brain_sources = [np.load(op.join(base_path, 'meg', 'labels_data_{}_{}_lh.npz'.format(atlas, labels_extract_method))),
                         np.load(op.join(base_path, 'meg', 'labels_data_{}_{}_rh.npz'.format(atlas, labels_extract_method)))]
        subcorticals_sources = [np.load(op.join(base_path, 'meg', 'subcortical_meg_activity.npz'))]
        add_data_to_brain(brain_sources)
        if bpy.context.scene.add_meg_labels_data:
            brain_obj = bpy.data.objects['Brain']
            add_data_to_parent_obj(brain_obj, brain_sources, STAT_DIFF)
        if bpy.context.scene.add_meg_subcorticals_data:
            subcorticals_obj = bpy.data.objects['Subcortical_structures']
            add_data_to_parent_obj(subcorticals_obj, subcorticals_sources, STAT_DIFF)

        _addon().select_all_rois()
        mu.view_all_in_graph_editor()
        bpy.types.Scene.brain_data_exist = True
        return {"FINISHED"}


class AddfMRIDynamicsToBrain(bpy.types.Operator):
    bl_idname = "mmvt.add_fmri_dynamics_to_brain"
    bl_label = "add_fmri_dynamics_to_brain"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        add_fmri_dynamics_to_parent_obj()
        return {"FINISHED"}


# class SelectExternalMEGEvoked(bpy.types.Operator):
#     bl_idname = "mmvt.select_external_meg_evoked"
#     bl_label = "select_external_meg_evoked"
#     bl_options = {"UNDO"}
#
#     def invoke(self, context, event=None):
#         evoked_name = '{}_{}'.format(bpy.context.scene.meg_evoked_files, bpy.context.scene.evoked_objects)
#         evoked_obj = bpy.data.objects.get(evoked_name)
#         if not evoked_obj is None:
#             evoked_obj.select = not evoked_obj.select
#         mu.view_all_in_graph_editor(context)
#         selected_objects = mu.get_selected_objects()
#         mu.change_fcurves_colors(selected_objects)
#         return {"FINISHED"}


# def get_external_meg_evoked_selected():
#     evoked_name = '{}_{}'.format(bpy.context.scene.meg_evoked_files, bpy.context.scene.evoked_objects)
#     evoked_obj = bpy.data.objects.get(evoked_name)
#     if not evoked_obj is None:
#         return evoked_obj.select
#     else:
#         return False


# def get_meg_evoked_source_files(base_path, files_prefix):
#     source_files = [op.join(base_path, '{}labels_data_lh.npz'.format(files_prefix)),
#                     op.join(base_path, '{}labels_data_rh.npz'.format(files_prefix)),
#                     op.join(base_path, '{}sub_cortical_activity.npz'.format(files_prefix))]
#     return source_files


# class AddOtherSubjectMEGEvokedResponse(bpy.types.Operator):
#     bl_idname = "mmvt.other_subject_meg_evoked"
#     bl_label = "other_subject_meg_evoked"
#     bl_options = {"UNDO"}
#
#     def invoke(self, context, event=None):
#         evoked_name = bpy.context.scene.meg_evoked_files
#         files_prefix = '{}_'.format(evoked_name)
#         base_path = op.join(mu.get_user_fol(), 'meg_evoked_files')
#         source_files = get_meg_evoked_source_files(base_path, files_prefix)
#         empty_layer = _addon().BRAIN_EMPTY_LAYER
#         layers_array = bpy.context.scene.layers
#         parent_obj_name = 'External'
#         create_empty_if_doesnt_exists(parent_obj_name, empty_layer, layers_array, parent_obj_name)
#         for input_file in source_files:
#             if not op.isfile(input_file):
#                 continue
#             f = np.load(input_file)
#             for label_name in f['names']:
#                 mu.create_empty_in_vertex((0, 0, 0), '{}_{}'.format(evoked_name, label_name),
#                     _addon().BRAIN_EMPTY_LAYER, parent_obj_name)
#
#         add_data_to_brain(base_path, files_prefix, files_prefix)
#         _meg_evoked_files_update()
#         return {"FINISHED"}

@mu.tryit
def add_data_to_electrodes(all_data, meta_data, window_len=None):
    print('Adding data to Electrodes')
    now = time.time()
    N = len(meta_data['names'])
    T = all_data.shape[1] if window_len is None or not 'dt' in meta_data else int(window_len / meta_data['dt'])
    for obj_counter, (obj_name, data) in enumerate(zip(meta_data['names'], all_data)):
        mu.time_to_go(now, obj_counter, N, runs_num_to_print=10)
        obj_name = obj_name.astype(str)
        # print(obj_name)
        if bpy.data.objects.get(obj_name, None) is None:
            mu.log_err("{} doesn't exist!".format(obj_name), logging)
            continue
        cur_obj = bpy.data.objects[obj_name]
        fcurves_num = mu.count_fcurves(cur_obj)
        if fcurves_num < len(meta_data['conditions']):
            for cond_ind, cond_str in enumerate(meta_data['conditions']):
                cond_str = cond_str.astype(str)
                # Set the values to zeros in the first and last frame for current object(current label)
                mu.insert_keyframe_to_custom_prop(cur_obj, obj_name + '_' + cond_str, 0, 1)
                mu.insert_keyframe_to_custom_prop(cur_obj, obj_name + '_' + cond_str, 0, T + 2)

                print('keyframing ' + obj_name + ' object in condition ' + cond_str)
                # For every time point insert keyframe to current object
                for ind, t in enumerate(data[:T, cond_ind]):
                    mu.insert_keyframe_to_custom_prop(cur_obj, obj_name + '_' + str(cond_str), t, ind + 2)
                # remove the orange keyframe sign in the fcurves window
                fcurves = bpy.data.objects[obj_name].animation_data.action.fcurves[cond_ind]
                mod = fcurves.modifiers.new(type='LIMITS')
        else:
            for fcurve_ind, fcurve in enumerate(cur_obj.animation_data.action.fcurves):
                fcurve.keyframe_points[0].co[1] = 0
                fcurve.keyframe_points[-1].co[1] = 0
                for t in range(T):
                    fcurve.keyframe_points[t + 1].co[1] = data[t, fcurve_ind]


    conditions = meta_data['conditions']
    print('Finished keyframing!!')
    return conditions


@mu.tryit
def add_data_to_electrodes_parent_obj(parent_obj, all_data, meta, stat=STAT_DIFF, window_len=None):
    # todo: merge with add_data_to_brain_parent_obj, same code
    sources = {}
    # for obj_name, data in zip(f['names'], f['data']):
    all_data_stat = meta['stat'] if 'stat' in meta else [None] * len(meta['names'])
    T = all_data.shape[1] if window_len is None or 'dt' not in meta else int(window_len / meta['dt'])
    for obj_name, data, data_stat in zip(meta['names'], all_data, all_data_stat):
        obj_name = obj_name.astype(str)
        if data_stat is None:
            if stat == STAT_AVG or data.shape[1] == 1:
                data_stat = np.squeeze(np.mean(data, axis=1))
            elif stat == STAT_DIFF:
                data_stat = np.squeeze(np.diff(data, axis=1))
        sources[obj_name] = data_stat

    sources_names = sorted(list(sources.keys()))
    N = len(sources_names)
    # T = _addon().get_max_time_steps() # len(sources[sources_names[0]]) + 2
    fcurves_num = mu.count_fcurves(parent_obj)
    if fcurves_num < len(sources_names):
        now = time.time()
        parent_obj.animation_data_clear()
        for obj_counter, source_name in enumerate(sources_names):
            mu.time_to_go(now, obj_counter, N, runs_num_to_print=10)
            data = sources[source_name]
            mu.insert_keyframe_to_custom_prop(parent_obj, source_name, 0, 1)
            mu.insert_keyframe_to_custom_prop(parent_obj, source_name, 0, T + 2)

            for ind in range(T):
                mu.insert_keyframe_to_custom_prop(parent_obj, source_name, data[ind], ind + 2)

            fcurves = parent_obj.animation_data.action.fcurves[obj_counter]
            mod = fcurves.modifiers.new(type='LIMITS')
    else:
        for fcurve_ind, fcurve in enumerate(parent_obj.animation_data.action.fcurves):
            fcurve_name = mu.get_fcurve_name(fcurve)
            fcurve.keyframe_points[0].co[1] = 0
            fcurve.keyframe_points[-1].co[1] = 0
            for t in range(T):
                fcurve.keyframe_points[t + 1].co[1] = sources[fcurve_name][t]

    mu.view_all_in_graph_editor()
    print('Finished keyframing {}!!'.format(parent_obj.name))


def load_electrodes_data(stat='diff'):
    # stat = 'diff' 'avg' if bpy.context.scene.selection_type == 'conds' else 'diff'
    fol = op.join(mu.get_user_fol(), 'electrodes')
    data_file = op.join(fol, 'electrodes_data_{}.npz'.format(stat))
    if op.isfile(data_file):
        f = np.load(data_file)
        data = f['data']
        names = f['names']
        conditions = f['conditions']
    elif op.isfile(op.join(fol, 'electrodes_data_{}.npy'.format(stat))) and op.isfile(
            op.join(fol, 'electrodes_data_{}_meta.npz'.format(stat))):
        data_file = op.join(fol, 'electrodes_data_{}.npy'.format(stat))
        f = np.load(data_file)
        data = f['data']
        meta_data = np.load(op.join(fol, 'electrodes_data_{}_meta.npz'.format(stat)))
        names = meta_data['names']
        conditions = meta_data['conditions']
    else:
        data, names, conditions = None, None, None
    return data, names, conditions


def meg_evoked_files_update(self, context):
    _meg_evoked_files_update()


def _meg_evoked_files_update():
    external_obj = bpy.data.objects.get('External', None)
    if not external_obj is None:
        evoked_name = bpy.context.scene.meg_evoked_files
        DataMakerPanel.externals = [ext.name[len(evoked_name) + 1:] for ext in external_obj.children \
                                    if ext.name.startswith(evoked_name)]
        items = [(name, name, '', ind) for ind, name in enumerate(DataMakerPanel.externals)]
        bpy.types.Scene.evoked_objects = bpy.props.EnumProperty(items=items, description="meg_evoked_types")


class AddDataToEEGSensors(bpy.types.Operator):
    bl_idname = "mmvt.eeg_add_data"
    bl_label = "add data eeg"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        parnet_name = 'EEG_sensors'
        parent_obj = bpy.data.objects.get(parnet_name)
        if parent_obj is None:
            layers_array = [False] * 20
            create_empty_if_doesnt_exists(parnet_name, _addon().BRAIN_EMPTY_LAYER, layers_array, parnet_name)
        data_fname = op.join(mu.get_user_fol(), 'eeg', 'eeg_data.npy')
        meta_fname = op.join(mu.get_user_fol(), 'eeg', 'eeg_data_meta.npz')
        if not op.isfile(data_fname) or not op.isfile(meta_fname):
            mu.log_err('EEG data should be here {} (data) and here {} (meta data)'.format(data_fname, meta_fname), logging)
        else:
            data = DataMakerPanel.eeg_data = np.load(data_fname, mmap_mode='r')
            meta = DataMakerPanel.eeg_meta = np.load(meta_fname)
            add_data_to_electrodes(data, meta, window_len=2)
            add_data_to_electrodes_parent_obj(parent_obj, data, meta, window_len=2)
            bpy.types.Scene.eeg_data_exist = True
        if bpy.data.objects.get(' '):
            bpy.context.scene.objects.active = bpy.data.objects[' ']
        return {"FINISHED"}


class AddDataToMEGSensors(bpy.types.Operator):
    bl_idname = "mmvt.meg_sensors_add_data"
    bl_label = "add meg sensors data"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        add_data_to_meg_sensors()
        return {"FINISHED"}


class AddDataToElectrodes(bpy.types.Operator):
    bl_idname = "mmvt.electrodes_add_data"
    bl_label = "add_data2 electrodes"
    bl_options = {"UNDO"}
    current_root_path = ''

    def invoke(self, context, event=None):
        # self.current_root_path = bpy.path.abspath(bpy.context.scene.conf_path)
        parent_obj = bpy.data.objects['Deep_electrodes']
        base_path = mu.get_user_fol()
        data, meta = None, None
        source_file = op.join(base_path, 'electrodes', 'electrodes{}_data_diff.npz'.format(
            '_bipolar' if bpy.context.scene.bipolar else ''))
            # 'avg' if bpy.context.scene.selection_type == 'conds' else 'diff'))
        if op.isfile(source_file):
            meta = np.load(source_file)
            data = meta['data']
        else:
            source_file = op.join(base_path, 'electrodes', 'electrodes{}_data_{}_data.npy'.format(
                '_bipolar' if bpy.context.scene.bipolar else '',
                'avg' if bpy.context.scene.selection_type == 'conds' else 'diff'))
            meta_file = op.join(base_path, 'electrodes', 'electrodes{}_data_{}_meta.npz'.format(
                '_bipolar' if bpy.context.scene.bipolar else '',
                'avg' if bpy.context.scene.selection_type == 'conds' else 'diff'))
            if op.isfile(source_file) and op.isfile(meta_file):
                data = np.load(source_file)
                meta = np.load(meta_file)
            else:
                mu.log_err('No electrodes data file!', logging)
        if not data is None and not meta is None:
            print('Loading electordes data from {}'.format(source_file))
            conditions = add_data_to_electrodes(data, meta)
            selection_panel.set_conditions_enum(conditions)
            add_data_to_electrodes_parent_obj(parent_obj, data, meta)
            bpy.types.Scene.electrodes_data_exist = True
        if bpy.data.objects.get(' '):
            bpy.context.scene.objects.active = bpy.data.objects[' ']
        return {"FINISHED"}


class DataMakerPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "Data Panel"
    addon = None
    meg_evoked_files = []
    evoked_files = []
    externals = []
    eeg_data, eeg_meta = None, None
    meg_labels_data_exist = False
    subcortical_meg_data_exist = False
    fMRI_dynamic_exist = False

    def draw(self, context):
        layout = self.layout
        # layout.prop(context.scene, 'conf_path')
        col = self.layout.column(align=True)
        col.prop(context.scene, 'atlas', text="Atlas")
        # if not bpy.types.Scene.brain_imported:
        # col.operator("mmvt.anatomy_preproc", text="Run Preporc", icon='BLENDER')
        col.operator(ImportBrain.bl_idname, text="Import Brain", icon='MATERIAL_DATA')
        col.prop(context.scene, 'inflated_morphing', text="Include inflated morphing")
        # if not bpy.types.Scene.electrodes_imported:
        electrodes_positions_files = glob.glob(op.join(mu.get_user_fol(), 'electrodes', 'electrodes*positions*.npz'))
        meg_sensors_positions_file = op.join(mu.get_user_fol(), 'meg', 'meg_sensors_positions.npz')
        meg_data_npz = op.join(mu.get_user_fol(), 'meg', 'meg_sensors_evoked_data_meta.npz')
        meg_data_npy = op.join(mu.get_user_fol(), 'meg', 'meg_sensors_evoked_data.npy')
        eeg_sensors_positions_file = op.join(mu.get_user_fol(), 'eeg', 'eeg_positions.npz')
        eeg_data_npz = op.join(mu.get_user_fol(), 'eeg', 'eeg_data.npz')
        eeg_data_npy = op.join(mu.get_user_fol(), 'eeg', 'eeg_data.npy')
        if len(electrodes_positions_files) > 0:
            col.prop(context.scene, 'electrodes_radius', text="Electrodes' radius")
            col.prop(context.scene, 'electrodes_positions_files', text="")
            col.prop(context.scene, 'bipolar', text="Bipolar")
            col.operator(ImportElectrodes.bl_idname, text="Import Electrodes", icon='COLOR_GREEN')

        if DataMakerPanel.meg_labels_data_exist:
            col = self.layout.column(align=True)
            col.prop(context.scene, 'labels_data_files', text="")
            col.operator(AddDataToBrain.bl_idname, text="Add MEG data to Brain", icon='FCURVE')
            col.prop(context.scene, 'add_meg_labels_data', text="labels")
            col.prop(context.scene, 'import_unknown', text="Import unknown")
        if DataMakerPanel.subcortical_meg_data_exist:
            col.prop(context.scene, 'add_meg_subcorticals_data', text="subcorticals")
        if DataMakerPanel.fMRI_dynamic_exist:
            col.prop(context.scene, 'fMRI_dynamic_files', text="")
            col.operator(AddfMRIDynamicsToBrain.bl_idname, text="Add fMRI data", icon='FCURVE')
        # if bpy.types.Scene.electrodes_imported and (not bpy.types.Scene.electrodes_data_exist):
        col.operator("mmvt.electrodes_add_data", text="Add data to Electrodes", icon='FCURVE')
        # if len(DataMakerPanel.evoked_files) > 0:
        #     layout.label(text='External MEG evoked files:')
        #     layout.prop(context.scene, 'meg_evoked_files', text="")
        #     layout.operator(AddOtherSubjectMEGEvokedResponse.bl_idname, text="Add MEG evoked response", icon='FCURVE')
        #     if len(DataMakerPanel.externals) > 0:
        #         layout.prop(context.scene, 'evoked_objects', text="")
        #         select_text = 'Deselect' if get_external_meg_evoked_selected() else 'Select'
        #         select_icon = 'BORDER_RECT' if select_text == 'Select' else 'PANEL_CLOSE'
        #         layout.operator(SelectExternalMEGEvoked.bl_idname, text=select_text, icon=select_icon)
        if op.isfile(meg_sensors_positions_file):
            col.operator(ImportMEGSensors.bl_idname, text="Import MEG sensors", icon='COLOR_GREEN')
            # col.operator("mmvt.meg_mesh", text="Creating MEG mesh", icon='COLOR_GREEN')
        if op.isfile(meg_data_npy) or op.isfile(meg_data_npz) and bpy.data.objects.get('MEG_sensors'):
            col.operator(AddDataToMEGSensors.bl_idname, text="Add data to MEG sensors", icon='FCURVE')

        if op.isfile(eeg_sensors_positions_file):
            col.operator(ImportEEG.bl_idname, text="Import EEG sensors", icon='COLOR_GREEN')
            col.operator(CreateEEGMesh.bl_idname, text="Creating EEG mesh", icon='COLOR_GREEN')
        # if op.isfile(eeg_data):
        if op.isfile(eeg_data_npy) or op.isfile(eeg_data_npz):
            col.operator(AddDataToEEGSensors.bl_idname, text="Add data to EEG", icon='FCURVE')


# def load_meg_evoked():
#     evoked_fol = op.join(mu.get_user_fol(), 'meg_evoked_files')
#     if op.isdir(evoked_fol):
#         DataMakerPanel.evoked_files = evoked_files = glob.glob(op.join(evoked_fol, '*_labels_data_rh.npz'))
#         basenames = [mu.namebase(fname).split('_')[0] for fname in evoked_files]
#         files_items = [(name, name, '', ind) for ind, name in enumerate(basenames)]
#         bpy.types.Scene.meg_evoked_files = bpy.props.EnumProperty(
#             items=files_items, description="meg_evoked_files", update=meg_evoked_files_update)


def init(addon):
    DataMakerPanel.addon = addon
    logging.basicConfig(filename='mmvt_addon.log', level=logging.DEBUG)
    bpy.context.scene.electrodes_radius = 0.15
    atlas = bpy.context.scene.atlas
    # load_meg_evoked()
    # _meg_evoked_files_update()
    labels_data_files = glob.glob(op.join(mu.get_user_fol(), 'meg', 'labels_data_{}_*_rh.npz'.format(atlas)))
    if len(labels_data_files) > 0:
        DataMakerPanel.meg_labels_data_exist = True
        files_names = [mu.namebase(fname)[len('labels_data_{}_'.format(atlas)):-3] for fname in labels_data_files]
        items = [(c, c, '', ind) for ind, c in enumerate(files_names)]
        bpy.types.Scene.labels_data_files = bpy.props.EnumProperty(items=items, description="labels data files")
        bpy.context.scene.labels_data_files = files_names[0]
    if op.isfile(op.join(mu.get_user_fol(), 'meg', 'subcortical_meg_activity.npz')):
        DataMakerPanel.subcortical_meg_data_exist = True

    electrodes_positions_files = glob.glob(op.join(mu.get_user_fol(), 'electrodes', 'electrodes*positions*.npz'))
    if len(electrodes_positions_files) > 0:
        files_names = [mu.namebase(fname) for fname in electrodes_positions_files]
        items = [(c, c, '', ind) for ind, c in enumerate(files_names)]
        bpy.types.Scene.electrodes_positions_files = bpy.props.EnumProperty(
            items=items,description="Electrodes positions")
        bpy.context.scene.electrodes_positions_files = files_names[0]
    if bpy.data.objects.get('Deep_electrodes'):
        bpy.context.scene.bipolar = np.all(['-' in o.name for o in bpy.data.objects['Deep_electrodes'].children])
    fMRI_labels_sources_files = glob.glob(
        op.join(mu.get_user_fol(), 'fmri', 'labels_data_{}_*_rh.npz'.format(bpy.context.scene.atlas)))
    if len(fMRI_labels_sources_files) > 0:
        DataMakerPanel.fMRI_dynamic_exist = True
        files_names = ['fMRI {}'.format(mu.namebase(fname)[len('labels_data_'):-len('_rh')].replace('_', ' '))
                       for fname in fMRI_labels_sources_files if atlas in mu.namebase(fname)]
        items = [(c, c, '', ind) for ind, c in enumerate(files_names)]
        bpy.types.Scene.fMRI_dynamic_files = bpy.props.EnumProperty(
            items=items,description="fMRI_dynamic")
        bpy.context.scene.fMRI_dynamic_files = files_names[0]
    register()


def register():
    try:
        unregister()
        bpy.utils.register_class(DataMakerPanel)
        bpy.utils.register_class(AddDataToElectrodes)
        # bpy.utils.register_class(AddDataNoCondsToBrain)
        bpy.utils.register_class(AddDataToBrain)
        bpy.utils.register_class(AddfMRIDynamicsToBrain)
        bpy.utils.register_class(AddDataToEEGSensors)
        bpy.utils.register_class(AddDataToMEGSensors)
        bpy.utils.register_class(ImportMEGSensors)
        bpy.utils.register_class(ImportElectrodes)
        bpy.utils.register_class(ImportEEG)
        bpy.utils.register_class(ImportRois)
        bpy.utils.register_class(ImportBrain)
        bpy.utils.register_class(CreateEEGMesh)
        bpy.utils.register_class(AnatomyPreproc)
        # bpy.utils.register_class(AddOtherSubjectMEGEvokedResponse)
        # bpy.utils.register_class(SelectExternalMEGEvoked)
        # print('Data Panel was registered!')
    except:
        print("Can't register Data Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(DataMakerPanel)
        bpy.utils.unregister_class(AddDataToElectrodes)
        # bpy.utils.unregister_class(AddDataNoCondsToBrain)
        bpy.utils.unregister_class(AddDataToBrain)
        bpy.utils.unregister_class(AddfMRIDynamicsToBrain)
        bpy.utils.unregister_class(AddDataToEEGSensors)
        bpy.utils.unregister_class(AddDataToMEGSensors)
        bpy.utils.unregister_class(ImportMEGSensors)
        bpy.utils.unregister_class(ImportElectrodes)
        bpy.utils.unregister_class(ImportRois)
        bpy.utils.unregister_class(ImportEEG)
        bpy.utils.unregister_class(ImportBrain)
        bpy.utils.unregister_class(CreateEEGMesh)
        bpy.utils.unregister_class(AnatomyPreproc)
        # bpy.utils.unregister_class(AddOtherSubjectMEGEvokedResponse)
        # bpy.utils.unregister_class(SelectExternalMEGEvoked)
    except:
        pass

