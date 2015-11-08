import bpy
import numpy as np
import os
import time
import mathutils
import glob
import math

# http://www.blender.org/api/blender_python_api_2_66_release/bpy.props.html
print("Neuroscience add on started!")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ data Panel ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
bpy.types.Scene.conf_path = bpy.props.StringProperty(name="Root Path", default="", description="Define the root path of the project", subtype='DIR_PATH')

# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv Import Brain - START vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv


def import_brain(base_path):
	brain_layer = 5
	for ii in range(len(bpy.context.scene.layers)):
		bpy.context.scene.layers[ii] = (ii == brain_layer)

	layers_array = bpy.context.scene.layers
	emptys_names = ["Functional maps", "Subcortical_activity_map"]
	for name in emptys_names:
		create_empty_if_doesnt_exists(name, brain_layer, layers_array, 'Functional maps')

	brain_layer = 11

	for ii in range(len(bpy.context.scene.layers)):
		bpy.context.scene.layers[ii] = (ii == brain_layer)

	for ii in range(len(bpy.context.scene.layers)):
		bpy.context.scene.layers[ii] = (ii == brain_layer)

	print("importing Hemispheres")
	# for cur_val in bpy.context.scene.layers:
	#     print(cur_val)
	#  print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	for i in os.listdir(base_path):
		bpy.ops.object.select_all(action='DESELECT')
		if i.endswith(".ply"):
			print(i)
			bpy.ops.import_mesh.ply(filepath=os.path.join(base_path, i))
			cur_obj = bpy.context.selected_objects[0]
			cur_obj.select = True
			bpy.ops.object.shade_smooth()
			cur_obj.scale = [0.1]*3
			cur_obj.hide = False
			cur_obj.name = cur_obj.name.split(sep='.')[0]
			cur_obj.active_material = bpy.data.materials['Activity_map_mat']
			cur_obj.parent = bpy.data.objects["Functional maps"]
	bpy.ops.object.select_all(action='DESELECT')


def create_subcortical_activity_mat(name):
	cur_mat = bpy.data.materials['subcortical_activity_mat'].copy()
	cur_mat.name = name+'_Mat'


def import_subcorticals(base_path):
	brain_layer = 5
	for ii in range(len(bpy.context.scene.layers)):
		bpy.context.scene.layers[ii] = (ii == brain_layer)

	layers_array = bpy.context.scene.layers
	emptys_names = ["Functional maps", "Subcortical_activity_map"]
	for name in emptys_names:
		create_empty_if_doesnt_exists(name, brain_layer, layers_array, 'Functional maps')

	brain_layer = 11

	for ii in range(len(bpy.context.scene.layers)):
		bpy.context.scene.layers[ii] = (ii == brain_layer)

	for ii in range(len(bpy.context.scene.layers)):
		bpy.context.scene.layers[ii] = (ii == brain_layer)

	print("importing Subcortical structures")
	# for cur_val in bpy.context.scene.layers:
	#     print(cur_val)
	#  print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	for i in os.listdir(base_path):
		bpy.ops.object.select_all(action='DESELECT')
		if i.endswith(".ply"):
			print(i)
			bpy.ops.import_mesh.ply(filepath=os.path.join(base_path, i))
			cur_obj = bpy.context.selected_objects[0]
			cur_obj.select = True
			bpy.ops.object.shade_smooth()
			cur_obj.scale = [0.1]*3
			cur_obj.hide = False
			cur_obj.name = cur_obj.name.split(sep='.')[0]
			create_subcortical_activity_mat(cur_obj.name)
			cur_obj.active_material = bpy.data.materials[cur_obj.name+'_Mat']
			cur_obj.parent = bpy.data.objects["Subcortical_activity_map"]
	bpy.ops.object.select_all(action='DESELECT')


class ImportBrain(bpy.types.Operator):
	bl_idname = "ohad.brain_importing"
	bl_label = "import2 brain"
	bl_options = {"UNDO"}
	current_root_path = ''

	def invoke(self, context, event=None):
		self.current_root_path = bpy.path.abspath(bpy.context.scene.conf_path)
		print("importing ROIs")
		import_rois(self.current_root_path)
		import_brain(self.current_root_path)
		import_subcorticals(os.path.join(self.current_root_path, 'subcortical_ply_names'))
		last_obj = context.active_object.name
		print('last obj is -'+last_obj)

		bpy.data.objects[' '].select = True
		bpy.data.objects[last_obj].select = False
		context.scene.objects.active = bpy.data.objects[' ']
		set_appearance_show_rois_layer(bpy.context.scene, True)
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
    anatomy_inputs = {'Cortex-rh': os.path.join(base_path, 'rh.aparc.pial.names'), 'Cortex-lh': os.path.join(base_path, 'lh.aparc.pial.names'), 'Subcortical structures': os.path.join(base_path, 'subcortical_ply_names')}
    brain_layer = 5

    for ii in range(len(bpy.context.scene.layers)):
        bpy.context.scene.layers[ii] = (ii == brain_layer)

    layers_array = bpy.context.scene.layers
    emptys_names = ["Brain", "Subcortical structures", "Cortex-lh", "Cortex-rh"]
    for name in emptys_names:
        create_empty_if_doesnt_exists(name, brain_layer, layers_array)

    bpy.context.scene.layers[10] = True
    for ii in range(len(bpy.context.scene.layers)):
        bpy.context.scene.layers[ii] = ii == 10

    for anatomy_name, base_path in anatomy_inputs.items():
        current_mat = bpy.data.materials['unselected_label_Mat_cortex']
        if anatomy_name == 'Subcortical structures':
            current_mat = bpy.data.materials['unselected_label_Mat_subcortical']
        for i in os.listdir(base_path):
            bpy.ops.object.select_all(action='DESELECT')
            if i.endswith(".ply"):
                print(i)
                bpy.ops.import_mesh.ply(filepath=os.path.join(base_path, i))
                cur_obj = bpy.context.selected_objects[0]
                cur_obj.select = True
                # print('~~~~~~~~~~~~~~~~~~~~~')
                # print(bpy.context.active_object)
                # print('~~~~~~~~~~~~~~~~~~~~~')
                bpy.ops.object.shade_smooth()
                cur_obj.parent = bpy.data.objects[anatomy_name]
                cur_obj.scale = [0.1]*3
                cur_obj.active_material = current_mat
                cur_obj.hide = False
                cur_obj.name = cur_obj.name.split(sep='.')[0]
                # time.sleep(0.3)
    bpy.ops.object.select_all(action='DESELECT')


class ImportRoisClass(bpy.types.Operator):
    bl_idname = "ohad.roi_importing"
    bl_label = "import2 ROIs"
    bl_options = {"UNDO"}
    current_root_path = ''

    def invoke(self, context, event=None):
        self.current_root_path = bpy.path.abspath(bpy.context.scene.conf_path)
        import_brain(self.current_root_path)
        return {"FINISHED"}

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Import Brain - END^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv Import Electrodes - START vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv


def create_sphere(loc, rad, my_layers, name):
    bpy.ops.mesh.primitive_uv_sphere_add(ring_count=30, size=rad, view_align=False, enter_editmode=False, location=loc,
                                         layers=my_layers)
    bpy.ops.object.shade_smooth()

    # Rename the object
    bpy.context.active_object.name = name


def create_and_set_material(obj):
    # curMat = bpy.data.materials['OrigPatchesMat'].copy()
    cur_mat = bpy.data.materials['Deep_electrode_mat'].copy()
    cur_mat.name = obj.name+'_Mat'
    cur_mat.node_tree.nodes["RGB"].outputs[0].default_value = (0, 0, 1, 1)
    obj.active_material = cur_mat


def import_electrodes(base_path):
    input_file = os.path.join(base_path, "electrodes.npz")

    print('Adding deep electrodes')
    f = np.load(input_file)
    print('loaded')

    deep_electrodes_layer = 1
    electrode_size = 0.25

    layers_array = [False]*20

    if bpy.data.objects.get("Deep_electrodes") is None:
        layers_array[5] = True
        bpy.ops.object.empty_add(type='PLAIN_AXES', radius=1, view_align=False, location=(0, 0, 0), layers=layers_array)
        bpy.data.objects['Empty'].name = 'Deep_electrodes'

    layers_array = [False]*20
    layers_array[deep_electrodes_layer] = True

    for (x, y, z), name in zip(f['pos'], f['names']):
        print('creating '+str(name)[2:-1])
        create_sphere((x*0.1, y*0.1, z*0.1), electrode_size, layers_array, str(name)[2:-1])
        cur_obj = bpy.data.objects[str(name)[2:-1]]
        cur_obj.select = True
        cur_obj.parent = bpy.data.objects['Deep_electrodes']
        # cur_obj.active_material = bpy.data.materials['Deep_electrode_mat']
        create_and_set_material(cur_obj)


class ImportElectrodes(bpy.types.Operator):
    bl_idname = "ohad.electrodes_importing"
    bl_label = "import2 electrodes"
    bl_options = {"UNDO"}
    current_root_path = ''

    def invoke(self, context, event=None):
        self.current_root_path = bpy.path.abspath(bpy.context.scene.conf_path)
        import_electrodes(self.current_root_path)
        print('Electrodes importing is Finished ')
        return {"FINISHED"}

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Import Electrodes - END^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv Add data to brain - START vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv


def insert_keyframe_to_custom_prop(obj, prop_name, value, keyframe):
    bpy.context.scene.objects.active = obj
    obj.select = True
    obj[prop_name] = value
    obj.keyframe_insert(data_path='['+'"'+prop_name+'"'+']', frame=keyframe)


def add_data_to_brain(base_path):
    source_files = [os.path.join(base_path, 'labels_data_lh.npz'), os.path.join(base_path, 'labels_data_rh.npz'),
                    os.path.join(base_path, 'sub_cortical_activity.npz')]
    print('Adding data to Brain')
    brain_obj = bpy.data.objects['Brain']
    obj_counter = 0
    for input_file in source_files:
        f = np.load(input_file)
        print('loaded')

        for obj_name, data in zip(f['names'], f['data']):
            # print('in label loop')
            obj_name = str(obj_name)
            if obj_name[1] == "'":
                obj_name = obj_name[2:-1]
            print(obj_name)
            cur_obj = bpy.data.objects[obj_name]

            for cond_ind, cond_str in enumerate(f['conditions']):
                cond_str = str(cond_str)
                if cond_str[1] == "'":
                    cond_str = cond_str[2:-1]
                # Set the values to zeros in the first and last frame for current object(current label)
                insert_keyframe_to_custom_prop(cur_obj, obj_name+'_'+cond_str, 0, 1)
                insert_keyframe_to_custom_prop(cur_obj, obj_name+'_'+cond_str, 0, len(f['data'][0])+2)

                print('keyframing '+obj_name+' object')
                # For every time point insert keyframe to current object
                for ind, timepoint in enumerate(data[:, cond_ind]):
                    # print('keyframing '+obj_name+' object')
                    insert_keyframe_to_custom_prop(cur_obj, obj_name+'_'+cond_str, timepoint, ind+2)

                # remove the orange keyframe sign in the fcurves window
                fcurves = bpy.data.objects[obj_name].animation_data.action.fcurves[cond_ind]
                mod = fcurves.modifiers.new(type='LIMITS')

            # Brain object handling
            # Set the values to zeros in the first and last frame for Brain object
            insert_keyframe_to_custom_prop(brain_obj, obj_name+' mean', 0, 1)
            insert_keyframe_to_custom_prop(brain_obj, obj_name+' mean', 0, len(f['data'][0])+2)

            # For every time point insert keyframe to the main Brain object
            # If you want to delete prints make sure no sleep is needed
            print('keyframing Brain object')
            for ind in range(len(data[:, cond_ind])):
                if len(data[ind]) == 2:
                    # print('keyframing Brain object')
                    insert_keyframe_to_custom_prop(brain_obj, obj_name+' mean', (data[ind][0]+data[ind][1])/2, ind+2)
                    # print('keyframed')

            # remove the orange keyframe sign in the fcurves window
            fcurves = bpy.data.objects['Brain'].animation_data.action.fcurves[obj_counter]
            mod = fcurves.modifiers.new(type='LIMITS')
            obj_counter += 1
    try:
        bpy.ops.graph.previewrange_set()
    except:
        pass
    print('Finished keyframing!!')


class AddDataToBrain(bpy.types.Operator):
    bl_idname = "ohad.brain_add_data"
    bl_label = "add_data2 brain"
    bl_options = {"UNDO"}
    current_root_path = ''

    def invoke(self, context, event=None):
        self.current_root_path = bpy.path.abspath(bpy.context.scene.conf_path)
        add_data_to_brain(self.current_root_path)
        return {"FINISHED"}
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Add data to brain - END^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv Add data to Electrodes - START vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv


def insert_keyframe_to_custom_prop(obj, prop_name, value, keyframe):
    bpy.context.scene.objects.active = obj
    obj.select = True
    obj[prop_name] = value
    obj.keyframe_insert(data_path='['+'"'+prop_name+'"'+']', frame=keyframe)


def add_data_to_electrodes(base_path):
    source_files = [os.path.join(base_path, "electrodes_data.npz")]

    print('Adding data to Electrodes')
    deep_electrodes_obj = bpy.data.objects['Deep_electrodes']
    obj_counter = 0
    for input_file in source_files:
        f = np.load(input_file)
        print('loaded')

        for obj_name, data in zip(f['names'], f['data']):
            obj_name = str(obj_name)[2:-1]
            print(obj_name)
            cur_obj = bpy.data.objects[obj_name]

            for cond_ind, cond_str in enumerate(f['conditions']):
                cond_str = str(cond_str)
                if cond_str[1] == "'":
                    cond_str = cond_str[2:-1]
                # Set the values to zeros in the first and last frame for current object(current label)
                insert_keyframe_to_custom_prop(cur_obj, obj_name+'_'+cond_str, 0, 1)
                insert_keyframe_to_custom_prop(cur_obj, obj_name+'_'+str(cond_str), 0, len(f['data'][0])+2)

                print('keyframing '+obj_name+' object in condition '+ cond_str)
                # For every time point insert keyframe to current object
                for ind, timepoint in enumerate(data[:, cond_ind]):
                    # print('keyframing '+obj_name+' object in condition '+ cond_str)
                    insert_keyframe_to_custom_prop(cur_obj, obj_name+'_'+str(cond_str), timepoint, ind+2)
                # remove the orange keyframe sign in the fcurves window
                fcurves = bpy.data.objects[obj_name].animation_data.action.fcurves[cond_ind]
                mod = fcurves.modifiers.new(type='LIMITS')

            # Brain object handling
            # Set the values to zeros in the first and last frame for Brain object
            insert_keyframe_to_custom_prop(deep_electrodes_obj, obj_name+' mean', 0, 1)
            insert_keyframe_to_custom_prop(deep_electrodes_obj, obj_name+' mean', 0, len(f['data'][0])+2)

            print('keyframing Deep_electrodes object')
            # For every time point insert keyframe to the main Brain object
            for ind in range(len(data[:, cond_ind])):
                # print('keyframing Deep_electrodes object')
                insert_keyframe_to_custom_prop(deep_electrodes_obj, obj_name+' mean', (data[ind][0]+data[ind][1])/2,
                                               ind+2)
                # print('keyframed')

            # remove the orange keyframe sign in the fcurves window
            fcurves = deep_electrodes_obj.animation_data.action.fcurves[obj_counter]
            mod = fcurves.modifiers.new(type='LIMITS')
            obj_counter += 1

    print('Finished keyframing!!')

class AddDataToElectrodes(bpy.types.Operator):
    bl_idname = "ohad.electrodes_add_data"
    bl_label = "add_data2 electrodes"
    bl_options = {"UNDO"}
    current_root_path = ''

    def invoke(self, context, event=None):
        self.current_root_path = bpy.path.abspath(bpy.context.scene.conf_path)
        add_data_to_electrodes(self.current_root_path)
        return {"FINISHED"}
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Add data to Electrodes - END^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


class DataMakerPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
    bl_label = "Data Panel"

    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene, 'conf_path')
        col1 = self.layout.column(align=True)
        col1.operator("ohad.brain_importing", text="Import Brain")
        # col1.operator("ohad.roi_importing", text="Import ROIs")
        col1.operator("ohad.electrodes_importing", text="Import Electrodes")
        col2 = self.layout.column(align=True)
        col2.operator("ohad.brain_add_data", text="Add data to Brain")
        col2.operator("ohad.electrodes_add_data", text="Add data to Electrodes")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ data Panel ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Selection Panel ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# select all ROIs
# Select all Electrodes
# select brain
# select Deep_electrodes
# clear selections


def deselect_all():
    for obj in bpy.data.objects:
        obj.select = False


def select_all_rois():
	for subHierarchy in bpy.data.objects['Brain'].children:
		for obj in subHierarchy.children:
			obj.select = True


class SelectionMakerPanel(bpy.types.Panel):
	bl_space_type = "GRAPH_EDITOR"
	bl_region_type = "UI"
	bl_context = "objectmode"
	bl_category = "Ohad"
	bl_label = "Selection Panel"

	@staticmethod
	def draw(self, context):
		col1 = self.layout.column(align=True)
		# col1.operator("select.ROIs", text="ROIs")
		col1.operator("ohad.roi_selection", text="Select all ROIs", icon='BORDER_RECT')
		col1.operator("ohad.electrodes_selection", text="Select all Electrodes", icon='BORDER_RECT')
		col1.operator("ohad.clear_selection", text="Deselect all", icon='PANEL_CLOSE')


class SelectAllRois(bpy.types.Operator):
	bl_idname = "ohad.roi_selection"
	bl_label = "select2 ROIs"
	bl_options = {"UNDO"}

	@staticmethod
	def invoke(self, context, event=None):
		select_all_rois()
		return {"FINISHED"}


def select_all_electrodes():
	for obj in bpy.data.objects['Deep_electrodes'].children:
			obj.select = True


class SelectAllElectrodes(bpy.types.Operator):
	bl_idname = "ohad.electrodes_selection"
	bl_label = "select2 Electrodes"
	bl_options = {"UNDO"}

	@staticmethod
	def invoke(self, context, event=None):
		select_all_electrodes()
		return {"FINISHED"}


class ClearSelection(bpy.types.Operator):
	bl_idname = "ohad.clear_selection"
	bl_label = "deselect all"
	bl_options = {"UNDO"}

	@staticmethod
	def invoke(self, context, event=None):
		for obj in bpy.data.objects:
			obj.select = False
		return {"FINISHED"}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Selection Panel ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Filter Panel ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def filter_draw(self, context):
	layout = self.layout
	layout.prop(context.scene, "filter_topK", text="Top K")
	row = layout.row(align=0)
	row.prop(context.scene, "filter_from", text="From")
	# row.label(str(GrabFromFiltering.value))
	row.operator(GrabFromFiltering.bl_idname, text="", icon = 'BORDERMOVE')
	# row.operator("ohad.grab_from", text="", icon = 'BORDERMOVE')
	row.prop(context.scene, "filter_to", text="To")
	row.operator("ohad.grab_to", text="", icon = 'BORDERMOVE')
	layout.prop(context.scene, "filter_curves_type", text="")
	layout.prop(context.scene, "filter_curves_func", text="")
	layout.operator("ohad.filter", text="Filter " + bpy.context.scene.filter_curves_type, icon = 'BORDERMOVE')
	layout.operator("ohad.filter_clear", text="Clear Filtering", icon='PANEL_CLOSE')

	# bpy.context.area.type = 'GRAPH_EDITOR'
	# filter_to = bpy.context.scence.frame_preview_end

files_names = {'MEG': 'labels_data_', 'Electrodes':'electrodes_data.npz'}

bpy.types.Scene.filter_topK = bpy.props.IntProperty(default=1, min=0, description="The top K elements to be shown")
bpy.types.Scene.filter_from = bpy.props.IntProperty(default=0, min=0, description="When to filter from")
bpy.types.Scene.filter_to = bpy.props.IntProperty(default=bpy.data.scenes['Scene'].frame_preview_end, min=0, description="When to filter to")
bpy.types.Scene.filter_curves_type = bpy.props.EnumProperty(items=[("MEG", "MEG time course", "", 1), ("Electrodes", " Electrodes time course", "", 2)], description="Type of curve to be filtered", update=filter_draw)
bpy.types.Scene.filter_curves_func = bpy.props.EnumProperty(items=[("RMS", "RMS", "RMS between the two conditions", 1), ("SumAbs", "SumAbs", "Sum of the abs values", 2)], description="Filtering function", update=filter_draw)


class FilteringMakerPanel(bpy.types.Panel):
	bl_space_type = "GRAPH_EDITOR"
	bl_region_type = "UI"
	bl_context = "objectmode"
	bl_category = "Ohad"
	bl_label = "Filter number of curves"

	def draw(self, context):
		filter_draw(self, context)


class GrabFromFiltering(bpy.types.Operator):
	bl_idname = "ohad.grab_from"
	bl_label = "grab from"
	bl_options = {"UNDO"}

	def invoke(self, context, event=None):
		print(bpy.context.scene.frame_current)
		context.scene.filter_from = bpy.context.scene.frame_current
		print(bpy.context.scene.filter_from)
		return {"FINISHED"}


class GrabToFiltering(bpy.types.Operator):
	bl_idname = "ohad.grab_to"
	bl_label = "grab to"
	bl_options = {"UNDO"}

	def invoke(self, context, event=None):
		print(bpy.context.scene.frame_current)
		context.scene.filter_to = bpy.context.scene.frame_current
		print(bpy.context.scene.filter_to)
		return {"FINISHED"}


class ClearFiltering(bpy.types.Operator):
	bl_idname = "ohad.filter_clear"
	bl_label = "filter clear"
	bl_options = {"UNDO"}

	def invoke(self, context, event=None):
		for subHierchy in bpy.data.objects['Brain'].children:
			new_mat = bpy.data.materials['unselected_label_Mat_cortex']
			if subHierchy.name == 'Subcortical structures':
				new_mat = bpy.data.materials['unselected_label_Mat_subcortical']
			for obj in subHierchy.children:
				obj.active_material = new_mat

		if bpy.data.objects.get('Deep_electrodes'):
			for obj in bpy.data.objects['Deep_electrodes'].children:
				obj.active_material.node_tree.nodes["Layer Weight"].inputs[0].default_value = 1

		type_of_filter = bpy.context.scene.filter_curves_type
		if type_of_filter == 'MEG':
			select_all_rois()
		elif type_of_filter == 'MEG':
			select_all_electrodes()
		return {"FINISHED"}


class Filtering(bpy.types.Operator):
    bl_idname = "ohad.filter"
    bl_label = "Filter deep elctrodes"
    bl_options = {"UNDO"}
    topK = -1

    def get_object_to_filter(self, source_files):
        data, names = [], []
        for input_file in source_files:
            f = np.load(input_file)
            data.append(f['data'])
            temp_names = [str(name) for name in f['names']]
            for ind in range(len(temp_names)):
                if temp_names[ind][1] == "'":
                    temp_names[ind] = temp_names[ind][2:-1]
            names.extend(temp_names)

        print('filtering {}-{}'.format(self.filter_from, self.filter_to))
        t_range = range(self.filter_from, self.filter_to+1)

        print(self.type_of_func)
        d = np.vstack((d for d in data))
        if self.type_of_func == 'RMS':
            dd = d[:, t_range, 0] - d[:, t_range, 1]
            dd = np.sqrt(np.sum(np.power(dd, 2), 1))
        elif self.type_of_func == 'SumAbs':
            dd = np.sum(abs(d[:, t_range, :]), (1, 2))

        if self.topK > 0:
            self.topK = min(self.topK, len(names))
        else:
            self.topK = sum(dd > 0)
        objects_to_filtter_in = np.argsort(dd)[::-1][:self.topK]
        print(dd[objects_to_filtter_in])
        return objects_to_filtter_in, names

    def filter_electrodes(self):
        print('filter_electrodes')
        source_files = [os.path.join(self.current_root_path, self.current_file_to_upload)]
        objects_to_filtter_in, names = self.get_object_to_filter(source_files)

        for obj in bpy.data.objects:
            obj.select = False

        for obj in bpy.data.objects['Deep_electrodes'].children:
            obj.active_material.node_tree.nodes["Layer Weight"].inputs[0].default_value = 1

        for ind in range(self.topK-1,-1,-1):
            # print(str(names[objects_to_filtter_in[ind]]))
            orig_name = bpy.data.objects[str(names[objects_to_filtter_in[ind]])].name
            # print(orig_name)
            # new_name = '*'+orig_name
            # print(new_name)
            # bpy.data.objects[orig_name].name = new_name
            bpy.data.objects[orig_name].active_material.node_tree.nodes["Layer Weight"].inputs[0].default_value = 0.3

            bpy.data.objects[orig_name].select = True
            bpy.context.scene.objects.active = bpy.data.objects[orig_name]

        bpy.context.object.parent.select = False

    def filter_ROIs(self):
        print('filter_ROIs')
        source_files = [os.path.join(self.current_activity_path, self.current_file_to_upload+hemi+'.npz') for hemi in ['lh', 'rh']]
        objects_to_filtter_in, names = self.get_object_to_filter(source_files)
        for obj in bpy.data.objects:
            obj.select = False
            if obj.parent == bpy.data.objects['Subcortical structures']:
                obj.active_material = bpy.data.materials['unselected_label_Mat_subcortical']
            elif obj.parent == bpy.data.objects['Cortex-lh'] or obj.parent == bpy.data.objects['Cortex-rh']:
                obj.active_material = bpy.data.materials['unselected_label_Mat_cortex']

        for ind in range(self.topK-1, -1, -1):
            orig_name = bpy.data.objects[str(names[objects_to_filtter_in[ind]])].name
            print(orig_name)
            # new_name = '*'+orig_name
            # print(new_name)
            # bpy.data.objects[orig_name].name = new_name
            bpy.data.objects[orig_name].select = True
            bpy.context.scene.objects.active = bpy.data.objects[orig_name]
            # if bpy.data.objects[orig_name].parent != bpy.data.objects[orig_name]:
            if bpy.data.objects[orig_name].active_material == bpy.data.materials['unselected_label_Mat_subcortical']:
                bpy.data.objects[orig_name].active_material = bpy.data.materials['selected_label_Mat_subcortical']
            else:
                bpy.data.objects[orig_name].active_material = bpy.data.materials['selected_label_Mat']

    def invoke(self, context, event=None):
        change_view3d()
        setup_layers()
        self.topK = bpy.context.scene.filter_topK
        self.filter_from = bpy.context.scene.filter_from
        self.filter_to = bpy.context.scene.filter_to
        self.current_activity_path = bpy.path.abspath(bpy.context.scene.conf_path)
        # self.current_activity_path = bpy.path.abspath(bpy.context.scene.activity_path)
        self.type_of_filter = bpy.context.scene.filter_curves_type
        self.type_of_func = bpy.context.scene.filter_curves_func
        self.current_file_to_upload = files_names[self.type_of_filter]

        # print(self.current_root_path)
        # source_files = ["/homes/5/npeled/space3/ohad/mg79/electrodes_data.npz"]
        if self.type_of_filter == 'Electrodes':
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~invoke~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            self.filter_electrodes()
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~invoke2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        elif self.type_of_filter == 'MEG':
            self.filter_ROIs()

        # bpy.context.screen.areas[2].spaces[0].dopesheet.filter_fcurve_name = '*'
        return {"FINISHED"}
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Filter Panel ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Filter Panel ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
# def filter_draw(self, context):
# 	layout = self.layout
# 	layout.prop(context.scene, "Filter_electrodes", text="Top K")
# 	layout.prop(context.scene, "filter_curves_type", text="")
# 	layout.operator("ohad.filter", text="Filter " + bpy.context.scene.filter_curves_type, icon = 'BORDERMOVE')
# 	layout.operator("ohad.filter_clear", text="Clear Filtering", icon = 'PANEL_CLOSE')
#
# bpy.types.Scene.Filter_electrodes = bpy.props.IntProperty(default=1, min=1,
#                                                           description="The top K electrodes to be shown")
# bpy.types.Scene.filter_curves_type = bpy.props.EnumProperty(items=[("MEG", "MEG time course", "", 1),
#                                                                    ("Electrodes", " Electrodes time course", "", 2)],
#                                                             description="Type of curve to be filtered",
#                                                             update=filter_draw)
#
#
# class FilteringMakerPanel(bpy.types.Panel):
#     bl_space_type = "GRAPH_EDITOR"
#     bl_region_type = "UI"
#     bl_context = "objectmode"
#     bl_category = "Ohad"
#     bl_label = "Filter number of curves"
#
#     def draw(self, context):
#         filter_draw(self, context)
#
# files_names = {'MEG': 'labels_data_', 'Electrodes': 'electrodes_data.npz'}
#
#
# class ClearFiltering(bpy.types.Operator):
#     bl_idname = "ohad.filter_clear"
#     bl_label = "filter clear"
#     bl_options = {"UNDO"}
#
#     @staticmethod
#     def invoke(self, context, event=None):
#         for subHierarchy in bpy.data.objects['Brain'].children:
#             new_mat = bpy.data.materials['unselected_label_Mat_cortex']
#             if subHierarchy.name == 'Subcortical structures':
#                 new_mat = bpy.data.materials['unselected_label_Mat_subcortical']
#
#             for obj in subHierarchy.children:
#                  obj.active_material = new_mat
#                  if obj.name == 'Left cerebellum cortex' or obj.name == 'Right cerebellum cortex':
#                     obj.active_material = bpy.data.materials['unselected_label_Mat_cerebellum']
#         for obj in bpy.data.objects['Deep_electrodes'].children:
#             obj.active_material.node_tree.nodes["Layer Weight"].inputs[0].default_value = 1
#
#         return {"FINISHED"}
#
#
# class Filtering(bpy.types.Operator):
#     bl_idname = "ohad.filter"
#     bl_label = "Filter deep electrodes"
#     bl_options = {"UNDO"}
#     topK = -1
#     current_root_path = ''
#     type_of_filter = ''
#     current_file_to_upload = ''
#
#     def get_object_to_filter(self, source_files):
#         data, names = [], []
#         for input_file in source_files:
#             f = np.load(input_file)
#             data.append(f['data'])
#             temp_names = [str(name) for name in f['names']]
#             for ind in range(len(temp_names)):
#                 if temp_names[ind][1] == "'":
#                     temp_names[ind] = temp_names[ind][2:-1]
#             names.extend(temp_names)
#         self.topK = min(self.topK, len(names))
#         d = np.vstack((d for d in data))
#         dd = d[:, :, 0]-d[:, :, 1]
#         dd = np.sqrt(np.sum(np.power(dd, 2), 1))
#         # dd = d[:, :, 0]- d[:, :, 1]
#         # amps = np.max(dd,1) - np.min(dd, 1)
#         objects_to_filter_in = np.argsort(dd)[::-1][:self.topK]
#         print(objects_to_filter_in, names)
#         return objects_to_filter_in, names
#
#     def filter_electrodes(self):
#         print('filter_electrodes')
#         source_files = [os.path.join(self.current_root_path, self.current_file_to_upload)]
#         objects_to_filter_in, names = self.get_object_to_filter(source_files)
#
#         for obj in bpy.data.objects:
#             obj.select = False
#
#         for obj in bpy.data.objects['Deep_electrodes'].children:
#             obj.active_material.node_tree.nodes["Layer Weight"].inputs[0].default_value = 1
#
#         for ind in range(self.topK-1, -1, -1):
#             # print(str(names[objects_to_filter_in[ind]]))
#             orig_name = bpy.data.objects[str(names[objects_to_filter_in[ind]])].name
#             # print(orig_name)
#             # new_name = '*'+orig_name
#             # print(new_name)
#             # bpy.data.objects[orig_name].name = new_name
#             bpy.data.objects[orig_name].active_material.node_tree.nodes["Layer Weight"].inputs[0].default_value = 0.3
#
#             bpy.data.objects[orig_name].select = True
#             bpy.context.scene.objects.active = bpy.data.objects[orig_name]
#
#         bpy.context.object.parent.select = False
#
#     def filter_rois(self):
#         print('filter_rois')
#         source_files = [os.path.join(self.current_root_path, self.current_file_to_upload+hemi+'.npz') for hemi
#                         in ['lh', 'rh']]
#         objects_to_filter_in, names = self.get_object_to_filter(source_files)
#         for obj in bpy.data.objects:
#             obj.select = False
#             if obj.name == 'Left cerebellum cortex' or obj.name == 'Right cerebellum cortex':
#                 obj.active_material = bpy.data.materials['unselected_label_Mat_cerebellum']
#             elif obj.parent == bpy.data.objects['Subcortical structures']:
#                 obj.active_material = bpy.data.materials['unselected_label_Mat_subcortical']
#             elif obj.parent == bpy.data.objects['Cortex-lh'] or obj.parent == bpy.data.objects['Cortex-rh']:
#                 obj.active_material = bpy.data.materials['unselected_label_Mat_cortex']
#
#         for ind in range(self.topK-1, -1, -1):
#             orig_name = bpy.data.objects[str(names[objects_to_filter_in[ind]])].name
#             print(orig_name)
#             # new_name = '*'+orig_name
#             # print(new_name)
#             # bpy.data.objects[orig_name].name = new_name
#             bpy.data.objects[orig_name].select = True
#             bpy.context.scene.objects.active = bpy.data.objects[orig_name]
#             # if bpy.data.objects[orig_name].parent != bpy.data.objects[orig_name]:
#             if bpy.data.objects[orig_name].active_material == bpy.data.materials['unselected_label_Mat_subcortical']:
#                  bpy.data.objects[orig_name].active_material = bpy.data.materials['selected_label_Mat_subcortical']
#             else:
#                 bpy.data.objects[orig_name].active_material = bpy.data.materials['selected_label_Mat']
#
#             if obj.name == 'Left cerebellum cortex' or obj.name == 'Right cerebellum cortex':
#                 obj.active_material = bpy.data.materials['unselected_label_Mat_cerebellum']
#
#     def invoke(self, context, event=None):
#         change_view3d()
#         setup_layers()
#         self.topK = bpy.context.scene.Filter_electrodes
#         self.current_root_path = bpy.path.abspath(bpy.context.scene.conf_path)
#         self.type_of_filter = bpy.context.scene.filter_curves_type
#         self.current_file_to_upload = files_names[self.type_of_filter]
#
#         # print(self.current_root_path)
#         # source_files = ["/homes/5/npeled/space3/ohad/mg79/electrodes_data.npz"]
#         if self.type_of_filter == 'Electrodes':
#             print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~invoke~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#             self.filter_electrodes()
#             print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~invoke2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#         elif self.type_of_filter == 'MEG':
#             self.filter_rois()
#
#         # bpy.context.screen.areas[2].spaces[0].dopesheet.filter_fcurve_name = '*'
#         return {"FINISHED"}
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Filter Panel ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Show / Hide objects ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def show_hide_hierarchy(val, obj):
    bpy.data.objects[obj].hide = val
    for child in bpy.data.objects[obj].children:
        child.hide = val


def show_hide_hemi(val, obj_func_name, obj_brain_name):
    bpy.data.objects[obj_func_name].hide = val
    show_hide_hierarchy(val, obj_brain_name)


def show_hide_rh(self, context):
    show_hide_hemi(bpy.context.scene.objects_show_hide_rh, "rh", "Cortex-rh")


def show_hide_lh(self, context):
    show_hide_hemi(bpy.context.scene.objects_show_hide_lh, "lh", "Cortex-lh")


def show_hide_sub_cortical(self, context):
    show_hide_hierarchy(bpy.context.scene.objects_show_hide_sub_cortical, "Subcortical structures")
    show_hide_hierarchy(bpy.context.scene.objects_show_hide_sub_cortical, "Subcortical_activity_map")


bpy.types.Scene.objects_show_hide_lh = bpy.props.BoolProperty(default=True, description="Show left hemisphere",
                                                              update=show_hide_lh)
bpy.types.Scene.objects_show_hide_rh = bpy.props.BoolProperty(default=True, description="Show right hemisphere",
                                                              update=show_hide_rh)
bpy.types.Scene.objects_show_hide_sub_cortical = bpy.props.BoolProperty(default=True, description="Show sub cortical",
                                                                        update=show_hide_sub_cortical)


class ShowHideObjectsPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
    bl_label = "Show Hide Objects"

    def draw(self, context):
        col1 = self.layout.column(align=True)
        col1.prop(context.scene, 'objects_show_hide_lh', text="Left Hemisphere", icon='RESTRICT_VIEW_OFF')
        col1.prop(context.scene, 'objects_show_hide_rh', text="Right Hemisphere", icon='RESTRICT_VIEW_OFF')
        col1.prop(context.scene, 'objects_show_hide_sub_cortical', text="Sub Cortical", icon='RESTRICT_VIEW_OFF')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Show / Hide objects ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Appearance Panel ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def setup_layers(self=None, context=None):
    empty_layer = 14

    for layer_ind in range(len(bpy.context.scene.layers)):
        bpy.context.scene.layers[layer_ind] = layer_ind == empty_layer

    bpy.context.scene.layers[1] = bpy.context.scene.appearance_show_electrodes_layer
    bpy.context.scene.layers[10] = bpy.context.scene.appearance_show_ROIs_layer
    bpy.context.scene.layers[11] = bpy.context.scene.appearance_show_activity_layer


def change_view3d(self=None, context=None):
    viewport_shade = bpy.context.scene.filter_view_type
    if viewport_shade == 'RENDERED':
        bpy.context.scene.layers[12] = True
    else:
        bpy.context.scene.layers[12] = False

    for ii in range(len(bpy.context.screen.areas)):
        if bpy.context.screen.areas[ii].type == 'VIEW_3D':
            bpy.context.screen.areas[ii].spaces[0].viewport_shade = viewport_shade


def get_appearance_show_electrodes_layer(self):
    return self['appearance_show_electrodes_layer']


def set_appearance_show_electrodes_layer(self, value):
    self['appearance_show_electrodes_layer'] = value
    bpy.context.scene.layers[1] = value


def get_appearance_show_rois_layer(self):
    return self['appearance_show_ROIs_layer']


def set_appearance_show_rois_layer(self, value):
    self['appearance_show_ROIs_layer'] = value
    bpy.context.scene.layers[10] = value
    if value:
        set_appearance_show_activity_layer(self,False)
        # bpy.context.scene.layers[12] = False


def get_appearance_show_activity_layer(self):
    return self['appearance_show_activity_layer']


def set_appearance_show_activity_layer(self, value):
    self['appearance_show_activity_layer'] = value
    bpy.context.scene.layers[11] = value
    if value:
        set_appearance_show_rois_layer(self, False)
        # bpy.context.scene.layers[12] = True


def get_filter_view_type(self):
    return self['filter_view_type']


def set_filter_view_type(self, value):
    self['filter_view_type'] = value
    change_view3d()


def make_brain_solid_or_transparent(self=None, context=None):
    bpy.data.materials['Activity_map_mat'].node_tree.nodes['transparency_node'].inputs['Fac'].default_value = bpy.context.scene.appearance_solid_slider


def update_layers():
    if bpy.context.scene.appearance_depth_Bool:
        bpy.data.materials['Activity_map_mat'].node_tree.nodes["layers_depth"].inputs[1].default_value = bpy.context.scene.appearance_depth_slider
    else:
        bpy.data.materials['Activity_map_mat'].node_tree.nodes["layers_depth"].inputs[1].default_value = 0


def appearance_draw(self, context):
    layout = self.layout
    col1 = self.layout.column(align=True)
    col1.prop(context.scene, 'appearance_show_ROIs_layer', text="Show ROIs", icon='RESTRICT_VIEW_OFF')
    col1.prop(context.scene, 'appearance_show_activity_layer', text="Show activity maps", icon='RESTRICT_VIEW_OFF')
    col1.prop(context.scene, 'appearance_show_electrodes_layer', text="Show electrodes", icon='RESTRICT_VIEW_OFF')
    split = layout.split()
    split.prop(context.scene, "filter_view_type", text="")
    # print(context.scene.filter_view_type)
    if context.scene.filter_view_type == 'RENDERED' and bpy.context.scene.appearance_show_activity_layer == True:
        # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        layout.prop(context.scene, 'appearance_solid_slider', text="Show solid brain")
        split2 = layout.split()
        split2.prop(context.scene, 'appearance_depth_Bool', text="Show cortex deep layers")
        split2.prop(context.scene, 'appearance_depth_slider', text="Depth")
        layout.operator("ohad.appearance_update", text="Update")


class UpdateAppearance(bpy.types.Operator):
    bl_idname = "ohad.appearance_update"
    bl_label = "filter clear"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        make_brain_solid_or_transparent()
        update_layers()
        return {"FINISHED"}


bpy.types.Scene.appearance_show_electrodes_layer = bpy.props.BoolProperty(default=False, description="Show electrodes",
                                                                          get=get_appearance_show_electrodes_layer,
                                                                          set=set_appearance_show_electrodes_layer)
bpy.types.Scene.appearance_show_ROIs_layer = bpy.props.BoolProperty(default=True, description="Show ROIs",
                                                                    get=get_appearance_show_rois_layer,
                                                                    set=set_appearance_show_rois_layer)
bpy.types.Scene.appearance_show_activity_layer = bpy.props.BoolProperty(default=False, description="Show activity maps",
                                                                        get=get_appearance_show_activity_layer,
                                                                        set=set_appearance_show_activity_layer)
bpy.types.Scene.filter_view_type = bpy.props.EnumProperty(items=[("RENDERED", "Rendered Brain", "", 1),
                                                                 ("SOLID", " Solid Brain", "", 2)],
                                                          description="Brain appearance", get=get_filter_view_type,
                                                          set=set_filter_view_type)
bpy.types.Scene.appearance_solid_slider = bpy.props.FloatProperty(default=0.0, min=0, max=1, description="",
                                                                  update=appearance_draw)
bpy.types.Scene.appearance_depth_slider = bpy.props.IntProperty(default=1, min=1, max=10, description="")
bpy.types.Scene.appearance_depth_Bool = bpy.props.BoolProperty(default=False, description="")


class AppearanceMakerPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
    bl_label = "Appearance"

    def draw(self, context):
        # make_brain_solid_or_transparent(self, context)
        appearance_draw(self, context)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Appearance Panel ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Coloring Panel ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def object_coloring(obj, rgb):
    bpy.context.scene.objects.active = obj
    obj.select = True
    cur_mat = obj.active_material
    print('***************************************************************')
    print(cur_mat)
    print('***************************************************************')
    new_color = (rgb[0], rgb[1], rgb[2], 1)
    cur_mat.node_tree.nodes["RGB"].outputs[0].default_value = new_color


def color_object_homogeneously(path, postfix_str=''):
    default_color = (1, 1, 1)
    cur_frame = bpy.context.scene.frame_current
    print('start')

    obj_counter = 0

    f = np.load(path)
    print('loaded')
    for obj_name, object_colors in zip(f['names'], f['colors']):
        obj_name = str(obj_name)
        if obj_name[1] == "'":
            obj_name = obj_name[2:-1]
        print(obj_name)
        cur_obj = bpy.data.objects[obj_name]

        new_color = object_colors[cur_frame]
        object_coloring(bpy.data.objects[obj_name+postfix_str], new_color)

    print('Finished coloring!!')


def activity_map_coloring(map_type):
    override_current_mat = True
    # setup_environment_settings()
    set_appearance_show_activity_layer(bpy.context.scene, True)
    set_filter_view_type(bpy.context.scene, 'RENDERS')
    # change_view3d()

    threshold = bpy.context.scene.coloring_threshold
    frame_str = str(bpy.context.scene.frame_current)
    d = {}
    start_time1 = time.time()
    current_root_path = bpy.path.abspath(bpy.context.scene.conf_path)
    d['lh'] = np.load(os.path.join(current_root_path, 'faces_verts_lh.npy'))  # .astype(np.int)
    d['rh'] = np.load(os.path.join(current_root_path, 'faces_verts_rh.npy'))  # .astype(np.int)
    print('load time = '+str(time.time()-start_time1))

    start_time = time.time()
    hemispheres = ['lh', 'rh']
    # hemispheres = ['rh']

    for hemisphere in hemispheres:
        if map_type == 'MEG':
            f = np.load(os.path.join(current_root_path, 'activity_map_'+hemisphere, 't'+frame_str+'.npy'))
        elif map_type == 'FMRI':
            f = np.load(os.path.join(current_root_path, 'fmri_'+hemisphere+'.npy'))
        cur_obj = bpy.data.objects[hemisphere]
        mesh = cur_obj.data
        scn = bpy.context.scene

        valid_verts = np.where(np.abs(f[:, 0]) > threshold)[0]
        # check if our mesh already has Vertex Colors, and if not add some... (first we need to make sure it's the active object)
        scn.objects.active = cur_obj
        cur_obj.select = True
        if override_current_mat:
            bpy.ops.mesh.vertex_color_remove()

        # if mesh.vertex_colors:
        #    vcol_layer = mesh.vertex_colors.active
        # else:
        #    vcol_layer = mesh.vertex_colors.new()
        vcol_layer = mesh.vertex_colors.new()

        for vert in valid_verts:
            x = d[hemisphere][vert]
            for loop_ind in x[x > -1]:
                vcol_layer.data[loop_ind].color = f[vert, 1:]
    # select_all_rois()

    if map_type == 'MEG':
        color_object_homogeneously(os.path.join(current_root_path, 'sub_cortical_activity.npz'), '.001')

    deselect_all()
    bpy.data.objects['Brain'].select = True


class ColorElectrodes(bpy.types.Operator):
    bl_idname = "ohad.electrodes_color"
    bl_label = "ohad electrodes color"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        current_root_path = bpy.path.abspath(bpy.context.scene.conf_path)
        color_object_homogeneously(os.path.join(current_root_path, 'electrodes_data.npz'))
        deselect_all()
        set_appearance_show_electrodes_layer(bpy.context.scene, True)
        bpy.data.objects['Deep_electrodes'].select = True

        return {"FINISHED"}


class ColorMeg(bpy.types.Operator):
    bl_idname = "ohad.meg_color"
    bl_label = "ohad meg color"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        activity_map_coloring('MEG')
        return {"FINISHED"}


class ColorFmri(bpy.types.Operator):
    bl_idname = "ohad.fmri_color"
    bl_label = "ohad fmri color"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        activity_map_coloring('FMRI')
        return {"FINISHED"}


class ClearColors(bpy.types.Operator):
    bl_idname = "ohad.colors_clear"
    bl_label = "ohad colors clear"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        hemispheres = ['lh', 'rh']
        for hemisphere in hemispheres:
            cur_obj = bpy.data.objects[hemisphere]
            mesh = cur_obj.data
            scn = bpy.context.scene
            scn.objects.active = cur_obj
            cur_obj.select = True
            bpy.ops.mesh.vertex_color_remove()
            vcol_layer = mesh.vertex_colors.new()
        for obj in bpy.data.objects['Subcortical_activity_map'].children:
            print('in clear subcortical '+obj.name)
            obj.active_material.node_tree.nodes['rgb'].outputs['Color'].default_value = (1, 1, 1, 1)
        for obj in bpy.data.objects['Deep_electrodes'].children:
            obj.active_material.node_tree.nodes['RGB'].outputs['Color'].default_value = (1, 1, 1, 1)
        return {"FINISHED"}

bpy.types.Scene.coloring_fmri = bpy.props.BoolProperty(default=True, description="Plot FMRI")
bpy.types.Scene.coloring_electrodes = bpy.props.BoolProperty(default=False, description="Plot Deep electrodes")
bpy.types.Scene.coloring_threshold = bpy.props.FloatProperty(default=0.5, min=0, description="")


class ColoringMakerPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
    bl_label = "Activity Maps"

    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene, 'coloring_threshold', text = "Threshold")
        layout.operator("ohad.meg_color", text="Plot MEG ", icon = 'POTATO')
        layout.operator("ohad.fmri_color", text="Plot FMRI ", icon = 'POTATO')
        layout.operator("ohad.electrodes_color", text="Plot Electrodes ", icon = 'POTATO')
        # layout.prop(context.scene, 'coloring_electrodes', text = "Plot Deep electrodes", icon = 'BLANK1')
        layout.operator("ohad.colors_clear", text="Clear", icon = 'PANEL_CLOSE')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Coloring Panel ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Where am I Panel ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class WhereAmI(bpy.types.Operator):
    bl_idname = "ohad.where_i_am"
    bl_label = "ohad where i am"
    bl_options = {"UNDO"}

    @staticmethod
    def setup_environment(self):
        set_appearance_show_rois_layer(bpy.context.scene, True)
        # pass

    @staticmethod
    def main_func(self):
        distances = []
        names = []

        bpy.data.objects['Brain'].select = False
        for subHierarchy in bpy.data.objects['Brain'].children:
            if subHierarchy == bpy.data.objects['Subcortical structures']:
                cur_material = bpy.data.materials['unselected_label_Mat_subcortical']
            else:
                cur_material = bpy.data.materials['unselected_label_Mat_cortex']
            for obj in subHierarchy.children:
                obj.active_material = cur_material
                obj.select = False
                obj.hide = subHierarchy.hide

                # 3d cursor relative to the object data
                cursor = bpy.context.scene.cursor_location
                if bpy.context.object.parent == bpy.data.objects['Deep_electrodes']:
                    cursor = bpy.context.object.location
                co_find = cursor * obj.matrix_world.inverted()

                mesh = obj.data
                size = len(mesh.vertices)
                kd = mathutils.kdtree.KDTree(size)

                for i, v in enumerate(mesh.vertices):
                    kd.insert(v.co, i)

                kd.balance()

                # Find the closest 10 points to the 3d cursor
                # print("Close 1 points")
                for (co, index, dist) in kd.find_n(co_find, 1):
                    # print("    ", obj.name,co, index, dist)
                    distances.append(dist)
                    names.append(obj.name)

        # print(np.argmin(np.array(distances)))
        closest_area = names[np.argmin(np.array(distances))]

        print('closest area is:'+closest_area)
        bpy.context.scene.objects.active = bpy.data.objects[closest_area]
        bpy.data.objects[closest_area].select = True
        bpy.data.objects[closest_area].hide = False
        bpy.data.objects[closest_area].active_material = bpy.data.materials['selected_label_Mat']

    def invoke(self, context, event=None):
        self.setup_environment()
        self.main_func()
        return {"FINISHED"}


class ClearWhereAmI(bpy.types.Operator):
    bl_idname = "ohad.where_am_i_clear"
    bl_label = "where am i clear"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        for subHierarchy in bpy.data.objects['Brain'].children:
            new_mat = bpy.data.materials['unselected_label_Mat_cortex']
            if subHierarchy.name == 'Subcortical structures':
                new_mat = bpy.data.materials['unselected_label_Mat_subcortical']
            for obj in subHierarchy.children:
                obj.active_material = new_mat

        for obj in bpy.data.objects['Deep_electrodes'].children:
            obj.active_material.node_tree.nodes["Layer Weight"].inputs[0].default_value = 1

        for obj in bpy.data.objects:
            obj.select = False
        return {"FINISHED"}


class WhereAmIMakerPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
    bl_label = "Where Am I"

    def draw(self, context):
        layout = self.layout
        layout.operator("ohad.where_i_am", text="Where Am I?", icon='SNAP_SURFACE')
        layout.operator("ohad.where_am_i_clear", text="Clear", icon='PANEL_CLOSE')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Where am I Panel ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Show data of vertex Panel ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class ClearVertexData(bpy.types.Operator):
    bl_idname = "ohad.vertex_data_clear"
    bl_label = "vertex data clear"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        for obj in bpy.data.objects:
            if obj.name.startswith('Activity_in_vertex'):
                obj.select = True
                bpy.context.scene.objects.unlink(obj)
                bpy.data.objects.remove(obj)

        return {"FINISHED"}


class CreateVertexData(bpy.types.Operator):
    bl_idname = "ohad.vertex_data_create"
    bl_label = "vertex data create"
    bl_options = {"UNDO"}

    @staticmethod
    def find_vertex_index_and_mesh_closest_to_cursor(self):
        # 3d cursor relative to the object data
        print('cursor at:'+str(bpy.context.scene.cursor_location))
        # co_find = context.scene.cursor_location * obj.matrix_world.inverted()
        distances = []
        names = []
        vertices_idx = []
        vertices_co = []

        # base_obj = bpy.data.objects['Functional maps']
        meshes = ['lh', 'rh']
#        for obj in base_obj.children:
        for cur_obj in meshes:
            obj = bpy.data.objects[cur_obj]
            co_find = bpy.context.scene.cursor_location * obj.matrix_world.inverted()
            mesh = obj.data
            size = len(mesh.vertices)
            kd = mathutils.kdtree.KDTree(size)

            for i, v in enumerate(mesh.vertices):
                kd.insert(v.co, i)

            kd.balance()
            print(obj.name)
            for (co, index, dist) in kd.find_n(co_find, 1):
                print("cursor at "+str(co_find)+',vertex '+str(co)+',index '+str(index)+',dist '+str(dist))
                distances.append(dist)
                names.append(obj.name)
                vertices_idx.append(index)
                vertices_co.append(co)

        closest_mesh_name = names[np.argmin(np.array(distances))]
        print('closest_mesh ='+str(closest_mesh_name))
        vertex_ind = vertices_idx[np.argmin(np.array(distances))]
        print('vertex_ind ='+str(vertex_ind))
        vertex_co = vertices_co[np.argmin(np.array(distances))] * obj.matrix_world
        return closest_mesh_name, vertex_ind, vertex_co

    @staticmethod
    def create_empty_in_vertex_location(self, vertex_location):
        layer = [False]*20
        layer[11] = True
        bpy.ops.object.empty_add(type='PLAIN_AXES', radius=1, view_align=False, location=vertex_location, layers=layer)
        bpy.context.object.name = "Activity_in_vertex"

    @staticmethod
    def insert_keyframe_to_custom_prop(self, obj, prop_name, value, keyframe):
        bpy.context.scene.objects.active = obj
        obj.select = True
        obj[prop_name] = value
        obj.keyframe_insert(data_path='['+'"'+prop_name+'"'+']', frame=keyframe)

    @staticmethod
    def keyframe_empty(self, empty_name, closest_mesh_name, vertex_ind, data_path):
        obj = bpy.data.objects[empty_name]
        number_of_time_points = len(glob.glob(os.path.join(data_path, 'activity_map_'+closest_mesh_name+'2', '',)+'*.npy'))
        insert_keyframe_to_custom_prop(obj, 'data', 0, 0)
        insert_keyframe_to_custom_prop(obj, 'data', 0, number_of_time_points+1)
        for ii in range(number_of_time_points):
            print(ii)
            frame_str = str(ii)
            f = np.load(os.path.join(data_path, 'activity_map_'+closest_mesh_name+'2', 't'+frame_str+'.npy'))
            insert_keyframe_to_custom_prop(obj, 'data', float(f[vertex_ind, 0]), ii+1)

        fcurves = bpy.data.objects[empty_name].animation_data.action.fcurves[0]
        mod = fcurves.modifiers.new(type='LIMITS')

    def keyframe_empty_test(self, empty_name, closest_mesh_name, vertex_ind, data_path):
        obj = bpy.data.objects[empty_name]
        lookup = np.load(os.path.join(data_path, 'activity_map_'+closest_mesh_name+'_verts_lookup.npy'))
        file_num_str = str(int(lookup[vertex_ind, 0]))
        line_num = int(lookup[vertex_ind, 1])
        data_file = np.load(os.path.join(data_path, 'activity_map_'+closest_mesh_name+'_verts', file_num_str+'.npy'))
        data = data_file[line_num, :].squeeze()

        number_of_time_points = len(data)
        self.insert_keyframe_to_custom_prop(obj, 'data', 0, 0)
        self.insert_keyframe_to_custom_prop(obj, 'data', 0, number_of_time_points+1)
        for ii in range(number_of_time_points):
            print(ii)
            frame_str = str(ii)
            self.insert_keyframe_to_custom_prop(obj, 'data', float(data[ii]), ii+1)
            # insert_keyframe_to_custom_prop(obj,'data',0,ii+1)
        fcurves = bpy.data.objects[empty_name].animation_data.action.fcurves[0]
        mod = fcurves.modifiers.new(type='LIMITS')

    def invoke(self, context, event=None):
        closest_mesh_name, vertex_ind, vertex_co = self.find_vertex_index_and_mesh_closest_to_cursor()
        self.create_empty_in_vertex_location(vertex_co)
        # data_path = '/homes/5/npeled/space3/MEG/ECR/mg79'
        data_path = bpy.path.abspath(bpy.context.scene.conf_path)
        # keyframe_empty('Activity_in_vertex',closest_mesh_name,vertex_ind,data_path)
        self.keyframe_empty_test('Activity_in_vertex', closest_mesh_name, vertex_ind, data_path)
        return {"FINISHED"}


class DataInVertMakerPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
    bl_label = "Data in vertex"

    def draw(self, context):
        layout = self.layout
        layout.operator("ohad.vertex_data_create", text="Get data in vertex", icon='ROTATE')
        layout.operator("ohad.vertex_data_clear", text="Clear", icon='PANEL_CLOSE')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Show data of vertex Panel ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


bpy.context.scene.appearance_show_electrodes_layer = False
bpy.context.scene.appearance_show_activity_layer = False
bpy.context.scene.appearance_show_ROIs_layer = True

setup_layers()

bpy.utils.register_class(AddDataToElectrodes)
bpy.utils.register_class(AddDataToBrain)
bpy.utils.register_class(ImportElectrodes)
bpy.utils.register_class(ImportBrain)
bpy.utils.register_class(ImportRoisClass)
bpy.utils.register_class(SelectAllRois)
bpy.utils.register_class(SelectAllElectrodes)
bpy.utils.register_class(ClearSelection)
bpy.utils.register_class(ClearFiltering)
bpy.utils.register_class(ColorMeg)
bpy.utils.register_class(ColorFmri)
bpy.utils.register_class(Filtering)
bpy.utils.register_class(UpdateAppearance)
bpy.utils.register_class(WhereAmI)
bpy.utils.register_class(ClearWhereAmI)
bpy.utils.register_class(ClearColors)
bpy.utils.register_class(ColorElectrodes)
bpy.utils.register_class(CreateVertexData)
bpy.utils.register_class(ClearVertexData)
bpy.utils.register_class(GrabFromFiltering)
bpy.utils.register_class(GrabToFiltering)


bpy.utils.register_class(AppearanceMakerPanel)
bpy.utils.register_class(ShowHideObjectsPanel)
bpy.utils.register_class(SelectionMakerPanel)
bpy.utils.register_class(FilteringMakerPanel)
bpy.utils.register_class(ColoringMakerPanel)
bpy.utils.register_class(WhereAmIMakerPanel)
bpy.utils.register_class(DataInVertMakerPanel)
bpy.utils.register_class(DataMakerPanel)

# ###############################################################
# bpy.types.Scene.conf_path = bpy.props.StringProperty(name = "Root Path",default = "",description = "Define the root path of the project",subtype = 'DIR_PATH')