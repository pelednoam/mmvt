import bpy
import mmvt_utils as mu
import colors_utils as cu
import numpy as np
import os.path as op
import glob
import numbers
import importlib
from collections import defaultdict

try:
    import connections_panel
    connections_panel_exist = True
except:
    connections_panel_exist = False


def find_obj_with_val():
    cur_objects = []
    for obj in bpy.data.objects:
        if obj.select is True:
            cur_objects.append(obj)

    graph_editor = mu.get_the_graph_editor()
    target = graph_editor.cursor_position_y

    values, names, obj_names = [], [], []
    for cur_obj in cur_objects:
        for name, val in cur_obj.items():
            if bpy.context.scene.selection_type == 'spec_cond' and bpy.context.scene.conditions_selection not in name:
                continue
            if isinstance(val, numbers.Number):
                values.append(val)
                names.append(name)
                obj_names.append(cur_obj.name)
            # print(name)
    np_values = np.array(values) - target
    try:
        index = np.argmin(np.abs(np_values))
        closest_curve_name = names[index]
        closet_object_name = obj_names[index]
    except ValueError:
        closest_curve_name = ''
        closet_object_name = ''
        print('ERROR - Make sure you select all objects in interest')
    # print(closest_curve_name, closet_object_name)
    bpy.types.Scene.closest_curve_str = closest_curve_name
    # object_name = closest_curve_str
    # if bpy.data.objects.get(object_name) is None:
    #     object_name = object_name[:object_name.rfind('_')]
    print('object name: {}, curve name: {}'.format(closet_object_name, closest_curve_name))
    parent_obj = bpy.data.objects[closet_object_name].parent
    # print('parent: {}'.format(bpy.data.objects[object_name].parent))
    # try:
    if parent_obj.name == 'Deep_electrodes':
        print('filtering electrodes')
        filter_electrode_func(closet_object_name)
        bpy.context.scene.cursor_location = bpy.data.objects[closet_object_name].location
    elif connections_panel_exist and parent_obj.name == connections_panel.PARENT_OBJ:
        connections_panel.find_connections_closest_to_target_value(closet_object_name, closest_curve_name, target)
    else:
        filter_roi_func(closet_object_name, closest_curve_name)
    # except KeyError:
    #     filter_roi_func(object_name)


def filter_draw(self, context):
    layout = self.layout
    layout.prop(context.scene, "filter_topK", text="Top K")
    row = layout.row(align=0)
    row.prop(context.scene, "filter_from", text="From")
    row.operator(GrabFromFiltering.bl_idname, text="", icon='BORDERMOVE')
    row.prop(context.scene, "filter_to", text="To")
    row.operator(GrabToFiltering.bl_idname, text="", icon='BORDERMOVE')
    layout.prop(context.scene, "filter_curves_type", text="")
    layout.prop(context.scene, "filter_curves_func", text="")
    layout.prop(context.scene, 'mark_filter_items', text="Mark selected items")
    layout.operator(Filtering.bl_idname, text="Filter " + bpy.context.scene.filter_curves_type, icon='BORDERMOVE')
    # if bpy.types.Scene.filter_is_on:
    layout.operator("ohad.filter_clear", text="Clear Filtering", icon='PANEL_CLOSE')
    col = layout.column(align=0)
    col.operator("ohad.curve_close_to_cursor", text="closest curve to cursor", icon='SNAP_SURFACE')
    if bpy.types.Scene.closest_curve_str != '':
        col.label(text=bpy.types.Scene.closest_curve_str)
    layout.prop(context.scene, 'filter_items_one_by_one', text="Show one by one")
    if bpy.context.scene.filter_items_one_by_one:
        row = layout.row(align=0)
        row.operator(PrevFilterItem.bl_idname, text="", icon='PREV_KEYFRAME')
        row.prop(context.scene, 'filter_items', text="")
        row.operator(NextFilterItem.bl_idname, text="", icon='NEXT_KEYFRAME')

        # bpy.context.area.type = 'GRAPH_EDITOR'
    # filter_to = bpy.context.scence.frame_preview_end


def clear_filtering():
    for subhierarchy in bpy.data.objects['Brain'].children:
        new_mat = bpy.data.materials['unselected_label_Mat_cortex']
        if subhierarchy.name == 'Subcortical_structures':
            new_mat = bpy.data.materials['unselected_label_Mat_subcortical']
        for obj in subhierarchy.children:
            obj.active_material = new_mat

    if bpy.data.objects.get('Deep_electrodes'):
        for obj in bpy.data.objects['Deep_electrodes'].children:
            de_select_electrode(obj)


def de_select_electrode(obj, call_create_and_set_material=True):
    obj.active_material.node_tree.nodes["Layer Weight"].inputs[0].default_value = 1
    # safety check, if something happened to the electrode's material
    if call_create_and_set_material:
        mu.create_and_set_material(obj)
    # Sholdn't change to color here. If user plot the electrodes, we don't want to change it back to white.
    if obj.name in FilteringMakerPanel.electrodes_colors:
        FilteringMakerPanel.addon.object_coloring(obj, FilteringMakerPanel.electrodes_colors[obj.name])
    # obj.active_material.node_tree.nodes["RGB"].outputs[0].default_value = (1, 1, 1, 1)


def filter_roi_func(closet_object_name, closest_curve_name=None, mark='mark_green'):
    if bpy.context.scene.selection_type == 'conds':
        bpy.data.objects[closet_object_name].select = True

    bpy.context.scene.objects.active = bpy.data.objects[closet_object_name]
    if bpy.context.scene.mark_filter_items:
        if bpy.data.objects[closet_object_name].active_material == bpy.data.materials['unselected_label_Mat_subcortical']:
            bpy.data.objects[closet_object_name].active_material = bpy.data.materials['selected_label_Mat_subcortical']
        else:
            if mark == 'mark_green':
                bpy.data.objects[closet_object_name].active_material = bpy.data.materials['selected_label_Mat']
            elif mark == 'mark_blue':
                bpy.data.objects[closet_object_name].active_material = bpy.data.materials['selected_label_Mat_blue']
    bpy.types.Scene.filter_is_on = True


def filter_electrode_func(elec_name):
    elec_obj = bpy.data.objects[elec_name]
    if bpy.context.scene.mark_filter_items:
        FilteringMakerPanel.electrodes_colors[elec_name] = FilteringMakerPanel.addon.get_obj_color(elec_obj)
        FilteringMakerPanel.addon.object_coloring(elec_obj, cu.name_to_rgb('green'))
    else:
        elec_obj.active_material.node_tree.nodes["Layer Weight"].inputs[0].default_value = 0.3

    # todo: selecting the electrode will show both of their conditions time series
    # We don't want it to happen if selection_type == 'conds'...
    if bpy.context.scene.selection_type == 'conds' or bpy.context.scene.filter_items_one_by_one:
        bpy.data.objects[elec_name].select = True
    bpy.context.scene.objects.active = bpy.data.objects[elec_name]
    bpy.types.Scene.filter_is_on = True


def deselect_all_objects():
    for obj in bpy.data.objects:
        obj.select = False
        if obj.parent == bpy.data.objects['Subcortical_structures']:
            obj.active_material = bpy.data.materials['unselected_label_Mat_subcortical']
        elif obj.parent == bpy.data.objects['Cortex-lh'] or obj.parent == bpy.data.objects['Cortex-rh']:
            obj.active_material = bpy.data.materials['unselected_label_Mat_cortex']
        elif bpy.data.objects.get('Deep_electrodes', None) and obj.parent == bpy.data.objects['Deep_electrodes']:
            de_select_electrode(obj)


def filter_items_update(self, context):
    deselect_all_objects()
    if bpy.context.scene.filter_curves_type == 'Electrodes':
        filter_electrode_func(bpy.context.scene.filter_items)
    elif bpy.context.scene.filter_curves_type == 'MEG':
        filter_roi_func(bpy.context.scene.filter_items)
    mu.view_all_in_graph_editor(context)


def update_filter_items(topk, objects_indices, filter_objects):
    filter_items = []
    FilteringMakerPanel.filter_items = []
    if topk == 0:
        topk = len(objects_indices)
    for loop_ind, ind in enumerate(range(min(topk, len(objects_indices)) - 1, -1, -1)):
        filter_item = filter_objects[objects_indices[ind]]
        if 'unknown' in filter_item:
            continue
        filter_items.append((filter_item, '{}) {}'.format(ind + 1, filter_item), '', loop_ind))
        FilteringMakerPanel.filter_items.append(filter_item)
    bpy.types.Scene.filter_items = bpy.props.EnumProperty(
        items=filter_items, description="filter items", update=filter_items_update)
    if len(filter_items) > 0:
        bpy.context.scene.filter_items = FilteringMakerPanel.filter_items[-1]


def show_one_by_one_update(self, context):
    if bpy.context.scene.filter_items_one_by_one:
        update_filter_items(bpy.context.scene.filter_topK, Filtering.objects_indices, Filtering.filter_objects)
    else:
        #todo: do something here
        pass


def next_filter_item():
    index = FilteringMakerPanel.filter_items.index(bpy.context.scene.filter_items)
    next_item = FilteringMakerPanel.filter_items[index + 1] if index < len(FilteringMakerPanel.filter_items) -1 \
        else FilteringMakerPanel.filter_items[0]
    bpy.context.scene.filter_items = next_item


def prev_filter_item():
    index = FilteringMakerPanel.filter_items.index(bpy.context.scene.filter_items)
    prev_cluster = FilteringMakerPanel.filter_items[index - 1] if index > 0 else FilteringMakerPanel.filter_items[-1]
    bpy.context.scene.filter_items = prev_cluster


def get_func(module_name):
    lib_name = 'traces_filters.{}'.format(module_name)
    lib = importlib.import_module(lib_name)
    importlib.reload(lib)
    return getattr(lib, 'filter_traces')


def module_has_func(module_name):
    try:
        f = get_func(module_name)
        return True
    except:
        return False


def get_filter_functions():
    functions_names = []
    files = glob.glob(op.join(op.sep.join(__file__.split(op.sep)[:-1]), 'traces_filters', '*.py'))
    for fname in files:
        file_name = op.splitext(op.basename(fname))[0]
        if module_has_func(file_name):
            functions_names.append(file_name)
    bpy.types.Scene.filter_curves_func = bpy.props.EnumProperty(
        items=[(module_name, module_name, '', k + 1) for k, module_name in enumerate(functions_names)],
        description="Filtering function")


class FindCurveClosestToCursor(bpy.types.Operator):
    bl_idname = "ohad.curve_close_to_cursor"
    bl_label = "curve close to cursor"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        find_obj_with_val()
        return {"FINISHED"}


class GrabFromFiltering(bpy.types.Operator):
    bl_idname = "ohad.grab_from"
    bl_label = "grab from"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        # print(bpy.context.scene.frame_current)
        context.scene.filter_from = bpy.context.scene.frame_current
        # print(bpy.context.scene.filter_from)
        bpy.data.scenes['Scene'].frame_preview_start = context.scene.frame_current
        return {"FINISHED"}


class GrabToFiltering(bpy.types.Operator):
    bl_idname = "ohad.grab_to"
    bl_label = "grab to"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        # print(bpy.context.scene.frame_current)
        context.scene.filter_to = bpy.context.scene.frame_current
        # print(bpy.context.scene.filter_to)
        bpy.data.scenes['Scene'].frame_preview_end = context.scene.frame_current
        return {"FINISHED"}


class ClearFiltering(bpy.types.Operator):
    bl_idname = "ohad.filter_clear"
    bl_label = "filter clear"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        clear_filtering()
        type_of_filter = bpy.context.scene.filter_curves_type
        if type_of_filter == 'MEG':
            FilteringMakerPanel.addon.select_all_rois()
        elif type_of_filter == 'Electrodes':
            FilteringMakerPanel.addon.select_all_electrodes()
        bpy.data.scenes['Scene'].frame_preview_end = FilteringMakerPanel.addon.get_max_time_steps()
        bpy.data.scenes['Scene'].frame_preview_start = 1
        bpy.types.Scene.closest_curve_str = ''
        bpy.types.Scene.filter_is_on = False
        return {"FINISHED"}


class PrevFilterItem(bpy.types.Operator):
    bl_idname = 'ohad.prev_filter_item'
    bl_label = 'prev_filter_item'
    bl_options = {'UNDO'}

    def invoke(self, context, event=None):
        prev_filter_item()
        return {'FINISHED'}


class NextFilterItem(bpy.types.Operator):
    bl_idname = 'ohad.next_filter_item'
    bl_label = 'next_filter_item'
    bl_options = {'UNDO'}

    def invoke(self, context, event=None):
        next_filter_item()
        return {'FINISHED'}


class Filtering(bpy.types.Operator):
    bl_idname = "ohad.filter"
    bl_label = "Filter deep elctrodes"
    bl_options = {"UNDO"}
    topK = -1
    filter_from = 100000
    filter_to = -100000
    current_activity_path = ''
    type_of_filter = None
    type_of_func = None
    current_file_to_upload = ''
    current_root_path = mu.get_user_fol()
    objects_indices = []
    filter_objects = []
    filter_values = []

    def get_object_to_filter(self, source_files):
        data, names = [], []
        for input_file in source_files:
            try:
                f = np.load(input_file)
                data.append(f['data'])
                names.extend([name.astype(str) for name in f['names']])
            except:
                mu.message(self, "Can't load {}!".format(input_file))
                return None, None

        print('filtering {}-{}'.format(self.filter_from, self.filter_to))

        # t_range = range(self.filter_from, self.filter_to + 1)
        if self.topK > 0:
            self.topK = min(self.topK, len(names))
        print(self.type_of_func)
        filter_func = get_func(self.type_of_func)
        d = np.vstack((d for d in data))
        print('%%%%%%%%%%%%%%%%%%%' + str(len(d[0, :, 0])))
        t_range = range(max(self.filter_from, 1), min(self.filter_to, len(d[0, :, 0])) - 1)
        objects_to_filtter_in, dd = filter_func(d, t_range, self.topK, bpy.context.scene.coloring_threshold)
        print(dd[objects_to_filtter_in])
        return objects_to_filtter_in, names, dd

    def filter_electrodes(self, current_file_to_upload):
        print('filter_electrodes')
        source_files = [op.join(self.current_activity_path, current_file_to_upload)]
        objects_indices, names, self.filter_values = self.get_object_to_filter(source_files)
        Filtering.objects_indices, Filtering.filter_objects = objects_indices, names
        if objects_indices is None:
            return

        for obj in bpy.data.objects:
            obj.select = False

        deep_electrodes_obj = bpy.data.objects['Deep_electrodes']
        for obj in deep_electrodes_obj.children:
            obj.active_material.node_tree.nodes["Layer Weight"].inputs[0].default_value = 1

        if bpy.context.scene.selection_type == 'diff':
            filter_obj_names = [names[ind] for ind in objects_indices]
            for fcurve in deep_electrodes_obj.animation_data.action.fcurves:
                con_name = mu.fcurve_name(fcurve)
                fcurve.hide = con_name not in filter_obj_names
                fcurve.select = not fcurve.hide
            deep_electrodes_obj.select = True
        else:
            deep_electrodes_obj.select = False

        for ind in range(min(self.topK, len(objects_indices)) - 1, -1, -1):
            if bpy.data.objects.get(names[objects_indices[ind]]):
                orig_name = bpy.data.objects[names[objects_indices[ind]]].name
                filter_electrode_func(orig_name)
            else:
                print("Can't find {}!".format(names[objects_indices[ind]]))

    def filter_rois(self, current_file_to_upload):
        print('filter_ROIs')
        FilteringMakerPanel.addon.show_rois()
        source_files = [op.join(self.current_activity_path, current_file_to_upload.format(hemi=hemi)) for hemi
                        in mu.HEMIS]
        objects_indices, names, self.filter_values = self.get_object_to_filter(source_files)
        Filtering.objects_indices, Filtering.filter_objects = objects_indices, names
        deselect_all_objects()

        if bpy.context.scene.selection_type == 'diff':
            filter_obj_names = [names[ind] for ind in objects_indices]
            brain_obj = bpy.data.objects['Brain']
            for fcurve in brain_obj.animation_data.action.fcurves:
                con_name = mu.fcurve_name(fcurve)
                fcurve.hide = con_name not in filter_obj_names
                fcurve.select = not fcurve.hide
            brain_obj.select = True

        curves_num = 0
        for ind in range(min(self.topK, len(objects_indices))):
            orig_name = names[objects_indices[ind]]
            if 'unknown' not in orig_name:
                obj = bpy.data.objects.get(orig_name)
                if not obj is None and not obj.animation_data is None:
                    curves_num += len(obj.animation_data.action.fcurves)

        colors = cu.get_distinct_colors(curves_num)
        objects_names, objects_colors, objects_data = defaultdict(list), defaultdict(list), defaultdict(list)
        for ind in range(min(self.topK, len(objects_indices)) - 1, -1, -1):
            if bpy.data.objects.get(names[objects_indices[ind]]):
                orig_name = names[objects_indices[ind]]
                obj_type = mu.check_obj_type(orig_name)
                objects_names[obj_type].append(orig_name)
                objects_colors[obj_type].append(cu.name_to_rgb('green'))
                objects_data[obj_type].append(1.0)
                if 'unknown' not in orig_name:
                    filter_roi_func(orig_name)
                    for fcurve in bpy.data.objects[orig_name].animation_data.action.fcurves:
                        fcurve.color_mode = 'CUSTOM'
                        fcurve.color = tuple(next(colors))
            else:
                print("Can't find {}!".format(names[objects_indices[ind]]))

        FilteringMakerPanel.addon.color_objects(objects_names, objects_colors, objects_data)

    def invoke(self, context, event=None):
        FilteringMakerPanel.addon.change_view3d()
        #todo: why should we call setup layers here??
        # FilteringMakerPanel.addon.setup_layers()
        self.topK = bpy.context.scene.filter_topK
        self.filter_from = bpy.context.scene.filter_from
        self.filter_to = bpy.context.scene.filter_to
        self.current_activity_path = mu.get_user_fol() # bpy.path.abspath(bpy.context.scene.conf_path)
        # self.current_activity_path = bpy.path.abspath(bpy.context.scene.activity_path)
        self.type_of_filter = bpy.context.scene.filter_curves_type
        self.type_of_func = bpy.context.scene.filter_curves_func
        files_names = {'MEG': 'labels_data_{hemi}.npz',
                       'Electrodes': op.join('electrodes', 'electrodes_data_{stat}.npz')}
        current_file_to_upload = files_names[self.type_of_filter]

        # print(self.current_root_path)
        # source_files = ["/homes/5/npeled/space3/ohad/mg79/electrodes_data.npz"]
        if self.type_of_filter == 'Electrodes':
            data_files = glob.glob(op.join(mu.get_user_fol(), 'electrodes', 'electrodes_data_*.npz'))
            if len(data_files) == 0:
                print('No data files!')
            elif len(data_files) == 1:
                current_file_to_upload = data_files[0]
            else:
                print('More the one data file!')
                current_file_to_upload = data_files[0]
                # todo: should decide which one to pick
                # current_file_to_upload = current_file_to_upload.format(
                #     stat='avg' if bpy.context.scene.selection_type == 'conds' else 'diff')
            self.filter_electrodes(current_file_to_upload)
        elif self.type_of_filter == 'MEG':
            self.filter_rois(current_file_to_upload)

        if bpy.context.scene.filter_items_one_by_one:
            update_filter_items(self.topK, self.objects_indices, self.filter_objects)
        else:
            mu.view_all_in_graph_editor(context)
        # bpy.context.screen.areas[2].spaces[0].dopesheet.filter_fcurve_name = '*'
        return {"FINISHED"}


bpy.types.Scene.closest_curve_str = ''
bpy.types.Scene.filter_is_on = False

bpy.types.Scene.closest_curve = bpy.props.StringProperty(description="Find closest curve to cursor")
bpy.types.Scene.filter_topK = bpy.props.IntProperty(default=1, min=0, description="The top K elements to be shown")
bpy.types.Scene.filter_from = bpy.props.IntProperty(default=0, min=0, description="When to filter from")
bpy.types.Scene.filter_to = bpy.props.IntProperty(default=bpy.context.scene.frame_end, min=0,
                                                  description="When to filter to")
bpy.types.Scene.filter_curves_type = bpy.props.EnumProperty(
    items=[("MEG", "MEG time course", "", 1), ("Electrodes", " Electrodes time course", "", 2)],
    description="Type of curve to be filtered")
bpy.types.Scene.filter_curves_func = bpy.props.EnumProperty(items=[], description="Filtering function")
    # items=[("RMS", "RMS", "RMS between the two conditions", 1), ("SumAbs", "SumAbs", "Sum of the abs values", 2),
    #        ("threshold", "Above threshold", "", 3)],
bpy.types.Scene.mark_filter_items = bpy.props.BoolProperty(default=False, description="Mark selected items")
bpy.types.Scene.filter_items = bpy.props.EnumProperty(items=[], description="Filtering items")
bpy.types.Scene.filter_items_one_by_one = bpy.props.BoolProperty(
    default=False, description="Show one by one", update=show_one_by_one_update)


class FilteringMakerPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
    bl_label = "Filter number of curves"
    addon = None
    electrodes_colors = {}
    filter_items = []

    def draw(self, context):
        filter_draw(self, context)


def init(addon):
    FilteringMakerPanel.addon = addon
    get_filter_functions()
    register()


def register():
    try:
        unregister()
        bpy.utils.register_class(Filtering)
        bpy.utils.register_class(FilteringMakerPanel)
        bpy.utils.register_class(ClearFiltering)
        bpy.utils.register_class(GrabToFiltering)
        bpy.utils.register_class(GrabFromFiltering)
        bpy.utils.register_class(FindCurveClosestToCursor)
        bpy.utils.register_class(PrevFilterItem)
        bpy.utils.register_class(NextFilterItem)
        # print('Filtering Panel was registered!')
    except:
        print("Can't register Filtering Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(Filtering)
        bpy.utils.unregister_class(FilteringMakerPanel)
        bpy.utils.unregister_class(ClearFiltering)
        bpy.utils.unregister_class(GrabToFiltering)
        bpy.utils.unregister_class(GrabFromFiltering)
        bpy.utils.unregister_class(FindCurveClosestToCursor)
        bpy.utils.unregister_class(PrevFilterItem)
        bpy.utils.unregister_class(NextFilterItem)
    except:
        pass

