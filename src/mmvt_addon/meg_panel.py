import bpy
import os.path as op
import glob
import numpy as np
import mmvt_utils as mu

try:
    import mne
    MNE_EXIST = True
except:
    MNE_EXIST = False


PARENT_OBJ_NAME = 'meg_clusters'


def _addon():
    return MEGPanel.addon


def clusters_update(self, context):
    if MEGPanel.addon is not None and MEGPanel.init:
        _clusters_update()


def get_cluster_name(cluster):
    return 'cluster_size_{}_max_{:.2f}_{}'.format(cluster.size, cluster.max, cluster.name)


def get_cluster_fcurve_name(cluster):
    return '{}_{}_{:.2f}'.format(cluster.name, cluster.size, cluster.max)


def get_cluster_verts_co(cluster):
    inflated_mesh = 'inflated_{}'.format(cluster.hemi)
    me = bpy.data.objects[inflated_mesh].to_mesh(bpy.context.scene, True, 'PREVIEW')
    vertex_cos = np.zeros((len(cluster.vertices), 3))
    for vert_ind, vert in enumerate(cluster.vertices):
        vertex_cos[vert_ind] = tuple(me.vertices[vert].co)
    bpy.data.meshes.remove(me)
    return vertex_cos


def _clusters_update():
    MEGPanel.current_cluster = cluster = MEGPanel.clusters_lookup[bpy.context.scene.meg_clusters]
    # set_cluster_time_series(cluster)
    cluster_name = get_cluster_name(cluster)
    cluster_max_vert_co = mu.get_vert_co(cluster.max_vert, cluster.hemi)
    bpy.context.scene.cursor_location = cluster_max_vert_co
    _addon().set_cursor_pos()
    _addon().set_closest_vertex_and_mesh_to_cursor(cluster.max_vert, 'inflated_{}'.format(cluster.hemi))
    _addon().save_cursor_position()
    _addon().create_slices()
    if bpy.context.scene.plot_current_meg_cluster:
        _addon().color_contours(
            [cluster_name], cluster.hemi, MEGPanel.contours, bpy.context.scene.cumulate_meg_cluster, False)
    bpy.data.objects[PARENT_OBJ_NAME].select = True
    # mu.view_all_in_graph_editor()
    fcurves = mu.get_fcurves(PARENT_OBJ_NAME)
    cluster_fcurve_name = get_cluster_fcurve_name(cluster)
    if bpy.context.scene.cumulate_meg_cluster:
        fcurve_ind = [mu.get_fcurve_name(f) for f in fcurves].index(cluster_fcurve_name)
        fcurves[fcurve_ind].hide = False
    else:
        for fcurve in fcurves:
            fcurve_name = mu.get_fcurve_name(fcurve)
            fcurve.hide = fcurve_name != cluster_fcurve_name

    MEGPanel.prev_cluster = bpy.context.scene.meg_clusters


def plot_all_clusters():
    for hemi in mu.INF_HEMIS:
        # Not sure why, but this is needed, otherwise the coloring layer is being erased
        _addon().recreate_coloring_layers(bpy.data.objects[hemi].data, 'contours')
    for cluster in MEGPanel.clusters_labels_filtered:
        _addon().color_contours([get_cluster_name(cluster)], cluster.hemi, MEGPanel.contours, True, False)


def get_selected_clusters_data():
    names = []
    data = None
    fcurves = mu.get_fcurves(PARENT_OBJ_NAME)
    filtered_fcurves_names = [get_cluster_fcurve_name(c) for c in MEGPanel.clusters_labels_filtered]
    filtered_clusters = {fcurve_name:cluster for fcurve_name,cluster in zip(
        filtered_fcurves_names, MEGPanel.clusters_labels_filtered)}
    data_ind = 0
    for fcurve in fcurves:
        fcurve_name = mu.get_fcurve_name(fcurve)
        if fcurve_name in filtered_fcurves_names:
            x = filtered_clusters[fcurve_name].label_data
            if data is None:
                data = np.zeros((len(filtered_fcurves_names), len(x)))
            data[data_ind] = x
            names.append(fcurve_name)
            data_ind += 1
    return data, names, ['all']


# def clear_cluster(cluster):
#     if isinstance(cluster, str):
#         cluster = MEGPanel.clusters_lookup[cluster]
#     cluster_name = get_cluster_name(cluster)
#     contours = MEGPanel.contours[cluster.hemi]
#     label_ind = np.where(np.array(contours['labels']) == cluster_name)[0][0] + 1
#     clustes_contour_vertices = np.where(contours['contours'] == label_ind)[0]
#     hemi_obj_name = 'inflated_{}'.format(cluster.hemi)
#     if hemi_obj_name not in _addon().get_prev_colors():
#         _addon().color_prev_colors(clustes_contour_vertices, hemi_obj_name)


def set_cluster_time_series(cluster):
    cluster_uid_name = get_cluster_fcurve_name(cluster)
    _addon().create_empty_if_doesnt_exists(PARENT_OBJ_NAME, _addon().EMPTY_LAYER, bpy.context.scene.layers, 'Functional maps')
    parent_obj = bpy.data.objects[PARENT_OBJ_NAME]
    T = len(cluster.label_data)
    cluster.label_data = np.array(cluster.label_data, dtype=np.float64)
    fcurves_names = mu.get_fcurves_names(parent_obj)
    if not cluster_uid_name in fcurves_names:
        # Set the values to zeros in the first and last frame for current object(current label)
        mu.insert_keyframe_to_custom_prop(parent_obj, cluster_uid_name, 0, 1)
        mu.insert_keyframe_to_custom_prop(parent_obj, cluster_uid_name, 0, T + 2)

        # For every time point insert keyframe to current object
        for ind, t_data in enumerate(cluster.label_data):
            mu.insert_keyframe_to_custom_prop(parent_obj, cluster_uid_name, t_data, ind + 2)

        # remove the orange keyframe sign in the fcurves window
        fcurve_ind = len(fcurves_names)
        fcurves = parent_obj.animation_data.action.fcurves[fcurve_ind]
        mod = fcurves.modifiers.new(type='LIMITS')
    else:
        fcurve_ind = fcurves_names.index(cluster_uid_name)
        fcurve = parent_obj.animation_data.action.fcurves[fcurve_ind]
        fcurve.keyframe_points[0].co[1] = 0
        fcurve.keyframe_points[-1].co[1] = 0
        for t in range(T):
            if fcurve.keyframe_points[t + 1].co[1] != cluster.label_data[t]:
                fcurve.keyframe_points[t + 1].co[1] = cluster.label_data[t]
    # mu.view_all_in_graph_editor()


def filter_clusters(val_threshold=None, size_threshold=None, clusters_label=None):
    if val_threshold is None:
        val_threshold = bpy.context.scene.meg_cluster_val_threshold
    if size_threshold is None:
        size_threshold = bpy.context.scene.meg_cluster_size_threshold
    if clusters_label is None:
        clusters_label = bpy.context.scene.meg_clusters_label
    return [c for c in MEGPanel.clusters_labels.values
            if abs(c.max) >= val_threshold and c.size >= size_threshold and \
            clusters_label in c.name]
            # any([clusters_label in inter_label['name'] for inter_label in c.intersects])]


def meg_clusters_labels_files_update(self, context):
    if MEGPanel.init:
        # fname = MEGPanel.clusters_labels_fnames[bpy.context.scene.meg_clusters_labels_files]
        # MEGPanel.clusters_labels = mu.load(fname)
        # MEGPanel.clusters_labels = mu.Bag(MEGPanel.clusters_labels)
        # for ind in range(len(MEGPanel.clusters_labels.values)):
        #     MEGPanel.clusters_labels.values[ind] = mu.Bag(MEGPanel.clusters_labels.values[ind])
        _meg_clusters_labels_files_update()


def _meg_clusters_labels_files_update():
    if MEGPanel.should_load_clusters_file:
        MEGPanel.clusters_labels = load_clusters_file()
    load_contours()
    update_clusters()
    load_stc()


def load_clusters_file(clusters_name=''):
    if clusters_name == '':
        clusters_name = bpy.context.scene.meg_clusters_labels_files
    fname = MEGPanel.clusters_labels_fnames[clusters_name]
    c = mu.Bag(mu.load(fname))
    for ind in range(len(c.values)):
        c.values[ind] = mu.Bag(c.values[ind])
    return c


def load_stc():
    if MNE_EXIST:
        stc_fname = op.join(mu.get_user_fol(), 'meg', '{}-lh.stc'.format(MEGPanel.clusters_labels.stc_name))
        MEGPanel.stc = mne.read_source_estimate(stc_fname)
        MEGPanel.stc._data *= np.power(10, 9) # from Amp to nAmp
        MEGPanel.max_stc = get_max_stc_t(MEGPanel.stc, MEGPanel.clusters_labels.time)


def load_contours():
    contours = mu.Bag({hemi:None for hemi in mu.HEMIS})
    for hemi in mu.HEMIS:
        verts_num = len(bpy.data.objects[hemi].data.vertices)
        contours[hemi] = mu.Bag(dict(contours=np.zeros(verts_num), labels=[], max=0))
        contours_fnames = glob.glob(op.join(mu.get_user_fol(), 'labels', 'clusters-{}*_contours_{}.npz'.format(
            bpy.context.scene.meg_clusters_labels_files, hemi)))
        for contours_fname in contours_fnames:
            d = np.load(contours_fname)
            # print(contours_fname, d['labels'], np.unique(d['contours']))
            contours_data = d['contours']
            contours_data[np.where(contours_data)] += len(contours[hemi].labels)
            contours[hemi].contours += contours_data
            labels = [l for l in d['labels'] if 'unknown' not in l]
            contours[hemi].labels.extend(labels)
    for hemi in mu.HEMIS:
        contours[hemi].max = len(contours[hemi].labels)
        # print('contours in {} {}: {}'.format(hemi, bpy.context.scene.meg_clusters_labels_files,
        #                                      np.unique(contours[hemi].contours)))
    MEGPanel.contours = contours


def get_clusters_files(user_fol=''):
    clusters_labels_files = glob.glob(op.join(user_fol, 'meg', 'clusters', 'clusters_labels_*.pkl'))
    files_names = [mu.namebase(fname)[len('clusters_labels_'):] for fname in clusters_labels_files]
    clusters_labels_items = [(c, c, '', ind) for ind, c in enumerate(list(set(files_names)))]
    return files_names, clusters_labels_files, clusters_labels_items


def meg_how_to_sort_update(self, context):
    if MEGPanel.init:
        update_clusters()


def cluster_name(x):
    return _cluster_name(x, bpy.context.scene.meg_how_to_sort)


def _cluster_name(x, sort_mode):
    if sort_mode == 'val':
        return '{}_{:.2f}'.format(x.name, x.max)
    elif sort_mode == 'size':
        return '{}_{:.2f}'.format(x.name, x.size)
    elif sort_mode == 'label':
        return x.name


def update_clusters(val_threshold=None, size_threshold=None, clusters_label=''):
    if val_threshold is None:
        val_threshold = bpy.context.scene.meg_clusters_val_threshold
    if size_threshold is None:
        size_threshold = bpy.context.scene.meg_clusters_size_threshold
    if clusters_label == '':
        clusters_label = bpy.context.scene.meg_clusters_label
    MEGPanel.clusters_labels_filtered = filter_clusters(val_threshold, size_threshold, clusters_label)
    for cluster in MEGPanel.clusters_labels_filtered:
        set_cluster_time_series(cluster)
    if bpy.context.scene.meg_how_to_sort == 'val':
        sort_func = lambda x: abs(x.max)
    elif bpy.context.scene.meg_how_to_sort == 'size':
        sort_func = lambda x: x.size
    elif bpy.context.scene.meg_how_to_sort == 'label':
        sort_func = lambda x: x.name
    clusters_tup = sorted([(sort_func(x), cluster_name(x)) for x in MEGPanel.clusters_labels_filtered])[::-1]
    MEGPanel.clusters = [x_name for x_size, x_name in clusters_tup]
    clusters_names = [cluster_name(x) for x in MEGPanel.clusters_labels_filtered]
    MEGPanel.clusters_lookup = {x_name:cluster for x_name, cluster in
                                zip(clusters_names, MEGPanel.clusters_labels_filtered)}
    # MEGPanel.clusters.sort(key=mu.natural_keys)
    clusters_items = [(c, c, '', ind + 1) for ind, c in enumerate(MEGPanel.clusters)]
    bpy.types.Scene.meg_clusters = bpy.props.EnumProperty(
        items=clusters_items, description="meg clusters", update=clusters_update)
    if len(MEGPanel.clusters) > 0:
        bpy.context.scene.meg_clusters = MEGPanel.clusters[0]


def next_cluster():
    index = MEGPanel.clusters.index(bpy.context.scene.meg_clusters)
    next_cluster = MEGPanel.clusters[index + 1] if index < len(MEGPanel.clusters) - 1 else MEGPanel.clusters[0]
    bpy.context.scene.meg_clusters = next_cluster


def prev_cluster():
    index = MEGPanel.clusters.index(bpy.context.scene.meg_clusters)
    prev_cluster = MEGPanel.clusters[index - 1] if index > 0 else MEGPanel.clusters[-1]
    bpy.context.scene.meg_clusters = prev_cluster


def plot_clusters():
    if MEGPanel.stc is not None:
        if bpy.context.scene.coloring_lower_threshold > MEGPanel.max_stc:
            bpy.context.scene.coloring_lower_threshold = 0
        _addon().plot_stc(MEGPanel.stc, MEGPanel.clusters_labels.time,
                 threshold=bpy.context.scene.coloring_lower_threshold, save_image=False, save_prev_colors=True)
        if not _addon().colorbar_values_are_locked():
            _addon().set_colorbar_title('MEG activity (nAmp)')
        bpy.context.scene.frame_current = MEGPanel.clusters_labels.time


def get_max_stc_t(stc, t):
    C = max([stc.rh_data.shape[0], stc.lh_data.shape[0]])
    stc_lh_data = stc.lh_data[:, t:t + 1] if stc.lh_data.shape[0] > 0 else np.zeros((C, 1))
    stc_rh_data = stc.rh_data[:, t:t + 1] if stc.rh_data.shape[0] > 0 else np.zeros((C, 1))
    data = np.concatenate([stc_lh_data, stc_rh_data])
    return np.max(np.abs(data))


def select_meg_cluster(event, context, pos=None):
    if not MEGPanel.init:
        return

    # Should be checked in the appearence modal loop
    # area = mu.get_click_area(event, context)
    # if area.type != 'VIEW_3D':
    #     return

    # if pos is None:    #
    # from bpy_extras.view3d_utils import region_2d_to_vector_3d, region_2d_to_location_3d
    #     region = [r for r in area.regions if r.type == 'WINDOW'][0]
    #     rv3d = area.spaces.active.region_3d
    #     coord = (event.mouse_x - area.x, event.mouse_y - area.y)
    #     vec = region_2d_to_vector_3d(region, rv3d, coord)
    #     pos = region_2d_to_location_3d(region, rv3d, coord, vec)
    #     # pos = mu.mouse_coo_to_3d_loc(event, context)
    #     print('pos, ', pos)
    # else:
    #     print('cursor, ', pos)
    if pos is None:
        return
    # bpy.context.scene.cumulate_meg_cluster = event.shift
    for clusters_name in MEGPanel.clusters_labels_fnames.keys():
        clusters = load_clusters_file(clusters_name)
        for cluster in clusters.values: #MEGPanel.clusters_labels.values:
            cluster_hemi_mesh_name = 'inflated_{}'.format(cluster.hemi)
            cluster_hemi_obj = bpy.data.objects[cluster_hemi_mesh_name]
            cluster_pos = pos * cluster_hemi_obj.matrix_world.inverted()

            cluster_vertices_co = get_cluster_verts_co(cluster)
            # dist = np.linalg.norm(cluster_max_vert_co - cluster_pos)
            # print(cluster.name, dist, cluster_max_vert_co, cluster_pos)
            co, index, dist = mu.min_cdist(cluster_vertices_co, [cluster_pos])
            # print(cluster.name, co, index, dist, cluster_pos)
            if dist < 1:
            # if cluster_hemi_mesh_name == closest_mesh_name and vertex_ind in cluster.vertices:
                if bpy.context.scene.meg_clusters_labels_files != clusters_name:
                    MEGPanel.should_load_clusters_file = False
                    MEGPanel.clusters_labels = clusters
                    bpy.context.scene.meg_clusters_labels_files = clusters_name
                    MEGPanel.should_load_clusters_file = True
                bpy.context.scene.meg_clusters = cluster_name(cluster)
                return cluster
    return None


def select_all_clusters():
    parent_obj = bpy.data.objects.get(PARENT_OBJ_NAME, None)
    if parent_obj is not None:
        parent_obj.select = True

    fcurves = mu.get_fcurves(PARENT_OBJ_NAME)
    filtered_fcurves_names = [get_cluster_fcurve_name(c) for c in MEGPanel.clusters_labels_filtered]
    for fcurve in fcurves:
        fcurve_name = mu.get_fcurve_name(fcurve)
        fcurve.hide = fcurve_name not in filtered_fcurves_names
    plot_all_clusters()
    mu.view_all_in_graph_editor()
    # mu.change_selected_fcurves_colors(mu.OBJ_TYPE_ELECTRODE)


def flip_meg_clusters_ts():
    fcurves = mu.get_fcurves(PARENT_OBJ_NAME, only_not_hiden=True)
    for fcurve, cluster in zip(fcurves, MEGPanel.clusters_labels_filtered):
        if cluster.ts_max < 0:
            for t in range(len(cluster.label_data)):
                fcurve.keyframe_points[t].co[1] = -fcurve.keyframe_points[t].co[1]
    MEGPanel.data_is_flipped = not MEGPanel.data_is_flipped


def deselect_all_clusters():
    for fcurve in mu.get_fcurves(PARENT_OBJ_NAME):
        fcurve.hide = True
    clear_all_clusters()


def clear_all_clusters():
    for obj_name in mu.INF_HEMIS:
        mesh = bpy.data.objects[obj_name].data
        _addon().recreate_coloring_layers(mesh, 'contours')
    # for cluster in MEGPanel.clusters:
    #     clear_cluster(cluster)


def meg_draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'meg_clusters_labels_files', text='')
    if MNE_EXIST:
        layout.operator(PlotMEGClusters.bl_idname, text='Plot STC', icon='POTATO')
    row = layout.row(align=True)
    row.operator(PrevMEGCluster.bl_idname, text="", icon='PREV_KEYFRAME')
    row.prop(context.scene, 'meg_clusters', text='')
    row.operator(NextMEGCluster.bl_idname, text="", icon='NEXT_KEYFRAME')
    row = layout.row(align=True)
    row.label(text='Sort: ')
    row.prop(context.scene, 'meg_how_to_sort', expand=True)
    layout.prop(context.scene, 'meg_show_filtering', text='Refine clusters')
    if bpy.context.scene.meg_show_filtering:
        layout.prop(context.scene, 'meg_clusters_val_threshold', text='Val threshold')
        layout.prop(context.scene, 'meg_clusters_size_threshold', text='Size threshold')
        layout.prop(context.scene, 'meg_clusters_label', text='Label filter')
        layout.operator(FilterMEGClusters.bl_idname, text="Filter clusters", icon='FILTER')
    layout.prop(context.scene, 'plot_current_meg_cluster', text="Plot current cluster's contour")
    layout.prop(context.scene, 'cumulate_meg_cluster', text="Cumulate contours")
    if not MEGPanel.current_cluster is None and len(MEGPanel.current_cluster) > 0: # and not MEGPanel.dont_show_clusters_info:
        cluster_size = MEGPanel.current_cluster['size']
        col = layout.box().column()
        mu.add_box_line(col, 'Max val', '{:.2f}'.format(MEGPanel.current_cluster['max']), 0.7)
        mu.add_box_line(col, 'Size', str(cluster_size), 0.7)
        col = layout.box().column()
        labels_num_to_show = min(7, len(MEGPanel.current_cluster['intersects']))
        for inter_labels in MEGPanel.current_cluster['intersects'][:labels_num_to_show]:
            mu.add_box_line(col, inter_labels['name'], '{:.0%}'.format(inter_labels['num'] / float(cluster_size)), 0.8)
        if labels_num_to_show < len(MEGPanel.current_cluster['intersects']):
            layout.label(text='Out of {} labels'.format(len(MEGPanel.current_cluster['intersects'])))
    layout.operator(SelectAllClusters.bl_idname, text="Select all", icon='BORDER_RECT')
    text = 'Flip time series' if not MEGPanel.data_is_flipped else 'Unflip time series'
    layout.operator(FlipMEGClustersTS.bl_idname, text=text, icon='FORCE_MAGNETIC')
    layout.operator(DeselecAllClusters.bl_idname, text="Deselect all", icon='PANEL_CLOSE')
    layout.operator(ClearClusters.bl_idname, text="Clear all clusters", icon='PANEL_CLOSE')
    layout.operator(_addon().ClearColors.bl_idname, text="Clear activity", icon='PANEL_CLOSE')


bpy.types.Scene.meg_clusters_labels_files = bpy.props.EnumProperty(
    items=[], description="meg files", update=meg_clusters_labels_files_update)
bpy.types.Scene.meg_clusters = bpy.props.EnumProperty(
    items=[], description="meg clusters", update=clusters_update)
bpy.types.Scene.meg_clusters_val_threshold = bpy.props.FloatProperty(min=0)
bpy.types.Scene.meg_clusters_size_threshold = bpy.props.FloatProperty(min=0)
bpy.types.Scene.meg_clusters_label = bpy.props.StringProperty()
bpy.types.Scene.meg_how_to_sort = bpy.props.EnumProperty(
    items=[('val', 'val', '', 1), ('size', 'size', '', 2), ('label', 'label', '', 3)],
    description='How to sort', update=meg_how_to_sort_update)
bpy.types.Scene.meg_show_filtering = bpy.props.BoolProperty(default=False)
bpy.types.Scene.plot_current_meg_cluster = bpy.props.BoolProperty(
    default=True, description="Plot current cluster's contour")
bpy.types.Scene.cumulate_meg_cluster = bpy.props.BoolProperty(default=False, description="Cumulate contours")


class NextMEGCluster(bpy.types.Operator):
    bl_idname = 'mmvt.next_meg_cluster'
    bl_label = 'nextMEGCluster'
    bl_options = {'UNDO'}

    def invoke(self, context, event=None):
        next_cluster()
        return {'FINISHED'}


class PrevMEGCluster(bpy.types.Operator):
    bl_idname = 'mmvt.prev_meg_cluster'
    bl_label = 'prevMEGCluster'
    bl_options = {'UNDO'}

    def invoke(self, context, event=None):
        prev_cluster()
        return {'FINISHED'}


class FlipMEGClustersTS(bpy.types.Operator):
    bl_idname = "mmvt.flip_meg_clusters_ts"
    bl_label = "Flip meg clusters ts"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        flip_meg_clusters_ts()
        return {'PASS_THROUGH'}


class SelectAllClusters(bpy.types.Operator):
    bl_idname = "mmvt.select_all_meg_clusters"
    bl_label = "Select all meg clusters"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        select_all_clusters()
        return {'PASS_THROUGH'}


class DeselecAllClusters(bpy.types.Operator):
    bl_idname = "mmvt.deselect_all_meg_clusters"
    bl_label = "Deselect all meg clusters"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        deselect_all_clusters()
        return {'PASS_THROUGH'}


class ClearClusters(bpy.types.Operator):
    bl_idname = "mmvt.clear_all_meg_clusters"
    bl_label = "Clear all meg clusters"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        clear_all_clusters()
        return {'PASS_THROUGH'}


class FilterMEGClusters(bpy.types.Operator):
    bl_idname = "mmvt.filter_meg_clusters"
    bl_label = "Filter MEG clusters"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        update_clusters()
        return {'PASS_THROUGH'}



class PlotMEGClusters(bpy.types.Operator):
    bl_idname = "mmvt.plot_meg_clusters"
    bl_label = "Plot MEG clusters"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        plot_clusters()
        return {'PASS_THROUGH'}


class MEGPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "MEG"
    addon = None
    init = False
    clusters_labels = None
    clusters_labels_fnames = {}
    stc = None
    prev_cluster = ''
    current_cluster = {}
    should_load_clusters_file = True
    data_is_flipped = False

    def draw(self, context):
        if MEGPanel.init:
            meg_draw(self, context)


def init(addon):
    MEGPanel.addon = addon
    user_fol = mu.get_user_fol()
    meg_files = glob.glob(op.join(user_fol, 'meg', 'meg*.npz'))
    if len(meg_files) == 0:
        print('No MEG clusters files')
        return None

    files_names, clusters_labels_files, clusters_labels_items = get_clusters_files(user_fol)
    MEGPanel.meg_clusters_files_exist = len(files_names) > 0 
    if not MEGPanel.meg_clusters_files_exist:
        print('No MEG_clusters_files_exist')
        return None
    for fname, name in zip(clusters_labels_files, files_names):
        MEGPanel.clusters_labels_fnames[name] = fname

    register()
    MEGPanel.init = True

    bpy.types.Scene.meg_clusters_labels_files = bpy.props.EnumProperty(
        items=clusters_labels_items, description="meg files", update=meg_clusters_labels_files_update)
    bpy.context.scene.meg_clusters_labels_files = files_names[0]
    bpy.context.scene.meg_clusters_val_threshold = MEGPanel.clusters_labels.min_cluster_max
    bpy.context.scene.meg_clusters_size_threshold = MEGPanel.clusters_labels.min_cluster_size
    bpy.context.scene.meg_clusters_label = MEGPanel.clusters_labels.clusters_label
    bpy.context.scene.meg_how_to_sort = 'val'
    bpy.context.scene.meg_show_filtering = False
    bpy.context.scene.cumulate_meg_cluster = False


def register():
    try:
        unregister()
        bpy.utils.register_class(MEGPanel)
        bpy.utils.register_class(NextMEGCluster)
        bpy.utils.register_class(PlotMEGClusters)
        bpy.utils.register_class(PrevMEGCluster)
        bpy.utils.register_class(FilterMEGClusters)
        bpy.utils.register_class(SelectAllClusters)
        bpy.utils.register_class(FlipMEGClustersTS)
        bpy.utils.register_class(DeselecAllClusters)
        bpy.utils.register_class(ClearClusters)
    except:
        print("Can't register MEG Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(MEGPanel)
        bpy.utils.unregister_class(PlotMEGClusters)
        bpy.utils.unregister_class(NextMEGCluster)
        bpy.utils.unregister_class(PrevMEGCluster)
        bpy.utils.unregister_class(FilterMEGClusters)
        bpy.utils.unregister_class(SelectAllClusters)
        bpy.utils.unregister_class(FlipMEGClustersTS)
        bpy.utils.unregister_class(DeselecAllClusters)
        bpy.utils.unregister_class(ClearClusters)
    except:
        pass
