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


def _addon():
    return MEGPanel.addon


def clusters_update(self, context):
    if MEGPanel.addon is not None and MEGPanel.init:
        _clusters_update()


def get_cluster_name(cluster):
    return 'cluster_size_{}_max_{:.2f}_{}'.format(cluster.size, cluster.max, cluster.name)


def _clusters_update():
    cluster = MEGPanel.clustser_lookup[bpy.context.scene.meg_clusters]
    create_empty_and_insert_time_series(cluster)
    cluster_name = get_cluster_name(cluster)
    max_vert_ind = cluster.max_vert
    inflated_mesh = 'inflated_{}'.format(cluster.hemi)
    me = bpy.data.objects[inflated_mesh].to_mesh(bpy.context.scene, True, 'PREVIEW')
    bpy.context.scene.cursor_location = me.vertices[max_vert_ind].co / 10.0
    _addon().set_cursor_pos()
    bpy.data.meshes.remove(me)
    _addon().set_closest_vertex_and_mesh_to_cursor(max_vert_ind, inflated_mesh)
    _addon().save_cursor_position()
    _addon().create_slices()
    if bpy.context.scene.plot_current_meg_cluster:
        _addon().color_contours(
            [cluster_name], cluster.hemi, MEGPanel.contours)
        if not bpy.context.scene.cumulate_meg_cluster and MEGPanel.prev_cluster not in ['', bpy.context.scene.meg_clusters]:
            prev_cluster = MEGPanel.clustser_lookup[MEGPanel.prev_cluster]
            prev_cluster_name = 'cluster_size_{}_max_{:.2f}_{}'.format(prev_cluster.size, prev_cluster.max, prev_cluster.name)
            contures = MEGPanel.contours[prev_cluster.hemi]
            label_ind = np.where(np.array(contures['labels']) == prev_cluster_name)[0][0] + 1
            prev_clustes_contour_vertices = np.where(contures['contours'] == label_ind)[0]
            _addon().color_prev_colors(prev_clustes_contour_vertices, 'inflated_{}'.format(prev_cluster.hemi))
    MEGPanel.prev_cluster = bpy.context.scene.meg_clusters


def create_empty_and_insert_time_series(cluster):
    cluster_uid_name = '{}_{}_{:.2f}'.format(cluster.name, cluster.size, cluster.max)
    _addon().create_empty_if_doesnt_exists('meg_clusters', _addon().EMPTY_LAYER, bpy.context.scene.layers, 'Functional maps')
    # _addon().create_empty_if_doesnt_exists(cluster_uid_name, _addon().EMPTY_LAYER, bpy.context.scene.layers, 'meg_clusters')
    # cur_obj = bpy.data.objects[cluster_uid_name]
    parent_obj = bpy.data.objects['meg_clusters']
    T = len(cluster.label_data)
    fcurves_num = mu.count_fcurves(parent_obj)
    cluster.label_data = np.array(cluster.label_data, dtype=np.float64)
    if fcurves_num == 0:
        # Set the values to zeros in the first and last frame for current object(current label)
        mu.insert_keyframe_to_custom_prop(parent_obj, cluster_uid_name, 0, 1)
        mu.insert_keyframe_to_custom_prop(parent_obj, cluster_uid_name, 0, T + 2)

        # For every time point insert keyframe to current object
        for ind, t_data in enumerate(cluster.label_data):
            mu.insert_keyframe_to_custom_prop(parent_obj, cluster_uid_name, t_data, ind + 2)

        # remove the orange keyframe sign in the fcurves window
        fcurves = parent_obj.animation_data.action.fcurves[0]
        mod = fcurves.modifiers.new(type='LIMITS')
    else:
        fcurve = parent_obj.animation_data.action.fcurves[0]
        fcurve.keyframe_points[0].co[1] = 0
        fcurve.keyframe_points[-1].co[1] = 0
        for t in range(T):
            fcurve.keyframe_points[t + 1].co[1] = cluster.label_data[t]
    mu.view_all_in_graph_editor()

def filter_clusters(val_threshold=None, size_threshold=None, clusters_label=None):
    if val_threshold is None:
        val_threshold = bpy.context.scene.meg_cluster_val_threshold
    if size_threshold is None:
        size_threshold = bpy.context.scene.meg_cluster_size_threshold
    if clusters_label is None:
        clusters_label = bpy.context.scene.meg_clusters_label
    return [c for c in MEGPanel.clusters_labels.values
            if abs(c.max) >= val_threshold and c.size >= size_threshold and \
            any([clusters_label in inter_label['name'] for inter_label in c.intersects])]


def meg_clusters_labels_files_update(self, context):
    if MEGPanel.init:
        fname = MEGPanel.clusters_labels_fnames[bpy.context.scene.meg_clusters_labels_files]
        MEGPanel.clusters_labels = mu.load(fname)
        MEGPanel.clusters_labels = mu.Bag(MEGPanel.clusters_labels)
        for ind in range(len(MEGPanel.clusters_labels.values)):
            MEGPanel.clusters_labels.values[ind] = mu.Bag(MEGPanel.clusters_labels.values[ind])
        load_contours()
        update_clusters()
        load_stc()


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
            print(contours_fname, d['labels'], np.unique(d['contours']))
            contours_data = d['contours']
            contours_data[np.where(contours_data)] += len(contours[hemi].labels)
            contours[hemi].contours += contours_data
            labels = [l for l in d['labels'] if 'unknown' not in l]
            contours[hemi].labels.extend(labels)
    for hemi in mu.HEMIS:
        contours[hemi].max = len(contours[hemi].labels)
        print('contours in {} {}: {}'.format(hemi, bpy.context.scene.meg_clusters_labels_files,
                                             np.unique(contours[hemi].contours)))
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
    if bpy.context.scene.meg_how_to_sort == 'val':
        sort_func = lambda x: abs(x.max)
    elif bpy.context.scene.meg_how_to_sort == 'size':
        sort_func = lambda x: x.size
    elif bpy.context.scene.meg_how_to_sort == 'label':
        sort_func = lambda x: x.name
    clusters_tup = sorted([(sort_func(x), cluster_name(x)) for x in MEGPanel.clusters_labels_filtered])[::-1]
    MEGPanel.clusters = [x_name for x_size, x_name in clusters_tup]
    clusters_names = [cluster_name(x) for x in MEGPanel.clusters_labels_filtered]
    MEGPanel.clustser_lookup = {x_name:cluster for x_name, cluster in
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
        if bpy.context.scene.coloring_threshold > MEGPanel.max_stc:
            bpy.context.scene.coloring_threshold = 0
        _addon().plot_stc(MEGPanel.stc, MEGPanel.clusters_labels.time,
                 threshold=bpy.context.scene.coloring_threshold, save_image=False, save_prev_colors=True)


def get_max_stc_t(stc, t):
    C = max([stc.rh_data.shape[0], stc.lh_data.shape[0]])
    stc_lh_data = stc.lh_data[:, t:t + 1] if stc.lh_data.shape[0] > 0 else np.zeros((C, 1))
    stc_rh_data = stc.rh_data[:, t:t + 1] if stc.rh_data.shape[0] > 0 else np.zeros((C, 1))
    data = np.concatenate([stc_lh_data, stc_rh_data])
    return np.max(np.abs(data))


def meg_draw(self, context):
    layout = self.layout
    user_fol = mu.get_user_fol()
    layout.prop(context.scene, 'meg_clusters_labels_files', text='')
    if MNE_EXIST:
        layout.operator(PlotMEGClusters.bl_idname, text="Plot STC", icon='POTATO')
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
    except:
        print("Can't register MEG Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(MEGPanel)
        bpy.utils.unregister_class(PlotMEGClusters)
        bpy.utils.unregister_class(NextMEGCluster)
        bpy.utils.unregister_class(PrevMEGCluster)
        bpy.utils.unregister_class(FilterMEGClusters)

    except:
        pass
