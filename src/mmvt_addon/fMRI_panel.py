import bpy
import os.path as op
import numpy as np
import mmvt_utils as mu
import glob
from queue import Empty


def _addon():
    return fMRIPanel.addon


def get_clusters_file_names():
    return fMRIPanel.clusters_labels_file_names


def fMRI_clusters_files_exist():
    return fMRIPanel.fMRI_clusters_files_exist


def clusters_update(self, context):
    _clusters_update()


def _clusters_update():
    if fMRIPanel.addon is None or not fMRIPanel.init:
        return
    clusters_labels_file = bpy.context.scene.fmri_clusters_labels_files
    fMRIPanel.cluster_labels = fMRIPanel.lookup[clusters_labels_file][bpy.context.scene.fmri_clusters]
    if bpy.context.scene.plot_current_cluster and not fMRIPanel.blobs_plotted:
        faces_verts = fMRIPanel.addon.get_faces_verts()
        if bpy.context.scene.fmri_what_to_plot == 'blob':
            plot_blob(fMRIPanel.cluster_labels, faces_verts)


def fmri_blobs_percentile_min_update(self, context):
    if bpy.context.scene.fmri_blobs_percentile_min > bpy.context.scene.fmri_blobs_percentile_max:
        bpy.context.scene.fmri_blobs_percentile_min = bpy.context.scene.fmri_blobs_percentile_max


def fmri_blobs_percentile_max_update(self, context):
    if bpy.context.scene.fmri_blobs_percentile_max < bpy.context.scene.fmri_blobs_percentile_min:
        bpy.context.scene.fmri_blobs_percentile_max = bpy.context.scene.fmri_blobs_percentile_min


def plot_blob(cluster_labels, faces_verts):
    fMRIPanel.dont_show_clusters_info = False
    _addon().init_activity_map_coloring('FMRI', subcorticals=False)
    blob_vertices = cluster_labels['vertices']
    hemi = cluster_labels['hemi']
    real_hemi = hemi
    if _addon().is_inflated():
        hemi = 'inflated_{}'.format(hemi)
    # fMRIPanel.blobs_plotted = True
    fMRIPanel.colors_in_hemis[hemi] = True
    activity = fMRIPanel.constrast[real_hemi]
    blob_activity = np.ones(activity.shape)
    blob_activity[blob_vertices] = activity[blob_vertices]
    if fMRIPanel.blobs_activity is None:
        fMRIPanel.blobs_activity, _ = calc_blobs_activity(
            fMRIPanel.constrast, fMRIPanel.clusters_labels_filtered, fMRIPanel.colors_in_hemis)
    data_min, colors_ratio = calc_colors_ratio(fMRIPanel.blobs_activity)
    threshold = bpy.context.scene.fmri_clustering_threshold
    cur_obj = bpy.data.objects[hemi]
    _addon().activity_map_obj_coloring(
        cur_obj, blob_activity, faces_verts[real_hemi], threshold, True,
        data_min=data_min, colors_ratio=colors_ratio)
    other_real_hemi = mu.other_hemi(real_hemi)
    other_hemi = mu.other_hemi(hemi)
    if fMRIPanel.colors_in_hemis[other_hemi]:
        _addon().clear_cortex([other_real_hemi])
        fMRIPanel.colors_in_hemis[other_hemi] = False


# @mu.profileit()
def find_closest_cluster(only_within=False):
    cursor = np.array(bpy.context.scene.cursor_location) * 10
    if bpy.context.scene.search_closest_cluster_only_in_filtered:
        cluster_to_search_in = fMRIPanel.clusters_labels_filtered
    else:
        clusters_labels_file = bpy.context.scene.fmri_clusters_labels_files
        cluster_to_search_in = fMRIPanel.clusters_labels[clusters_labels_file]['values']
        unfilter_clusters()
    dists, indices = [], []
    for ind, cluster in enumerate(cluster_to_search_in):
        _, _, dist = mu.min_cdist(cluster['coordinates'], [cursor])[0]
        dists.append(dist)
    if len(dists) == 0:
        print('No cluster was found!')
    else:
        min_index = np.argmin(np.array(dists))
        min_dist = dists[min_index]
        if not (only_within and min_dist > 1):
            fMRIPanel.dont_show_clusters_info = False
            closest_cluster = cluster_to_search_in[min_index]
            bpy.context.scene.fmri_clusters = cluster_name(closest_cluster)
            fMRIPanel.cluster_labels = closest_cluster
            print('Closest cluster: {}, dist: {}'.format(bpy.context.scene.fmri_clusters, min_dist))
            _clusters_update()
        else:
            print('only within: dist to big ({})'.format(min_dist))


class NextCluster(bpy.types.Operator):
    bl_idname = 'mmvt.next_cluster'
    bl_label = 'nextCluster'
    bl_options = {'UNDO'}

    def invoke(self, context, event=None):
        next_cluster()
        return {'FINISHED'}


def next_cluster():
    index = fMRIPanel.clusters.index(bpy.context.scene.fmri_clusters)
    next_cluster = fMRIPanel.clusters[index + 1] if index < len(fMRIPanel.clusters) - 1 else fMRIPanel.clusters[0]
    bpy.context.scene.fmri_clusters = next_cluster


class PrevCluster(bpy.types.Operator):
    bl_idname = 'mmvt.prev_cluster'
    bl_label = 'prevcluster'
    bl_options = {'UNDO'}

    def invoke(self, context, event=None):
        prev_cluster()
        return {'FINISHED'}


def prev_cluster():
    index = fMRIPanel.clusters.index(bpy.context.scene.fmri_clusters)
    prev_cluster = fMRIPanel.clusters[index - 1] if index > 0 else fMRIPanel.clusters[-1]
    bpy.context.scene.fmri_clusters = prev_cluster


def fmri_clusters_update(self, context):
    if fMRIPanel.init:
        update_clusters()


def set_fmri_clusters_file_name(file_name):
    bpy.context.scene.fmri_clusters_labels_files = file_name


def fmri_clusters_labels_files_update(self, context):
    constrast_name = bpy.context.scene.fmri_clusters_labels_files
    fMRIPanel.constrast = {}
    for hemi in mu.HEMIS:
        contrast_fname = op.join(mu.get_user_fol(), 'fmri', 'fmri_{}_{}.npy'.format(constrast_name, hemi))
        fMRIPanel.constrast[hemi] = np.load(contrast_fname)
    if fMRIPanel.init:
        update_clusters()


def fmri_how_to_sort_update(self, context):
    if fMRIPanel.init:
        update_clusters()


def update_clusters(val_threshold=None, size_threshold=None):
    fMRIPanel.dont_show_clusters_info = False
    if val_threshold is None:
        val_threshold = bpy.context.scene.fmri_cluster_val_threshold
    if size_threshold is None:
        size_threshold = bpy.context.scene.fmri_cluster_size_threshold
    clusters_labels_file = bpy.context.scene.fmri_clusters_labels_files
    if isinstance(fMRIPanel.clusters_labels[clusters_labels_file], dict):
        bpy.context.scene.fmri_clustering_threshold = fMRIPanel.clusters_labels[clusters_labels_file]['threshold']
    else:
        bpy.context.scene.fmri_clustering_threshold = 2
    # bpy.context.scene.fmri_cluster_val_threshold = bpy.context.scene.fmri_clustering_threshold
    fMRIPanel.clusters_labels_filtered = filter_clusters(clusters_labels_file, val_threshold, size_threshold)
    sort_field = 'max' if bpy.context.scene.fmri_how_to_sort == 'tval' else 'size'
    clusters_tup = sorted([(abs(x[sort_field]), cluster_name(x)) for x in fMRIPanel.clusters_labels_filtered])[::-1]
    fMRIPanel.clusters = [x_name for x_size, x_name in clusters_tup]
    # fMRIPanel.clusters.sort(key=mu.natural_keys)
    clusters_items = [(c, c, '', ind) for ind, c in enumerate(fMRIPanel.clusters)]
    bpy.types.Scene.fmri_clusters = bpy.props.EnumProperty(
        items=clusters_items, description="fmri clusters", update=clusters_update)
    if len(fMRIPanel.clusters) > 0:
        bpy.context.scene.fmri_clusters = fMRIPanel.current_cluster = fMRIPanel.clusters[0]
        if bpy.context.scene.fmri_clusters in fMRIPanel.lookup:
            fMRIPanel.cluster_labels = fMRIPanel.lookup[clusters_labels_file][bpy.context.scene.fmri_clusters]


def unfilter_clusters():
    update_clusters(2, 1)


def plot_all_blobs():
    # fMRIPanel.dont_show_clusters_info = False
    faces_verts = _addon().get_faces_verts()
    _addon().init_activity_map_coloring('FMRI', subcorticals=False)
    blobs_activity, hemis = calc_blobs_activity(
        fMRIPanel.constrast, fMRIPanel.clusters_labels_filtered, fMRIPanel.colors_in_hemis)
    data_min, colors_ratio = calc_colors_ratio(blobs_activity)
    threshold = bpy.context.scene.fmri_clustering_threshold
    for hemi in hemis:
        if bpy.context.scene.coloring_both_pial_and_inflated:
            for inf_hemi in [hemi, 'inflated_{}'.format(hemi)]:
                _addon().activity_map_obj_coloring(
                    bpy.data.objects[inf_hemi], blobs_activity[hemi], faces_verts[hemi], threshold, True,
                    data_min=data_min, colors_ratio=colors_ratio)
        else:
            inf_hemi = hemi if _addon().is_pial() else 'inflated_{}'.format(hemi)
            _addon().activity_map_obj_coloring(
                bpy.data.objects[inf_hemi],blobs_activity[hemi], faces_verts[hemi], threshold, True,
                data_min=data_min, colors_ratio=colors_ratio)
    for hemi in set(mu.HEMIS) - hemis:
        _addon().clear_cortex([hemi])
    fMRIPanel.blobs_plotted = True


def calc_blobs_activity(constrast, clusters_labels_filtered, colors_in_hemis={}):
    fmri_contrast, blobs_activity = {}, {}
    for hemi in mu.HEMIS:
        fmri_contrast[hemi] = constrast[hemi]
        blobs_activity[hemi] = np.zeros(fmri_contrast[hemi].shape)
    hemis = set()
    for cluster_labels in clusters_labels_filtered:
        if bpy.context.scene.fmri_what_to_plot == 'blob':
            blob_vertices = cluster_labels['vertices']
            hemi = cluster_labels['hemi']
            hemis.add(hemi)
            inf_hemi = hemi if _addon().is_pial() else 'inflated_{}'.format(hemi)
            #todo: check if colors_in_hemis should be initialized (I guess it should be...)
            colors_in_hemis[inf_hemi] = True
            blobs_activity[hemi][blob_vertices] = fmri_contrast[hemi][blob_vertices]
    return blobs_activity, hemis


def calc_colors_ratio(activity):
    if _addon().colorbar_values_are_locked():
        data_max, data_min = _addon().get_colorbar_max_min()
    else:
        data_max, data_min = get_activity_max_min(activity)
        # output_fname = op.join(mu.get_user_fol(), 'fmri','fmri_blobs_{}_minmax.pkl'.format(
        #         bpy.context.scene.fmri_clusters_labels_files))
        # print('Saving {}'.format(output_fname))
        # mu.save((data_min, data_max), output_fname)
        if data_max == 0 and data_min == 0:
            print('Both data max and min are zeros!')
            return 0, 0
        _addon().set_colorbar_max_min(data_max, data_min)

    colors_ratio = 256 / (data_max - data_min)
    _addon().set_colorbar_title('fMRI')
    return data_min, colors_ratio


def get_activity_max_min(activity):
    norm_percs = (bpy.context.scene.fmri_blobs_percentile_min, bpy.context.scene.fmri_blobs_percentile_max)
    data_max, data_min = mu.get_data_max_min(
        activity, bpy.context.scene.fmri_blobs_norm_by_percentile, norm_percs=norm_percs, data_per_hemi=True,
        symmetric=True)
    return data_max, data_min


def cluster_name(x):
    return _cluster_name(x, bpy.context.scene.fmri_how_to_sort)


def _cluster_name(x, sort_mode):
    return '{}_{:.2f}'.format(x['name'], x['max']) if sort_mode == 'tval' else\
        '{}_{:.2f}'.format(x['name'], x['size'])


def support_old_verions(clusters_labels):
    # support old versions
    if not isinstance(clusters_labels, dict):
        data = clusters_labels
        new_clusters_labels = dict(values=data, threshold=2)
    else:
        new_clusters_labels = clusters_labels
    if not 'size' in new_clusters_labels['values'][0]:
        for cluster_labels in new_clusters_labels['values']:
            if not 'size' in cluster_labels:
                cluster_labels['size'] = len(cluster_labels['vertices'])
    return new_clusters_labels


def find_fmri_files_min_max():
    _addon().lock_colorbar_values()
    abs_values = []
    for constrast_name in fMRIPanel.clusters_labels_file_names:
        constrast = {}
        for hemi in mu.HEMIS:
            contrast_fname = op.join(mu.get_user_fol(), 'fmri', 'fmri_{}_{}.npy'.format(constrast_name, hemi))
            constrast[hemi] = np.load(contrast_fname)
        clusters_labels_filtered = filter_clusters(constrast_name)
        blobs_activity, _ = calc_blobs_activity(constrast, clusters_labels_filtered)
        data_max, data_min = get_activity_max_min(blobs_activity)
        abs_values.extend([abs(data_max), abs(data_min)])
    data_max = max(abs_values)
    _addon().set_colorbar_max_min(data_max, -data_max)


def filter_clusters(constrast_name, val_threshold=None, size_threshold=None):
    if val_threshold is None:
        val_threshold = bpy.context.scene.fmri_cluster_val_threshold
    if size_threshold is None:
        size_threshold = bpy.context.scene.fmri_cluster_size_threshold
    return [c for c in fMRIPanel.clusters_labels[constrast_name]['values']
            if abs(c['max']) >= val_threshold and len(c['vertices']) >= size_threshold]


def fMRI_draw(self, context):
    layout = self.layout
    user_fol = mu.get_user_fol()
    # clusters_labels_files = glob.glob(op.join(user_fol, 'fmri', 'clusters_labels_*.npy'))
    # if len(clusters_labels_files) > 1:
    layout.prop(context.scene, 'fmri_clusters_labels_files', text='')
    row = layout.row(align=True)
    row.prop(context.scene, 'fmri_clustering_threshold', text='Threshold')
    row.operator(RefinefMRIClusters.bl_idname, text="Find clusters", icon='GROUP_VERTEX')
    layout.prop(context.scene, 'fmri_cluster_val_threshold', text='clusters t-val threshold')
    layout.prop(context.scene, 'fmri_cluster_size_threshold', text='clusters size threshold')
    layout.operator(FilterfMRIBlobs.bl_idname, text="Filter blobs", icon='FILTER')
    row = layout.row(align=True)
    row.operator(PrevCluster.bl_idname, text="", icon='PREV_KEYFRAME')
    row.prop(context.scene, 'fmri_clusters', text="")
    row.operator(NextCluster.bl_idname, text="", icon='NEXT_KEYFRAME')
    layout.prop(context.scene, 'plot_current_cluster', text="Plot current cluster")
    layout.prop(context.scene, 'plot_fmri_cluster_per_click', text="Listen to left clicks")
    # layout.prop(context.scene, 'fmri_what_to_plot', expand=True)
    row = layout.row(align=True)
    row.label(text='Sort: ')
    row.prop(context.scene, 'fmri_how_to_sort', expand=True)
    if not fMRIPanel.cluster_labels is None and len(fMRIPanel.cluster_labels) > 0 and not fMRIPanel.dont_show_clusters_info:
        if 'size' not in fMRIPanel.cluster_labels:
            fMRIPanel.cluster_labels['size'] = len(fMRIPanel.cluster_labels['vertices'])
        blob_size = fMRIPanel.cluster_labels['size']
        col = layout.box().column()
        mu.add_box_line(col, 'Max val', '{:.2f}'.format(fMRIPanel.cluster_labels['max']), 0.7)
        mu.add_box_line(col, 'Size', str(blob_size), 0.7)
        col = layout.box().column()
        labels_num_to_show = min(7, len(fMRIPanel.cluster_labels['intersects']))
        for inter_labels in fMRIPanel.cluster_labels['intersects'][:labels_num_to_show]:
            mu.add_box_line(col, inter_labels['name'], '{:.0%}'.format(inter_labels['num'] / float(blob_size)), 0.8)
        if labels_num_to_show < len(fMRIPanel.cluster_labels['intersects']):
            layout.label(text='Out of {} labels'.format(len(fMRIPanel.cluster_labels['intersects'])))
    # row = layout.row(align=True)
    layout.prop(context.scene, 'fmri_blobs_norm_by_percentile', text="Norm by percentiles")
    if bpy.context.scene.fmri_blobs_norm_by_percentile:
        layout.prop(context.scene, 'fmri_blobs_percentile_min', text="Percentile min")
        layout.prop(context.scene, 'fmri_blobs_percentile_max', text="Percentile max")
    layout.operator(PlotAllBlobs.bl_idname, text="Plot all blobs", icon='POTATO')
    layout.operator(NearestCluster.bl_idname, text="Nearest cluster", icon='MOD_SKIN')
    layout.prop(context.scene, 'search_closest_cluster_only_in_filtered', text="Seach only in filtered blobs")
    # layout.operator(LoadMEGData.bl_idname, text="Save as functional ROIs", icon='IPO')
    layout.operator(FindfMRIFilesMinMax.bl_idname, text="Calc minmax for all files", icon='IPO')
    layout.operator(fmriClearColors.bl_idname, text="Clear", icon='PANEL_CLOSE')


class FindfMRIFilesMinMax(bpy.types.Operator):
    bl_idname = "mmvt.find_fmri_files_min_max"
    bl_label = "mmvt find_fmri_files_min_max"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        find_fmri_files_min_max()
        return {"FINISHED"}


class fmriClearColors(bpy.types.Operator):
    bl_idname = "mmvt.fmri_colors_clear"
    bl_label = "mmvt fmri colors clear"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        _addon().clear_cortex()
        _addon().clear_subcortical_fmri_activity()
        fMRIPanel.blobs_plotted = False
        fMRIPanel.dont_show_clusters_info = True
        return {"FINISHED"}


class LoadMEGData(bpy.types.Operator):
    bl_idname = "mmvt.load_meg_data"
    bl_label = "Load MEG"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):

        return {'PASS_THROUGH'}


class RefinefMRIClusters(bpy.types.Operator):
    bl_idname = "mmvt.refine_fmri_clusters"
    bl_label = "Calc clusters"
    bl_options = {"UNDO"}
    in_q, out_q = None, None
    _timer = None

    def modal(self, context, event):
        if event.type == 'TIMER':
            if not self.out_q is None:
                try:
                    fMRI_preproc = self.out_q.get(block=False)
                    print('fMRI_preproc: {}'.format(fMRI_preproc))
                except Empty:
                    pass
        return {'PASS_THROUGH'}

    def invoke(self, context, event=None):
        subject = mu.get_user()
        threshold = bpy.context.scene.fmri_clustering_threshold
        contrast = bpy.context.scene.fmri_clusters_labels_files
        atlas = bpy.context.scene.atlas
        task = contrast.split('_')[0]
        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.1, context.window)
        mu.change_fol_to_mmvt_root()
        cmd = '{} -m src.preproc.fMRI_preproc -s {} -T {} -c {} -t {} -a {} -f find_clusters'.format(
            bpy.context.scene.python_cmd, subject, task, contrast, threshold, atlas)
        print('Running {}'.format(cmd))
        self.in_q, self.out_q = mu.run_command_in_new_thread(cmd)
        return {'RUNNING_MODAL'}


class NearestCluster(bpy.types.Operator):
    bl_idname = "mmvt.nearest_cluster"
    bl_label = "Nearest Cluster"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        find_closest_cluster()
        return {'PASS_THROUGH'}


class PlotAllBlobs(bpy.types.Operator):
    bl_idname = "mmvt.plot_all_blobs"
    bl_label = "Plot all blobs"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        plot_all_blobs()
        return {'PASS_THROUGH'}


class FilterfMRIBlobs(bpy.types.Operator):
    bl_idname = "mmvt.filter_fmri_blobs"
    bl_label = "Filter fMRI blobs"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        update_clusters()
        return {'PASS_THROUGH'}


bpy.types.Scene.plot_current_cluster = bpy.props.BoolProperty(
    default=False, description="Plot current cluster")
bpy.types.Scene.plot_fmri_cluster_per_click = bpy.props.BoolProperty(
    default=False, description="Plot cluster per left click")
bpy.types.Scene.search_closest_cluster_only_in_filtered = bpy.props.BoolProperty(
    default=False, description="Plot current cluster")
bpy.types.Scene.fmri_what_to_plot = bpy.props.EnumProperty(
    items=[('blob', 'Plot blob', '', 1)], description='What do plot') # ('cluster', 'Plot cluster', '', 1)
bpy.types.Scene.fmri_how_to_sort = bpy.props.EnumProperty(
    items=[('tval', 't-val', '', 1), ('size', 'size', '', 2)],
    description='How to sort', update=fmri_how_to_sort_update)
bpy.types.Scene.fmri_cluster_val_threshold = bpy.props.FloatProperty(default=2,
    description='clusters t-val threshold', min=0, max=20, update=fmri_clusters_update)
bpy.types.Scene.fmri_cluster_size_threshold = bpy.props.FloatProperty(default=50,
    description='clusters size threshold', min=1, max=2000, update=fmri_clusters_update)
bpy.types.Scene.fmri_clustering_threshold = bpy.props.FloatProperty(default=2,
    description='clustering threshold', min=0, max=20)
bpy.types.Scene.fmri_clusters_labels_files = bpy.props.EnumProperty(
    items=[], description="fMRI files", update=fmri_clusters_labels_files_update)
bpy.types.Scene.fmri_blobs_norm_by_percentile = bpy.props.BoolProperty(default=True)
bpy.types.Scene.fmri_blobs_percentile_min = bpy.props.FloatProperty(
    default=1, min=0, max=100, update=fmri_blobs_percentile_min_update)
bpy.types.Scene.fmri_blobs_percentile_max = bpy.props.FloatProperty(
    default=99, min=0, max=100, update=fmri_blobs_percentile_max_update)


class fMRIPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "fMRI"
    addon = None
    python_bin = 'python'
    init = False
    clusters_labels = None
    cluster_labels = None
    clusters = []
    clusters_labels_filtered = []
    colors_in_hemis = {'rh':False, 'lh':False}
    blobs_activity = None
    blobs_plotted = False
    fMRI_clusters_files_exist = False
    constrast = {'rh':None, 'lh':None}
    clusters_labels_file_names = []

    def draw(self, context):
        if fMRIPanel.init:
            fMRI_draw(self, context)


def init(addon):
    user_fol = mu.get_user_fol()
    clusters_labels_files = glob.glob(op.join(user_fol, 'fmri', 'clusters_labels_*.pkl'))
    # old code was saving those files as npy instead of pkl
    clusters_labels_files.extend(glob.glob(op.join(user_fol, 'fmri', 'clusters_labels_*.npy')))
    # fmri_blobs = glob.glob(op.join(user_fol, 'fmri', 'blobs_*_rh.npy'))
    fMRIPanel.fMRI_clusters_files_exist = len(clusters_labels_files) > 0 # and len(fmri_blobs) > 0
    if not fMRIPanel.fMRI_clusters_files_exist:
        return None
    fMRIPanel.addon = addon
    fMRIPanel.lookup, fMRIPanel.clusters_labels = {}, {}
    fMRIPanel.cluster_labels = {}
    files_names = [mu.namebase(fname)[len('clusters_labels_'):] for fname in clusters_labels_files]
    fMRIPanel.clusters_labels_file_names = files_names
    clusters_labels_items = [(c, c, '', ind) for ind, c in enumerate(files_names)]
    bpy.types.Scene.fmri_clusters_labels_files = bpy.props.EnumProperty(
        items=clusters_labels_items, description="fMRI files", update=fmri_clusters_labels_files_update)
    bpy.context.scene.fmri_clusters_labels_files = files_names[0]
    for file_name, clusters_labels_file in zip(files_names, clusters_labels_files):
        # Check if the constrast files exist
        if mu.hemi_files_exists(op.join(user_fol, 'fmri', 'fmri_{}_{}.npy'.format(file_name, '{hemi}'))):
            fMRIPanel.clusters_labels[file_name] = np.load(clusters_labels_file)
            fMRIPanel.clusters_labels[file_name] = support_old_verions(fMRIPanel.clusters_labels[file_name])
            fMRIPanel.lookup[file_name] = create_lookup_table(fMRIPanel.clusters_labels[file_name])

    bpy.context.scene.fmri_cluster_val_threshold = 3
    bpy.context.scene.fmri_cluster_size_threshold = 50
    bpy.context.scene.search_closest_cluster_only_in_filtered = True
    bpy.context.scene.fmri_what_to_plot = 'blob'
    bpy.context.scene.fmri_how_to_sort = 'tval'

    update_clusters()
    fMRIPanel.blobs_activity, _ = calc_blobs_activity(
        fMRIPanel.constrast, fMRIPanel.clusters_labels_filtered, fMRIPanel.colors_in_hemis)
    bpy.context.scene.plot_fmri_cluster_per_click = False
    fMRIPanel.dont_show_clusters_info = True
    # addon.clear_cortex()
    register()
    fMRIPanel.init = True
    # print('fMRI panel initialization completed successfully!')


def create_lookup_table(clusters_labels):
    lookup = {}
    values = clusters_labels['values'] if 'values' in clusters_labels else clusters_labels
    for cluster_label in values:
        lookup[_cluster_name(cluster_label, 'tval')] = cluster_label
        lookup[_cluster_name(cluster_label, 'size')] = cluster_label
    return lookup


def register():
    try:
        unregister()
        bpy.utils.register_class(fMRIPanel)
        bpy.utils.register_class(NextCluster)
        bpy.utils.register_class(PrevCluster)
        bpy.utils.register_class(NearestCluster)
        bpy.utils.register_class(FilterfMRIBlobs)
        bpy.utils.register_class(PlotAllBlobs)
        bpy.utils.register_class(RefinefMRIClusters)
        bpy.utils.register_class(LoadMEGData)
        bpy.utils.register_class(FindfMRIFilesMinMax)
        bpy.utils.register_class(fmriClearColors)
        # print('fMRI Panel was registered!')
    except:
        print("Can't register fMRI Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(fMRIPanel)
        bpy.utils.unregister_class(NextCluster)
        bpy.utils.unregister_class(PrevCluster)
        bpy.utils.unregister_class(NearestCluster)
        bpy.utils.unregister_class(FilterfMRIBlobs)
        bpy.utils.unregister_class(PlotAllBlobs)
        bpy.utils.unregister_class(RefinefMRIClusters)
        bpy.utils.unregister_class(LoadMEGData)
        bpy.utils.unregister_class(FindfMRIFilesMinMax)
        bpy.utils.unregister_class(fmriClearColors)
    except:
        pass
        # print("Can't unregister fMRI Panel!")
