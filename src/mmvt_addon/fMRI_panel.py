import bpy
import os.path as op
import numpy as np
import mmvt_utils as mu
import glob
from queue import Empty

def clusters_update(self, context):
    _clusters_update()


def _clusters_update():
    if fMRIPanel.addon is None or not fMRIPanel.init:
        return
    clusters_labels_file = bpy.context.scene.fmri_clusters_labels_files
    fMRIPanel.cluster_labels = fMRIPanel.lookup[clusters_labels_file][bpy.context.scene.fmri_clusters]
    if bpy.context.scene.plot_current_cluster:
        faces_verts = fMRIPanel.addon.get_faces_verts()
        if bpy.context.scene.fmri_what_to_plot == 'blob':
            plot_blob(fMRIPanel.cluster_labels, faces_verts)


def plot_blob(cluster_labels, faces_verts):
    fMRIPanel.addon.init_activity_map_coloring('FMRI', subcorticals=False)
    blob_vertices = cluster_labels['vertices']
    hemi = cluster_labels['hemi']
    fMRIPanel.colors_in_hemis[hemi] = True
    activity = fMRIPanel.addon.get_fMRI_activity(hemi)
    blob_activity = np.ones((len(activity), 4))
    blob_activity[blob_vertices] = activity[blob_vertices]
    cur_obj = bpy.data.objects[hemi]
    fMRIPanel.addon.activity_map_obj_coloring(cur_obj, blob_activity, faces_verts[hemi], 0, True)
    other_hemi = mu.other_hemi(hemi)
    if fMRIPanel.colors_in_hemis[other_hemi]:
        fMRIPanel.addon.clear_cortex([other_hemi])
        fMRIPanel.colors_in_hemis[other_hemi] = False


# @mu.profileit()
def find_closest_cluster():
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
        closest_cluster = cluster_to_search_in[min_index]
        bpy.context.scene.fmri_clusters = cluster_name(closest_cluster)
        fMRIPanel.cluster_labels = closest_cluster
        print('Closest cluster: {}'.format(bpy.context.scene.fmri_clusters))
        _clusters_update()


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


def fmri_clusters_labels_files_update(self, context):
    if fMRIPanel.init:
        update_clusters()


def fmri_how_to_sort_update(self, context):
    if fMRIPanel.init:
        update_clusters()


def update_clusters(val_threshold=None, size_threshold=None):
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
    fMRIPanel.clusters_labels_filtered = [c for c in fMRIPanel.clusters_labels[clusters_labels_file]['values']
                           if abs(c['max']) >= val_threshold and len(c['vertices']) >= size_threshold]
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
    faces_verts = fMRIPanel.addon.get_faces_verts()
    fMRIPanel.addon.init_activity_map_coloring('FMRI', subcorticals=False)
    fmri_contrast, blobs_activity = {}, {}
    for hemi in mu.HEMIS:
        fmri_contrast[hemi] = fMRIPanel.addon.get_fMRI_activity(hemi)
        blobs_activity[hemi] = np.zeros((len(fmri_contrast[hemi]), 4))

    hemis = set()
    for cluster_labels in fMRIPanel.clusters_labels_filtered:
        if bpy.context.scene.fmri_what_to_plot == 'blob':
            blob_vertices = cluster_labels['vertices']
            hemi = cluster_labels['hemi']
            hemis.add(hemi)
            fMRIPanel.colors_in_hemis[hemi] = True
            blobs_activity[hemi][blob_vertices] = fmri_contrast[hemi][blob_vertices]

    for hemi in hemis:
        fMRIPanel.addon.activity_map_obj_coloring(
            bpy.data.objects[hemi],blobs_activity[hemi], faces_verts[hemi], 2, True)

    for hemi in set(mu.HEMIS) - hemis:
        fMRIPanel.addon.clear_cortex([hemi])


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
    # layout.prop(context.scene, 'fmri_what_to_plot', expand=True)
    row = layout.row(align=True)
    row.label(text='Sort: ')
    row.prop(context.scene, 'fmri_how_to_sort', expand=True)
    if not fMRIPanel.cluster_labels is None and len(fMRIPanel.cluster_labels) > 0:
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
    layout.operator(PlotAllBlobs.bl_idname, text="Plot all blobs", icon='POTATO')
    layout.operator(NearestCluster.bl_idname, text="Nearest cluster", icon='MOD_SKIN')
    layout.prop(context.scene, 'search_closest_cluster_only_in_filtered', text="Seach only in filtered blobs")
    layout.operator(LoadMEGData.bl_idname, text="Save as functional ROIs", icon='IPO')


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
bpy.types.Scene.search_closest_cluster_only_in_filtered = bpy.props.BoolProperty(
    default=False, description="Plot current cluster")
bpy.types.Scene.fmri_what_to_plot = bpy.props.EnumProperty(
    items=[('cluster', 'Plot cluster', '', 1), ('blob', 'Plot blob', '', 2)],
    description='What do plot')
bpy.types.Scene.fmri_how_to_sort = bpy.props.EnumProperty(
    items=[('tval', 't-val', '', 1), ('size', 'size', '', 2)],
    description='How to sort', update=fmri_how_to_sort_update)
bpy.types.Scene.fmri_cluster_val_threshold = bpy.props.FloatProperty(default=2,
    description='clusters t-val threshold', min=0, max=20, update=fmri_clusters_labels_files_update)
bpy.types.Scene.fmri_cluster_size_threshold = bpy.props.FloatProperty(default=50,
    description='clusters size threshold', min=1, max=2000, update=fmri_clusters_labels_files_update)
bpy.types.Scene.fmri_clustering_threshold = bpy.props.FloatProperty(default=2,
    description='clustering threshold', min=0, max=20)
bpy.types.Scene.fmri_clusters_labels_files = bpy.props.EnumProperty(
    items=[], description="fMRI files", update=fmri_clusters_labels_files_update)


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

    def draw(self, context):
        if fMRIPanel.init:
            fMRI_draw(self, context)


def init(addon):
    user_fol = mu.get_user_fol()
    clusters_labels_files = glob.glob(op.join(user_fol, 'fmri', 'clusters_labels_*.pkl'))
    # old code was saving those files as npy instead of pkl
    clusters_labels_files.extend(glob.glob(op.join(user_fol, 'fmri', 'clusters_labels_*.npy')))
    # fmri_blobs = glob.glob(op.join(user_fol, 'fmri', 'blobs_*_rh.npy'))
    fMRI_clusters_files_exist = len(clusters_labels_files) > 0 # and len(fmri_blobs) > 0
    if not fMRI_clusters_files_exist:
        return None
    fMRIPanel.addon = addon
    fMRIPanel.lookup, fMRIPanel.clusters_labels = {}, {}
    fMRIPanel.cluster_labels = {}
    files_names = [mu.namebase(fname)[len('clusters_labels_'):] for fname in clusters_labels_files]
    clusters_labels_items = [(c, c, '', ind) for ind, c in enumerate(files_names)]
    bpy.types.Scene.fmri_clusters_labels_files = bpy.props.EnumProperty(
        items=clusters_labels_items, description="fMRI files", update=fmri_clusters_labels_files_update)
    bpy.context.scene.fmri_clusters_labels_files = files_names[0]
    for file_name, clusters_labels_file in zip(files_names, clusters_labels_files):
        fMRIPanel.clusters_labels[file_name] = np.load(clusters_labels_file)
        fMRIPanel.clusters_labels[file_name] = support_old_verions(fMRIPanel.clusters_labels[file_name])
        fMRIPanel.lookup[file_name] = create_lookup_table(fMRIPanel.clusters_labels[file_name])

    bpy.context.scene.fmri_cluster_val_threshold = 3
    bpy.context.scene.fmri_cluster_size_threshold = 50
    bpy.context.scene.search_closest_cluster_only_in_filtered = True
    bpy.context.scene.fmri_what_to_plot = 'blob'
    bpy.context.scene.fmri_how_to_sort = 'tval'

    update_clusters()
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
    except:
        pass
        # print("Can't unregister fMRI Panel!")
