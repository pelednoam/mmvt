import bpy
import os.path as op
import numpy as np
import mmvt_utils as mu
import glob

def clusters_update(self, context):
    _clusters_update()


def _clusters_update():
    if fMRIPanel.addon is None or not fMRIPanel.init:
        return
    clusters_labels_file = bpy.context.scene.fmri_clusters_labels_files
    fMRIPanel.cluster_labels = fMRIPanel.lookup[clusters_labels_file][bpy.context.scene.fmri_clusters]
    # prev_cluster = fMRIPanel.current_electrode

    if bpy.context.scene.plot_current_cluster:
        faces_verts = fMRIPanel.addon.get_faces_verts()
        if bpy.context.scene.fmri_what_to_plot == 'blob':
            plot_blob(fMRIPanel.cluster_labels, faces_verts)
    # other_hemi = 'lh' if fMRIPanel.cluster_labels['hemi'] == 'rh' else 'rh'


def plot_blob(cluster_labels, faces_verts):
    # todo: clear the cortex if the hemi flip
    # fMRIPanel.addon.clear_cortex()
    fMRIPanel.addon.init_activity_map_coloring('FMRI', subcorticals=False)
    blob_vertices = cluster_labels['vertices']
    hemi = cluster_labels['hemi']
    fMRIPanel.colors_in_hemis[hemi] = True
    activity = fMRIPanel.addon.get_fMRI_activity(hemi)
    blob_activity = np.zeros((len(activity), 4))
    blob_activity[blob_vertices] = activity[blob_vertices]
    cur_obj = bpy.data.objects[hemi]
    fMRIPanel.addon.activity_map_obj_coloring(cur_obj, blob_activity, faces_verts[hemi], 2, True)
    other_hemi = mu.other_hemi(hemi)
    if fMRIPanel.colors_in_hemis[other_hemi]:
        fMRIPanel.addon.clear_cortex([other_hemi])
        fMRIPanel.colors_in_hemis[other_hemi] = False


# @mu.profileit()
def find_closest_cluster():
    cursor = np.array(bpy.context.scene.cursor_location) * 10
    # hemis_objs = [bpy.data.objects[hemi_obj] for hemi_obj in ['rh', 'lh']]
    # dists, indices = [], []
    # for hemi_obj, hemi in zip(hemis_objs, mu.HEMIS):
    #     _, _, dist = mu.min_cdist_from_obj(hemi_obj, [cursor])[0]
    #     dists.append(dist)
    # min_index = np.argmin(np.array(dists))
    # closest_hemi = mu.HEMIS[min_index]
    # print('closest hemi: {}'.format(closest_hemi))

    if bpy.context.scene.search_closest_cluster_only_in_filtered:
        cluster_to_search_in = fMRIPanel.clusters_labels_filtered
    else:
        clusters_labels_file = bpy.context.scene.fmri_clusters_labels_files
        cluster_to_search_in = fMRIPanel.clusters_labels[clusters_labels_file]
        unfilter_clusters()
    dists, indices = [], []
    # for ind, cluster in enumerate(fMRIPanel.clusters_labels[clusters_labels_file]):
    for ind, cluster in enumerate(cluster_to_search_in):
        # if cluster['hemi'] != closest_hemi:
        #     continue
        # co_find = cursor * hemi_obj.matrix_world.inverted()
        # clusters_hemi = fMRIPanel.clusters_labels['rh']
        _, _, dist = mu.min_cdist(cluster['coordinates'], [cursor])[0]
        dists.append(dist)
    min_index = np.argmin(np.array(dists))
    closest_cluster = cluster_to_search_in[min_index]
    bpy.context.scene.fmri_clusters = cluster_name(closest_cluster)
    fMRIPanel.cluster_labels = closest_cluster
    print('Closest cluster: {}'.format(bpy.context.scene.fmri_clusters))
    _clusters_update()


class NextCluster(bpy.types.Operator):
    bl_idname = 'ohad.next_cluster'
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
    bl_idname = 'ohad.prev_cluster'
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


def update_clusters(val_threshold=None, size_threshold=None):
    if val_threshold is None:
        val_threshold = bpy.context.scene.fmri_cluster_val_threshold
    if size_threshold is None:
        size_threshold = bpy.context.scene.fmri_cluster_size_threshold
    clusters_labels_file = bpy.context.scene.fmri_clusters_labels_files
    fMRIPanel.clusters_labels_filtered = [c for c in fMRIPanel.clusters_labels[clusters_labels_file]
                           if abs(c['max']) >= val_threshold and len(c['vertices']) >= size_threshold]
    clusters_tup = sorted([(abs(x['max']), cluster_name(x)) for x in fMRIPanel.clusters_labels_filtered])[::-1]
    fMRIPanel.clusters = [x_name for x_size, x_name in clusters_tup]
    # fMRIPanel.clusters.sort(key=mu.natural_keys)
    clusters_items = [(c, c, '', ind) for ind, c in enumerate(fMRIPanel.clusters)]
    bpy.types.Scene.fmri_clusters = bpy.props.EnumProperty(
        items=clusters_items, description="fmri clusters", update=clusters_update)
    if len(fMRIPanel.clusters) > 0:
        bpy.context.scene.fmri_clusters = fMRIPanel.current_cluster = fMRIPanel.clusters[0]
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


def fMRI_draw(self, context):
    layout = self.layout
    user_fol = mu.get_user_fol()
    clusters_labels_files = glob.glob(op.join(user_fol, 'fmri', 'clusters_labels_*.npy'))
    if len(clusters_labels_files) > 1:
        layout.prop(context.scene, 'fmri_clusters_labels_files', text='')
    layout.prop(context.scene, 'fmri_cluster_val_threshold', text='clusters t-val threshold')
    layout.prop(context.scene, 'fmri_cluster_size_threshold', text='clusters size threshold')
    layout.operator(FilterfMRIBlobs.bl_idname, text="Filter blobs", icon='FILTER')
    row = layout.row(align=True)
    row.operator(PrevCluster.bl_idname, text="", icon='PREV_KEYFRAME')
    row.prop(context.scene, 'fmri_clusters', text="")
    row.operator(NextCluster.bl_idname, text="", icon='NEXT_KEYFRAME')
    layout.prop(context.scene, 'plot_current_cluster', text="Plot current cluster")
    # layout.prop(context.scene, 'fmri_what_to_plot', expand=True)
    if not fMRIPanel.cluster_labels is None:
        blob_size = len(fMRIPanel.cluster_labels['vertices'])
        col = layout.box().column()
        mu.add_box_line(col, 'Max val', '{:.2f}'.format(fMRIPanel.cluster_labels['max']), 0.8)
        mu.add_box_line(col, 'Size', str(blob_size), 0.8)
        col = layout.box().column()
        for inter_labels in fMRIPanel.cluster_labels['intersects']:
            mu.add_box_line(col, inter_labels['name'], '{:.0%}'.format(inter_labels['num'] / float(blob_size)), 0.8)
    # row = layout.row(align=True)
    layout.operator(PlotAllBlobs.bl_idname, text="Plot all blobs", icon='POTATO')
    layout.operator(NearestCluster.bl_idname, text="Nearest cluster", icon='MOD_SKIN')
    layout.prop(context.scene, 'search_closest_cluster_only_in_filtered', text="Seach only in filtered blobs")


class NearestCluster(bpy.types.Operator):
    bl_idname = "ohad.nearest_cluster"
    bl_label = "Nearest Cluster"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        find_closest_cluster()
        return {'PASS_THROUGH'}


class PlotAllBlobs(bpy.types.Operator):
    bl_idname = "ohad.plot_all_blobs"
    bl_label = "Plot all blobs"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        plot_all_blobs()
        return {'PASS_THROUGH'}


class FilterfMRIBlobs(bpy.types.Operator):
    bl_idname = "ohad.filter_fmri_blobs"
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
bpy.types.Scene.fmri_cluster_val_threshold = bpy.props.FloatProperty(default=2,
    description='clusters t-val threshold', min=2, max=20, update=fmri_clusters_labels_files_update)
bpy.types.Scene.fmri_cluster_size_threshold = bpy.props.FloatProperty(default=50,
    description='clusters size threshold', min=1, max=2000, update=fmri_clusters_labels_files_update)


class fMRIPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
    bl_label = "fMRI"
    addon = None
    init = False
    clusters_labels = None
    cluster_labels = None
    clusters = []
    clusters_labels_filtered = []
    colors_in_hemis = {'rh':False, 'lh':False}

    def draw(self, context):
        if fMRIPanel.init:
            fMRI_draw(self, context)


def cluster_name(x):
    return '{}_{:.2f}'.format(x['name'], x['max'])


def init(addon):
    user_fol = mu.get_user_fol()
    clusters_labels_files = glob.glob(op.join(user_fol, 'fmri', 'clusters_labels_*.npy'))
    fmri_blobs = glob.glob(op.join(user_fol, 'fmri', 'blobs_*_rh.npy'))
    fMRI_clusters_files_exist = len(clusters_labels_files) > 0 and len(fmri_blobs) > 0
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
        fMRIPanel.lookup[file_name] = create_lookup_table(fMRIPanel.clusters_labels[file_name])

    bpy.context.scene.fmri_cluster_val_threshold = 3
    bpy.context.scene.fmri_cluster_size_threshold = 50
    bpy.context.scene.search_closest_cluster_only_in_filtered = True
    bpy.context.scene.fmri_what_to_plot = 'blob'

    update_clusters()
    # addon.clear_cortex()
    register()
    fMRIPanel.init = True
    print('fMRI panel initialization completed successfully!')


def create_lookup_table(clusters_labels):
    lookup = {}
    for cluster_label in clusters_labels:
            lookup[cluster_name(cluster_label)] = cluster_label
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
        print('fMRI Panel was registered!')
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
    except:
        pass
        # print("Can't unregister fMRI Panel!")
