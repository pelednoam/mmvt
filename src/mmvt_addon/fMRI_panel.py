import bpy
import os.path as op
import numpy as np
import mmvt_utils as mu


def clusters_update(self, context):
    _clusters_update()


def _clusters_update():
    if fMRIPanel.addon is None or not fMRIPanel.init:
        return


class NextCluster(bpy.types.Operator):
    bl_idname = 'ohad.next_cluster'
    bl_label = 'nextCluster'
    bl_options = {'UNDO'}

    def invoke(self, context, event=None):
        next_cluster()
        return {'FINISHED'}


def next_cluster():
    index = fMRIPanel.clusters.index(bpy.context.scene.clusters)
    next_cluster = fMRIPanel.clusters[index + 1] if index < len(fMRIPanel.clusters) - 1 else fMRIPanel.clusters[0]
    bpy.context.scene.clusters = next_cluster


class PrevCluster(bpy.types.Operator):
    bl_idname = 'ohad.prev_cluster'
    bl_label = 'prevcluster'
    bl_options = {'UNDO'}

    def invoke(self, context, event=None):
        prev_cluster()
        return {'FINISHED'}


def prev_cluster():
    index = fMRIPanel.clusters.index(bpy.context.scene.clusters)
    prev_cluster = fMRIPanel.clusters[index - 1] if index > 0 else fMRIPanel.clusters[-1]
    bpy.context.scene.clusters = prev_cluster


def fMRI_draw(self, context):
    layout = self.layout
    row = layout.row(align=True)
    row.operator(PrevCluster.bl_idname, text="", icon='PREV_KEYFRAME')
    row.prop(context.scene, "clusters", text="")
    row.operator(NextCluster.bl_idname, text="", icon='NEXT_KEYFRAME')
    box = layout.box()
    col = box.column()
    # for cluster_label in fMRIPanel.clusters_labels
    #     row = col.split(percentage=0.8, align=True)
    #     row.label(text=cortical_name)
    #     row.label(text='{:.2f}'.format(cortical_prob))



class fMRIPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Ohad"
    bl_label = "fMRI"
    addon = None

    def draw(self, context):
        fMRI_draw(self, context)


def cluster_name(x):
    return '{}_{}'.format(x['name'], len(x['vertices']))


def init(addon):
    fMRI_clusters_files_exist = mu.hemi_files_exists(
        op.join(mu.get_user_fol(), 'fmri_clusters_{hemi}.npy')) and \
        op.isfile(op.join(mu.get_user_fol(), 'fmri_cluster_labels.npy'))
    if not fMRI_clusters_files_exist:
        return None
    fMRIPanel.addon = addon
    fMRIPanel.clusters_labels = np.load(op.join(mu.get_user_fol(), 'fmri_cluster_labels.npy'))
    fMRIPanel.clusters = [cluster_name(x) for x in fMRIPanel.clusters_labels['rh']]
    fMRIPanel.clusters.extend([cluster_name(x) for x in fMRIPanel.clusters_labels['lh']])
    fMRIPanel.clusters.sort(key=mu.natural_keys)
    clusters_items = [(c, c, '', ind) for ind, c in enumerate(fMRIPanel.clusters)]
    bpy.types.Scene.clusters = bpy.props.EnumProperty(
        items=clusters_items, description="electrodes", update=clusters_update)
    bpy.context.scene.clusters = fMRIPanel.current_cluster = fMRIPanel.clusters[0]
    addon.clear_cortex()
    register()
    fMRIPanel.init = True
    print('fMRI panel initialization completed successfully!')


def create_lookup_table(clusters_labels):
    lookup = {}
    for hemi in mu.HEMIS:
        for cluster_label in clusters_labels[hemi]:
            lookup[cluster_label['name']] = cluster_label


def register():
    try:
        unregister()
        bpy.utils.register_class(fMRIPanel)
        bpy.utils.register_class(NextCluster)
        bpy.utils.register_class(PrevCluster)
        print('fMRI Panel was registered!')
    except:
        print("Can't register fMRI Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(fMRIPanel)
        bpy.utils.unregister_class(NextCluster)
        bpy.utils.unregister_class(PrevCluster)
    except:
        print("Can't unregister fMRI Panel!")
