import bpy
import os.path as op
import numpy as np
import time

def run(mmvt):
    mu = mmvt.utils
    perspectives = mmvt.slicer.get_perspectives()
    for pers in perspectives:
        mu.make_dir(op.join(mmvt.utils.get_user_fol(), 'slices', pers))
    images_names = mmvt.slicer.get_slices_names()
    now = time.time()
    voxels_range = range(bpy.context.scene.save_slices_from_vox, bpy.context.scene.save_slices_to_vox)
    N = len(voxels_range)
    for ind, voxel_ind in enumerate(voxels_range):
        mu.time_to_go(now, ind, N, 10)
        pos = np.ones(3) * voxel_ind
        mmvt.where_am_i.create_slices(pos=pos, plot_cross=False, mark_voxel=False, pos_in_vox=True)
        save_slices(mmvt, voxel_ind, images_names, perspectives)


def save_slices(mmvt, voxel_ind, images_names, perspectives):
    mu = mmvt.utils
    current_images_names = mmvt.slicer.get_slices_names(voxel_ind)
    ind = 0
    for area in mu.get_images_areas():
        override = mu.get_image_area_override(area)
        image = bpy.data.images[images_names[ind]]
        area.spaces.active.image = image

        fname = op.join(mmvt.utils.get_user_fol(), 'slices', perspectives[ind], current_images_names[ind].lower())
        scene = bpy.context.scene
        scene.render.image_settings.file_format = 'JPEG'
        image.save_render(fname, scene)

        # bpy.ops.image.save_as(override, filepath=fname)
        ind += 1


bpy.types.Scene.save_slices_from_vox = bpy.props.IntProperty(default=0, min=0, max=256)
bpy.types.Scene.save_slices_to_vox = bpy.props.IntProperty(default=256, min=0, max=256)


def draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'save_slices_from_vox', text='from')
    layout.prop(context.scene, 'save_slices_to_vox', text='to')
