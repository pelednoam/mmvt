import bpy
import os.path as op


def run(mmvt):
    mu = mmvt.utils
    images_names = mmvt.slicer.get_slices_names()
    ind = 0
    for area in mu.get_images_areas():
        override = mu.get_image_area_override(area)
        image = bpy.data.images[images_names[ind]]
        area.spaces.active.image = image
        bpy.ops.image.save_as(override, filepath=op.join(mmvt.utils.get_user_fol(), 'figures', images_names[ind]))
        ind += 1