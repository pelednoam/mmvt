import bpy
import os.path as op


def run(mmvt):
    screen = bpy.data.screens['Neuro']
    perspectives = ['sagital', 'coronal', 'axial']
    images_names = ['{}.{}'.format(pres, mmvt.render.get_figure_format()) for pres in perspectives]
    ind = 0
    for area in screen.areas:
        if area.type == 'IMAGE_EDITOR':
            override = bpy.context.copy()
            override['area'] = area
            override['screen'] = screen
            image = bpy.data.images[images_names[ind]]
            area.spaces.active.image = image
            bpy.ops.image.save_as(override, filepath=op.join(mmvt.utils.get_user_fol(), 'figures', images_names[ind]))
            ind += 1