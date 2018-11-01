import os.path as op
from itertools import product
from src.mmvt_addon.scripts.render_image import render_image
from src.utils import utils

LINKS_DIR = utils.get_links_dir()
MMVT_DIR = op.join(LINKS_DIR, 'mmvt')


def render_perspectives(subject, atlas, quality=10, inflated=False, inflated_ratio=0, background_color='black',
                        lighting=1.0, **kargs):
    perpectives = ['lateral_lh', 'lateral_rh', 'medial_lh', 'medial_rh']
    output_fol = op.join(MMVT_DIR, subject, 'figures')
    render_image(subject, atlas, perpectives, output_fol, quality, inflated, inflated_ratio,
                 background_color, lighting, hide_subs=True, interactive=False)


def render_perspectives_product(subject, atlas, quality=10, inflated_ratio=0.5, **kargs):
    for inflated, background_color in product([True, False], ['black', 'white']):
        lighting = 1.0 if background_color == 'black' else 0.7
        render_perspectives(subject, atlas, quality, inflated, inflated_ratio, background_color, lighting)


# def combine_four_brain_perspectives():
#     data_min, data_max = _addon().get_colorbar_max_min()
#     background = bpy.context.scene.background_color
#     figure_name = 'splitted_lateral_medial_{}_{}.png'.format(
#         'inflated' if _addon().is_inflated() else 'pial', background)
#     figure_fname = op.join(mu.get_user_fol(), 'figures', figure_name)
#     colors_map = _addon().get_colormap().replace('-', '_')
#     x_left_crop, x_right_crop, y_top_crop, y_buttom_crop = (300, 300, 0, 0)
#     w_fac, h_fac = (1.5, 1)
#     cmd = '{} -m src.utils.figures_utils '.format(bpy.context.scene.python_cmd) + \
#         '-f combine_four_brain_perspectives,combine_brain_with_color_bar --fol {} --data_max {} --data_min {} '.format(
#         op.join(mu.get_user_fol(), 'figures'), data_max, data_min) + \
#         '--figure_fname {} --colors_map {} --x_left_crop {} --x_right_crop {} --y_top_crop {} --y_buttom_crop {} '.format(
#         figure_fname, colors_map, x_left_crop, x_right_crop, y_top_crop, y_buttom_crop) + \
#         '--w_fac {} --h_fac {} --facecolor {}'.format(w_fac, h_fac, background)
#     mu.run_command_in_new_thread(cmd, False)


if __name__ == '__main__':
    import argparse
    from src.utils import args_utils as au
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='aparc.DKTatlas')
    parser.add_argument('-q', '--quality', help='quality', required=False, default=20, type=int)
    parser.add_argument('-f', '--function', help='function name', required=False, default='render_perspectives')
    args = utils.Bag(au.parse_parser(parser))
    subjects = args.subject
    for subject in subjects:
        args.subject = subject
        locals()[args.function](**args)