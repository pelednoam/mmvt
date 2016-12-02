import os.path as op
from src.mmvt_addon.scripts.render_image import render_image
from src.utils import utils

LINKS_DIR = utils.get_links_dir()
MMVT_DIR = op.join(LINKS_DIR, 'mmvt')


def render_perspectives(subject, atlas, quality=20, inflated=False, set_inflated_ratio=0, background_color='black'):
    perpectives = ['lateral_lh', 'lateral_rh', 'medial_lh', 'medial_rh']
    output_fol = op.join(MMVT_DIR, subject, 'figures')
    render_image(subject, atlas, perpectives, output_fol, quality, inflated, set_inflated_ratio,
                 background_color, hide_subs=True, interactive=False)


if __name__ == '__main__':
    import argparse
    from src.utils import args_utils as au
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='aparc.DKTatlas40')
    parser.add_argument('-f', '--function', help='function name', required=False, default='render_perspectives')
    args = utils.Bag(au.parse_parser(parser))
    for subject in args.subject:
        locals()[args.function](subject, args.atlas)