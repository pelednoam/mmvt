import sys
import os
import os.path as op
from itertools import product


try:
    from src.mmvt_addon.scripts import scripts_utils as su
except:
    # Add current folder the imports path
    sys.path.append(os.path.split(__file__)[0])
    import scripts_utils as su


def wrap_blender_call():
    args = read_args()
    should_render_figures = check_if_all_figures_were_rendered(args)
    if should_render_figures:
        su.call_script(__file__, args)
    post_script(args)


def check_if_all_figures_were_rendered(args):
    from src.utils import figures_utils as fu
    from src.mmvt_addon import fMRI_panel as fmri

    subject_fol = op.join(su.get_mmvt_dir(), args.subject)
    figures_fol = op.join(subject_fol, 'figures')
    clusters_file_names, _ = fmri.get_clusters_files(subject_fol)
    clusters_names = [f for f in clusters_file_names if args.clusters_type in f]
    render_figures = False
    all_figures = []
    for clusters_name, inflated, background_color in product(clusters_names, args.inflated, args.background_color):
        figures_fnames = fu.get_brain_perspectives_figures(
            figures_fol, inflated, background_color, clusters_name, args.inflated_ratio)
        all_figures.extend(figures_fnames)
        if any(not op.isfile(f) for f in figures_fnames):
            render_figures = True
            break
    return render_figures


def read_args(argv=None):
    parser = su.add_default_args()
    # Add more args here
    parser.add_argument('-q', '--quality', help='render quality', required=False, default=60, type=int)
    parser.add_argument('--inflated', required=False, default='0,1', type=su.bool_arr_type)
    parser.add_argument('--inflated_ratio', required=False, default=0.5, type=float)
    parser.add_argument('--background_color', required=False, default='black,white', type=su.str_arr_type)
    parser.add_argument('--lighting', required=False, default='1.0,0.7', type=su.float_arr_type)
    parser.add_argument('--transparency', required=False, default=0.0, type=float)
    parser.add_argument('--light_layers_depth', required=False, default=0, type=int)
    parser.add_argument('--rendering_in_the_background', required=False, default=0, type=su.is_true)
    parser.add_argument('--clusters_type', required=False, default='')
    parser.add_argument('--dpi', required=False, default=100, type=int)
    parser.add_argument('--overwrite', required=False, default=1, type=su.is_true)

    parser.add_argument('--colors_map', required=False, default='')
    parser.add_argument('--x_left_crop', required=False, default=400, type=float)
    parser.add_argument('--x_right_crop', required=False, default=400, type=float)
    parser.add_argument('--y_top_crop', required=False, default=0, type=float)
    parser.add_argument('--y_buttom_crop', required=False, default=0, type=float)
    parser.add_argument('--w_fac', required=False, default=2, type=float)
    parser.add_argument('--h_fac', required=False, default=3/2, type=float)

    return su.parse_args(parser, argv)


def run_script(subject_fname):
    args = read_args(su.get_python_argv())
    if args.debug:
        su.debug()
    mmvt = su.init_mmvt_addon()
    mmvt.find_fmri_files_min_max()
    clusters_names = [f for f in mmvt.get_clusters_file_names() if args.clusters_type in f]
    print('clusters_names: {}'.format(clusters_names))
    if len(args.lighting) == 1 and len(args.background_color) == 2:
        args.lighting = [args.lighting] * 2
    for clusters_file_name in clusters_names:
        for inflated, background_color in product(args.inflated, args.background_color):
            lighting = args.lighting[0] if background_color == 'black' else args.lighting[1]
            mmvt.init_rendering(inflated, args.inflated_ratio, args.transparency, args.light_layers_depth,
                                lighting, background_color, args.rendering_in_the_background)
            mmvt.load_fmri_cluster(clusters_file_name)
            mmvt.plot_all_blobs()
            mmvt.render_lateral_medial_split_brain(clusters_file_name, args.quality, args.overwrite)
    su.save_blend_file(subject_fname)
    su.exit_blender()


def post_script(args):
    from src.utils import figures_utils as fu
    from src.mmvt_addon import fMRI_panel as fmri
    from src.utils import utils

    subject_fol = op.join(su.get_mmvt_dir(), args.subject)
    figures_fol = op.join(subject_fol, 'figures')
    clusters_file_names, _ = fmri.get_clusters_files(subject_fol)
    clusters_names = [f for f in clusters_file_names if args.clusters_type in f]
    print('clusters_names: {}'.format(clusters_names))
    fmri_files_minmax_fname = op.join(subject_fol, 'fmri', 'fmri_files_minmax_cm.pkl')
    data_min, data_max, colors_map_name = utils.load(fmri_files_minmax_fname)
    for clusters_name, inflated, background_color in product(clusters_names, args.inflated, args.background_color):
        print('Combing figures for {}, inflated: {}, background color: {}'.format(
            clusters_name, inflated, background_color))
        perspectives_image_fname = fu.combine_four_brain_perspectives(
            figures_fol, inflated, args.dpi, background_color,
            clusters_name, args.inflated_ratio, True, args.overwrite)
        fu.combine_brain_with_color_bar(
            data_max, data_min, perspectives_image_fname, colors_map_name, args.overwrite, args.dpi,
            args.x_left_crop, args.x_right_crop, args.y_top_crop, args.y_buttom_crop,
            args.w_fac, args.h_fac, background_color)

if __name__ == '__main__':
    import sys
    subject_fname = sys.argv[1]
    if sys.argv[2] == '--background':
        run_script(subject_fname)
    else:
        wrap_blender_call()
