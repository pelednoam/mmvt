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
    su.call_script(__file__, args)
    post_script(args)


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
    parser.add_argument('--overwrite', required=False, default=1, type=su.is_true)
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

    subject_fol = op.join(su.get_mmvt_dir(), args.subject)
    figures_fol = op.join(subject_fol, 'figures')
    clusters_file_names, _ = fmri.get_clusters_files(subject_fol)
    clusters_names = [f for f in clusters_file_names if args.clusters_type in f]
    print('clusters_names: {}'.format(clusters_names))
    for clusters_name, inflated, background_color in product(clusters_names, args.inflated, args.background_color):
        fu.combine_four_brain_perspectives(figures_fol, inflated=args.inflated, facecolor=background_color,
                                           clusters_name=clusters_name, inflated_ratio=args.inflated_ratio,
                                           overwrite=args.overwrite)

if __name__ == '__main__':
    import sys
    subject_fname = sys.argv[1]
    if sys.argv[2] == '--background':
        run_script(subject_fname)
    else:
        wrap_blender_call()
