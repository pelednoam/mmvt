import sys
import os
import os.path as op


try:
    from src.mmvt_addon.scripts import scripts_utils as su
except:
    # Add current folder the imports path
    sys.path.append(os.path.split(__file__)[0])
    import scripts_utils as su


def wrap_blender_call(args=None):
    if args is None:
        args = read_args()
    su.call_script(__file__, args, run_in_background=True, err_pipe=sys.stdin)


def read_args(argv=None):
    print('play_electrodes argv: {}'.format(argv))
    parser = su.add_default_args()
    parser = add_args(parser)
    return su.parse_args(parser, argv)


def add_args(parser):
    parser.add_argument('-q', '--quality', help='render quality', required=False, default=60, type=int)
    parser.add_argument('--play_from', help='from when to play', required=True, type=int)
    parser.add_argument('--play_to', help='until when to play', required=True, type=int)
    parser.add_argument('--output_fol_name', help='output_fol_name', required=False, default='electrodes')
    return parser


def wrap_mmvt_calls(subject_fname):
    args = read_args(su.get_python_argv())
    if args.debug:
        su.debug()
    mmvt = su.init_mmvt_addon()

    mmvt_calls(mmvt, args, subject_fname)
    su.exit_blender()


def mmvt_calls(mmvt, args, subject_fname):
    mu = mmvt.utils
    mmvt.data.set_bipolar = args.bipolar
    mmvt.coloring.set_lower_threshold(0)
    mmvt.colorbar.set_colorbar_max_min(1, -1)
    mmvt.colorbar.set_colormap('viridis-YlOrRd')
    mmvt.colorbar.lock_colorbar_values()
    mmvt.show_hide.show_hemis()
    mmvt.show_hide.hide_subcorticals()
    mmvt.appearance.show_electrodes()
    for elc_obj in mmvt.utils.children('Deep_electrodes'):
        mmvt.utils.show_hide_obj(elc_obj, elc_obj.animation_data is not None)
    mmvt.render.set_render_quality(args.quality)
    args.output_path = mu.make_dir(op.join(mu.get_user_fol(), 'movies', args.output_fol_name))
    mu.write_to_stderr('Writing to {}'.format(args.output_path))
    mmvt.set_render_output_path(args.output_path)
    mmvt.play.render_movie('elecs', args.play_from, args.play_to)


if __name__ == '__main__':
    import sys
    if op.isfile(sys.argv[0]) and sys.argv[0][-2:] == 'py':
        wrap_blender_call()
    else:
        wrap_mmvt_calls(sys.argv[1])
