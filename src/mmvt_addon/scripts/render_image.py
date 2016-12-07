import sys
import os
import os.path as op


try:
    from src.mmvt_addon.scripts import scripts_utils as su
except:
    # Add current folder the imports path
    sys.path.append(os.path.split(__file__)[0])
    import scripts_utils as su


def wrap_blender_call():
    args = read_args()
    su.call_script(__file__, args)


def render_image(subject, atlas, image_name, output_path, quality=60, inflated=False, inflated_ratio=1,
                 background_color='black', lighting=1.0, smooth_figure=False, hide_lh=False, hide_rh=False,
                 hide_subs=False, show_elecs=False, bipolar=False, show_only_lead=False, curr_elec='',
                 show_connections=False, interactive=True, blender_fol='', subjects=''):
    image_name = [image_name] if isinstance(image_name, str) else image_name
    camera = [op.join(su.get_mmvt_dir(), subject, 'camera', 'camera_{}{}.pkl'.format(
        camera_name, '_inf' if inflated else '')) for camera_name in image_name]
    image_name = ['{}_{}_{}'.format(name, 'inflated_{}'.format(inflated_ratio)if inflated else 'pial',
                                    background_color) for name in image_name]
    camera = ','.join(camera)
    image_name = ','.join(image_name)
    args = su.Bag(locals())
    su.call_script(__file__, args)


def read_args(argv=None):
    parser = su.add_default_args()
    parser.add_argument('-o', '--output_path', help='output path', required=False, default='')
    parser.add_argument('-i', '--image_name', help='image name', required=False, default='', type=su.str_arr_type)
    parser.add_argument('-q', '--quality', help='render quality', required=False, default=60, type=int)
    parser.add_argument('-c', '--camera', help='camera fname', required=False, default='', type=su.str_arr_type)
    parser.add_argument('--inflated', required=False, default=0, type=su.is_true)
    parser.add_argument('--inflated_ratio', required=False, default=1.0, type=float)
    parser.add_argument('--background_color', required=False, default='black')
    parser.add_argument('--lighting', required=False, default=1.0, type=float)
    parser.add_argument('--smooth_figure', help='smooth figure', required=False, default=False, type=su.is_true)
    parser.add_argument('--hide_lh', help='hide left hemi', required=False, default=None, type=su.is_true_or_none)
    parser.add_argument('--hide_rh', help='hide right hemi', required=False, default=None, type=su.is_true_or_none)
    parser.add_argument('--hide_subs', help='hide sub corticals', required=False, default=None, type=su.is_true_or_none)
    parser.add_argument('--show_elecs', help='show electrodes', required=False, default=None, type=su.is_true_or_none)
    parser.add_argument('--show_only_lead', help='show only current lead', required=False, default=None, type=su.is_true_or_none)
    parser.add_argument('--curr_elec', help='current electrode', required=False, default='')
    parser.add_argument('--show_connections', help='show connections', required=False, default=None, type=su.is_true_or_none)
    parser.add_argument('--interactive', required=False, default=1, type=su.is_true)
    parser.add_argument('--overwrite', required=False, default=1, type=su.is_true)
    args = su.parse_args(parser, argv)
    return args


def render_image_blender(subject_fname):
    args = read_args(su.get_python_argv())
    mmvt_dir = op.join(su.get_links_dir(), 'mmvt')
    subject = su.get_subject_name(subject_fname)
    if args.output_path == '':
        args.output_path = op.join(mmvt_dir, args.subject, 'figures')
    su.make_dir(args.output_path)
    mmvt = su.init_mmvt_addon()
    mmvt.set_render_quality(args.quality)
    mmvt.set_render_output_path(args.output_path)
    mmvt.set_render_smooth_figure(args.smooth_figure)
    if not args.hide_lh is None:
        mmvt.show_hide_hemi(args.hide_lh, 'lh')
    if not args.hide_rh is None:
        mmvt.show_hide_hemi(args.hide_rh, 'rh')
    if not args.hide_subs is None:
        mmvt.show_hide_sub_corticals(args.hide_subs)
    if not args.show_elecs is None:
        mmvt.show_electrodes(args.show_elecs)
    if not args.show_only_lead is None:
        print('Showing only the current lead: {}'.format(args.show_only_lead))
        mmvt.set_show_only_lead(args.show_only_lead)
    if args.curr_elec not in ['', 'None']:
        print('Setting current electrode to {}'.format(args.curr_elec))
        mmvt.set_current_electrode(args.curr_elec)
    if not args.show_connections is None:
        mmvt.show_hide_connections(args.show_connections)
    if args.inflated:
        mmvt.show_inflated()
        mmvt.set_inflated_ratio = args.inflated_ratio
    mmvt.set_background_color(args.background_color)
    mmvt.set_lighting(args.lighting)
    for image_name, camera in zip(args.image_name, args.camera):
        if camera == '':
            camera = op.join(mmvt_dir, subject, 'camera', 'camera.pkl')
        print('Camera fname: {}'.format(camera))
        if op.isfile(camera):
            mmvt.load_camera(camera)
        else:
            if args.interactive:
                cont = input('No camera file was detected in the output folder, continue?')
                if not su.is_true(cont):
                    continue
            else:
                su.stdout_print('No camera file was detected in the output folder!!!')
                continue
        # su.save_blend_file(subject_fname)
        mmvt.render_image(image_name, args.output_path, args.quality, args.smooth_figure, render_background=False,
                          overwrite=args.overwrite)
    su.stdout_print('*** finish rendering! ***')
    su.exit_blender()


if __name__ == '__main__':
    import sys
    subject_fname = sys.argv[1]
    if sys.argv[2] == '--background':
        render_image_blender(subject_fname)
    else:
        wrap_blender_call()
