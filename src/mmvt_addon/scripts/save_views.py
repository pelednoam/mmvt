import sys
import os
import os.path as op
import glob
import time
from itertools import product

try:
    from src.mmvt_addon.scripts import scripts_utils as su
except:
    # Add current folder the imports path
    sys.path.append(os.path.split(__file__)[0])
    import scripts_utils as su


def wrap_blender_call():
    args = read_args()
    # todo: check args
    if args.output_path == '':
        mmvt_dir = su.get_link_dir(su.get_links_dir(), 'mmvt')
        args.output_path = op.join(mmvt_dir, args.subject, 'figures')
    su.make_dir(args.output_path)
    log_fname = op.join(args.output_path, 'save_views.log')
    images_fname = op.join(args.output_path, 'images_names.txt')
    if op.isfile(log_fname):
        os.remove(log_fname)

    su.call_script(__file__, args, run_in_background=False)

    if args.add_cb or args.join_hemis:
        log_exist = op.isfile(log_fname)
        while not log_exist:
            log_exist = op.isfile(log_fname)
            time.sleep(.1)
        with open(images_fname, 'r') as text_file:
            images_names = text_file.readlines()
        images_names = [l.strip() for l in images_names]
        post_script(args, images_names)


def add_args():
    parser = su.add_default_args()
    parser.add_argument('--hemi', help='hemis ("rh", "lh", "rh,lh" or "both" (default))', required=False, default='both',
                        type=su.str_arr_type)
    parser.add_argument('--rot_lh_axial', help='Rotate lh to allign with rh in axial view (default False)',
        required=False, default=False, type=su.is_true)
    parser.add_argument('--views',
        help='views, any combination of 1-6 (sagittal_left/right, coronal_anterior/posterior, axial_superior/inferior',
        required=False, default='1,2,3,4,5,6', type=su.str_arr_type)
    parser.add_argument('--surf', help='surface ("pial", "inflated" or "pial,inflated")', required=False,
        default='pial,inflated', type=su.str_arr_type)
    parser.add_argument('--sub',
        help='When to show subcorticals (1:always, 2:never, 3:only when showing both hemis (default))',
        required=False, default=3, type=int)
    parser.add_argument('--join_hemis', help='Join hemis to one file (default False)', required=False, default=False,
        type=su.is_true)
    parser.add_argument('-o', '--output_path',
        help='Output path. The default is op.join(mmvt_dir, args.subject, "figures")', required=False, default='')
    parser.add_argument('--inflated', help='Infalted ratio (0-1) (default:1)', required=False, default=1.0, type=float)
    parser.add_argument('--inflated_ratio_in_file_name', help='', required=False, default=False, type=su.is_true)
    parser.add_argument('--add_cb', help='Add colorbar (default False)', required=False, default=False, type=su.is_true)
    parser.add_argument('--cm', help='Colormap (default RdOrYl)', required=False, default='RdOrYl')
    parser.add_argument('--cb_vals', help='Colorbar min and max (default "0,1")', required=False, default='0,1',
                        type=su.float_arr_type)
    parser.add_argument('--cb_ticks', help='Colorbar ticks (default None)', required=False, default=None,
                        type=su.float_arr_type)
    parser.add_argument('--background_color', help='Set the background color in solid mode (default black)',
                        required=False, default='0,0,0', type=su.float_arr_type)
    return parser


def read_args(argv=None):
    parser = add_args()
    return su.parse_args(parser, argv)


def save_views(subject_fname):
    args = read_args(su.get_python_argv())
    if args.debug:
        su.debug()
    mmvt = su.init_mmvt_addon()
    mmvt.set_render_output_path(args.output_path)
    views = [int(v)-1 for v in args.views]
    if args.sub == 1:
        mmvt.show_subcorticals()
    elif args.sub == 2:
        mmvt.hide_subcorticals()
    all_images_names = []
    args.render_images = False
    args.transparency = 0
    args.render_quality = 30
    if args.render_images:
        mmvt.set_brain_transparency(args.transparency)
    else:
        mmvt.change_background_color(args.background_color)
    for hemi, surface in product(args.hemi, args.surf):
        if hemi in su.HEMIS:
            su.get_hemi_obj(hemi).hide = False
            su.get_hemi_obj(su.other_hemi(hemi)).hide = True
            if args.sub == 3:
                mmvt.hide_subcorticals()
        if hemi == 'both':
            su.get_hemi_obj('rh').hide = False
            su.get_hemi_obj('lh').hide = False
            if args.sub == 3:
                mmvt.show_subcorticals()
        if surface == 'pial':
            mmvt.show_pial()
        elif surface == 'inflated':
            mmvt.show_inflated()
            mmvt.set_inflated_ratio(args.inflated - 1)
        images_names = mmvt.save_all_views(
            views=views, inflated_ratio_in_file_name=args.inflated_ratio_in_file_name, rot_lh_axial=args.rot_lh_axial,
            render_images=args.render_images, quality=args.render_quality)
        all_images_names.extend(images_names)
    with open(op.join(args.output_path, 'images_names.txt'), 'w') as text_file:
        for image_fname in all_images_names:
            print(image_fname, file=text_file)
    with open(op.join(args.output_path, 'save_views.log'), 'w') as text_file:
        print(args, file=text_file)
    su.exit_blender()


def post_script(args, images_names):
    from src.utils import figures_utils as fu
    from src.utils import utils

    print('post script')
    data_max, data_min = list(map(float, args.cb_vals))
    ticks = list(map(float, args.cb_ticks)) if args.cb_ticks is not None else None
    background = args.background_color # '#393939'
    if args.join_hemis:
        images_hemi_inv_list = set(
            [utils.namebase(fname)[3:] for fname in images_names if utils.namebase(fname)[:2] in ['rh', 'lh']])
        files = [[fname for fname in images_names if utils.namebase(fname)[3:] == img_hemi_inv] for img_hemi_inv in
                 images_hemi_inv_list]
        for files_coup in files:
            hemi = 'rh' if utils.namebase(files_coup[0]).startswith('rh') else 'lh'
            coup_template = files_coup[0].replace(hemi, '{hemi}')
            coup = {}
            for hemi in utils.HEMIS:
                coup[hemi] = coup_template.format(hemi=hemi)
            new_image_fname = op.join(utils.get_fname_folder(files_coup[0]),
                                      utils.namesbase_with_ext(files_coup[0])[3:])
            if args.add_cb:
                fu.crop_image(coup['lh'], coup['lh'], dx=150, dy=0, dw=150, dh=0)
                fu.crop_image(coup['rh'], coup['rh'], dx=150, dy=0, dw=0, dh=0)
                fu.combine_two_images(coup['lh'], coup['rh'], new_image_fname, facecolor=background)
                fu.combine_brain_with_color_bar(
                    data_max, data_min, new_image_fname, args.cm, dpi=200, overwrite=True, ticks=ticks,
                    w_fac=3, h_fac=3, ddh=0.7, dy=0.13, ddw=0.4, dx=-0.02)
            else:
                fu.crop_image(coup['lh'], coup['lh'], dx=150, dy=0, dw=150, dh=0)
                fu.crop_image(coup['rh'], coup['rh'], dx=150, dy=0, dw=150, dh=0)
                fu.combine_two_images(coup['lh'], coup['rh'], new_image_fname, facecolor=background)
            for hemi in utils.HEMIS:
                utils.remove_file(coup[hemi])
    elif args.add_cb and not args.join_hemis:
        for fig_name in images_names:
            fu.combine_brain_with_color_bar(
                data_max, data_min, fig_name, args.cm, dpi=100, overwrite=True, ticks=ticks,
                w_fac=1.2, h_fac=1.2, ddh=0.7, dy=0.13)


if __name__ == '__main__':
    import sys
    if op.isfile(sys.argv[0]) and sys.argv[0][-2:] == 'py':
        wrap_blender_call()
    else:
        # su.debug()
        save_views(sys.argv[1])

