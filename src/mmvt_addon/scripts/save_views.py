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


def pre_blender_call(args):
    if args.output_path == '':
        mmvt_dir = su.get_link_dir(su.get_links_dir(), 'mmvt')
        args.output_path = op.join(mmvt_dir, args.subject, 'figures')
    su.make_dir(args.output_path)
    args.log_fname = op.join(args.output_path, 'mmvt_calls.log')
    args.images_log_fname = op.join(args.output_path, 'images_names.txt')
    if op.isfile(args.log_fname):
        os.remove(args.log_fname)
    return args


def wrap_blender_call():
    args = read_args()
    # todo: check args
    args = pre_blender_call(args)
    if args.call_mmvt_calls:
        su.call_script(__file__, args, run_in_background=False)
    post_blender_call(args)        


def add_args(parser):
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
    parser.add_argument('--img_name_prefix', help='imges name prefix (default:"")', required=False, default='',)
    parser.add_argument('--inflated', help='Infalted ratio (0-1) (default:1)', required=False, default=1.0, type=float)
    parser.add_argument('--inflated_ratio_in_file_name', help='', required=False, default=False, type=su.is_true)
    parser.add_argument('--add_cb', help='Add colorbar (default False)', required=False, default=False, type=su.is_true)
    parser.add_argument('--cb_cm', help='Colormap (default RdOrYl)', required=False, default='RdOrYl')
    parser.add_argument('--cb_vals', help='Colorbar min and max (default "0,1")', required=False, default='0,1',
                        type=su.float_arr_type)
    parser.add_argument('--cb_ticks', help='Colorbar ticks (default None)', required=False, default=None,
                        type=su.float_arr_type)
    parser.add_argument('--background_color', help='Set the background color in solid mode (default black)',
                        required=False, default='0,0,0', type=su.float_arr_type)
    parser.add_argument('--render_images', help='Render the images (default False)', required=False, default=False,
                        type=su.is_true)
    parser.add_argument('--render_quality', help='Render quality (default 50)', required=False, default=50, type=int)
    parser.add_argument('--transparency', help='Rendering transparency, 0-1 (default 0)', required=False, default=0, type=float)
    parser.add_argument('--log_fname', help='For inner usage', required=False, default='', type=str)
    parser.add_argument('--images_log_fname', help='For inner usage', required=False, default='', type=str)
    parser.add_argument('--call_mmvt_calls', help='Run the mmvt calls (default True)',
                        required=False, default=True, type=su.is_true)
    parser.add_argument('--remove_temp_figures', help='Delete temp figures (default True)',
                        required=False, default=True, type=su.is_true)
    parser.add_argument('--crop_figures', help='Crop figures (default True)',
                        required=False, default=True, type=su.is_true)
    parser.add_argument('--cb_ticks_font_size', help='Colorbar ticks font size (default 10)', required=False,
                        default=10, type=int)
    return parser


def read_args(argv=None):
    parser = su.add_default_args()
    parser = add_args(parser)
    return su.parse_args(parser, argv)


def wrap_mmvt_calls(subject_fname):
    args = read_args(su.get_python_argv())
    if args.debug:
        su.debug()
    mmvt = su.init_mmvt_addon()
    mmvt_calls(mmvt, args, subject_fname)
    su.exit_blender()


def mmvt_calls(mmvt, args, subject_fname):
    mmvt.set_render_output_path(args.output_path)
    views = [int(v)-1 for v in args.views]
    if args.sub == 1:
        mmvt.show_subcorticals()
    elif args.sub == 2:
        mmvt.hide_subcorticals()
    all_images_names = []
    # args.render_images = False
    # args.transparency = 0
    # args.render_quality = 30
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
            mmvt.show_hemis()
            if args.sub == 3:
                mmvt.show_subcorticals()
        if surface == 'pial':
            mmvt.show_pial()
        elif surface == 'inflated':
            mmvt.show_inflated()
            mmvt.set_inflated_ratio(args.inflated - 1)
        images_names = mmvt.save_all_views(
            views=views, inflated_ratio_in_file_name=args.inflated_ratio_in_file_name, rot_lh_axial=args.rot_lh_axial,
            render_images=args.render_images, quality=args.render_quality, img_name_prefix=args.img_name_prefix)
        all_images_names.extend(images_names)

    with open(args.images_log_fname, 'w') as text_file:
        for image_fname in all_images_names:
            print(image_fname, file=text_file)
    with open(args.log_fname, 'w') as text_file:
        print(args, file=text_file)


def post_blender_call(args):
    if not args.add_cb and not args.join_hemis:
        return
    
    from src.utils import figures_utils as fu
    from src.utils import utils
    from PIL import Image

    if args.call_mmvt_calls:
        su.waits_for_file(args.log_fname)
    with open(args.images_log_fname, 'r') as text_file:
        images_names = text_file.readlines()
    images_names = [l.strip() for l in images_names]
    data_max, data_min = list(map(float, args.cb_vals))
    ticks = list(map(float, args.cb_ticks)) if args.cb_ticks is not None else None
    background = args.background_color # '#393939'
    if args.join_hemis:
        images_hemi_inv_list = set(
            [utils.namebase(fname)[3:] for fname in images_names if utils.namebase(fname)[:2] in ['rh', 'lh']])
        files = [[fname for fname in images_names if utils.namebase(fname)[3:] == img_hemi_inv] for img_hemi_inv in
                 images_hemi_inv_list]
        fol = utils.get_fname_folder(files[0][0])
        cb_fname = op.join(fol, '{}_colorbar.jpg'.format(args.cb_cm))
        # if not op.isfile(cb_fname):
        fu.plot_color_bar(data_max, data_min, args.cb_cm, do_save=True, ticks=ticks, fol=fol, facecolor=background,
                          ticks_font_size=args.cb_ticks_font_size)
        cb_img = Image.open(cb_fname)
        for files_coup in files:
            hemi = 'rh' if utils.namebase(files_coup[0]).startswith('rh') else 'lh'
            coup_template = files_coup[0].replace(hemi, '{hemi}')
            coup = {}
            for hemi in utils.HEMIS:
                coup[hemi] = coup_template.format(hemi=hemi)
            new_image_fname = op.join(fol, utils.namebase_with_ext(files_coup[0])[3:])
            if args.add_cb:
                if args.crop_figures:
                    fu.crop_image(coup['lh'], coup['lh'], dx=150, dy=0, dw=50, dh=70)
                    fu.crop_image(coup['rh'], coup['rh'], dx=150+50, dy=0, dw=0, dh=70)
                fu.combine_two_images(coup['lh'], coup['rh'], new_image_fname, facecolor=background,
                                      dpi=200, w_fac=1, h_fac=1)
                fu.combine_brain_with_color_bar(new_image_fname, cb_img, overwrite=True)
            else:
                if args.crop_figures:
                    fu.crop_image(coup['lh'], coup['lh'], dx=150, dy=0, dw=150, dh=0)
                    fu.crop_image(coup['rh'], coup['rh'], dx=150, dy=0, dw=150, dh=0)
                fu.combine_two_images(coup['lh'], coup['rh'], new_image_fname, facecolor=background)
            if args.remove_temp_figures:
                for hemi in utils.HEMIS:
                    utils.remove_file(coup[hemi])
    elif args.add_cb and not args.join_hemis:
        fol = utils.get_fname_folder(images_names[0])
        cb_fname = op.join(fol, '{}_colorbar.jpg'.format(args.cb_cm))
        if not op.isfile(cb_fname):
            fu.plot_color_bar(data_max, data_min, args.cb_cm, do_save=True, ticks=ticks, fol=fol, facecolor=background)
        cb_img = Image.open(cb_fname)
        for fig_name in images_names:
            fu.combine_brain_with_color_bar(fig_name, cb_img, overwrite=True)
            # fu.combine_brain_with_color_bar(
            #     data_max, data_min, fig_name, args.cb_cm, dpi=100, overwrite=True, ticks=ticks,
            #     w_fac=1.2, h_fac=1.2, ddh=0.7, dy=0.13)


if __name__ == '__main__':
    import sys
    if op.isfile(sys.argv[0]) and sys.argv[0][-2:] == 'py':
        wrap_blender_call()
    else:
        wrap_mmvt_calls(sys.argv[1])

