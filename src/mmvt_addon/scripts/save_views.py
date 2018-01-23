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
    if  op.isfile(log_fname):
        os.remove(log_fname)
    su.call_script(__file__, args, run_in_background=False)

    if args.add_cb:
        log_exist = op.isfile(log_fname)
        while not log_exist:
            time.sleep(.1)
            log_exist = op.isfile(log_fname)
        post_script(args)


def add_args():
    parser = su.add_default_args()
    parser.add_argument('--hemi', help='hemis ("rh", "lh", "rh,lh" or "both" (default))', required=False, default='both',
                        type=su.str_arr_type)
    parser.add_argument('--views',
        help='views, any combination of 1-6 (sagittal_left/right, coronal_anterior/posterior, axial_superior/inferior',
        required=False, default='1,2,3,4,5,6', type=su.str_arr_type)
    parser.add_argument('--surf', help='surface ("pial", "inflated" or "pial,inflated")', required=False,
        default='pial,inflated', type=su.str_arr_type)
    parser.add_argument('--sub',
        help='When to show subcorticals (1:always, 2:never, 3:only when showing both hemis (default))',
        required=False, default=3, type=int)
    parser.add_argument('-o', '--output_path',
        help='Output path. The default is op.join(mmvt_dir, args.subject, "figures")', required=False, default='')
    parser.add_argument('--inflated', help='Infalted ratio (0-1) (default:1)', required=False, default=1.0, type=float)
    parser.add_argument('--inflated_ratio_in_file_name', help='', required=False, default=False, type=su.is_true)
    parser.add_argument('--add_cb', help='Add colormap (default False)', required=False, default=False, type=su.is_true)
    parser.add_argument('--cb', help='Colormap (default RdOrYl)', required=False, default='RdOrYl')
    parser.add_argument('--cb_vals', help='Colormap min and max (default "0,1")', required=False, default='0,1',
                        type=su.float_arr_type)
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
        mmvt.save_all_views(views=views, inflated_ratio_in_file_name=args.inflated_ratio_in_file_name)
    with open(op.join(args.output_path, 'save_views.log'), 'w') as text_file:
        print(args, file=text_file)
    su.exit_blender()


def post_script(args):
    from src.utils import figures_utils as fu

    print('Adding colorbar')
    data_max, data_min = list(map(float, args.cb_vals))
    for fig_name in glob.glob(op.join(args.output_path, '*.png')):
        fu.combine_brain_with_color_bar(
            data_max, data_min, fig_name, args.cb, dpi=100, overwrite=True, w_fac=1.2, h_fac=1.2, ddh=0.7, dy=0.13)


if __name__ == '__main__':
    import sys
    if op.isfile(sys.argv[0]) and sys.argv[0][-2:] == 'py':
        wrap_blender_call()
    else:
        # su.debug()
        save_views(sys.argv[1])

