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
    # todo: check args
    su.call_script(__file__, args, run_in_background=False)


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
    return parser


def read_args(argv=None):
    parser = add_args()
    return su.parse_args(parser, argv)


def save_views(subject_fname):
    args = read_args(su.get_python_argv())
    if args.debug:
        su.debug()
    mmvt = su.init_mmvt_addon()
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
        mmvt.save_all_views(views=views)
    su.exit_blender()


if __name__ == '__main__':
    import sys
    if op.isfile(sys.argv[0]) and sys.argv[0][-2:] == 'py':
        wrap_blender_call()
    else:
        # su.debug()
        save_views(sys.argv[1])

