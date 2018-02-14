from src.mmvt_addon.scripts import load_nii


def main(subject, nii_fname):
    # -s nmr01218 --nii "/cluster/neuromind/mhibert/clinical/temp_analysis/freesurfer/nmr01218/sycabs/sycabs_self_rh/words_v_symbols/z.nii.gz"
    # --use_abs_threshold 0 --cb_min_max 2,6 --save_views 1 --hemi rh,lh --views 1,2,6 --add_cb 1 --cb_vals 2,6 --cb_ticks 2,4,6  --rot_lh_axial 1 --join_hemis 1
    args = load_nii.read_args(dict(
        subject=subject,
        nii=nii_fname,
        use_abs_threshold=False,
        cb_min_max='2,6',
        save_views=True,
        hemi='rh,lh',
        views='1,2,6',
        add_cb=True,
        cb_vals='2,6',
        cb_ticks='2,4,6',
        rot_lh_axial=True,
        join_hemis=True
    ))
    load_nii.wrap_blender_call(args)


if __name__ == '__main__':
    import argparse
    from src.utils import args_utils as au
    from src.utils import utils

    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True)
    parser.add_argument('-n', '--nii', help='nii fname', required=True)
    args = utils.Bag(au.parse_parser(parser))
    main(args.subject, args.nii)

