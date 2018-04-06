

def run(mmvt):
    # mmvt.coloring.plot_stc('stc_mag', view_selected=True, views=[mmvt.ROT_SAGITTAL_LEFT, mmvt.ROT_MEDIAL_LEFT],
    #                        cb_percentiles=[0, 99], hide_subcorticals=True, hide_right=True,
    #                        render_image=True, quality=40, light=0.7)
    mmvt.show_hide.hide_hemi('rh')
    mmvt.show_hide.show_hemi('lh')
    mmvt.show_hide.hide_subcorticals()
    mmvt.appearance.set_inflated_ratio(-0.5)
    mmvt.transparency.set_light_layers_depth(0)
    mmvt.transparency.set_brain_transparency(0)
    mmvt.render.set_background_color('white')
    mmvt.render.set_lighting(0.7)
    mmvt.coloring.set_lower_threshold(0)

    for stc_name in ['stc_mag', 'stc_grad', 'stc_eeg', 'stc_plot']:
        mmvt.coloring.plot_stc(stc_name, view_selected=True, cb_percentiles=(0, 99), cm='hot')
        mmvt.render.save_all_views((mmvt.ROT_SAGITTAL_LEFT, mmvt.ROT_MEDIAL_LEFT), render_images=True, quality=60,
                                   img_name_prefix=stc_name, add_colorbar=False)