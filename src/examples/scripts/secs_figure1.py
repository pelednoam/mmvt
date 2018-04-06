

def run(mmvt):
    # todo: save the orginal state, not only the background color
    org_background_color = mmvt.appearance.get_panels_background_color()
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
        # mmvt.render.save_all_views((mmvt.ROT_SAGITTAL_LEFT, mmvt.ROT_MEDIAL_LEFT), render_images=True, quality=60,
        #                            img_name_prefix=stc_name, add_colorbar=False)
        meg_data_min, meg_data_max = mmvt.coloring.get_meg_data_minmax()
        meg_data_min, meg_data_max= round(meg_data_min / 10.0) * 10, round(meg_data_max / 10.0) * 10
        mmvt.colorbar.save_colorbar(meg_data_min, meg_data_max, 'hot', ticks_num=3, ticks_font_size=16, prec=2,
                      title='', background_color_name='white', colorbar_name='{}_colobar'.format(stc_name))
    mmvt.appearance.set_panels_background_color(org_background_color)
