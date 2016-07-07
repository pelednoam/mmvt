from src.preproc import meg_preproc as meg


def read_epoches_and_calc_activity(subject, mri_subject):
    args = meg.read_cmd_args(['-s', subject, '-m', mri_subject])
    args.function = ['calc_stc_per_condition', 'calc_labels_avg_per_condition', 'smooth_stc', 'save_activity_map']
    args.pick_ori = 'normal'
    args.colors_map = 'jet'
    meg.main(subject, mri_subject, args)


if __name__ == '__main__':
    subject, mri_subject = 'ESZC25', 'KC'
    read_epoches_and_calc_activity(subject, mri_subject)