import utils
import os

preproc_sess = 'preproc-sess -s {subject} -fsd bold -stc up -surface {subject} lhrh -mni305 -fwhm 5 -per-run'
mkanalysis_sess = 'mkanalysis-sess -fsd bold -stc up -surface {subject} {hemi} -fwhm 5 -event-related -paradigm {par_file} -nconditions 2 -spmhrf 0 -TR {tr} -refeventdur 1 -nskip 4 -polyfit 2 -analysis {contrast_name}.sm05.{hemi}  -per-run -force'
mkanalysis_sess_mni = 'mkanalysis-sess -fsd bold -stc up -mni305 2 -fwhm 5 -event-related  -paradigm {par_file} -nconditions 2 -spmhrf 0 -TR {tr} -refeventdur 1 -nskip 4 -polyfit 2 -analysis {contrast_name}.sm05.mni305 -per-run -force'
mkcontrast_sess = 'mkcontrast-sess -analysis {contrast_name}.sm05.{hemi} -contrast {contrast} {constrast_flags}'
mkcontrast_sess_mni = 'mkcontrast-sess -analysis {contrast_name}.sm05.mni305 -contrast {contrast} {constrast_flags}'
selxavg3_sess = 'selxavg3-sess -s {subject} -analysis {contrast_name}.sm05.{hemi}'
selxavg3_sess_mni = 'selxavg3-sess -s {subject} -analysis {contrast_name}.sm05.mni305'
tksurfer_sess = 'tksurfer-sess -s {subject} -analysis {contrast_name}.sm05.{hemi} -c {contrasts_names[0]} -c {contrasts_names[1]} -c {contrasts_names[2]} -c {contrasts_names[3]}'
tksurfer_sess_mni = 'tksurfer-sess -s {subject} -analysis {contrast_name}.sm05.mni305 -c {contrasts_names[0]} -c {contrasts_names[1]} -c {contrasts_names[2]} -c {contrasts_names[3]}'

mni305_to_subject_reg = 'reg-mni305.2mm --s {subject} --reg mn305_to_{subject}.dat'
mni305_to_subject = 'mri_vol2vol --mov {mni305_sig_file} --reg mn305_to_{subject}.dat --o {subject_sig_file} --fstarg'


def run(subject, root_dir, par_file, contrast_name, contrasts, tr=2, run_preproc_sess=True, print_only=False):
    #todo: find the data TR instead of just guessing it's 2
    contrasts_names = contrasts.keys()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(root_dir)
    rs = utils.partial_run_script(locals(), print_only=print_only)
    # Start the actual calls
    if run_preproc_sess:
        rs(preproc_sess)
    for hemi in ['rh', 'lh']:
        rs(mkanalysis_sess, hemi=hemi)
        for contrast, constrast_flags in contrasts.iteritems():
            rs(mkcontrast_sess, hemi=hemi, contrast=contrast, constrast_flags=constrast_flags)
        rs(selxavg3_sess, hemi=hemi)
    rs(mkanalysis_sess_mni)
    for contrast, constrast_flags in contrasts.iteritems():
        rs(mkcontrast_sess_mni, contrast=contrast, constrast_flags=constrast_flags)
    rs(selxavg3_sess_mni)
    os.chdir(current_dir)

def plot_contrast(subject, root_dir, contrast_name, contrasts, hemi):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(root_dir)
    contrasts_names = contrasts.keys()
    rs = utils.partial_run_script(locals())
    rs(tksurfer_sess)
    os.chdir(current_dir)


def transform_mni_to_subject(subject, contrast_name, contrast, mn305_contrast_file_name='sig.mgz',
        subject_contrast_file_name='sig_subject.mgz', print_only=False):

    sig_fol = os.path.join('{}.sm05.mni305'.format(contrast_name), contrast)
    mni305_sig_file = os.path.join(sig_fol, mn305_contrast_file_name)
    subject_sig_file = os.path.join(sig_fol, subject_contrast_file_name)
    rs = utils.partial_run_script(locals(), print_only=print_only)
    rs(mni305_to_subject_reg)
    rs(mni305_to_subject)


if __name__ == '__main__':
    subject = 'ep001'
    root_dir = '/homes/5/npeled/space3/MSIT'
    contrast_name = 'interference'
    par_file = '014/msit.par'
    contrasts={'non-interference-v-base': '-a 1', 'interference-v-base': '-a 2', 'non-interference-v-interference': '-a 1 -c 2', 'task.avg-v-base': '-a 1 -a 2'}
    TR = 1.75

    run(subject, root_dir, par_file, contrast_name, contrasts, TR, run_preproc_sess=True)
    plot_contrast(subject, root_dir, contrast_name, contrasts, hemi='rh')