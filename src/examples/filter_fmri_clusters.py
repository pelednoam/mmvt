from src.preproc import fMRI

def filter_clusters(subject, atlas, filter_dic):
    # -s pp009 -a high.level.atlas -f fmri_pipeline_all
    fMRI.fmri_pipeline_all(subject, atlas, filter_dic=filter_dic, new_name='pp009-summary')

if __name__ == '__main__':
    subject = 'pp009'
    atlas = 'high.level.atlas'
    filter_dic = {'ECR-Interference':[{'name': 'dACC', 'hemi': 'lh', 'tval': -8.1},
                                      {'name': 'OFC', 'hemi': 'lh', 'tval': -5.44}],
                  'MSIT-Interference':[{'name': 'dmPFC', 'hemi': 'rh', 'tval': 3.10}],
                  'ARC-risk':[{'name': 'vlPFC', 'hemi': 'rh', 'tval': 5.22, 'new_name':'insula'}]}
    filter_clusters(subject, atlas, filter_dic)