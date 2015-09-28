import os
from surfer import Brain
import glob
import mne
import matplotlib.pyplot as plt
import matplotlib
import colorsys

subject_id = 'mg79' #"fsaverage"
hemi = "rh"
surf = "pial"
subjects_dir = os.environ.get('SUBJECTS_DIR', '/home/noam/subjects/mri')

# x=range(15)
# cm = plt.get_cmap('jet')
# cNorm = matplotlib.colors.Normalize(vmin=min(x) vmax=max(x))
# return cmx.ScalarMappable(norm=cNorm, cmap=cm)

def get_spaced_colors(n):
    HSV_tuples = [(x*1.0/n, 0.5, 0.5) for x in range(n)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return RGB_tuples

labels = glob.glob(os.path.join(subjects_dir,subject_id,'label', 'aparc250','*{}.label'.format(hemi)))
groups = set([l.split('/')[-1].split('_')[0] for l in labels])
brain = Brain(subject_id, hemi, surf, offscreen=True)
for group in groups:
    group_label = [l for l in labels if group in l]
    colors = get_spaced_colors(len(group_label))
    for label_id, label in enumerate(group_label):
        print(label)
        brain.add_label(label, color=colors[label_id])
    brain.save_imageset('{}-{}'.format(subject_id,group), ['med', 'lat', 'ros', 'caud'], 'jpg')
    break
brain.remove_labels()
print('sdf')