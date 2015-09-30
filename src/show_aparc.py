import os
from surfer import Brain
import glob
import mne
import colorsys
from PIL import Image
from mayavi import mlab
mlab.options.backend = 'auto'
import utils

subject = 'hc004' # 'mg79' #"fsaverage"
hemi = "rh"
surf = "pial"
subjects_dir = utils.get_exisiting_dir(['/home/noam/subjects', '/home/noam/subjects/mri', '/homes/5/npeled/space3/subjects'])
aparc_name = 'laus250'

os.environ['SUBJECTS_DIR'] = subjects_dir
os.environ['SUBJECT'] = subject

def annotation_to_labels():
    fol = os.path.join(subjects_dir, subject, 'label', aparc_name)
    if not(os.path.isdir(fol)):
        os.mkdir(fol)
    labels = mne.read_labels_from_annot(subject, parc=aparc_name, hemi='both', surf_name='pial')
    for label in labels:
        label.save(os.path.join(fol, label.name))


def get_spaced_colors(n):
    HSV_tuples = [(x*1.0/n, 0.5, 0.5) for x in range(n)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return RGB_tuples

def get_groups():
    labels = glob.glob(os.path.join(subjects_dir, subject, 'label', aparc_name,'*{}.label'.format(hemi)))
    groups = list(set([l.split('/')[-1].split('_')[0] for l in labels]))
    return labels, groups

def get_group_labels(group):
    return glob.glob('{}-{}*'.format(subject,group))

def get_group_images(group):
    images = []
    for view in get_views():
        images.append(os.path.join(subjects_dir, subject, 'label', '{}_figures'.format(aparc_name), '{}_{}.jpg'.format(group,view)))
    return images

def get_views():
    return ['med', 'lat', 'ros', 'caud']

def save_groups_labels_figures():
    labels, groups = get_groups()
    for group in groups:
        print(group)
        if len(get_group_labels(group)) < 4:
            group_label = [l for l in labels if group in l]
            colors = get_spaced_colors(len(group_label))
        brain = Brain(subject, hemi, surf, offscreen=False)
        for label_id, label in enumerate(group_label):
            print(label)
            brain.add_label(label, color=colors[label_id])
        fol = os.path.join(subjects_dir, subject, 'label', '{}_figures'.format(aparc_name))
        if not os.path.isdir(fol):
            os.mkdir(fol)
        brain.save_imageset(os.path.join(fol, group), get_views(), 'jpg')
        brain.remove_labels()
        brain.close()

def combine_images_into_groups():
    labels, groups = get_groups()
    fol = os.path.join(subjects_dir, subject, 'label', '{}_groups_figures'.format(aparc_name))
    utils.make_dir(fol)
    for group in groups:
        group_im = Image.new('RGB', (800,800))
        group_images = get_group_images(group)
        for view_image_file, coo in zip(group_images, [(0, 0), (0, 400), (400, 0), (400,400)]):
            view_img = Image.open(view_image_file)
            view_img.thumbnail((400,400))
            group_im.paste(view_img, coo)
        group_im.save(os.path.join(fol, '{}-{}.jpg'.format(subject, group)))

if __name__ == '__main__':
    # annotation_to_labels()
    # save_groups_labels_figures()
    combine_images_into_groups()
    print('finish!')
