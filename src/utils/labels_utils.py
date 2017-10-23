import mne.surface
import time
import os.path as op
import numpy as np
import os
import shutil
import glob
import traceback
from collections import defaultdict
import functools

from src.mmvt_addon import mmvt_utils as mu
read_labels_from_annots = mu.read_labels_from_annots
read_labels_from_annot = mu.read_labels_from_annot
Label = mu.Label

try:
    from src.utils import utils
    from src.utils import preproc_utils as pu
    from src.mmvt_addon import colors_utils as cu
    SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()
except:
    print("Sorry, no src libs...")

HEMIS = ['rh', 'lh']


def find_template_brain_with_annot_file(aparc_name, fsaverage, subjects_dir):
    fs_found = False
    if isinstance(fsaverage, str):
        fsaverage = [fsaverage]
    for fsav in fsaverage:
        fsaverage_annot_files_exist = utils.both_hemi_files_exist(op.join(
            subjects_dir, fsav, 'label', '{}.{}.annot'.format('{hemi}', aparc_name)))
        fsaverage_labels_exist = len(glob.glob(op.join(subjects_dir, fsav, 'label', aparc_name, '*.label'))) > 0
        if fsaverage_annot_files_exist or fsaverage_labels_exist:
            fsaverage = fsav
            fs_found = True
            break
    if not fs_found:
        print("Can't find the annot file for any of the templates brain!")
        return ''
    else:
        return fsaverage


def load_surf(subject, mmvt_dir, subjects_dir):
    verts = {}
    for hemi in HEMIS:
        if op.isfile(op.join(mmvt_dir, subject, 'surf', '{}.pial.npz'.format(hemi))):
            hemi_verts, _ = utils.read_pial(subject, mmvt_dir, hemi)
        elif op.isfile(op.join(subjects_dir, subject, 'surf', '{}.pial.ply'.format(hemi))):
            hemis_verts, _ = utils.read_ply_file(
                op.join(subjects_dir, subject, 'surf', '{}.pial.ply'.format(hemi)))
        else:
            print("Can't find {} pial ply/npz files!".format(hemi))
            return False
        verts[hemi] = hemi_verts
    return verts


def morph_labels_from_fsaverage(subject, subjects_dir, mmvt_dir, aparc_name='aparc250', fs_labels_fol='',
            sub_labels_fol='', n_jobs=6, fsaverage='fsaverage', overwrite=False):
    fsaverage = find_template_brain_with_annot_file(aparc_name, fsaverage, subjects_dir)
    if fsaverage == '':
        return False
    if subject == fsaverage:
        return True
    subject_dir = op.join(subjects_dir, subject)
    labels_fol = op.join(subjects_dir, fsaverage, 'label', aparc_name) if fs_labels_fol=='' else fs_labels_fol
    sub_labels_fol = op.join(subject_dir, 'label', aparc_name) if sub_labels_fol=='' else sub_labels_fol
    if not op.isdir(sub_labels_fol):
        os.makedirs(sub_labels_fol)
    labels = read_labels(fsaverage, subjects_dir, aparc_name, n_jobs=n_jobs)
    verts = load_surf(subject, mmvt_dir, subjects_dir)
    for fs_label in labels:
        label_file = op.join(labels_fol, '{}.label'.format(fs_label.name))
        local_label_name = op.join(sub_labels_fol, '{}.label'.format(op.splitext(op.split(label_file)[1])[0]))
        if not op.isfile(local_label_name) or overwrite:
            # fs_label = mne.read_label(label_file)
            fs_label.values.fill(1.0)
            sub_label = fs_label.morph(fsaverage, subject, grade=None, n_jobs=1, subjects_dir=subjects_dir)
            if np.all(sub_label.pos == 0):
                sub_label.pos = verts[sub_label.hemi][sub_label.vertices]
            sub_label.save(local_label_name)
    return True


def labels_to_annot(subject, subjects_dir='', aparc_name='aparc250', labels_fol='', overwrite=True, labels=[],
                    fix_unknown=True):
    from src.utils import labels_utils as lu
    if subjects_dir == '':
        subjects_dir = os.environ['SUBJECTS_DIR']
    subject_dir = op.join(subjects_dir, subject)
    if utils.both_hemi_files_exist(op.join(subject_dir, 'label', '{}.{}.annot'.format('{hemi}', aparc_name))) \
            and not overwrite:
        return True
    if len(labels) == 0:
        labels_fol = op.join(subject_dir, 'label', aparc_name) if labels_fol=='' else labels_fol
        labels = []
        labels_files = glob.glob(op.join(labels_fol, '*.label'))
        if len(labels_files) == 0:
            raise Exception('labels_to_annot: No labels files!')
        for label_file in labels_files:
            if fix_unknown and 'unknown' in utils.namebase(label_file):
                continue
            label = mne.read_label(label_file)
            # print(label.name)
            label.name = lu.get_label_hemi_invariant_name(label.name)
            labels.append(label)
        labels.sort(key=lambda l: l.name)
    if overwrite:
        for hemi in HEMIS:
            utils.remove_file(op.join(subject_dir, 'label', '{}.{}.annot'.format(hemi, aparc_name)))
    try:
        mne.write_labels_to_annot(subject=subject, labels=labels, parc=aparc_name, overwrite=overwrite,
                                  subjects_dir=subjects_dir)
    except:
        print('Error in writing annot file!')
        return False
    return utils.both_hemi_files_exist(op.join(subject_dir, 'label', '{}.{}.annot'.format('{hemi}', aparc_name)))


def solve_labels_collision(subject, subjects_dir, atlas, backup_atlas, surf_type='inflated', n_jobs=1):
    # now = time.time()
    # print('Read labels')
    # utils.read_labels_parallel(subject, subjects_dir, atlas, labels_fol='', n_jobs=n_jobs)
    # labels = read_labels(subject, subjects_dir, atlas, n_jobs=n_jobs)

    backup_labels_fol = op.join(subjects_dir, subject, 'label', backup_atlas)
    labels_fol = op.join(subjects_dir, subject, 'label', atlas)
    if op.isdir(backup_labels_fol):
        shutil.rmtree(backup_labels_fol)
    shutil.copytree(labels_fol, backup_labels_fol)

    # os.rename(labels_fol, backup_labels_fol)
    # utils.make_dir(labels_fol)
    save_labels_from_vertices_lookup(subject, atlas, subjects_dir, surf_type='pial', read_labels_from_fol=backup_labels_fol)
    return

    # hemis_verts, labels_hemi, pia_verts = {}, {}, {}
    # print('Read surface ({:.2f}s)'.format(time.time() - now))
    # for hemi in HEMIS:
    #     surf_fname = op.join(subjects_dir, subject, 'surf', '{}.{}'.format(hemi, surf_type))
    #     hemis_verts[hemi], _ = mne.surface.read_surface(surf_fname)
    #     labels_hemi[hemi] = [l for l in labels if l.hemi == hemi]
    # print('Calc centroids ({:.2f}s)'.format(time.time() - now))
    # centroids = calc_labels_centroids(labels_hemi, hemis_verts)
    # for hemi in HEMIS:
    #     print('Calc vertices labeling for {} ({:.2f}s)'.format(hemi, time.time() - now))
    #     hemi_centroids_dist = cdist(hemis_verts[hemi], centroids[hemi])
    #     vertices_labels_indices = np.argmin(hemi_centroids_dist, axis=1)
    #     labels_hemi_chunks = utils.chunks(list(enumerate(labels_hemi[hemi])), len(labels_hemi[hemi]) / n_jobs)
    #     params = [(labels_hemi_chunk, atlas, vertices_labels_indices, hemis_verts, labels_fol) for labels_hemi_chunk in labels_hemi_chunks]
    #     print('Save labels for {} ({:.2f}s)'.format(hemi, time.time() - now))
    #     utils.run_parallel(_save_new_labels_parallel, params, n_jobs)


# def _save_new_labels_parallel(params_chunk):
#     labels_hemi_chunk, atlas, vertices_labels_indices, hemis_verts, labels_fol = params_chunk
#     for ind, label in labels_hemi_chunk:
#         vertices = np.where(vertices_labels_indices == ind)[0]
#         pos = hemis_verts[label.hemi][vertices]
#         new_label = mne.Label(vertices, pos, hemi=label.hemi, name=label.name, filename=None,
#             subject=label.subject, color=label.color, verbose=None)
#         if not op.isfile(op.join(labels_fol, new_label.name)):
#             new_label.save(op.join(labels_fol, new_label.name))

def create_unknown_labels(subject, atlas):
    labels_fol = op.join(SUBJECTS_DIR, subject, 'label', atlas)
    utils.make_dir(labels_fol)
    unknown_labels_fname_template = op.join(labels_fol,  'unknown-{}.label'.format('{hemi}'))
    if utils.both_hemi_files_exist(unknown_labels_fname_template):
        unknown_labels = {hemi:mne.read_label(unknown_labels_fname_template.format(hemi=hemi), subject)
                          for hemi in utils.HEMIS}
        return unknown_labels

    unknown_labels = {}
    for hemi in utils.HEMIS:
        labels = read_labels(subject, SUBJECTS_DIR, atlas, hemi=hemi)
        unknown_label_name = 'unknown-{}'.format(hemi)
        labels_names = [l.name for l in labels]
        if unknown_label_name not in labels_names:
            verts, _ = utils.read_pial(subject, MMVT_DIR, hemi)
            unknown_verts = set(range(verts.shape[0]))
            for label in labels:
                unknown_verts -= set(label.vertices)
            unknown_verts = np.array(sorted(list(unknown_verts)))
            unknown_label = mne.Label(unknown_verts, hemi=hemi, name=unknown_label_name, subject=subject)
        else:
            unknown_label = labels[labels_names.index(unknown_label_name)]
        unknown_labels[hemi] = unknown_label
        if not op.isfile(unknown_labels_fname_template.format(hemi=hemi)):
            unknown_label.save(unknown_labels_fname_template.format(hemi=hemi))
    return unknown_labels


def fix_unknown_labels(subject, atlas):
    for hemi in utils.HEMIS:
        labels = read_labels(subject, SUBJECTS_DIR, atlas, hemi=hemi)
        labels_names = [l.name for l in labels]
        while 'unknown-{}'.format(hemi) in labels_names:
            del labels[labels_names.index('unknown-{}'.format(hemi))]
            labels_names = [l.name for l in labels]
        mne.write_labels_to_annot(
                labels, subject=subject, parc=atlas, overwrite=True, subjects_dir=SUBJECTS_DIR, hemi=hemi)
    return utils.both_hemi_files_exist(op.join(SUBJECTS_DIR, subject, 'label', '{}-{}.annot'.format(atlas, '{hemi}')))


def create_vertices_labels_lookup(subject, atlas, save_labels_ids=False, overwrite=False, read_labels_from_fol=''):

    def check_loopup_is_ok(lookup):
        unique_values_num = sum([len(set(lookup[hemi].values())) for hemi in utils.HEMIS])
        # check it's not only the unknowns
        lookup_ok = unique_values_num > 2
        if lookup_ok:
            for hemi in utils.HEMIS:
                verts, _ = utils.read_pial(subject, MMVT_DIR, hemi)
                lookup_ok = lookup_ok and len(lookup[hemi].keys()) == len(verts)
        return lookup_ok

    output_fname = op.join(MMVT_DIR, subject, '{}_vertices_labels_{}lookup.pkl'.format(
        atlas, 'ids_' if save_labels_ids else ''))
    if op.isfile(output_fname) and not overwrite:
        lookup = utils.load(output_fname)
        if check_loopup_is_ok(lookup):
            return lookup
    lookup = {}

    for hemi in utils.HEMIS:
        lookup[hemi] = {}
        create_unknown_label = False
        if read_labels_from_fol != '':
            labels = read_labels(subject, SUBJECTS_DIR, atlas, hemi=hemi, try_first_from_annotation=False,
                                 labels_fol=read_labels_from_fol)
        else:
            labels = read_labels(subject, SUBJECTS_DIR, atlas, hemi=hemi)
        if len(labels) == 0:
            raise Exception("Can't read labels from {} {}".format(subject, atlas))
        labels_names = [l.name for l in labels]
        # if 'unknown-{}'.format(hemi) not in [l.name for l in labels]:
        # create_unknown_label = True
        verts, _ = utils.read_pial(subject, MMVT_DIR, hemi)
        # unknown_verts = set(range(verts.shape[0]))
        for label in labels:
            for vertice in label.vertices:
                lookup[hemi][vertice] = labels_names.index(label.name) if save_labels_ids else label.name
            # if create_unknown_label and label.name != 'unknown-{}'.format(hemi):
            #     unknown_verts -= set(label.vertices)
        # if create_unknown_label:
        #     for vertice in unknown_verts:
        #         lookup[hemi][vertice] = len(labels_names) if save_labels_ids else 'unknown-{}'.format(hemi)
        #     if 'unknown-{}'.format(hemi) in labels_names:
        #         unknown_label = labels[labels_names.index('unknown-{}'.format(hemi))]
        #         recreate_annot = set(unknown_label.vertices) != unknown_verts
        #     if recreate_annot:
        #         while 'unknown-{}'.format(hemi) in labels_names:
        #             del labels[labels_names.index('unknown-{}'.format(hemi))]
        #             labels_names = [l.name for l in labels]
                    # labels[labels_names.index('unknown-{}'.format(hemi))].vertices = sorted(list(unknown_verts))
                # else:s
                #     unknown_label = mne.Label(
                #         sorted(list(unknown_verts)), hemi=hemi, name='unknown-{}'.format(hemi), subject=subject)
                #     labels.append(unknown_label)
                # try:
                #     mne.write_labels_to_annot(
                #         labels, subject=subject, parc=atlas, overwrite=True, subjects_dir=SUBJECTS_DIR, hemi=hemi)
                #     annot_fname = op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format(hemi, atlas))
                #     utils.make_dir(op.join(SUBJECTS_DIR, subject, 'label'))
                #     shutil.copy(annot_fname, op.join(SUBJECTS_DIR, subject, 'label', '{}.{}_backup.annot'.format(hemi, atlas)))
                # except:
                #     print(traceback.format_exc())
                #     return None
                    # labels = mne.read_labels_from_annot(subject, annot_fname=annot_fname)
    if check_loopup_is_ok(lookup):
        utils.save(lookup, output_fname)
        return lookup
    else:
        raise Exception('Error in vertices_labels_lookup!')


def find_label_vertices(subject, atlas, hemi, vertices, label_template):
    import re
    vertices_labels_lookup = create_vertices_labels_lookup(subject, atlas)
    label_re_template = re.compile(label_template)
    label_vertices, label_vertices_indices = [], []
    for vert_ind, vert in enumerate(vertices):
        vert_label = vertices_labels_lookup[hemi].get(vert, '')
        if vert_label == '':
            print('find_pick_activity: No label for vert {}'.format(vert))
            continue
        if label_re_template.search(vert_label) is not None:
            label_vertices.append(vert)
            label_vertices_indices.append(vert_ind)
    return label_vertices, label_vertices_indices


def save_labels_from_vertices_lookup(subject, atlas, subjects_dir, surf_type='pial', read_labels_from_fol=''):
    lookup = create_vertices_labels_lookup(subject, atlas, read_labels_from_fol=read_labels_from_fol)
    labels_fol = op.join(subjects_dir, subject, 'label', atlas)
    utils.delete_folder_files(labels_fol)
    for hemi in utils.HEMIS:
        labels_vertices = defaultdict(list)
        surf_fname = op.join(subjects_dir, subject, 'surf', '{}.{}'.format(hemi, surf_type))
        surf, _ = mne.surface.read_surface(surf_fname)
        for vertice, label in lookup[hemi].items():
            labels_vertices[label].append(vertice)
        for label, vertices in labels_vertices.items():
            if label == 'unknown-{}'.format(hemi):
                # Don't save the unknown label, the labales_to_annot will do that, otherwise there will be 2 unknown labels
                continue
            new_label = mne.Label(sorted(vertices), surf[vertices], hemi=hemi, name=label, subject=subject)
            new_label.save(op.join(labels_fol, label))



def calc_labels_centroids(labels_hemi, hemis_verts):
    centroids = {}
    for hemi in HEMIS:
        centroids[hemi] = np.zeros((len(labels_hemi[hemi]), 3))
        for ind, label in enumerate(labels_hemi[hemi]):
            coo = hemis_verts[label.hemi][label.vertices]
            centroids[label.hemi][ind, :] = np.mean(coo, axis=0)
    return centroids


def backup_annotation_files(subject, subjects_dic, aparc_name, backup_str='backup'):
    # Backup annotation files
    for hemi in HEMIS:
        annot_fname = op.join(subjects_dic, subject, 'label', '{}.{}.annot'.format(hemi, aparc_name))
        if op.isfile(annot_fname):
            shutil.copyfile(op.join(subjects_dic, subject, 'label', '{}.{}.annot'.format(hemi, aparc_name)),
                            op.join(subjects_dic, subject, 'label', '{}.{}.{}.annot'.format(hemi, aparc_name, backup_str)),)


def get_atlas_labels_names(subject, atlas, subjects_dir, delim='-', pos='end', return_flat_labels_list=False, include_unknown=False,
                           include_corpuscallosum=False, n_jobs=1):
    annot_fname_hemi = op.join(subjects_dir, subject, 'label', '{}.{}.annot'.format('{hemi}', atlas))
    labels_names_hemis = dict(lh=[], rh=[])
    all_labels = []
    if utils.both_hemi_files_exist(annot_fname_hemi):
        for hemi in ['rh', 'lh']:
            annot_fname = op.join(subjects_dir, subject, 'label', '{}.{}.annot'.format(hemi, atlas))
            _, _, labels_names = mne.label._read_annot(annot_fname)
            labels_names = fix_labels_names(labels_names, hemi, delim, pos)
            all_labels.extend(labels_names)
            labels_names_hemis[hemi] = labels_names
    else:
        all_labels = read_labels_parallel(subject, subjects_dir, atlas, labels_fol='' , n_jobs=n_jobs)
        for label in all_labels:
            labels_names_hemis[label.hemi].append(label.name)
    if len(labels_names_hemis['rh']) == 0 or len(labels_names_hemis['lh']) == 0:
        raise Exception("Can't read {} labels for atlas {}".format(subject, atlas))
    if return_flat_labels_list:
        if not include_unknown:
            all_labels = [l for l in all_labels if 'unknown' not in l]
        if not include_corpuscallosum:
            all_labels = [l for l in all_labels if 'corpuscallosum' not in l]
        return all_labels
    else:
        if not include_unknown:
            for hemi in HEMIS:
                labels_names_hemis[hemi] = [l for l in labels_names_hemis[hemi] if 'unknown' not in l]
        if not include_corpuscallosum:
            for hemi in HEMIS:
                labels_names_hemis[hemi] = [l for l in labels_names_hemis[hemi] if 'corpuscallosum' not in l]
        return labels_names_hemis


def fix_labels_names(labels_names, hemi, delim='-', pos='end'):
    fixed_labels_names = []
    for label_name in labels_names:
        if isinstance(label_name, bytes):
            label_name = label_name.decode('utf-8')
        if not '{}-'.format(hemi) in label_name or \
            not '{}.'.format(hemi) in label_name or \
            not '-{}'.format(hemi) in label_name or \
            not '.{}'.format(hemi) in label_name:
                if pos == 'end':
                    label_name = '{}{}{}'.format(label_name, delim, hemi)
                elif pos == 'start':
                    label_name = '{}{}{}'.format(hemi, delim, label_name)
                else:
                    raise Exception("pos can be 'end' or 'start'")
        fixed_labels_names.append(label_name)
    return fixed_labels_names


def get_hemi_delim_and_pos(label_name):
    delim, pos, label, label_hemi = '', '', label_name, ''
    for hemi in ['rh', 'lh']:
        if label_name.startswith('{}-'.format(hemi)):
            delim, pos, label = '-', 'start', label_name[3:]
            label_hemi = hemi
            break
        if label_name.startswith('{}_'.format(hemi)):
            delim, pos, label = '_', 'start', label_name[3:]
            label_hemi = hemi
            break
        if label_name.startswith('{}.'.format(hemi)):
            delim, pos, label = '.', 'start', label_name[3:]
            label_hemi = hemi
            break
        if label_name.endswith('-{}'.format(hemi)):
            delim, pos, label = '-', 'end', label_name[:-3]
            label_hemi = hemi
            break
        if label_name.endswith('_{}'.format(hemi)):
            delim, pos, label = '_', 'end', label_name[:-3]
            label_hemi = hemi
            break
        if label_name.endswith('.{}'.format(hemi)):
            label_hemi = hemi
            delim, pos, label = '.', 'end', label_name[:-3]
            break
    return delim, pos, label, label_hemi


def get_label_hemi(label_name):
    _, _, _, hemi = get_hemi_delim_and_pos(label_name)
    return hemi


def get_label_hemi_invariant_name(label_name):
    _, _, label_inv_name, _ = get_hemi_delim_and_pos(label_name)
    while label_inv_name != label_name:
        label_name = label_inv_name
        _, _, label_inv_name, _ = get_hemi_delim_and_pos(label_name)
    return label_inv_name


def remove_duplicate_hemis(label_name):
    delim, pos, label, hemi = get_hemi_delim_and_pos(label_name)
    while label != label_name:
        label_name = label
        _, _, label, _ = get_hemi_delim_and_pos(label_name)
    res = build_label_name(delim, pos, label, hemi)
    return res


def get_other_hemi(hemi):
    return 'rh' if hemi == 'lh' else 'lh'


def get_other_hemi_label_name(label_name):
    delim, pos, label, hemi = get_hemi_delim_and_pos(label_name)
    other_hemi = get_other_hemi(hemi)
    res_label_name = build_label_name(delim, pos, label, other_hemi)
    return res_label_name


def get_template_hemi_label_name(label_name, wild_char=False):
    delim, pos, label, hemi = get_hemi_delim_and_pos(label_name)
    hemi_temp = '?h' if wild_char else '{hemi}'
    res_label_name = build_label_name(delim, pos, label, hemi_temp)
    return res_label_name


def build_label_name(delim, pos, label, hemi):
    if pos == 'end':
        return '{}{}{}'.format(label, delim, hemi)
    else:
        return '{}{}{}'.format(hemi, delim, label)


def get_hemi_from_name(label_name):
    _, _, _, hemi = get_hemi_delim_and_pos(label_name)
    return hemi


@functools.lru_cache(maxsize=None)
def read_labels(subject, subjects_dir, atlas, try_first_from_annotation=True, only_names=False,
                output_fname='', exclude=None, rh_then_lh=False, lh_then_rh=False, sorted_according_to_annot_file=False,
                hemi='both', surf_name='pial', labels_fol='', read_only_from_annot=False, n_jobs=1):
    try:
        labels = []
        if try_first_from_annotation:
            try:
                labels = mne.read_labels_from_annot(
                    subject, atlas, subjects_dir=subjects_dir, surf_name=surf_name, hemi=hemi)
            except:
                print(traceback.format_exc())
                print("read_labels_from_annot failed! subject {} atlas {} surf name {} hemi {}.".format(
                    subject, atlas, surf_name, hemi))
                print('Trying to read labels files')
                if not read_only_from_annot:
                    labels_fol = op.join(subjects_dir, subject, 'label', atlas) if labels_fol == '' else labels_fol
                    labels = read_labels_parallel(subject, subjects_dir, atlas, labels_fol=labels_fol, n_jobs=n_jobs)
        else:
            if not read_only_from_annot:
                labels = read_labels_parallel(
                    subject, subjects_dir, atlas, hemi=hemi, labels_fol=labels_fol, n_jobs=n_jobs)
        if len(labels) == 0:
            raise Exception("Can't read the {} labels!".format(atlas))
        if exclude is None:
            exclude = []
        labels = [l for l in labels if not np.any([e in l.name for e in exclude])]
        if rh_then_lh or lh_then_rh:
            rh_labels = [l for l in labels if l.hemi == 'rh']
            lh_labels = [l for l in labels if l.hemi == 'lh']
            labels = rh_labels + lh_labels if rh_then_lh else lh_labels + rh_labels
        if sorted_according_to_annot_file:
            annot_labels = get_atlas_labels_names(
                subject, atlas, subjects_dir, return_flat_labels_list=True,
                include_corpuscallosum=True, include_unknown=True)
            try:
                labels.sort(key=lambda x: annot_labels.index(x.name))
            except ValueError:
                print("Can't sort labels according to the annot file")
                print(traceback.format_exc())
        if output_fname != '':
            with open(output_fname, 'w') as output_file:
                for label in labels:
                    output_file.write('{}\n'.format(label.name))
        if only_names:
            labels = [l.name for l in labels]
        return labels
    except:
        print(traceback.format_exc())
        return []


def read_labels_parallel(subject, subjects_dir, atlas, hemi='', labels_fol='', n_jobs=1):
    try:
        labels_fol = op.join(subjects_dir, subject, 'label', atlas) if labels_fol == '' else labels_fol
        if hemi != '':
            labels_files = glob.glob(op.join(labels_fol, '*{}.label'.format(hemi)))
            labels_files.extend(glob.glob(op.join(labels_fol, '{}.*label'.format(hemi))))
        else:
            labels_files = glob.glob(op.join(labels_fol, '*.label'))
        files_chunks = utils.chunks(labels_files, len(labels_files) / n_jobs)
        results = utils.run_parallel(_read_labels_parallel, files_chunks, njobs=n_jobs)
        labels = []
        for labels_chunk in results:
            labels.extend(labels_chunk)
        return labels
    except:
        print(traceback.format_exc())
        return []


def _read_labels_parallel(files_chunk):
    labels = []
    for label_fname in files_chunk:
        label = mne.read_label(label_fname)
        labels.append(label)
    return labels


# def read_hemi_labels(subject, subjects_dir, atlas, hemi, surf_name='pial', labels_fol=''):
#     # todo: replace with labels utils read labels function
#     labels_fol = op.join(subjects_dir, subject, 'label', atlas) if labels_fol=='' else labels_fol
#     annot_fname_template = op.join(subjects_dir, subject, 'label', '{}.{}.annot'.format('{hemi}', atlas))
#     if utils.both_hemi_files_exist(annot_fname_template):
#         try:
#             labels = mne.read_labels_from_annot(subject, atlas, hemi, surf_name)
#         except:
#             labels = read_labels_from_annots(subject, subjects_dir, atlas, hemi)
#         if len(labels) == 0:
#             raise Exception('No labels were found in the {} annot file!'.format(annot_fname_template))
#     else:
#         labels = []
#         for label_file in glob.glob(op.join(labels_fol, '*{}.label'.format(hemi))):
#             label = mne.read_label(label_file)
#             labels.append(label)
#         if len(labels) == 0:
#             print('No labels were found in {}!'.format(labels_fol))
#             return []
#     return labels


def calc_center_of_mass(labels, ret_mat=False):
    center_of_mass = np.zeros((len(labels), 3)) if ret_mat else {}
    for ind, label in enumerate(labels):
        if ret_mat:
            center_of_mass[ind] = np.mean(label.pos, 0)
        else:
            center_of_mass[label.name] = np.mean(label.pos, 0)
    return center_of_mass


def label_is_excluded(label_name, compiled_excludes):
    return not compiled_excludes.search(label_name) is None


def label_name(l):
    return l if isinstance(l, str) else l.name


def remove_exclude_labels(labels, excludes=()):
    from functools import partial
    import re

    _label_is_excluded = partial(label_is_excluded, compiled_excludes=re.compile('|'.join(excludes)))
    labels_tup = [(l, ind) for ind, l in enumerate(labels) if not _label_is_excluded(label_name(l))]
    labels = [t[0] for t in labels_tup]
    indices = [t[1] for t in labels_tup]
    return labels, indices


def remove_exclude_labels_and_data(labels_names, labels_data, excludes=()):
    if len(excludes) > 0:
        org_labels_names = labels_names
        labels_names, indices = remove_exclude_labels(labels_names, excludes)
        remove_indices = list(set(range(len(org_labels_names))) - set(indices))
        if len(remove_indices) > len(excludes):
            raise Exception('Error in removing excludes')
        if len(remove_indices) > 0:
            labels_data = np.delete(labels_data, remove_indices, 0)
    if len(labels_names) != labels_data.shape[0]:
        raise Exception(
            'Error in remove_exclude_labels_and_data! len(labels_names) {} != labels_data.shape {}'.format(
                labels_names, labels_data.shape[0]))
    return labels_names, labels_data


def calc_time_series_per_label(x, labels, measure, excludes=(),
                               figures_dir='', do_plot=False, do_plot_all_vertices=False):
    import sklearn.decomposition as deco
    import matplotlib.pyplot as plt

    labels, _ = remove_exclude_labels(labels, excludes)
    if measure.startswith('pca'):
        comps_num = 1 if '_' not in measure else int(measure.split('_')[1])
        labels_data = np.zeros((len(labels), x.shape[-1], comps_num))
    else:
        labels_data = np.zeros((len(labels), x.shape[-1]))
    labels_names = []
    if do_plot_all_vertices:
        all_vertices_plots_dir = op.join(figures_dir, 'all_vertices')
        utils.make_dir(all_vertices_plots_dir)
    if do_plot:
        measure_plots_dir = op.join(figures_dir, measure)
        utils.make_dir(measure_plots_dir)
    for ind, label in enumerate(labels):
        if measure == 'mean':
            labels_data[ind, :] = np.mean(x[label.vertices, 0, 0, :], 0)
        elif measure.startswith('pca'):
            print(label)
            _x = x[label.vertices, 0, 0, :].T
            remove_cols = np.where(np.all(_x == np.mean(_x, 0), 0))[0]
            _x = np.delete(_x, remove_cols, 1)
            _x = (_x - np.mean(_x, 0)) / np.std(_x, 0)
            comps = 1 if '_' not in measure else int(measure.split('_')[1])
            pca = deco.PCA(comps)
            x_r = pca.fit(_x).transform(_x)
            # if x_r.shape[1] == 1:
            labels_data[ind, :] = x_r
            # else:
            #     labels_data[ind, :] = x_r
        elif measure == 'cv': #''coef_of_variation':
            label_mean = np.mean(x[label.vertices, 0, 0, :], 0)
            label_std = np.std(x[label.vertices, 0, 0, :], 0)
            labels_data[ind, :] = label_std / label_mean
        labels_names.append(label.name)
        if do_plot_all_vertices:
            plt.figure()
            plt.plot(x[label.vertices, 0, 0, :].T)
            plt.savefig(op.join(all_vertices_plots_dir, '{}.jpg'.format(label.name)))
            plt.close()
        if do_plot:
            plt.figure()
            plt.plot(labels_data[ind, :])
            plt.savefig(op.join(measure_plots_dir, '{}_{}.jpg'.format(measure, label.name)))
            plt.close()

    return labels_data, labels_names


def morph_labels(morph_from_subject, morph_to_subject, atlas, hemi, n_jobs=1):
    labels_fol = op.join(SUBJECTS_DIR, morph_to_subject, 'label')
    labels_fname = op.join(labels_fol, '{}.{}.pkl'.format(hemi, atlas,morph_from_subject))
    annot_file = op.join(SUBJECTS_DIR, morph_from_subject, 'label', '{}.{}.annot'.format(hemi, atlas))
    if not op.isfile(annot_file):
        print("Can't find the annot file in {}!".format(annot_file))
        return []
    if not op.isfile(labels_fname):
        labels = mne.read_labels_from_annot(morph_from_subject, atlas, subjects_dir=SUBJECTS_DIR, hemi=hemi)
        if morph_from_subject != morph_to_subject:
            morphed_labels = []
            for label in labels:
                label.values.fill(1.0)
                morphed_label = label.morph(morph_from_subject, morph_to_subject, 5, None, SUBJECTS_DIR, n_jobs)
                morphed_labels.append(morphed_label)
            labels = morphed_labels
        utils.save(labels, labels_fname)
    else:
        labels = utils.load(labels_fname)
    return labels


def create_atlas_coloring(subject, atlas, n_jobs=-1):
    ret = False
    coloring_dir = op.join(MMVT_DIR, subject, 'coloring')
    utils.make_dir(coloring_dir)
    coloring_fname = op.join(coloring_dir, 'labels_{}_coloring.csv'.format(atlas))
    coloring_names_fname = op.join(coloring_dir, 'labels_{}_colors_names.txt'.format(atlas))
    try:
        labels = read_labels(subject, SUBJECTS_DIR, atlas, n_jobs=n_jobs)
        colors_rgb_and_names = cu.get_distinct_colors_and_names()
        labels_colors_rgb, labels_colors_names = {}, {}
        for label in labels:
            label_inv_name = get_label_hemi_invariant_name(label.name)
            if label_inv_name not in labels_colors_rgb:
                labels_colors_rgb[label_inv_name], labels_colors_names[label_inv_name] = next(colors_rgb_and_names)
        print('Writing to {} and {}'.format(coloring_fname, coloring_names_fname))
        with open(coloring_fname, 'w') as colors_file, open(coloring_names_fname, 'w') as col_names_file:
            for label in labels:
                label_inv_name = get_label_hemi_invariant_name(label.name)
                color_rgb = labels_colors_rgb[label_inv_name]
                color_name = labels_colors_names[label_inv_name]
                colors_file.write('{},{},{},{}\n'.format(label.name, *color_rgb))
                col_names_file.write('{},{}\n'.format(label.name, color_name))
        ret = op.isfile(coloring_fname)
    except:
        print('Error in save_labels_coloring!')
        print(traceback.format_exc())
    return ret


def create_labels_coloring(subject, labels_names, labels_values, coloring_name, norm_percs=(3, 99),
                           norm_by_percentile=True, colors_map='jet'):
    coloring_dir = op.join(MMVT_DIR, subject, 'coloring')
    utils.make_dir(coloring_dir)
    coloring_fname = op.join(coloring_dir, '{}.csv'.format(coloring_name))
    ret = False
    try:
        labels_colors = utils.arr_to_colors(
            labels_values, norm_percs=norm_percs, norm_by_percentile=norm_by_percentile, colors_map=colors_map)
        print('Saving coloring to {}'.format(coloring_fname))
        with open(coloring_fname, 'w') as colors_file:
            for label_name, label_color, label_value in zip(labels_names, labels_colors, labels_values):
                colors_file.write('{},{},{},{},{}\n'.format(label_name, *label_color[:3], label_value))
        ret = op.isfile(coloring_fname)
    except:
        print('Error in create_labels_coloring!')
        print(traceback.format_exc())
    return ret


def join_labels(new_name, labels):
    from functools import reduce
    import operator
    labels = list(labels)
    new_label = reduce(operator.add, labels[1:], labels[0])
    new_label.name = new_name
    return new_label


def get_lh_rh_indices(labels):
    get_hemi_delim_and_pos(labels[0])
    indices = {hemi:[ind for ind, l in enumerate(labels) if get_label_hemi(label_name(l))==hemi] for hemi in utils.HEMIS}
    labels_arr = np.array(labels)
    if sum([len(labels_arr[indices[hemi]]) for hemi in utils.HEMIS]) != len(labels):
        raise Exception('len(rh_labels) ({}) + len(lh_labels) ({}) != len(labels) ({})'.format(
            len(labels_arr[indices['rh']]), len(labels_arr[indices['lh']]), len(labels)))
    return indices


def grow_label(subject, vertice_indice, hemi, new_label_name, new_label_r=5, n_jobs=6):
    new_label = mne.grow_labels(subject, vertice_indice, new_label_r, 0 if hemi == 'lh' else 1, SUBJECTS_DIR,
                                n_jobs, names=new_label_name, surface='pial')[0]
    utils.make_dir(op.join(MMVT_DIR, subject, 'labels'))
    new_label_fname = op.join(MMVT_DIR, subject, 'labels', '{}.label'.format(new_label_name))
    new_label.save(new_label_fname)
    return new_label


def find_clusters_overlapped_labeles(subject, clusters, data, atlas, hemi, verts,
                                     min_cluster_max=0, min_cluster_size=0, clusters_label='', n_jobs=6):
    cluster_labels = []
    if not op.isfile(op.join(SUBJECTS_DIR, subject, 'surf', '{}.pial'.format(hemi))):
        from src.utils import freesurfer_utils as fu
        verts, faces = utils.read_pial(subject, MMVT_DIR, hemi)
        fu.write_surf(op.join(SUBJECTS_DIR, subject, 'surf', '{}.pial'.format(hemi)), verts, faces)
    labels = read_labels(subject, SUBJECTS_DIR, atlas, hemi=hemi, n_jobs=n_jobs)
    if len(labels) == 0:
        print('No labels!')
        return None
    for cluster in clusters:
        x = data[cluster]
        cluster_max = np.min(x) if abs(np.min(x)) > abs(np.max(x)) else np.max(x)
        inter_labels, inter_labels_tups = [], []
        for label in labels:
            overlapped_vertices = np.intersect1d(cluster, label.vertices)
            if len(overlapped_vertices) > 0:
                if 'unknown' not in label.name:
                    inter_labels_tups.append((len(overlapped_vertices), label.name))
                    # inter_labels.append(dict(name=label.name, num=len(overlapped_vertices)))
        inter_labels_tups = sorted(inter_labels_tups)[::-1]
        for inter_labels_tup in inter_labels_tups:
            inter_labels.append(dict(name=inter_labels_tup[1], num=inter_labels_tup[0]))
        if len(inter_labels) > 0 and abs(cluster_max) >= min_cluster_max and len(cluster) >= min_cluster_size and \
                        clusters_label in inter_labels[0]['name']:
            # max_inter = max([(il['num'], il['name']) for il in inter_labels])
            cluster_labels.append(utils.Bag(dict(vertices=cluster, intersects=inter_labels, name=inter_labels[0]['name'],
                coordinates=verts[cluster], max=cluster_max, hemi=hemi, size=len(cluster))))
        else:
            print('No intersected labels!')
    return cluster_labels


if __name__ == '__main__':
    subject = 'DC'
    atlas = 'laus250'
    # label_name = 'bankssts_1-lh'
    n_jobs = 6
    # check_labels(subject, SUBJECTS_DIR, atlas, label_name)
    save_labels_from_vertices_lookup(
        subject, atlas, SUBJECTS_DIR, surf_type='pial',
        read_labels_from_fol=op.join(SUBJECTS_DIR, subject, 'label', '{}_before_solve_collision'.format(atlas)))
    pass


