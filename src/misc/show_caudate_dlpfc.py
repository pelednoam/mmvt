import os.path as op
import csv
from collections import defaultdict
from src.utils import utils

LINKS_DIR = utils.get_links_dir()
MMVT_DIR = op.join(LINKS_DIR, 'mmvt')


def read_xlsx(xlsx_fname):
    electrodes = {}
    for line_ind, line in enumerate(utils.xlsx_reader(xlsx_fname)):
        if line_ind == 0:
            header = line
            continue
        subject = line[0].lower()
        electrodes[subject] = defaultdict(list)
        for k in range(len(header) - 1):
            electrodes[subject][header[k + 1].lower()].extend(read_electrodes_from_cell(line[k + 1]))
    return electrodes


def create_coloring_files(electrodes, template_subject):
    fol = utils.make_dir(op.join(MMVT_DIR, template_subject))
    regions_csv_fname = op.join(fol, 'coloring', 'caudate_dlpfc.csv')
    all_electrodes_csv_fname = op.join(fol, 'coloring', 'caudate_dlpfc_all.csv')
    template_electrodes = utils.load(op.join(fol, 'electrodes', 'template_electrodes.pkl'))
    regions, colors = None, None
    with open(all_electrodes_csv_fname, 'w') as all_csv, open(regions_csv_fname, 'w') as regions_csv:
        wr_regions = csv.writer(regions_csv, quoting=csv.QUOTE_NONE)
        wr_all = csv.writer(all_csv, quoting=csv.QUOTE_NONE)
        for subject, subject_electrodes in electrodes.items():
            if regions is None:
                regions = list(subject_electrodes.keys())
                colors = utils.get_distinct_colors(len(regions) + 1)
            all_subject_electrodes = [elc_name.split('_')[1] for (elc_name, _) in template_electrodes[subject]]
            for elc_name in all_subject_electrodes:
                for region, color in zip(regions, colors[:-1]):
                    if elc_name in electrodes[subject][region]:
                        print('{}: {} {}'.format(subject, elc_name, region))
                        wr_all.writerow(['{}_{}'.format(subject, elc_name), *color])
                        wr_regions.writerow(['{}_{}'.format(subject, elc_name), *color])
                        break
                else:
                    wr_all.writerow(['{}_{}'.format(subject, elc_name), *colors[-1]])
                    wr_regions.writerow(['{}_{}'.format(subject, elc_name), 1, 1, 1])


def read_electrodes_from_cell(cell):
    electrodes = []
    for elc_name in cell.replace(' ', '').split(','):
        group, num = utils.elec_group_number(elc_name)
        electrodes.append('{}{}'.format(group, num))
    return electrodes


if __name__ == '__main__':
    xlsx_fname = '/autofs/space/thibault_001/users/npeled/Documents/SarahBick/DLPFC Caudate Channels.xlsx'
    template_subject = 'fsaverage'
    electrodes = read_xlsx(xlsx_fname)
    create_coloring_files(electrodes, template_subject)
    print('Finish!')