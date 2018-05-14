import os.path as op
import glob
import time
import numpy as np
import re
import mne
from mne.connectivity import spectral_connectivity
import scipy.io as sio
import scipy
import matplotlib.pyplot as plt
from src.utils import utils
from src.preproc import meg
from src.preproc import electrodes as elec_pre

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
FREESURFER_HOME = utils.get_link_dir(LINKS_DIR, 'freesurfer', 'FREESURFER_HOME')
MMVT_DIR = op.join(LINKS_DIR, 'mmvt')
SUBJECTS_MEG_DIR = op.join(LINKS_DIR, 'meg')
ELECTRODES_DIR = op.join(LINKS_DIR, 'electrodes')


def create_python_electrodes_evoked_response_file(subject, task, bipolar, conditions, matlab_input_file, meg_electrodes_data_length):
    d = sio.loadmat(matlab_input_file)
    conds_data = [d[conditions[0]], d[conditions[1]]]
    E = conds_data[0].shape[1]
    electrodes_data = np.zeros((E, meg_electrodes_data_length, 2))

    for cond, data in enumerate(conds_data):
        data = downsample_data(data)
        electrodes_data[:, :, cond] = np.mean(data[:, :, :meg_electrodes_data_length], 0)

    if bipolar:
        electrodes_names = [e[0][0] for e in d['electrodes']]
        electrodes_data, _ = elec_pre.bipolarize_data(electrodes_data, electrodes_names)

    np.save(op.join(ELECTRODES_DIR, subject, task, 'evoked_{}data').format('bipolar_' if bipolar else ''), electrodes_data)


def meg_recon_to_electrodes(subject, mri_subject, atlas, task ,bipolar, error_radius=3, elec_length=4,
                            vertices_num_threshold=0, meg_single_trials=False):
    elecs_probs = get_electrodes_with_cortical_vertices(mri_subject, atlas, bipolar, error_radius, elec_length)
    get_meg(subject, mri_subject, task, elecs_probs, bipolar, meg_single_trials=meg_single_trials)
    # elecs_probs_fol = op.split(probs_fname)[0]
    # elecs_probs_data_fname = op.join(elecs_probs_fol, '{}_meg.pkl'.format(utils.namebase(probs_fname)))
    # utils.save(elecs_probs, elecs_probs_data_fname)
    # for elec_probs in elecs_probs:
    #     if len(elec_probs['cortical_indices']) > vertices_num_threshold and elec_probs['approx'] == 3:
    #         print(elec_probs['name'], elec_probs['hemi'], len(elec_probs['cortical_indices']))
    #         # meg = get_meg(subject, elec_probs['cortical_indices'], elec_probs['hemi'])


def get_electrodes_with_cortical_vertices(mri_subject, atlas, bipolar, error_radius=3, elec_length=4):
    elecs_probs, _ = utils.get_electrodes_labeling(
        mri_subject, MMVT_DIR, atlas, bipolar, error_radius, elec_length)
    return [elec_probs for elec_probs in elecs_probs if
            len(elec_probs['cortical_indices_dists']) > 0 and len(elec_probs['cortical_rois']) > 0]


def get_meg(subject, mri_subject, task, elecs_probs, bipolar, vertices_num_threshold=30, read_from_stc=False,
            meg_single_trials=False, do_plot=True):
    # meg_files = glob.glob(op.join(MMVT_DIR, subject, 'activity_map_{}'.format(hemi), '*.npy'))
    # meg_data = np.zeros((len(meg_files), len(vertices)))
    # for meg_file in meg_files:
    #     t = int(re.findall('\d+', utils.namebase(meg_file))[0])
    #     meg_data_t = np.load(meg_file)
    #     meg_data[t] = meg_data_t[vertices, 0]
    #     print("sdf")
    electordes_data_fname = op.join(MMVT_DIR, mri_subject, 'electrodes', 'electrodes_{}data.npz'.format(
        'bipolar_' if bipolar else ''))
    electordes_evokde_data_fname = op.join(ELECTRODES_DIR, mri_subject, task, 'evoked_{}data.npy').format(
        'bipolar_' if bipolar else '')

    if not op.isfile(electordes_data_fname) or not op.isfile(electordes_evokde_data_fname):
        print('No electrodes data file!')
        print(electordes_data_fname)
        print(electordes_evokde_data_fname)
        return None
    f = np.load(electordes_data_fname)
    evoked_data = np.load(electordes_evokde_data_fname)
    conds = np.array([cond.decode() if isinstance(cond, np.bytes_) else cond for cond in f['conditions']])
    names = np.array([name.decode() if isinstance(name, np.bytes_) else name for name in f['names']])
    figs_fol = op.join(MMVT_DIR, mri_subject, 'figs', 'meg-electrodes2', 'bipolar' if bipolar else 'unipolar')
    utils.make_dir(figs_fol)
    fname_format, fname_format_cond, events_id = meg.get_fname_format(task)
    if task == 'MSIT':
        cleaning_method = 'nTSSS'
        constrast = 'interference'
        keys_dict = {'neutral': 'noninterference', 'interference': 'interference'}
    elif task == 'ECR':
        cleaning_method = ''
        constrast = ''
        keys_dict = {'C': 'congruent', 'I': 'incongruent'}
    inverse_method = 'dSPM'
    meg.init_globals(subject, mri_subject, fname_format, fname_format_cond, cleaning_method=cleaning_method,
                     task=task, subjects_meg_dir=SUBJECTS_MEG_DIR, subjects_mri_dir=SUBJECTS_DIR,
                     mmvt_dir=MMVT_DIR, files_includes_cond=True, fwd_no_cond=True,
                     constrast=constrast)

    # for elec_probs in elecs_probs:
    #     elec_probs['data'] = {cond:None for cond in events_id.keys()}
        # if len(elec_probs['cortical_indices_dists']) > 0:
        #     print(elec_probs['name'], len(elec_probs['cortical_indices']), np.min(elec_probs['cortical_indices_dists']))

    meg_elecs, errors, dists = [], [], []
    elec_meg_data_st = None
    for cond_id, cond in enumerate(events_id.keys()):
        meg_cond = keys_dict[cond]
        if read_from_stc:
            stc_fname = meg.STC_HEMI_SMOOTH.format(cond=cond, method=inverse_method, hemi='rh')
            stc = mne.read_source_estimate(stc_fname)
            cond_ind = np.where(meg_cond == conds)[0][0]
        else:
            if meg_single_trials:
                meg_data = np.load(op.join(SUBJECTS_MEG_DIR, task, subject, 'labels_ts_{}.npy'.format(cond)))
            meg_evo_data = {}
            for hemi in utils.HEMIS:
                meg_evo_data[hemi] = np.load(
                    op.join(MMVT_DIR, mri_subject, op.basename(meg.LBL.format(atlas, hemi))))
            meg_conds = np.array([cond.decode() if isinstance(cond, np.bytes_) else cond for cond in meg_evo_data['rh']['conditions']])
            meg_labels = {hemi:np.array([name.decode() if isinstance(name, np.bytes_) else name for name in meg_evo_data[hemi]['names']]) for hemi in utils.HEMIS}
            cond_ind = np.where(cond == meg_conds)[0][0]

        for elec_probs_ind, elec_probs in enumerate(elecs_probs):
            try:
                # len(elec_probs['cortical_indices']) > vertices_num_threshold
                if len(elec_probs['cortical_indices_dists']) > 0 and \
                                len(elec_probs['cortical_rois']) > 0:
                                # np.min(elec_probs['cortical_indices_dists']) < 1 and \
                                # elec_probs['approx'] == 3:
                    # print(elec_probs['name'], elec_probs['hemi'], len(elec_probs['cortical_indices']))
                    elec_inds = np.where(elec_probs['name'] == names)[0]
                    if len(elec_inds) == 1:
                        elec_ind = elec_inds[0]
                    else:
                        print('{} found {} in names'.format(elec_probs['name'], len(elec_inds)))
                        continue
                    if read_from_stc:
                        data = stc.rh_data if elec_probs['hemi'] == 'rh' else stc.lh_data
                        T = data.shape[1]
                        elec_meg_data = data[elec_probs['cortical_indices']]
                        dists = elec_probs['cortical_indices_dists']
                        norm_dists = dists / np.sum(dists)
                        norm_dists = np.reshape(norm_dists, (1, len(norm_dists)))
                        elec_meg_data = np.dot(norm_dists, elec_meg_data)
                        elec_meg_data = np.reshape(elec_meg_data, ((T)))
                    else:
                        meg_labels_inds = np.array([np.where(label == meg_labels[elec_probs['hemi']])[0][0] \
                                                    for label in elec_probs['cortical_rois']])
                        probs = elec_probs['cortical_probs']
                        probs = np.reshape(probs, (1, len(probs)))
                        if meg_single_trials:
                            data = meg_data[:, meg_labels_inds, :]
                            if elec_meg_data_st is None:
                                # n_epochs, n_signals, n_times
                                elec_meg_data_st = np.zeros((data.shape[0], len(elecs_probs), data.shape[2], 2))
                            for trial in range(data.shape[0]):
                                elec_meg_data_st[trial, elec_probs_ind, :, cond_id] = np.dot(probs, data[trial])[0, :]
                        else:
                            data = meg_evo_data[elec_probs['hemi']]['data'][meg_labels_inds, :, cond_ind]
                            elec_meg_data = np.dot(probs, data)[0, :]
                    if not meg_single_trials:
                        # elec_data = f['data'][elec_ind, :, cond_ind]
                        elec_data = evoked_data[elec_ind, :, cond_ind]
                        elec_data_diff = np.max(elec_data) - np.min(elec_data)
                        elec_meg_data *= elec_data_diff / (np.max(elec_meg_data) - np.min(elec_meg_data))
                        # elec_meg_data += elec_data[0] - elec_meg_data[0]
                        elec_meg_data += np.mean(elec_data) - np.mean(elec_meg_data)
                        # elec_probs['data'][cond] = elec_meg_data

                        elec_meg_data, elec_data = utils.trim_to_same_size(elec_meg_data, elec_data)
                        data_diff = elec_meg_data-elec_data
                        data_diff = data_diff / max(data_diff)
                        rms = np.sqrt(np.mean(np.power(data_diff, 2)))
                        dist = min(elec_probs['cortical_indices_dists'])
                        errors.append(rms)
                        dists.append(dist)
                        meg_elecs.append(dict(name=elec_probs['name'],rms=rms, dist=dist, cond=cond, approx=elec_probs['approx']))
                        if do_plot: # and elec_probs['name'] == 'RPT8':
                            plt.figure()
                            plt.plot(elec_meg_data, label='pred')
                            plt.plot(elec_data, label='elec')
                            plt.xlabel('Time(ms)')
                            plt.ylabel('Voltage (mV)')
                            plt.legend()
                            plt.title('{}-{}'.format(elec_probs['name'], cond))
                            plt.savefig(op.join(figs_fol, '{:.2f}-{}-{}.jpg'.format(rms, elec_probs['name'], cond)))
                            plt.close()
            except:
                print('Error with {}!'.format(elec_probs['name']))
    if meg_single_trials:
        np.save(op.join(ELECTRODES_DIR, mri_subject, task, 'meg_electrodes_{}ts'.format('bipolar_' if bipolar else '')), elec_meg_data_st)
    else:
        rmss, dists = [], []
        results_fname = op.join(figs_fol, 'results{}.csv'.format('_bipolar' if bipolar else ''))
        with open(results_fname, 'w') as output_file:
            for res in meg_elecs:
                output_file.write('{},{},{},{},{}\n'.format(res['name'], res['cond'], res['rms'], res['dist'], res['approx']))
                rmss.append(res['rms'])
                dists.append(res['dist'])
        plt.hist(rmss, 20)
        plt.xlabel('mV')
        plt.savefig(op.join(figs_fol, 'rmss{}.jpg'.format('_bipolar' if bipolar else '')))
        return elecs_probs


def calc_meg_electrodes_coh(subject, tmin=0, tmax=2.5, sfreq=1000, fmin=55, fmax=110, bw=15, n_jobs=6):
    input_file = op.join(ELECTRODES_DIR, mri_subject, task, 'meg_electrodes_ts.npy')
    output_file = op.join(ELECTRODES_DIR, mri_subject, task, 'meg_electrodes_ts_coh.npy')
    data = np.load(input_file)
    for cond in range(data.shape[3]):
        data_cond = data[:, :, :, cond]
        if cond == 0:
            coh_mat = np.zeros((data_cond.shape[1], data_cond.shape[1], 2))
        con_cnd, _, _, _, _ = spectral_connectivity(
            data_cond, method='coh', mode='multitaper', sfreq=sfreq,
            fmin=fmin, fmax=fmax, mt_adaptive=True, n_jobs=n_jobs, mt_bandwidth=bw, mt_low_bias=True,
            tmin=tmin, tmax=tmax)
        con_cnd = np.mean(con_cnd, axis=2)
        coh_mat[:, :, cond] = con_cnd
    np.save(output_file[:-4], coh_mat)
    return con_cnd


def calc_meg_electrodes_coh_windows(subject, tmin=0, tmax=2.5, sfreq=1000,
        freqs = ((8, 12), (12, 25), (25,55), (55,110)), bw=15, dt=0.1, window_len=0.2, n_jobs=6):
    input_file = op.join(ELECTRODES_DIR, mri_subject, task, 'meg_electrodes_ts.npy')
    output_file = op.join(ELECTRODES_DIR, mri_subject, task, 'meg_electrodes_ts_coh_windows_{}.npy'.format(window_len))
    data = np.load(input_file)
    windows = np.linspace(tmin, tmax - dt, tmax / dt)
    for cond in range(data.shape[3]):
        data_cond = data[:, :, :, cond]
        if cond == 0:
            coh_mat = np.zeros((data_cond.shape[1], data_cond.shape[1], len(windows), len(freqs), 2))

        for freq_ind, (fmin, fmax) in enumerate(freqs):
            for win, tmin in enumerate(windows):
                con_cnd, _, _, _, _ = spectral_connectivity(
                    data[:, :, :, cond], method='coh', mode='multitaper', sfreq=sfreq,
                    fmin=fmin, fmax=fmax, mt_adaptive=True, n_jobs=n_jobs, mt_bandwidth=bw, mt_low_bias=True,
                    tmin=tmin, tmax=tmin + window_len)
                con_cnd = np.mean(con_cnd, axis=2)
                coh_mat[:, :, win, freq_ind, cond] = con_cnd
    np.save(output_file[:-4], coh_mat)
    return con_cnd


def electrodes_tp_remove(electrodes, meg_electordes_names):
    # Remove and sort the electrodes according to the meg_electordes_names
    electrodes_to_remove = set(electrodes) - set(meg_electordes_names)
    indices_to_remove = [electrodes.index(e) for e in electrodes_to_remove]
    electrodes = scipy.delete(electrodes, indices_to_remove).tolist()
    electrodes_indices = np.array([electrodes.index(e) for e in meg_electordes_names])
    electrodes = np.array(electrodes)[electrodes_indices].tolist()
    assert(np.all(electrodes==meg_electordes_names))
    return indices_to_remove, electrodes_indices


def calc_electrodes_coh_windows(subject, input_fname, conditions, bipolar, meg_electordes_names, meg_electrodes_data, tmin=0, tmax=2.5, sfreq=1000,
                freqs=((8, 12), (12, 25), (25,55), (55,110)), bw=15, dt=0.1, window_len=0.2, n_jobs=6):
    output_file = op.join(ELECTRODES_DIR, subject, task, 'electrodes_coh_{}windows_{}.npy'.format('bipolar_' if bipolar else '', window_len))
    if input_fname[-3:] == 'mat':
        d = sio.loadmat(matlab_input_file)
        conds_data = [d[conditions[0]], d[conditions[1]]]
        electrodes = get_electrodes_names(subject, task)
    elif input_fname[-3:] == 'npz':
        d = np.load(input_fname)
        conds_data = d['data']
        conditions = d['conditions']
        electrodes = d['names'].tolist()
        pass

    indices_to_remove, electrodes_indices = electrodes_tp_remove(electrodes, meg_electordes_names)
    windows = np.linspace(tmin, tmax - dt, tmax / dt)
    for cond, data in enumerate(conds_data):
        data = scipy.delete(data, indices_to_remove, 1)
        data = data[:, electrodes_indices, :]
        data = downsample_data(data)
        data = data[:, :, :meg_electrodes_data.shape[2]]
        if cond == 0:
            coh_mat = np.zeros((data.shape[1], data.shape[1], len(windows), len(freqs), 2))
            # coh_mat = np.load(output_file)
            # continue
        now = time.time()
        for freq_ind, (fmin, fmax) in enumerate(freqs):
            for win, tmin in enumerate(windows):
                try:
                    print('cond {}, tmin {}'.format(cond, tmin))
                    utils.time_to_go(now, win + 1, len(windows))
                    con_cnd, _, _, _, _ = spectral_connectivity(
                        data, method='coh', mode='multitaper', sfreq=sfreq,
                        fmin=fmin, fmax=fmax, mt_adaptive=True, n_jobs=n_jobs, mt_bandwidth=bw, mt_low_bias=True,
                        tmin=tmin, tmax=tmin + window_len)
                    con_cnd = np.mean(con_cnd, axis=2)
                    coh_mat[:, :, win, freq_ind, cond] = con_cnd
                except:
                    print('Error with freq {} and win {}'.format(freq_ind, win))
    np.save(output_file[:-4], coh_mat)


def get_electrodes_names(subject, task):
    electrodes = sio.loadmat(op.join(ELECTRODES_DIR, subject, task, 'electrodes_data.mat'))['electrodes']
    return [e[0][0] for e in electrodes]


def calc_coh(subject, conditions, task, meg_electordes_names, meg_electrodes_data, tmin=0, tmax=2.5, sfreq=1000, fmin=55, fmax=110, bw=15, n_jobs=6):
    input_file = op.join(ELECTRODES_DIR, subject, task, 'electrodes_data_trials.mat')
    output_file = op.join(ELECTRODES_DIR, subject, task, 'electrodes_coh.npy')
    d = sio.loadmat(input_file)
    # Remove and sort the electrodes according to the meg_electordes_names
    electrodes = get_electrodes_names(subject, task)
    electrodes_to_remove = set(electrodes) - set(meg_electordes_names)
    indices_to_remove = [electrodes.index(e) for e in electrodes_to_remove]
    electrodes = scipy.delete(electrodes, indices_to_remove).tolist()
    electrodes_indices = np.array([electrodes.index(e) for e in meg_electordes_names])
    electrodes = np.array(electrodes)[electrodes_indices].tolist()
    assert(np.all(electrodes==meg_electordes_names))

    for cond, data in enumerate([d[conditions[0]], d[conditions[1]]]):
        data = scipy.delete(data, indices_to_remove, 1)
        data = data[:, electrodes_indices, :]
        data = downsample_data(data)
        data = data[:, :, :meg_electrodes_data.shape[2]]
        if cond == 0:
            coh_mat = np.zeros((data.shape[1], data.shape[1], 2))

        con_cnd, _, _, _, _ = spectral_connectivity(
            data, method='coh', mode='multitaper', sfreq=sfreq,
            fmin=fmin, fmax=fmax, mt_adaptive=True, n_jobs=n_jobs, mt_bandwidth=bw, mt_low_bias=True,
            tmin=tmin, tmax=tmax)
        con_cnd = np.mean(con_cnd, axis=2)
        coh_mat[:, :, cond] = con_cnd
    np.save(output_file[:-4], coh_mat)
    return con_cnd


def downsample_data(data):
    C, E, T = data.shape
    new_data = np.zeros((C, E, int(T/2)))
    for epoch in range(C):
        new_data[epoch, :, :] = utils.downsample_2d(data[epoch, :, :], 2)
    return new_data


def compare_coh(subject, task, conditions, do_plot=False):
    electrodes_coh = np.load(op.join(ELECTRODES_DIR, subject, task, 'electrodes_coh.npy'))
    meg_electrodes_coh = np.load(op.join(ELECTRODES_DIR, subject, task, 'meg_electrodes_ts_coh.npy'))
    for cond_id, cond in enumerate(conditions):
        # plt.matshow(electrodes_coh[:, :, cond_id])
        # plt.title('electrodes_coh ' + cond)
        # plt.colorbar()
        # plt.matshow(meg_electrodes_coh[:, :, cond_id])
        # plt.title('meg_electrodes_coh ' + cond)
        # plt.colorbar()
        plt.matshow(meg_electrodes_coh[:, :, cond_id]-electrodes_coh[:, :, cond_id])
        plt.title('meg_electrodes_coh-electrodes_coh ' + cond)
        plt.colorbar()
    plt.show()


def compare_coh_windows(subject, task, conditions, electrodes, freqs=((8, 12), (12, 25), (25,55), (55,110)), do_plot=False):
    electrodes_coh = np.load(op.join(ELECTRODES_DIR, subject, task, 'electrodes_coh_windows.npy'))
    meg_electrodes_coh = np.load(op.join(ELECTRODES_DIR, subject, task, 'meg_electrodes_ts_coh_windows.npy'))
    figs_fol = op.join(MMVT_DIR, subject, 'figs', 'coh_windows')
    utils.make_dir(figs_fol)
    results = []
    for cond_id, cond in enumerate(conditions):
        now = time.time()
        for freq_id, freq in enumerate(freqs):
            freq = '{}-{}'.format(*freq)
            indices = list(utils.lower_rec_indices(electrodes_coh.shape[0]))
            for ind, (i, j) in enumerate(indices):
                utils.time_to_go(now, ind, len(indices))
                meg = meg_electrodes_coh[i, j, :, freq_id, cond_id][:22]
                elc = electrodes_coh[i, j, :, freq_id, cond_id][:22]

                elc_diff = np.max(elc) - np.min(elc)
                meg *= elc_diff / (np.max(meg) - np.min(meg))
                meg += np.mean(elc) - np.mean(meg)

                if sum(meg) > len(meg) * 0.99:
                    continue
                data_diff = meg - elc
                # data_diff = data_diff / max(data_diff)
                rms = np.sqrt(np.mean(np.power(data_diff, 2)))
                corr = np.corrcoef(meg, elc)[0, 1]
                results.append(dict(elc1=electrodes[i], elc2=electrodes[j], cond=cond, freq=freq, rms=rms, corr=corr))
                if electrodes[i]=='RAF6' and electrodes[j] == 'LOF4': #corr > 10 and rms < 3:
                    plt.figure()
                    plt.plot(meg, label='prediction')
                    plt.plot(elc, label='electrode')
                    plt.legend()
                    # plt.title('{}-{} {} {}'.format(electrodes[i], electrodes[j], freq, cond)) # (rms:{:.2f})
                    plt.savefig(op.join(figs_fol, '{:.2f}-{}-{}-{}-{}.jpg'.format(rms, electrodes[i], electrodes[j], freq, cond)))
                    plt.close()

    results_fname = op.join(figs_fol, 'results{}.csv'.format('_bipolar' if bipolar else ''))
    rmss, corrs = [], []
    with open(results_fname, 'w') as output_file:
        for res in results:
            output_file.write('{},{},{},{},{},{}\n'.format(
                res['elc1'], res['elc2'], res['cond'], res['freq'], res['rms'], res['corr']))
            rmss.append(res['rms'])
            corrs.append(res['corr'])
    rmss = np.array(rmss)
    corrs = np.array(corrs)
    pass
    # plt.hist(rmss, 100)
    # plt.savefig(op.join(figs_fol, 'rmss_coh{}.jpg'.format('_bipolar' if bipolar else '')))


if __name__ == '__main__':
    subject = 'ep001' # 'ep009' # 'ep007'
    mri_subject = 'mg78' # 'mg99' # 'mg96'
    atlas = 'laus250'
    bipolar = False
    task = 'MSIT' # 'ECR'
    conditions = ['interference', 'noninterference']
    error_radius, elec_length = 3, 4

    matlab_input_file = op.join(ELECTRODES_DIR, mri_subject, task, 'electrodes_data_trials.mat')
    python_input_file = op.join(ELECTRODES_DIR, mri_subject, task, 'electrodes_{}data_st.npz'.format('bipolar_' if bipolar else ''))

    # meg_electrodes_data = np.load(op.join(ELECTRODES_DIR, mri_subject, task, 'meg_electrodes_ts.npy'))
    # T = meg_electrodes_data.shape[2]
    T = 2051
    # create_python_electrodes_evoked_response_file(mri_subject, task, bipolar, conditions, matlab_input_file, T)
    meg_recon_to_electrodes(subject, mri_subject, atlas, task, bipolar, meg_single_trials=False)

    elecs_probs = get_electrodes_with_cortical_vertices(mri_subject, atlas, bipolar, error_radius, elec_length)
    meg_electordes_names = [e['name'] for e in elecs_probs]
    # calc_coh(mri_subject, conditions, task, meg_electordes_names, meg_electrodes_data)
    # calc_electrodes_coh_windows(mri_subject, matlab_input_file, conditions, bipolar, meg_electordes_names, meg_electrodes_data, window_len=0.5)
    # calc_meg_electrodes_coh(mri_subject)
    # calc_meg_electrodes_coh_windows(mri_subject, window_len=0.5)
    # compare_coh_windows(mri_subject, task, conditions, meg_electordes_names, do_plot=False)
    print('finish!')