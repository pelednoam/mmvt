cd('/homes/5/npeled/space3/subjects/mg78/electrodes/FieldTripFormatMSIT_AlignedToImagePresent_MG78MSIT_NoStimLFP');
load 'FieldTripFormatMSIT_AlignedToImagePresent_MG78MSIT_NoStimLFP,mat'
d=load('MSITBehavioralDataandTiming.mat');
FIRST = 1526;
LAST = 1830;

el_indices=1:length(ElectrodeLabels);
el_indices(64)=[];
el_indices(81:end)=[];
electrodes = ElectrodeLabels(el_indices);

labels=cell2mat(d.msit_table(:,3))==cell2mat(d.msit_table(:,4));
T = size(cell2mat(ft_data3.trial(1,FIRST)),2);
N = LAST-FIRST+1;
E = length(el_indices);
trials = zeros(N,E,T); 
for trial_index=1:N
    trial = cell2mat(ft_data3.trial(1,FIRST+trial_index-1));
    trials(trial_index, :, :) = trial(el_indices, 1:T);
    fprintf('%d\n',trial_index);
end
%plot(mean(squeeze(mean(trials(:, :, :), 1))))
noninterference_evoked = squeeze(mean(trials(labels==1, :, :), 1));
interference_evoked = squeeze(mean(trials(labels==0, :, :), 1));
Tdurr = d.timing_info.Tdur;
Toffset = d.timing_info.Toffset;
dt = d.timing_info.dt;

save('/homes/5/npeled/space3/subjects/mg78/electrodes/electrodes_data', 'electrodes', 'noninterference_evoked', 'interference_evoked', 'Tdurr', 'Toffset', 'dt');
disp('finish!')