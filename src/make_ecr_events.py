import getopt, os, re, sys
import numpy as np
from pandas import read_csv
from mne import find_events, write_events
from mne.io import Raw

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Define help.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

h = '''
    Make an events file for an Emotion Conflict Resolution (ECR) run.
    
    Events are coded as a six-digit string, where each position (from
    left to right) encodes information about the trial for each event.
    
    This encoding is as follows:
    -- Position 1: Event [Stimulus=1, Response=2]
    -- Position 2: Congruence [Congruent=1, Incongruent=2]
    -- Position 3: Conflict [No Conflict = 1, Low Conflict=2, High Conflict=3]
    -- Position 4: Face Valence [Fear=1, Happy=2]
    -- Position 5: Word Valence [Fear=1, Happy=2]
    -- Position 6: Accuracy [Correct=1, Incorrect=2, Missed=3]
    
    To clarify is the following example: Event "122121" corresponds to the
    stimulus-onset of a trial for which the condition is incongruent; the 
    conflict is low; the face valence is anger; the word valence is happy;
    and the subject correctly responded (accurate).
    
    INPUTS
    -i: in_file, or M/EEG ECR file.
    -o: out_file, or name of events (should end in -eve.fif).
    -d: data_file, or behavior file.
    -p: pattern: event pattern to search for (accepts regexp).
    
        -- pattern: 
    
    PATTERN EXAMPLES
    >> '12' will return all events with '12' present.
    >> '.1.' will return all three-character events with '1' in the center.
    >> '^1' will return all events beginning with '1'.
    '''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Define functions.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def parser(argv):
    raw_file, data_file, out_file, pattern = False, False, False, []
    try: 
        opts, args = getopt.getopt(argv, 'hi:o:d:p:', ['in=','out=','data=','pattern='])
    except getopt.GetoptError:
        print(h)
    for opt, arg in opts:
        if opt == '-h':
            print(h)
            sys.exit()
        elif opt in ('-i', '--in'):
            raw_file = arg
        elif opt in ('-o', '--out'):
            out_file = arg
        elif opt in ('-d', '--data'):
            data_file = arg
        elif opt in ('-p', '--pattern'):
            pattern.append(arg)
    return raw_file, data_file, out_file, pattern

def make_ecr_events(raw_file, data_file, out_file, pattern=False):

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Load behavioral file.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    data = read_csv(data_file)
    n_events, _ = data.shape
    n_trials, _ = data[ data.Condition != 0 ].shape # <--- Excludes rest trials.
    n_responses = (~np.isnan(data[data.Condition!=0].ResponseOnset)).sum()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Load raw file.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    raw = Raw(raw_file,preload=False,verbose=False)
    stim_onsets = find_events(raw, stim_channel='STI001', output='onset', verbose=False)
    response_onsets = find_events(raw, stim_channel='STI002', output='onset', verbose=False)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Error catching.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    if n_events != stim_onsets.shape[0]:
        raise ValueError('Number of trial onsets in %s and %s do not match!' %(data_file,raw_file))
    elif n_responses != response_onsets.shape[0]:
        raise ValueError('Number of responses in %s and %s do not match!' %(data_file,raw_file))
    else:
        pass

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Make events file.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    # Ammend Conflict and ResponseAccuracy Categories.
    data['Conflict'] = np.where( data.Conflict == 0, 1, data.Conflict ) # No Conflict: [0,1] --> 1
    data['ResponseAccuracy'] = np.where( data.ResponseAccuracy == 0, 2, data.ResponseAccuracy ) # Accuracy: 0 --> 2
    data['ResponseAccuracy'] = np.where( data.ResponseAccuracy ==99, 3, data.ResponseAccuracy ) # Accuracy:99 --> 3


    # Append Word Valence category.
    data['WordValence'] = np.where(# Con + Angry Face = Angry Word
                                  (data.Condition == 1) & (data.Valence == 1), 1, np.where(
                                   # Con + Happy Face = Happy Word
                                  (data.Condition == 1) & (data.Valence == 2), 2, np.where( 
                                   # Incon + Angry Face = Happy Word
                                  (data.Condition == 2) & (data.Valence == 1), 2, np.where(
                                   # Incon + Happy Face = Angry Word
                                  (data.Condition == 2) & (data.Valence == 2), 1, 99 ))))

    # Make unique identifiers.
    data['StimIDs'] = '1' + data.Condition.map(str) + data.Conflict.map(str) + data.Valence.map(str) +\
                       data.WordValence.map(str) + data.ResponseAccuracy.map(str)
    data['RespIDs'] = '2' + data.Condition.map(str) + data.Conflict.map(str) + data.Valence.map(str) +\
                       data.WordValence.map(str) + data.ResponseAccuracy.map(str)

    # Add identifiers to onset arrays.
    stim_onsets = stim_onsets[np.where(data.Condition == 0, False, True), :]
    stim_onsets[:,2] = data.StimIDs[data.Condition != 0].astype(int)
    response_onsets[:,2] = data.RespIDs[data.ResponseKey != 99].astype(int)

    # Merge and sort.
    events = np.concatenate([stim_onsets,response_onsets])
    events = events[events[:,0].argsort(),:]
    
    # Reduce to pattern.
    if pattern:
        p = re.compile(pattern)
        idx, = np.where( [True if re.findall(p,event.astype(str)) else False for event in events[:,2]] )
        events = events[idx,:]

    # Insert first sample.
    events = np.insert(events, 0, [raw.first_samp, 0, 0], 0)

    # Write to fif file.
    write_events(out_file, events)

    # Write to text file.
    if out_file.endswith('.fif'): out_file = out_file[:-4] + '.txt'
    else: out_file = out_file + '.txt'
    for n, p in enumerate(pattern): events[:,2] = np.where( [True if re.findall(p,event.astype(str)) else False for event in events[:,2]], n+1, events[:,2])
    events = np.insert(events, 1, raw.index_as_time(events[:,0]), 1)
    np.savetxt(out_file, events, fmt = '%s', header = 'pattern = %s' %' | '.join(pattern))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Open file.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

if __name__ == '__main__':
    raw_file, data_file, out_file, pattern = parser(sys.argv[1:])

    if not raw_file: raise ValueError('Must supply raw file! Type "make_ecr_events.py -h" for details.')
    if not out_file: raise ValueError('Must supply out file! Type "make_ecr_events.py -h" for details.')
    if not data_file: raise ValueError('Must supply data file! Type "make_ecr_events.py -h" for details.')

    make_ecr_events(raw_file, data_file, out_file, pattern)
