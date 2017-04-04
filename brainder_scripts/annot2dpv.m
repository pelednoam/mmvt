%#!/usr/pubsw/packages/octave/current/bin/octave -q
function ret = annot2dpv(varargin)
% Convert an annotation file to a DPV file.
%
% Usage:
% annot2dpv(annotfile,dpvfile)
%
% Inputs:
% annotfile : Annotation file.
% dpvfile   : Output DPV file.
%
% Before running, be sure that ${FREESURFER_HOME}/matlab is
% in the OCTAVE/MATLAB path.
%
% _____________________________________
% Anderson M. Winkler
% Yale University / Institute of Living
% Aug/2011
% http://brainder.org

% Do some OCTAVE stuff, but use TRY to ensure MATLAB compatibility
try
    % Get the inputs
    varargin = argv();
    nargin = numel(varargin);
    
    % Disable memory dump on SIGTERM
    sigterm_dumps_octave_core(0);
    
    % Print usage if no inputs are given
    if isempty(varargin) || strcmp(varargin{1},'-q'),
        fprintf('Convert an annotation file to a DPV file.\n');
        fprintf('\n');
        fprintf('Usage:\n');
        fprintf('annot2dpv <annotfile> <dpvfile>\n');
        fprintf('\n');
        fprintf('Inputs:\n');
        fprintf('annotfile : Annotation file.\n');
        fprintf('dpvfile   : Output DPV file.\n');
        fprintf('\n');
        fprintf('Before running, be sure that ${FREESURFER_HOME}/matlab is\n');
        fprintf('in the OCTAVE/MATLAB path.\n');
        fprintf('\n');
        fprintf('_____________________________________\n');
        fprintf('Anderson M. Winkler\n');
        fprintf('Yale University / Institute of Living\n');
        fprintf('Aug/2011\n');
        fprintf('http://brainder.org\n');
        return;
    end
end

% Some FS commands are needed now
fspath = varargin{3}; %'/usr/local/freesurfer/stable5_3_0'; % getenv('FREESURFER_HOME');
if isempty(fspath)
    error('FREESURFER_HOME variable not correctly set');
else
    addpath(fullfile(fspath,'matlab'));
end

% Accept arguments
annotfile = varargin{1};
dpvfile   = varargin{2};

% Read the annotation file
[vertices,lab,ctab] = blender_read_annotation(annotfile);
if length(vertices) == 0
    ret = false;
    return
end
names = ctab.struct_names;

% For each structure, replace its coded colour by its index
no_indices = zeros(1, ctab.numEntries);
labels_indices = cell(1, ctab.numEntries);
labels_names = cell(1, ctab.numEntries);
for s = 1:ctab.numEntries
    indices = lab == ctab.table(s,5);
    if (sum(indices)==0 && ~strcmp(ctab.struct_names(s), 'unknown'))
        fprintf('%s has no vertices!\n', ctab.struct_names{s});
        no_indices(s) = 1;
        labels_indices{s} = [];
    else
        labels_indices{s} = find(indices);
        labels_names{s} = names{s};
    end
    labels_names{s} = names{s};
    lab(lab == ctab.table(s,5)) = s;
end
save(sprintf('%s_labels.m',annotfile), 'labels_indices', 'labels_names');
% Remove the areas without vertices from the names list
names(no_indices==1) = [];
c=cellstr(names);
fileID = fopen(sprintf('%s_names.txt',annotfile),'w');
fprintf(fileID,'%s\n',c{:});
fclose(fileID);

% Save the result
dpxwrite(dpvfile,lab)
ret = true;