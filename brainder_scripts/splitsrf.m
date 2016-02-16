%#!/usr/bin/octave -q
function splitsrf(varargin)
% Split a surface according to the labels given by a DPF/DPV file
% 
% Usage:
% splitsrf(srffile,dpxfile,srfprefix)
% 
% srffile   : Surface file to be split (*.srf).
% dpxfile   : Labels per vertex or per face, in DPF or DPF format.
% srfprefix : File prefix (may include path) to create the new surfaces.
% 
% The program outputs also a text file containing named
% [ srfprefix '.index.csv' ] containing the correspondence between file
% names and the values in the original labels.
% 
% _____________________________________
% Anderson M. Winkler
% Yale University / Institute of Living
% Sep/2011
% http://brainder.org

% Do the OCTAVE stuff, using TRY to ensure MATLAB compatibility
try
    % Get the inputs
    varargin = argv();

    % Disable memory dump on SIGTERM
    sigterm_dumps_octave_core(0);

    % Print usage if no inputs are given
    if isempty(varargin) || strcmp(varargin{1},'-q'),
        fprintf('Split a surface according to the labels given by a DPF/DPV file\n');
        fprintf('\n');
        fprintf('Usage:\n');
        fprintf('splitsrf <srffile> <dpxfile> <srfprefix>\n');
        fprintf('\n');
        fprintf('srffile   : Surface file to be split (*.srf).\n');
        fprintf('dpxfile   : Labels per vertex or per face, in DPF or DPF format.\n');
        fprintf('srfprefix : File prefix (may include path) to create the new surfaces.\n');
        fprintf('\n');
        fprintf('The program outputs also a text file containing named\n');
        fprintf('${srfprefix}.index.csv containing the correspondence between file\n');
        fprintf('names and the values in the original labels.\n');
        fprintf('\n');
        fprintf('_____________________________________\n');
        fprintf('Anderson M. Winkler\n');
        fprintf('Yale University / Institute of Living\n');
        fprintf('Sep/2011\n');
        fprintf('http://brainder.org\n');
        return;
    end
end

% Accept inputs
srffile   = varargin{1};
dpxfile   = varargin{2};
srfprefix = varargin{3};

% Read the surface file and DPV/DPF file with labels
[vtx,fac] = srfread(srffile);
dpx = dpxread(dpxfile);
nV = size(vtx,1);
nF = size(fac,1);
nX = numel(dpx);

% Make sure data contains only integers (labels)
if any(mod(dpx,1)~=0),
    error('The DPV or DPF file must contain only integers')
end

% Verify if this is facewise or vertexwise data
if nX == nV,
    fprintf('Working with vertexwise data.\n');
    facewise = false;
elseif nX == nF,
    fprintf('Working with facewise data.\n');
    facewise = true;
else
    error('The data does not match the surface.');
end

% Make a short list of labels, with no repetitions, and index using
% integers only, monotonically growing and with no intervals
udpx = unique(dpx);   % Unique labels
nL = numel(udpx);
uidx = 1:nL;          % Unique corresponding indices
fprintf('The number of unique labels is %d\n',nL);
dpxidx = zeros(size(dpx));
for lab = 1:nL,
    dpxidx(dpx == udpx(lab)) = uidx(lab); % Replace labels by indices
end

% Vars to store vtx and fac for each label
vtxL = cell(nL,1);
facL = cell(nL,1);

if facewise,
    % If facewise data, simply take the faces and assign
    % them to the corresponding labels
    for lab = 1:nL,
        facL{lab} = fac(dpxidx==lab,:);
    end
else
    % If vertexwise data
    for f = 1:nF,

        % Current face & labels
        Cfac = fac(f,:);
        Cidx = dpxidx(Cfac);

        % Depending on how many vertices of the current face
        % are in different labels, behave differently
        nuCidx = numel(unique(Cidx));
        if nuCidx == 1, % If all vertices share same label

            % Add the current face to the list of faces of the
            % respective label, and don't create new faces
            facL{Cidx(1)} = [facL{Cidx(1)}; Cfac];

        else % If 2 or 3 vertices are in different labels

            % Create 3 new vertices at the midpoints of the 3 edges
            vtxCfac = vtx(Cfac,:);
            vtxnew  = (vtxCfac + vtxCfac([2 3 1],:))./2;
            vtx = [vtx; vtxnew];

            % Define 4 new faces, with care preserve normals (all CCW)
            facnew = [ ...
                Cfac(1)  nV+1    nV+3;
                 nV+1   Cfac(2)  nV+2;
                 nV+3    nV+2   Cfac(3);
                 nV+1    nV+2    nV+3];

            % Update nV for the next loop
            nV = size(vtx,1);

            % Add the new faces to their respective labels
            facL{Cidx(1)}    = [facL{Cidx(1)};    facnew(1,:)];
            facL{Cidx(2)}    = [facL{Cidx(2)};    facnew(2,:)];
            facL{Cidx(3)}    = [facL{Cidx(3)};    facnew(3,:)];
            facL{mode(Cidx)} = [facL{mode(Cidx)}; facnew(4,:)]; % central face
        end
    end
end

% Having defined new faces and assigned all faces to labels, now
% select the vertices and redefine faces to use the new vertex indices
% Also, create the file for the indices
fidx = fopen(sprintf('%s.index.csv',srfprefix),'w');
for lab = 1:nL,

    % Vertices for the current label
    vidx = unique(facL{lab}(:));
    vtxL{lab} = vtx(vidx,:);

    % Reindex the faces
    tmp = zeros(nV,1);
    tmp(vidx) = 1:numel(vidx);
    facL{lab} = reshape(tmp(facL{lab}(:)),size(facL{lab}));

    % Save the resulting surface
    fname = sprintf('%s.%0.4d.srf',srfprefix,lab);
    srfwrite(vtxL{lab},facL{lab},fname)
    
    % Add the corresponding line to the index file
    fprintf(fidx,'%s,%g\n',fname,udpx(lab));
end
fclose(fidx);

% Done!
fprintf('Done!\n')
