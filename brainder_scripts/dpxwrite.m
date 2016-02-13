function dpxwrite(varargin)
% Write a curvature file (DPV or DPF), in ASCII format.
%
% dpxwrite(filename,dpx)
% dpxwrite(filename,dpx,crd,idx)
%
% - fname is the file name to be created
% - dpx contains the values for each vertex or face
% - crd contains the vertex coordinates or face indices
% - idx contains vertex or face sequential index
%
% _____________________________________
% Anderson M. Winkler
% Yale University / Institute of Living
% Oct/2011
% http://brainder.org

% File name
fname = varargin{1};

% Get the actual data
dpx = varargin{2}(:);
nX  = numel(dpx);

% Check if all are integers and use appropriate formating
if all(mod(dpx,1)==0),
    fstr = '%d';
else
    fstr = '%0.10f';
end

if nargin == 2,

    % Organise the data, fill the coords with zeros amd prep to save
    dpx = [(0:nX-1) ; zeros(3,nX) ; dpx'];

elseif nargin == 4,

    % Organise the coords
    crd = varargin{3};
    if size(crd,1) > size(crd,2),
        crd = crd';
    end

    % Take the indices
    idx = varargin{4}(:);

    % Prepare to save
    dpx = [idx' ; crd ; dpx'];

else
    error('Incorrect number of arguments');
end

% Save
fid = fopen(fname,'w');
fprintf(fid,['%d %g %g %g ' fstr '\n'],dpx);
fclose(fid);
