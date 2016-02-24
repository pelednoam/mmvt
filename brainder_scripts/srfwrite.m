function srfwrite(varargin)
% Write a surface file, in ASCII format.
% 
% srfwrite(vtx,fac,fname,comment);
% 
% - vtx contains the coordinates (x,y,z), one vertex per row
% - fac contains the indices for the three vertices of each face
% - fname is the file name to be created (with full path if needed)
% - comment is an optional, single line comment to be added.
% 
% The file format is not widely documented, so be parsimonious and use
% only very short comments (say, less than 25 chars or so) to be safe
% in other applications.
% 
% _____________________________________
% Anderson M. Winkler
% Yale University / Institute of Living
% Jan/2011
% http://brainder.org

% Accept inputs
if nargin < 3 || nargin > 4,
    error('Wrong number of arguments.');
else
    vtx = varargin{1};
    fac = varargin{2};
    fname = varargin{3};
    if nargin == 4,
        comment = varargin{4};
    else
        comment = '';
    end
end

% Make sure that the face indices start at zero
if min(fac(:)) == 1,
    fac = fac - 1;
end

% Add an extra col of zeros to vtx and fac
vtx = [vtx zeros(size(vtx,1),1)];
fac = [fac zeros(size(fac,1),1)];

% Write to the disk
fid = fopen(fname,'w');
fprintf(fid,'#!ascii %s\n',comment);             % signature and comment
fprintf(fid,'%g %g\n',size(vtx,1),size(fac,1));  % number of vertices and faces
fprintf(fid,'%f %f %f %g\n',vtx');               % vertex coords
fprintf(fid,'%g %g %g %g\n',fac');               % face indices
fclose(fid);
