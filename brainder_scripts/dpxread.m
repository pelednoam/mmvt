function varargout = dpxread(filename)
% Read a curvature file (DPV or DPF), in ASCII format.
% This function is much faster than 'dlmread' for large files,
% and works only in Linux and Mac.
%
% [dpx,crd,idx] = dpxread(filename)
% 
% - dpx contains the values for each vertex or face
% - crd contains the vertex coordinates or face indices
% - idx contains vertex or face sequential index
% 
% _____________________________________
% Anderson M. Winkler
% Yale University / Institute of Living
% Feb/2011
% http://brainder.org

% Count the number of lines. This won't work on MS-Windows, but who cares...
[~,result] = system(sprintf('wc -l %s', filename));
nL = str2double(strtok(result,' '));

% Open and read the whole file
fid = fopen(filename,'r');
dpx0 = fscanf(fid,'%f',nL*5);
fclose(fid);

% Reshape from vector to a matrix and get what matters
dpx0 = reshape(dpx0,[5 nL])';
varargout{1} = dpx0(:,5);
varargout{2} = dpx0(:,2:4);
varargout{3} = dpx0(:,1);
