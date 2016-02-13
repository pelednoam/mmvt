function [vtx,fac] = srfread(filename)
% Read a surface file, in ASCII format.
% 
% [vtx,fac] = srfread('filename');
% 
% - vtx contains the coordinates (x,y,z), one vertex per row
% - fac contains the indices for the three vertices of each face
% 
% The indices for the vertices start at 1, not 0.
% 
% _____________________________________
% Anderson M. Winkler
% Yale University / Institute of Living
% Jan/2011
% http://brainder.org

fid = fopen(filename,'r');
fgets(fid); % ignore 1st line
nV  = fscanf(fid,'%d',1); % num of vertices
nF  = fscanf(fid,'%d',1); % num of faces
vtx = fscanf(fid,'%f',nV*4);
fac = fscanf(fid,'%d',nF*4);
fclose(fid);
vtx = reshape(vtx,[4 nV])';
vtx(:,4) = [];
fac = reshape(fac,[4 nF])';
fac(:,4) = [];
fac = fac + 1; % indices start at 1, not 0 in OCTAVE/MATLAB.
