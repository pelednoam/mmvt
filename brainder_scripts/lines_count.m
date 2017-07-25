function count = lines_count(fname)
    fh = fopen(fname, 'rt');
    assert(fh ~= -1, 'Could not read: %s', fname);
    x = onCleanup(@() fclose(fh));
    count = 0;
    while ~feof(fh)
        count = count + sum( fread( fh, 16384, 'char' ) == char(10) );
    end
end

