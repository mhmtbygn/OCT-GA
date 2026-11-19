function write_error(f, ME)
%WRITE_ERROR  Write an error report to a text file (usable inside parfor).
fid = fopen(f, 'w');
fprintf(fid, '%s\n', getReport(ME, 'extended', 'hyperlinks', 'off'));
fclose(fid);
end
