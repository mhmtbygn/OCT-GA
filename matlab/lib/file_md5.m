function h = file_md5(f)
%FILE_MD5  MD5 hash of a file's bytes (hex string), used to find exact duplicates.
fid = fopen(f, 'r');
b = fread(fid, inf, '*uint8');
fclose(fid);
md = java.security.MessageDigest.getInstance('MD5');
md.update(typecast(b, 'int8'));
d = typecast(md.digest(), 'uint8');
h = lower(reshape(dec2hex(d, 2).', 1, []));
end
