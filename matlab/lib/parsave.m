function parsave(f, res)
%PARSAVE  save() wrapper usable inside parfor.
save(f, 'res', '-v7');
end
