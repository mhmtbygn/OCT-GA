function seed = job_seed(id)
%JOB_SEED  Deterministic seed derived from the job id.
v = double(char(id));
seed = mod(sum(v .* (1:numel(v))) * 7919 + numel(v), 2^31 - 2) + 1;
end
