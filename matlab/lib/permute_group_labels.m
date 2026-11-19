function yp = permute_group_labels(y, group, seed)
%PERMUTE_GROUP_LABELS  Permute labels at patient level (label-permutation
%   test). Every patient keeps one label for all of its images.
[~, ~, gi] = unique(group(:));
gl = accumarray(gi, y(:), [], @mode);
rs = RandStream('mt19937ar', 'Seed', seed);
glp = gl(randperm(rs, numel(gl)));
yp = glp(gi);
end
