function [ci, bs, v0] = cluster_boot_ci(metricFn, groups, B, seed)
%CLUSTER_BOOT_CI  Patient-cluster (group) bootstrap percentile 95% CI.
%   metricFn(idx) returns a row vector of metrics computed on the images
%   idx (idx may contain repeats). Groups (patients) are resampled with
%   replacement and all images of a drawn patient are kept.
[~, ~, gi] = unique(groups(:));
nG = max(gi);
members = accumarray(gi, (1:numel(gi)).', [nG 1], @(x) {x});
v0 = metricFn((1:numel(gi)).');
rs = RandStream('mt19937ar', 'Seed', seed);
bs = nan(B, numel(v0));
for b = 1:B
    pick = randi(rs, nG, nG, 1);
    bs(b, :) = metricFn(vertcat(members{pick}));
end
ci = prctile(bs, [2.5 97.5], 1);
end
