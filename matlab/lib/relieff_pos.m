function [keep, w] = relieff_pos(X, y, k, method)
%RELIEFF_POS  ReliefF weights and the indices of features with W > 0.
%   method = 'classification' (revision) or 'regression' (what MATLAB used
%   in the original code, because y was numeric).
[~, w] = relieff(X, y, k, 'method', method);
keep = find(w > 0);
if isempty(keep)
    [~, keep] = max(w);
end
keep = keep(:).';
w = single(w(:).');
end
