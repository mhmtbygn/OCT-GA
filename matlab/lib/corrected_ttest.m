function [p, t, mdiff, sdiff, n] = corrected_ttest(d, nTe, nTr)
%CORRECTED_TTEST  Corrected resampled t-test for repeated k-fold CV
%   (Nadeau & Bengio, 2003; Bouckaert & Frank, 2004). d holds the paired
%   per-fold differences (same repeat and fold); nTe/nTr are the test and
%   training sizes of those folds. The variance is inflated by nTe/nTr to
%   account for the overlap between training sets, so folds are not
%   treated as independent.
d = d(:); keep = ~isnan(d); d = d(keep);
nTe = nTe(:); nTr = nTr(:); nTe = nTe(keep); nTr = nTr(keep);
n = numel(d); mdiff = mean(d); sdiff = std(d);
rho = mean(nTe ./ nTr);
v = var(d);
if n < 2
    t = NaN; p = NaN;
elseif v == 0
    t = sign(mdiff) * Inf; p = double(mdiff == 0);
else
    t = mdiff / sqrt((1 / n + rho) * v);
    p = 2 * tcdf(-abs(t), n - 1);
end
end
