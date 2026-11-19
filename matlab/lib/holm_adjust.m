function padj = holm_adjust(p)
%HOLM_ADJUST  Holm-Bonferroni adjusted p-values (NaN entries ignored).
p = p(:); padj = nan(size(p));
ok = find(~isnan(p)); m = numel(ok);
if m == 0, return, end
[ps, i] = sort(p(ok));
adj = min(1, cummax((m - (1:m).' + 1) .* ps));
padj(ok(i)) = adj;
end
