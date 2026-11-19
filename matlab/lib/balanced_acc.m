function b = balanced_acc(y, pred, nCls)
%BALANCED_ACC  Mean per-class recall over the classes present in y.
y = y(:); pred = pred(:);
r = nan(1, nCls);
for c = 1:nCls
    m = y == c;
    if any(m), r(c) = mean(pred(m) == c); end
end
b = mean(r, 'omitnan');
end
