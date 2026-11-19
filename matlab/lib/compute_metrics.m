function M = compute_metrics(yTrue, yPred, nCls, pPos)
%COMPUTE_METRICS  Overall and per-class metrics.
%   yTrue, yPred : labels 1..nCls.  pPos : posterior of class 1 (binary), optional.
%   Binary convention: class 1 = Diseased (positive), class 2 = Normal.
if nargin < 4, pPos = []; end
yTrue = yTrue(:); yPred = yPred(:);
ok = ~isnan(yTrue) & ~isnan(yPred);
yTrue = yTrue(ok); yPred = yPred(ok);
if ~isempty(pPos), pPos = pPos(:); pPos = pPos(ok); end
cm = accumarray([yTrue yPred], 1, [nCls nCls]);
s = sum(cm(:)); tp = diag(cm); t = sum(cm, 2); p = sum(cm, 1).';
fn = t - tp; fp = p - tp; tn = s - tp - fn - fp;
sens = tp ./ t; spec = tn ./ (tn + fp); prec = tp ./ p;
f1 = 2 * tp ./ (2 * tp + fp + fn); gm = sqrt(sens .* spec);
M.cm = cm; M.n = s;
M.acc = trace(cm) / s;
M.bacc = mean(sens(t > 0));
M.macroF1 = mean(f1((t > 0) | (p > 0)), 'omitnan');
den = sqrt((s^2 - p.' * p) * (s^2 - t.' * t));
if den == 0, M.mcc = 0; else, M.mcc = (trace(cm) * s - t.' * p) / den; end
pe = (t.' * p) / s^2; M.kappa = (M.acc - pe) / (1 - pe);
M.sensC = sens.'; M.specC = spec.'; M.precC = prec.'; M.f1C = f1.'; M.gmeanC = gm.'; M.supportC = t.';
M.auc = NaN; M.ap = NaN; M.brier = NaN; M.ece = NaN; M.sensAtSpec90 = NaN; M.sensAtSpec95 = NaN;
if nCls == 2
    M.sensitivity = sens(1); M.specificity = sens(2); M.precision = prec(1);
    M.f1 = f1(1); M.gmean = sqrt(sens(1) * sens(2));
    if ~isempty(pPos) && all(~isnan(pPos)) && numel(unique(yTrue)) == 2
        pos = yTrue == 1;
        try
            [fpr, tpr, ~, auc] = perfcurve(double(pos), pPos, 1);
            M.auc = auc;
            M.sensAtSpec90 = max(tpr(fpr <= 0.10));
            M.sensAtSpec95 = max(tpr(fpr <= 0.05));
        catch
        end
        [~, o] = sort(pPos, 'descend'); ps = pos(o);
        cumTP = cumsum(ps); precAt = cumTP ./ (1:numel(ps)).';
        M.ap = sum(precAt(ps)) / max(1, sum(ps));            % average precision
        M.brier = mean((pPos - pos).^2);
        edges = linspace(0, 1, 11); b = discretize(pPos, edges); ece = 0;
        for k = 1:10
            m = b == k;
            if any(m), ece = ece + sum(m) / numel(pPos) * abs(mean(pos(m)) - mean(pPos(m))); end
        end
        M.ece = ece;
    end
end
end
