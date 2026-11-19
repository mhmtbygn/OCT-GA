function [mdl, best, cvTab] = svm_tune_fit(X, y, innerFold, cfg, nCls, fixed)
%SVM_TUNE_FIT  Choose kernel and C by inner (patient-grouped) CV on the
%   outer training set (criterion: balanced accuracy), then refit on the
%   whole outer training set. Pass 'fixed' = struct('kernel',..,'C',..) to
%   skip tuning.
if nargin >= 6 && ~isempty(fixed)
    best = fixed; cvTab = table();
else
    nK = numel(cfg.svm.kernels); nC = numel(cfg.svm.C);
    kern = cell(nK * nC, 1); Cs = zeros(nK * nC, 1); sc = -inf(nK * nC, 1);
    r = 0; bestScore = -inf; best = struct('kernel', cfg.svm.kernels{1}, 'C', cfg.svm.C(1));
    for ki = 1:nK
        for ci = 1:nC
            r = r + 1; kern{r} = cfg.svm.kernels{ki}; Cs(r) = cfg.svm.C(ci);
            pred = zeros(size(y)); ok = true;
            for f = 1:max(innerFold)
                tr = innerFold ~= f; va = ~tr;
                try
                    m = fit_svm(X(tr, :), y(tr), kern{r}, Cs(r), nCls);
                    pred(va) = predict(m, X(va, :));
                catch
                    ok = false; break
                end
            end
            if ok, sc(r) = balanced_acc(y, pred, nCls); end
            if sc(r) > bestScore
                bestScore = sc(r); best = struct('kernel', kern{r}, 'C', Cs(r));
            end
        end
    end
    cvTab = table(kern, Cs, sc, 'VariableNames', {'kernel', 'C', 'innerBAcc'});
end
mdl = fit_svm(X, y, best.kernel, best.C, nCls);
end
