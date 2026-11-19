function [mdl, bestLambda, cvTab] = lin_tune_fit(X, y, innerFold, lambdas, nCls)
%LIN_TUNE_FIT  L1-regularised logistic regression (lasso), lambda chosen by
%   inner (patient-grouped) CV on balanced accuracy, refit on all data.
%   Binary: fitclinear; multi-class: one-vs-all ECOC of fitclinear learners.
bacc = -inf(numel(lambdas), 1);
for li = 1:numel(lambdas)
    pred = zeros(size(y)); ok = true;
    for f = 1:max(innerFold)
        tr = innerFold ~= f; va = ~tr;
        try
            m = fit_lin(X(tr, :), y(tr), lambdas(li), nCls);
            pred(va) = predict(m, X(va, :));
        catch
            ok = false; break
        end
    end
    if ok, bacc(li) = balanced_acc(y, pred, nCls); end
end
[~, i] = max(bacc); bestLambda = lambdas(i);
mdl = fit_lin(X, y, bestLambda, nCls);
cvTab = table(lambdas(:), bacc, 'VariableNames', {'lambda', 'innerBAcc'});
end

function m = fit_lin(X, y, lam, nCls)
if nCls == 2
    m = fitclinear(X, y, 'Learner', 'logistic', 'Regularization', 'lasso', ...
        'Lambda', lam, 'Solver', 'sparsa', 'ClassNames', 1:2);
else
    t = templateLinear('Learner', 'logistic', 'Regularization', 'lasso', ...
        'Lambda', lam, 'Solver', 'sparsa');
    m = fitcecoc(X, y, 'Learners', t, 'Coding', 'onevsall', 'ClassNames', 1:nCls);
end
end
