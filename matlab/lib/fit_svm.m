function mdl = fit_svm(X, y, kernel, C, nCls)
%FIT_SVM  SVM with the settings used in the manuscript (auto kernel scale,
%   standardisation); binary -> fitcsvm, multi-class -> one-vs-one ECOC.
switch kernel
    case 'linear', kargs = {'KernelFunction', 'linear'};
    case 'poly2',  kargs = {'KernelFunction', 'polynomial', 'PolynomialOrder', 2};
    case 'poly3',  kargs = {'KernelFunction', 'polynomial', 'PolynomialOrder', 3};
    case 'rbf',    kargs = {'KernelFunction', 'gaussian'};
    otherwise, error('fit_svm:kernel', 'Unknown kernel %s', kernel);
end
args = [kargs, {'KernelScale', 'auto', 'BoxConstraint', C, 'Standardize', true}];
if nCls == 2
    mdl = fitcsvm(X, y, args{:}, 'ClassNames', 1:2);
else
    t = templateSVM(args{:});
    mdl = fitcecoc(X, y, 'Learners', t, 'Coding', 'onevsone', 'ClassNames', 1:nCls);
end
end
