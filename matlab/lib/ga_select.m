function out = ga_select(Xt, yt, Xv, yv, g, rs)
%GA_SELECT  Binary genetic-algorithm feature selection, k-NN hold-out fitness.
%   Same operators as GA_FeatureSelection in the original code: random
%   initial masks (p = 0.5), roulette-wheel parents, single-point crossover
%   applied with probability CR, bit-flip mutation with probability MR per
%   gene, (mu + lambda) elitist survival, fitness = 1 - hold-out accuracy.
%   Revision additions: stall rule (g.stall generations without improvement)
%   and an optional target accuracy (g.targetAcc; NaN = off).
%   g.MR < 0 means MR = 1/D.
t0 = tic;
yt = yt(:); yv = yv(:);
D = size(Xt, 2); N = g.N;
MR = g.MR; if MR < 0, MR = 1 / D; end
nCls = max([yt; yv]);
fitFn = @(m) knn_holdout_err(Xt, yt, Xv, yv, m, g.knnK, nCls, rs);
if D < 2
    out = struct('mask', true(1, D), 'bestFit', fitFn(true(1, D)), 'curve', [], ...
                 'nGen', 0, 'nEval', 1, 'stop', 'D<2', 'time', toc(t0));
    return
end
pop = rand(rs, N, D) > 0.5;
fit = zeros(1, N);
for i = 1:N, fit(i) = fitFn(pop(i, :)); end
[best, bi] = min(fit); bestSol = pop(bi, :);
curve = nan(1, g.maxIter); stallCnt = 0; nEval = N; stopReason = 'maxIter'; gen = 0;
for gen = 1:g.maxIter
    w = 1 - fit;
    if sum(w) <= 0, w = ones(1, N); end
    cumP = cumsum(w / sum(w)); cumP(end) = 1;
    children = false(N + 1, D); nc = 0;
    while nc < N
        if rand(rs) < g.CR
            p1 = pop(find(cumP >= rand(rs), 1), :);
            p2 = pop(find(cumP >= rand(rs), 1), :);
            cp = randi(rs, [1, D - 1]);
            children(nc + 1, :) = [p1(1:cp), p2(cp + 1:end)];
            children(nc + 2, :) = [p2(1:cp), p1(cp + 1:end)];
            nc = nc + 2;
        end
    end
    children = children(1:N, :);
    mut = rand(rs, N, D) < MR;
    children(mut) = ~children(mut);
    cfit = zeros(1, N); improved = false;
    for i = 1:N
        cfit(i) = fitFn(children(i, :));
        if cfit(i) < best
            best = cfit(i); bestSol = children(i, :); improved = true;
        end
    end
    nEval = nEval + N;
    allPop = [pop; children]; allFit = [fit, cfit];
    [allFit, ord] = sort(allFit, 'ascend');
    pop = allPop(ord(1:N), :); fit = allFit(1:N);
    curve(gen) = best;
    if improved, stallCnt = 0; else, stallCnt = stallCnt + 1; end
    if ~isnan(g.targetAcc) && (1 - best) >= g.targetAcc
        stopReason = 'targetAcc'; break
    end
    if stallCnt >= g.stall
        stopReason = 'stall'; break
    end
end
out = struct('mask', bestSol, 'bestFit', best, 'curve', curve(1:gen), ...
             'nGen', gen, 'nEval', nEval, 'stop', stopReason, 'time', toc(t0));
end
