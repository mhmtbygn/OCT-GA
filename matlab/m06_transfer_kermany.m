function m06_transfer_kermany(cfg)
%M06_TRANSFER_KERMANY  E13: frozen cross-dataset test. The proposed pipeline
%   (z-score, per-backbone ReliefF, GA, tuned SVM, Platt scaling) is fitted
%   ONLY on the local Kermany binary images (no patient IDs available, so the
%   GA hold-out and inner CV are image-level), frozen, and applied without
%   any refitting to all 2064 OCTDL images (Normal vs Diseased).
%   Repeated for cfg.e13.seeds (GA stochasticity).
K = load(fullfile(cfg.featDir, 'features_kermany.mat'), 'X', 'bb');
O = load(fullfile(cfg.featDir, 'features_octdl.mat'), 'X');
I = load(fullfile(cfg.featDir, 'inventory.mat'), 'kerm');
Xk = K.X; yk = I.kerm.y2(:); Xo = O.X; bb = K.bb;
seeds = cfg.e13.seeds; outDir = cfg.jobDir;
parfor si = 1:numel(seeds)
    s = seeds(si);
    id = sprintf('T2_TRANSFER_KtoO_s%d', s);
    out = fullfile(outDir, [id '.mat']);
    if isfile(out), continue, end
    warning('off', 'all'); rng(s, 'twister'); rs = RandStream.getGlobalStream;
    T = struct(); tAll = tic;
    mu = mean(Xk, 1); sd = std(Xk, 0, 1); sd(sd == 0) = 1;
    Xz = (Xk - mu) ./ sd;
    innerFold = strat_group_kfold(yk, (1:numel(yk)).', cfg.innerK, s);
    c1 = find(bb == 1); c2 = find(bb == 2);
    t = tic;
    k1 = relieff_pos(Xz(:, c1), yk, cfg.relieffK, cfg.relieffMethod);
    k2 = relieff_pos(Xz(:, c2), yk, cfg.relieffK, cfg.relieffMethod);
    T.relieff = toc(t);
    pool = [c1(k1), c2(k2)];
    gaVal = innerFold == 1;
    t = tic; ga = ga_select(Xz(~gaVal, pool), yk(~gaVal), Xz(gaVal, pool), yk(gaVal), cfg.ga, rs); T.ga = toc(t);
    sel = pool(ga.mask);
    t = tic; [mdl, best] = svm_tune_fit(Xz(:, sel), yk, innerFold, cfg, 2); T.fit = toc(t);
    mdlP = fitPosterior(mdl);
    Xoz = (Xo - mu) ./ sd;
    [yPred, post] = predict(mdlP, Xoz(:, sel));
    T.total = toc(tAll);
    res = struct('id', id, 'task', 2, 'config', 'F', 'protocol', 'TRANSFER', 'rep', s, 'fold', 0, ...
        'teIdx', (1:size(Xo, 1)).', 'yPred', yPred, 'pPos', post(:, 1), 'nTrain', numel(yk), ...
        'nPool', numel(pool), 'nSel', numel(sel), 'svm', best, 'ga', ga, 'T', T);
    parsave(out, res);
end
end
