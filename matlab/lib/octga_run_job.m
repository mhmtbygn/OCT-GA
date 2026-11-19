function res = octga_run_job(job, D, cfg)
%OCTGA_RUN_JOB  Run one outer fold of one configuration (nested protocol).
%   Everything that learns from data (z-score, ReliefF, GA, SVM kernel/C,
%   Platt scaling) is fitted on the outer TRAINING images only. The outer
%   test images are used once, for prediction.
%
%   job.protocol: 'PG'   patient-grouped outer and inner folds (main)
%                 'IL'   image-level outer and inner folds (leakage check)
%                 'ORIG' replica of the original code (see octga_run_orig)
warning('off', 'all');
tJob = tic;
seed = job_seed(job.id);
rng(seed, 'twister');
rs = RandStream.getGlobalStream;
if strcmp(job.protocol, 'ORIG')
    res = octga_run_orig(job, D, cfg, rs);
    res.T.total = toc(tJob);
    [res.peakMB, res.curMB] = peak_mem_mb();
    return
end

nCls = job.task;
if nCls == 2, y = D.y2; else, y = D.y7; end
if job.permIdx > 0
    y = permute_group_labels(y, D.group, seed);
end
if strcmp(job.protocol, 'PG')
    fv = D.foldsPG(:, job.rep); grp = D.group;
else
    fv = D.foldsIL(:, job.rep); grp = (1:numel(y)).';
end
te = find(fv == job.fold); tr = find(fv ~= job.fold);

% --- learning curve: keep a stratified fraction of the training patients
if job.lcFrac < 1
    [ug, ~, gi] = unique(grp(tr));
    gLab = accumarray(gi, D.y7(tr), [], @mode);
    keepG = [];
    for c = unique(gLab).'
        gc = ug(gLab == c);
        nk = max(1, round(job.lcFrac * numel(gc)));
        keepG = [keepG; gc(randperm(rs, numel(gc), nk))]; %#ok<AGROW>
    end
    tr = tr(ismember(grp(tr), keepG));
end
ytr = y(tr); yte = y(te);

% --- z-score fitted on the outer training set
T = struct();
t = tic;
Xtr = D.X(tr, :); Xte = D.X(te, :);
mu = mean(Xtr, 1); sd = std(Xtr, 0, 1); sd(sd == 0) = 1;
Xtr = (Xtr - mu) ./ sd; Xte = (Xte - mu) ./ sd;
T.zscore = toc(t);

% --- inner patient-grouped folds (fold 1 = GA fitness hold-out)
innerFold = strat_group_kfold(D.y7(tr), grp(tr), cfg.innerK, seed + 1);
gaVal = innerFold == 1; gaTr = ~gaVal;

c1 = find(D.bb == 1); c2 = find(D.bb == 2);
k1 = []; k2 = []; w1 = []; w2 = []; wG = [];
T.relieff = 0; T.ga = 0; ga = [];
gcfg = cfg.ga;
if job.variant > 0
    v = cfg.e12.variants{job.variant}; gcfg.MR = v.MR; gcfg.N = v.N;
end

t = tic;
switch job.config
    case {'F', 'D'}
        [k1, w1] = relieff_pos(Xtr(:, c1), ytr, cfg.relieffK, cfg.relieffMethod);
        [k2, w2] = relieff_pos(Xtr(:, c2), ytr, cfg.relieffK, cfg.relieffMethod);
        pool = [c1(k1), c2(k2)];
    case 'A'
        [k1, w1] = relieff_pos(Xtr(:, c1), ytr, cfg.relieffK, cfg.relieffMethod);
        pool = c1(k1);
    case 'B'
        [k2, w2] = relieff_pos(Xtr(:, c2), ytr, cfg.relieffK, cfg.relieffMethod);
        pool = c2(k2);
    case 'G'
        [kG, wG] = relieff_pos(Xtr, ytr, cfg.relieffK, cfg.relieffMethod);
        pool = kG;
    case {'C', 'E', 'Etuned', 'L1'}
        pool = 1:size(D.X, 2);
    case 'FZ'
        pool = find(D.bb == 2 & D.lay == 2);      % DenseNet201 GAP (1920)
    otherwise
        error('octga_run_job:config', 'Unknown config %s', job.config);
end
T.relieff = toc(t);

if ismember(job.config, cfg.gaConfigs)
    t = tic;
    ga = ga_select(Xtr(gaTr, pool), ytr(gaTr), Xtr(gaVal, pool), ytr(gaVal), gcfg, rs);
    T.ga = toc(t);
    sel = pool(ga.mask);
else
    sel = pool;
end

% --- classifier (tuned inside the outer training set)
t = tic;
switch job.config
    case 'E'
        best = struct('kernel', 'poly3', 'C', 1); cvTab = table();
        mdl = fit_svm(Xtr(:, sel), ytr, 'poly3', 1, nCls);
    case 'L1'
        [mdl, lam, cvTab] = lin_tune_fit(Xtr(:, sel), ytr, innerFold, cfg.lin.lambda, nCls);
        best = struct('kernel', 'L1-logistic', 'C', lam);
    case 'FZ'
        fz = cfg; fz.svm.kernels = {'linear'};
        [mdl, best, cvTab] = svm_tune_fit(Xtr(:, sel), ytr, innerFold, fz, nCls);
    otherwise
        [mdl, best, cvTab] = svm_tune_fit(Xtr(:, sel), ytr, innerFold, cfg, nCls);
end
T.fit = toc(t);

t = tic;
[yPred, score] = predict(mdl, Xte(:, sel));
T.predict = toc(t);
T.predictPerImage = T.predict / max(1, numel(te));

% --- posterior of class 1 (binary task) for AUC / calibration
pPos = [];
if nCls == 2
    t = tic;
    try
        if strcmp(job.config, 'L1')
            pPos = score(:, 1);                       % logistic scores are posteriors
        else
            mdlP = fitPosterior(mdl);                 % Platt scaling, outer-train only
            [~, post] = predict(mdlP, Xte(:, sel));
            pPos = post(:, 1);
        end
    catch
        pPos = [];
    end
    T.posterior = toc(t);
end

% --- random-subset control (E7): same size as the GA subset, same pool
rp = [];
if strcmp(job.config, 'F') && strcmp(job.protocol, 'PG') && job.lcFrac == 1 ...
        && job.permIdx == 0 && job.variant == 0
    rpIdx = pool(randperm(rs, numel(pool), numel(sel)));
    [mR, bR] = svm_tune_fit(Xtr(:, rpIdx), ytr, innerFold, cfg, nCls);
    rp.yPred = predict(mR, Xte(:, rpIdx)); rp.svm = bR; rp.nSel = numel(rpIdx);
end

% --- optional: keep the fitted model for the latency module (E11)
if job.saveModel
    model = struct('mu', mu, 'sd', sd, 'sel', sel, 'mdl', compact(mdl), ...
                   'task', nCls, 'config', job.config, 'id', job.id);
    save(fullfile(cfg.modelDir, [job.id '.mat']), 'model', '-v7');
end

% --- collect
res = struct();
res.id = job.id; res.task = nCls; res.config = job.config; res.protocol = job.protocol;
res.rep = job.rep; res.fold = job.fold; res.lcFrac = job.lcFrac;
res.permIdx = job.permIdx; res.variant = job.variant; res.seed = seed;
res.teIdx = te; res.yTrue = yte; res.yPred = yPred; res.score = single(score); res.pPos = pPos;
res.nTrain = numel(tr); res.nTest = numel(te);
res.nTrainGroups = numel(unique(grp(tr))); res.nTestGroups = numel(unique(grp(te)));
res.nPool1 = numel(k1); res.nPool2 = numel(k2); res.nPool = numel(pool); res.nSel = numel(sel);
res.sel = uint16(sel);
res.selDarkNet = sum(D.bb(sel) == 1); res.selDenseNet = sum(D.bb(sel) == 2);
res.ga = ga; res.svm = best; res.cvTab = cvTab; res.rp = rp;
if job.rep == 1 && job.lcFrac == 1 && job.permIdx == 0 && job.variant == 0 && strcmp(job.protocol, 'PG')
    res.relieffW1 = w1; res.relieffW2 = w2; res.relieffWG = wG;
end
T.total = toc(tJob);
res.T = T;
[res.peakMB, res.curMB] = peak_mem_mb();
end
