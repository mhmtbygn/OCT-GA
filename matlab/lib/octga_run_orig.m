function res = octga_run_orig(job, D, cfg, rs)
%OCTGA_RUN_ORIG  Replica of the ORIGINAL pipeline (Kod\GA_CNN_FeatureSelection_Code.m)
%   on OCTDL, used to (i) check that the published OCTDL numbers can be
%   reproduced and (ii) quantify the optimism of the original protocol.
%   - z-score on ALL images
%   - ReliefF on ALL images ('regression' = RReliefF, the MATLAB default
%     for numeric labels, as in the original code)
%   - GA (N = 50, MR = 0.3, targetAcc = 0.95) whose fitness hold-out is fold 1
%     of the SAME image-level 10-fold partition used for evaluation
%   - cubic SVM (C = 1, auto scale, standardised), image-level 10-fold CV
nCls = job.task;
if nCls == 2, y = D.y2; else, y = D.y7; end
fv = D.foldsIL(:, job.rep);
T = struct();
t = tic; Xz = normalize(D.X, 'zscore'); T.zscore = toc(t);
c1 = find(D.bb == 1); c2 = find(D.bb == 2);
t = tic;
[k1, w1] = relieff_pos(Xz(:, c1), y, cfg.relieffK, cfg.orig.relieffMethod);
[k2, w2] = relieff_pos(Xz(:, c2), y, cfg.relieffK, cfg.orig.relieffMethod);
T.relieff = toc(t);
pool = [c1(k1), c2(k2)];
gaVal = fv == 1; gaTr = ~gaVal;
t = tic;
ga = ga_select(Xz(gaTr, pool), y(gaTr), Xz(gaVal, pool), y(gaVal), cfg.orig.ga, rs);
T.ga = toc(t);
sel = pool(ga.mask);
yPred = zeros(size(y));
t = tic;
for k = 1:max(fv)
    te = fv == k; tr = ~te;
    m = fit_svm(Xz(tr, sel), y(tr), 'poly3', 1, nCls);
    yPred(te) = predict(m, Xz(te, sel));
end
T.fit = toc(t); T.predict = 0;
res = struct();
res.id = job.id; res.task = nCls; res.config = job.config; res.protocol = 'ORIG';
res.rep = job.rep; res.fold = 0; res.lcFrac = 1; res.permIdx = 0; res.variant = 0;
res.seed = job_seed(job.id);
res.teIdx = (1:numel(y)).'; res.yTrue = y; res.yPred = yPred; res.score = []; res.pPos = [];
res.nTrain = NaN; res.nTest = numel(y); res.nTrainGroups = NaN; res.nTestGroups = NaN;
res.nPool1 = numel(k1); res.nPool2 = numel(k2); res.nPool = numel(pool); res.nSel = numel(sel);
res.sel = uint16(sel); res.selDarkNet = sum(D.bb(sel) == 1); res.selDenseNet = sum(D.bb(sel) == 2);
res.ga = ga; res.svm = struct('kernel', 'poly3', 'C', 1); res.cvTab = table(); res.rp = [];
res.relieffW1 = w1; res.relieffW2 = w2; res.relieffWG = [];
res.T = T;
end
