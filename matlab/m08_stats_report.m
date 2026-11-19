function m08_stats_report(cfg)
%M08_STATS_REPORT  Aggregate all job results into the tables and figures
%   used in the revised manuscript and the response to reviewers.
%   Outputs: cfg.tabDir\*.csv, cfg.outDir\summary_tables.xlsx,
%            cfg.figDir\*.png (600 dpi), cfg.outDir\REPORT.txt
I = load(fullfile(cfg.featDir, 'inventory.mat'));
n = numel(I.y7); grp = I.group(:);
xls = fullfile(cfg.outDir, 'summary_tables.xlsx');
if isfile(xls), delete(xls); end
rep = fopen(fullfile(cfg.outDir, 'REPORT.txt'), 'w');
cleanupObj = onCleanup(@() fclose(rep));
say = @(varargin) fprintf_both(rep, varargin{:});

%% ---- load all job results -------------------------------------------------
L = dir(fullfile(cfg.jobDir, '*.mat'));
R = cell(numel(L), 1);
for i = 1:numel(L)
    S = load(fullfile(L(i).folder, L(i).name), 'res'); R{i} = S.res;
end
nR = numel(R);
meta = table('Size', [nR 8], 'VariableTypes', {'double', 'string', 'string', 'double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'task', 'config', 'protocol', 'rep', 'fold', 'lcFrac', 'permIdx', 'variant'});
for i = 1:nR
    r = R{i};
    meta.task(i) = r.task; meta.config(i) = string(r.config); meta.protocol(i) = string(r.protocol);
    meta.rep(i) = r.rep; meta.fold(i) = r.fold; meta.lcFrac(i) = gf(r, 'lcFrac', 1);
    meta.permIdx(i) = gf(r, 'permIdx', 0); meta.variant(i) = gf(r, 'variant', 0);
end
say('OCT-GA revision results (%s)\n%d result files in %s\n\n', char(datetime('now')), nR, cfg.jobDir);
writetable(I.inv7, xls, 'Sheet', 'E0_inventory7'); writetable(I.inv2, xls, 'Sheet', 'E0_inventory2');
Fo = load(fullfile(cfg.outDir, 'folds.mat'), 'foldTab'); writetable(Fo.foldTab, xls, 'Sheet', 'E0_folds_rep1');
say('E0  OCTDL: %d images, %d patients (patient_id), %d exact-duplicate images, %d multi-disease patients\n', ...
    n, numel(unique(I.T.patient_id)), height(I.dupTab), numel(I.multiDis));

mainConfs = [cfg.configs, {'RP', 'FT'}];
YPs = struct(); PPs = struct(); Tmain = cell(1, 7); FR = table();

%% ---- E2/E4/E5: main patient-grouped results ---------------------------------
for task = cfg.tasks
    y = yof(I, task); [names, scale, bnames] = mnames(task);
    rows = {};
    for c = 1:numel(mainConfs)
        conf = mainConfs{c}; src = conf; useRP = strcmp(conf, 'RP');
        if useRP, src = 'F'; end
        reps = 1:cfg.R; if strcmp(conf, 'FT'), reps = 1:cfg.ft.reps; end
        V = []; YP = []; PP = [];
        for r = reps
            ii = pick(meta, task, src, "PG", r, 1, 0, 0);
            if isempty(ii), continue, end
            [yp, pp] = pooled(R, ii, n, useRP);
            if any(isnan(yp))
                warning('m08: %s T%d r%d incomplete (%d/%d images) - skipped', conf, task, r, sum(~isnan(yp)), n); continue
            end
            if any(isnan(pp)), ppU = []; else, ppU = pp; end
            V = [V; mvec(compute_metrics(y, yp, task, ppU), task)]; %#ok<AGROW>
            YP = [YP, yp]; PP = [PP, pp]; %#ok<AGROW>
        end
        if isempty(V), continue, end
        YPs.(sprintf('T%d_%s', task, conf)) = YP; PPs.(sprintf('T%d_%s', task, conf)) = PP;
        ci = cluster_boot_ci(@(idx) boot_fn(y, YP, idx, task), grp, cfg.boot.B, 11);
        rows(end + 1, :) = summarize(conf, V, ci, scale); %#ok<AGROW>
    end
    vn = [{'Config', 'nReps'}, strcat(names, '_mean'), strcat(names, '_sd'), strcat(bnames, '_lo'), strcat(bnames, '_hi')];
    Tm = cell2table(rows, 'VariableNames', vn);
    Tm = Tm(:, [1 2 reshape([3:2 + numel(names); 3 + numel(names):2 + 2 * numel(names)], 1, [])  2 + 2 * numel(names) + (1:2 * numel(bnames))]);
    Tmain{task} = Tm;
    writetable(Tm, fullfile(cfg.tabDir, sprintf('E2E4E5_main_PG_T%d.csv', task)));
    writetable(Tm, xls, 'Sheet', sprintf('E2E4E5_main_T%d', task));
    say('\nE2/E4/E5  patient-grouped nested CV, task T%d (mean +- SD over repeats; 95%% patient-bootstrap CI)\n', task);
    for k = 1:height(Tm)
        say('  %-7s Acc %6.2f +- %4.2f [%6.2f, %6.2f]  BAcc %6.2f +- %4.2f  MacroF1 %6.2f +- %4.2f  MCC %5.3f\n', ...
            Tm.Config{k}, Tm.Acc_mean(k), Tm.Acc_sd(k), Tm.Acc_lo(k), Tm.Acc_hi(k), Tm.BAcc_mean(k), Tm.BAcc_sd(k), ...
            Tm.MacroF1_mean(k), Tm.MacroF1_sd(k), Tm.MCC_mean(k));
    end
end

%% ---- fold-level metrics and corrected resampled t-tests -------------------------
for i = 1:nR
    r = R{i};
    if ~(meta.protocol(i) == "PG" && meta.lcFrac(i) == 1 && meta.permIdx(i) == 0 && meta.variant(i) == 0), continue, end
    y = yof(I, r.task); yt = y(r.teIdx);
    M = compute_metrics(yt, r.yPred, r.task);
    FR = [FR; table(r.task, string(r.config), r.rep, r.fold, M.acc, M.bacc, M.macroF1, r.nTest, r.nTrain, ...
        'VariableNames', {'task', 'config', 'rep', 'fold', 'acc', 'bacc', 'macroF1', 'nTe', 'nTr'})]; %#ok<AGROW>
    if strcmp(r.config, 'F') && isfield(r, 'rp') && ~isempty(r.rp)
        M = compute_metrics(yt, r.rp.yPred, r.task);
        FR = [FR; table(r.task, "RP", r.rep, r.fold, M.acc, M.bacc, M.macroF1, r.nTest, r.nTrain, ...
            'VariableNames', FR.Properties.VariableNames)]; %#ok<AGROW>
    end
end
writetable(FR, fullfile(cfg.tabDir, 'fold_level_metrics_PG.csv'));
Tt = table();
for task = cfg.tasks
    for metric = ["acc", "bacc", "macroF1"]
        blk = table();
        for c = 1:numel(mainConfs)
            conf = string(mainConfs{c}); if conf == "F", continue, end
            Fm = FR(FR.task == task & FR.config == "F", :); Cm = FR(FR.task == task & FR.config == conf, :);
            if isempty(Cm), continue, end
            [tf, loc] = ismember(Cm.rep * 1000 + Cm.fold, Fm.rep * 1000 + Fm.fold);
            Cm = Cm(tf, :); Fm = Fm(loc(tf), :);
            d = 100 * (Fm.(metric) - Cm.(metric));
            [p, t, md, sdd, nn] = corrected_ttest(d, Fm.nTe, Fm.nTr);
            blk = [blk; table(task, metric, "F vs " + conf, nn, md, sdd, t, p, ...
                'VariableNames', {'task', 'metric', 'comparison', 'nFolds', 'meanDiff_pp', 'sdDiff_pp', 't', 'p'})]; %#ok<AGROW>
        end
        if isempty(blk), continue, end
        blk.pHolm = holm_adjust(blk.p);
        Tt = [Tt; blk]; %#ok<AGROW>
    end
end
writetable(Tt, fullfile(cfg.tabDir, 'E4_corrected_ttests.csv')); writetable(Tt, xls, 'Sheet', 'E4_ttests');
say('\nE4  corrected resampled t-tests (F minus comparator, percentage points, Holm-adjusted)\n');
for k = 1:height(Tt)
    if Tt.metric(k) ~= "bacc", continue, end
    say('  T%d %-14s BAcc diff %+6.2f pp (SD %5.2f), t = %6.2f, p = %.4f, pHolm = %.4f\n', Tt.task(k), ...
        Tt.comparison(k), Tt.meanDiff_pp(k), Tt.sdDiff_pp(k), Tt.t(k), Tt.p(k), Tt.pHolm(k));
end

%% ---- per-class metrics and confusion matrices (F, PG) ---------------------------
for task = cfg.tasks
    key = sprintf('T%d_F', task); if ~isfield(YPs, key), continue, end
    y = yof(I, task); YP = YPs.(key); nr = size(YP, 2);
    if task == 2, cn = cfg.classNames2; else, cn = cfg.classOrder7; end
    S = zeros(nr, task); P = S; F1 = S; SP = S; cmSum = zeros(task);
    for r = 1:nr
        M = compute_metrics(y, YP(:, r), task);
        S(r, :) = M.sensC; P(r, :) = M.precC; F1(r, :) = M.f1C; SP(r, :) = M.specC; cmSum = cmSum + M.cm;
        if r == 1, cm1 = M.cm; end
    end
    ci = cluster_boot_ci(@(idx) mean(cell2mat(arrayfun(@(r) getfield(compute_metrics(y(idx), YP(idx, r), task), 'sensC'), ...
        (1:nr).', 'UniformOutput', false)), 1), grp, cfg.boot.B, 12);
    [wl, wh] = wilson_ci(diag(cm1), sum(cm1, 2));
    Tc = table(string(cn(:)), sum(cm1, 2), 100 * mean(S, 1).', 100 * std(S, 0, 1).', 100 * ci(1, :).', 100 * ci(2, :).', ...
        100 * wl, 100 * wh, 100 * mean(SP, 1).', 100 * mean(P, 1, 'omitnan').', 100 * mean(F1, 1, 'omitnan').', ...
        'VariableNames', {'Class', 'Support', 'Sens_mean', 'Sens_sd', 'Sens_bootLo', 'Sens_bootHi', ...
        'Sens_rep1_WilsonLo', 'Sens_rep1_WilsonHi', 'Spec_mean', 'Prec_mean', 'F1_mean'});
    writetable(Tc, fullfile(cfg.tabDir, sprintf('E2_perclass_F_PG_T%d.csv', task)));
    writetable(Tc, xls, 'Sheet', sprintf('E2_perclass_T%d', task));
    writematrix(cm1, fullfile(cfg.tabDir, sprintf('E2_confusion_F_PG_T%d_rep1.csv', task)));
    writematrix(cmSum, fullfile(cfg.tabDir, sprintf('E2_confusion_F_PG_T%d_sumreps.csv', task)));
    fig = figure('Visible', 'off', 'Position', [100 100 560 480]);
    cc = confusionchart(cm1, cn); cc.Title = sprintf('Proposed (F), patient-grouped CV, repeat 1 (T%d)', task);
    save_fig(fig, cfg, sprintf('E2_confusion_F_PG_T%d_rep1', task));
end

%% ---- E3: protocol comparison --------------------------------------------------
pub = [NaN 96.12 NaN NaN NaN NaN 88.86];
T3 = table();
for task = cfg.tasks
    y = yof(I, task);
    for prot = ["PG", "IL", "ORIG"]
        V = [];
        for r = 1:cfg.R
            ii = pick(meta, task, "F", prot, r, 1, 0, 0); if isempty(ii), continue, end
            yp = pooled(R, ii, n, false); if any(isnan(yp)), continue, end
            M = compute_metrics(y, yp, task); V = [V; 100 * [M.acc M.bacc M.macroF1]]; %#ok<AGROW>
        end
        if isempty(V), continue, end
        T3 = [T3; table(task, prot, size(V, 1), mean(V(:, 1)), std(V(:, 1)), mean(V(:, 2)), std(V(:, 2)), ...
            mean(V(:, 3)), std(V(:, 3)), pub(task), 'VariableNames', {'task', 'protocol', 'nReps', 'Acc_mean', 'Acc_sd', ...
            'BAcc_mean', 'BAcc_sd', 'MacroF1_mean', 'MacroF1_sd', 'PublishedAcc'})]; %#ok<AGROW>
    end
end
writetable(T3, fullfile(cfg.tabDir, 'E3_protocol_comparison.csv')); writetable(T3, xls, 'Sheet', 'E3_protocols');
say('\nE3  protocol comparison (proposed pipeline F): PG = patient-grouped nested, IL = image-level nested, ORIG = original code\n');
for k = 1:height(T3)
    say('  T%d %-5s Acc %6.2f +- %4.2f  BAcc %6.2f +- %4.2f  MacroF1 %6.2f +- %4.2f  (published Acc %.2f)\n', T3.task(k), ...
        T3.protocol(k), T3.Acc_mean(k), T3.Acc_sd(k), T3.BAcc_mean(k), T3.BAcc_sd(k), T3.MacroF1_mean(k), T3.MacroF1_sd(k), T3.PublishedAcc(k));
end

%% ---- E6: selection statistics and stability ------------------------------------
T6 = table(); TW = table();
for task = cfg.tasks
    for conf = string(cfg.gaConfigs)
        ii = find(meta.task == task & meta.config == conf & meta.protocol == "PG" & meta.lcFrac == 1 & meta.permIdx == 0 & meta.variant == 0);
        if isempty(ii), continue, end
        g = @(f) cellfun(@(r) double(r.(f)), R(ii));
        sets = cellfun(@(r) double(r.sel(:)).', R(ii), 'UniformOutput', false);
        J = jaccard_pairwise(sets);
        nGen = cellfun(@(r) r.ga.nGen, R(ii)); holdAcc = cellfun(@(r) 1 - r.ga.bestFit, R(ii));
        stops = cellfun(@(r) string(r.ga.stop), R(ii));
        allSel = [sets{:}]; freq = accumarray(allSel(:), 1, [4944 1]) / numel(ii);
        T6 = [T6; table(task, conf, numel(ii), mean(g('nPool1')), std(g('nPool1')), mean(g('nPool2')), std(g('nPool2')), ...
            mean(g('nPool')), std(g('nPool')), mean(g('nSel')), std(g('nSel')), min(g('nSel')), max(g('nSel')), ...
            mean(g('selDarkNet')), mean(g('selDenseNet')), 100 * mean(g('selDarkNet') ./ g('nSel')), mean(J), std(J), ...
            sum(freq >= 0.9), mean(nGen), std(nGen), sum(stops == "stall"), sum(stops == "maxIter"), 100 * mean(holdAcc), ...
            'VariableNames', {'task', 'config', 'nFolds', 'ReliefF_DarkNet_mean', 'ReliefF_DarkNet_sd', ...
            'ReliefF_DenseNet_mean', 'ReliefF_DenseNet_sd', 'Pool_mean', 'Pool_sd', 'GA_sel_mean', 'GA_sel_sd', ...
            'GA_sel_min', 'GA_sel_max', 'Sel_from_DarkNet', 'Sel_from_DenseNet', 'Pct_from_DarkNet', ...
            'Jaccard_mean', 'Jaccard_sd', 'nFeat_selected_in_90pct_folds', 'GA_generations_mean', ...
            'GA_generations_sd', 'GA_stop_stall', 'GA_stop_maxIter', 'GA_holdout_acc_mean'})]; %#ok<AGROW>
        if conf == "F"
            ii1 = ii(meta.rep(ii) == 1);
            for b = 1:2
                W = cell2mat(cellfun(@(r) double(r.(sprintf('relieffW%d', b))), R(ii1), 'UniformOutput', false));
                if isempty(W), continue, end
                mx = max(abs(W), [], 2);
                fr = nan(size(W, 1), 1);
                for q = 1:size(W, 1)
                    wq = W(q, W(q, :) > 0); fr(q) = mean(wq <= 0.05 * mx(q));
                end
                TW = [TW; table(task, b, size(W, 2), 100 * mean(W(:) > 0), median(W(:)), ...
                    100 * mean(abs(W) <= 0.05 * mx, 'all'), 100 * mean(fr, 'omitnan'), ...
                    'VariableNames', {'task', 'backbone', 'nFeatures', 'PctPositive', 'MedianW', ...
                    'PctNearZero_abs_le_5pctMax', 'PctOfRetained_le_5pctMax'})]; %#ok<AGROW>
                if task == cfg.tasks(1)
                    fig = figure('Visible', 'off', 'Position', [100 100 560 360]);
                    histogram(W(1, :), 80); xline(0, 'r-', 'W = 0');
                    xlabel('ReliefF weight'); ylabel('Number of features');
                    title(sprintf('%s, fold 1, repeat 1 (T%d)', ternary(b == 1, 'DarkNet53', 'DenseNet201'), task));
                    save_fig(fig, cfg, sprintf('E6_relieff_weights_b%d_T%d', b, task));
                end
            end
            fig = figure('Visible', 'off', 'Position', [100 100 560 360]); hold on
            for j = ii1(:).', plot(100 * (1 - R{j}.ga.curve), 'LineWidth', 0.8); end
            xlabel('Generation'); ylabel('Best k-NN hold-out accuracy (%)');
            title(sprintf('GA convergence, proposed pipeline, repeat 1 (T%d)', task)); box on
            save_fig(fig, cfg, sprintf('E6_GA_convergence_T%d', task));
        end
    end
end
writetable(T6, fullfile(cfg.tabDir, 'E6_selection_stability.csv')); writetable(T6, xls, 'Sheet', 'E6_selection');
if ~isempty(TW), writetable(TW, fullfile(cfg.tabDir, 'E6_relieff_weights.csv')); writetable(TW, xls, 'Sheet', 'E6_relieff'); end
say('\nE6  selection statistics (mean over outer folds)\n');
for k = 1:height(T6)
    say('  T%d %-2s ReliefF %6.1f + %6.1f -> pool %6.1f; GA keeps %6.1f +- %5.1f (%.0f%% DarkNet53); Jaccard %.3f +- %.3f; GA gens %.0f\n', ...
        T6.task(k), T6.config(k), T6.ReliefF_DarkNet_mean(k), T6.ReliefF_DenseNet_mean(k), T6.Pool_mean(k), ...
        T6.GA_sel_mean(k), T6.GA_sel_sd(k), T6.Pct_from_DarkNet(k), T6.Jaccard_mean(k), T6.Jaccard_sd(k), T6.GA_generations_mean(k));
end

%% ---- E7: complementarity of the two backbones ------------------------------------
T7c = table();
for task = cfg.tasks
    ka = sprintf('T%d_A', task); kb = sprintf('T%d_B', task);
    if ~isfield(YPs, ka) || ~isfield(YPs, kb), continue, end
    y = yof(I, task); A = YPs.(ka); B = YPs.(kb); nr = min(size(A, 2), size(B, 2));
    for r = 1:nr
        ca = A(:, r) == y; cb = B(:, r) == y;
        n11 = sum(ca & cb); n10 = sum(ca & ~cb); n01 = sum(~ca & cb); n00 = sum(~ca & ~cb);
        Q = (n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10);
        T7c = [T7c; table(task, r, n11, n10, n01, n00, 100 * n00 / n, 100 * (n10 + n01) / n, Q, mcnemar_exact(n10, n01), ...
            'VariableNames', {'task', 'rep', 'bothCorrect', 'onlyDarkNet', 'onlyDenseNet', 'bothWrong', ...
            'DoubleFault_pct', 'Disagreement_pct', 'Qstatistic', 'McNemar_p'})]; %#ok<AGROW>
    end
end
writetable(T7c, fullfile(cfg.tabDir, 'E7_backbone_error_overlap.csv')); writetable(T7c, xls, 'Sheet', 'E7_complementarity');
say('\nE7  error overlap DarkNet53-only (A) vs DenseNet201-only (B), mean over repeats\n');
for task = cfg.tasks
    m = T7c.task == task; if ~any(m), continue, end
    say('  T%d double-fault %.2f%%, disagreement %.2f%%, Q = %.3f, McNemar p (rep 1) = %.4f\n', task, ...
        mean(T7c.DoubleFault_pct(m)), mean(T7c.Disagreement_pct(m)), mean(T7c.Qstatistic(m)), T7c.McNemar_p(find(m, 1)));
end

%% ---- E8: label-permutation test (binary task) --------------------------------------
if isfield(YPs, 'T2_F')
    y = yof(I, 2); obs = compute_metrics(y, YPs.T2_F(:, 1), 2);
    nullB = nan(cfg.perm.B, 2);
    for p = 1:cfg.perm.B
        ii = pick(meta, 2, "F", "PG", 1, 1, p, 0); if isempty(ii), continue, end
        yt = nan(n, 1); yp = nan(n, 1);
        for j = ii(:).', yt(R{j}.teIdx) = R{j}.yTrue; yp(R{j}.teIdx) = R{j}.yPred; end
        if any(isnan(yp)), continue, end
        M = compute_metrics(yt, yp, 2); nullB(p, :) = [M.acc M.bacc];
    end
    ok = ~isnan(nullB(:, 2)); nb = sum(ok);
    if nb > 0
        pB = (1 + sum(nullB(ok, 2) >= obs.bacc)) / (nb + 1);
        T8 = table(nb, 100 * obs.bacc, 100 * mean(nullB(ok, 2)), 100 * std(nullB(ok, 2)), 100 * max(nullB(ok, 2)), pB, ...
            'VariableNames', {'nPermutations', 'Observed_BAcc', 'Null_BAcc_mean', 'Null_BAcc_sd', 'Null_BAcc_max', 'p_value'});
        writetable(T8, fullfile(cfg.tabDir, 'E8_permutation_test.csv')); writetable(T8, xls, 'Sheet', 'E8_permutation');
        say('\nE8  permutation test (T2, repeat 1): observed BAcc %.2f%%, null %.2f +- %.2f%% (max %.2f%%), p = %.4f (B = %d)\n', ...
            T8.Observed_BAcc, T8.Null_BAcc_mean, T8.Null_BAcc_sd, T8.Null_BAcc_max, pB, nb);
        fig = figure('Visible', 'off', 'Position', [100 100 560 360]);
        histogram(100 * nullB(ok, 2), 20); xline(100 * obs.bacc, 'r-', 'observed', 'LineWidth', 1.5);
        xlabel('Balanced accuracy (%)'); ylabel('Permutations'); title('Label-permutation null distribution (T2)');
        save_fig(fig, cfg, 'E8_permutation_null');
    end
end

%% ---- E9: discrimination and calibration (binary task) ----------------------------
if isfield(PPs, 'T2_F') && ~all(isnan(PPs.T2_F(:)))
    y = yof(I, 2); pp = PPs.T2_F(:, 1); pos = y == 1;
    fig = figure('Visible', 'off', 'Position', [100 100 900 380]);
    subplot(1, 2, 1); [fx, fy, ~, auc] = perfcurve(double(pos), pp, 1);
    plot(fx, fy, 'LineWidth', 1.2); hold on; plot([0 1], [0 1], 'k:');
    xlabel('1 - specificity'); ylabel('Sensitivity'); title(sprintf('ROC, repeat 1 (AUC = %.3f)', auc)); axis square; box on
    subplot(1, 2, 2); edges = linspace(0, 1, 11); b = discretize(pp, edges);
    mp = accumarray(b, pp, [10 1], @mean, NaN); mo = accumarray(b, double(pos), [10 1], @mean, NaN);
    plot(mp, mo, 'o-', 'LineWidth', 1.2); hold on; plot([0 1], [0 1], 'k:');
    xlabel('Predicted probability (Diseased)'); ylabel('Observed proportion'); title('Reliability, repeat 1'); axis square; box on
    save_fig(fig, cfg, 'E9_ROC_reliability_T2');
    Tm = Tmain{2}; k = find(strcmp(Tm.Config, 'F'));
    say('\nE9  T2 discrimination/calibration (F, PG): AUC %.2f +- %.2f, AP %.2f +- %.2f, Brier %.4f, ECE %.4f, Sens@Spec90 %.2f, Sens@Spec95 %.2f\n', ...
        Tm.AUC_mean(k), Tm.AUC_sd(k), Tm.AP_mean(k), Tm.AP_sd(k), Tm.Brier_mean(k), Tm.ECE_mean(k), Tm.SensAtSpec90_mean(k), Tm.SensAtSpec95_mean(k));
end

%% ---- E10: learning curve ------------------------------------------------------------
T10 = table();
for task = cfg.tasks
    y = yof(I, task);
    for f = [cfg.lc.fracs, 1]
        V = []; nTr = [];
        for r = 1:cfg.lc.R
            ii = pick(meta, task, "F", "PG", r, f, 0, 0); if isempty(ii), continue, end
            yp = pooled(R, ii, n, false); if any(isnan(yp)), continue, end
            M = compute_metrics(y, yp, task); V = [V; 100 * [M.acc M.bacc M.macroF1]]; %#ok<AGROW>
            nTr = [nTr; mean(cellfun(@(q) q.nTrain, R(ii)))]; %#ok<AGROW>
        end
        if isempty(V), continue, end
        T10 = [T10; table(task, f, size(V, 1), mean(nTr), mean(V(:, 1)), std(V(:, 1)), mean(V(:, 2)), std(V(:, 2)), ...
            mean(V(:, 3)), std(V(:, 3)), 'VariableNames', {'task', 'trainFraction', 'nReps', 'meanTrainImages', ...
            'Acc_mean', 'Acc_sd', 'BAcc_mean', 'BAcc_sd', 'MacroF1_mean', 'MacroF1_sd'})]; %#ok<AGROW>
    end
end
if ~isempty(T10)
    writetable(T10, fullfile(cfg.tabDir, 'E10_learning_curve.csv')); writetable(T10, xls, 'Sheet', 'E10_learning_curve');
    fig = figure('Visible', 'off', 'Position', [100 100 560 380]); hold on
    for task = cfg.tasks
        m = T10.task == task;
        errorbar(T10.meanTrainImages(m), T10.BAcc_mean(m), T10.BAcc_sd(m), 'o-', 'LineWidth', 1.2, 'DisplayName', sprintf('T%d', task));
    end
    xlabel('Training images per outer fold'); ylabel('Balanced accuracy (%)'); legend('Location', 'southeast'); box on
    title('Learning curve (training patients subsampled)');
    save_fig(fig, cfg, 'E10_learning_curve');
    say('\nE10 learning curve (BAcc mean +- SD)\n');
    for k = 1:height(T10)
        say('  T%d frac %.2f (%.0f train images): BAcc %.2f +- %.2f\n', T10.task(k), T10.trainFraction(k), T10.meanTrainImages(k), T10.BAcc_mean(k), T10.BAcc_sd(k));
    end
end

%% ---- E11: computational cost ---------------------------------------------------------
T11 = table();
for task = cfg.tasks
    for c = 1:numel(mainConfs)
        conf = string(mainConfs{c}); if conf == "RP", continue, end
        ii = find(meta.task == task & meta.config == conf & meta.protocol == "PG" & meta.lcFrac == 1 & meta.permIdx == 0 & meta.variant == 0);
        if isempty(ii), continue, end
        tf = @(f) cellfun(@(r) gf(r.T, f, NaN), R(ii));
        T11 = [T11; table(task, conf, numel(ii), median(tf('relieff')), median(tf('ga')), median(tf('fit')), median(tf('train')), ...
            1000 * median(tf('predictPerImage')), median(tf('latencyMed')) * 1000, median(tf('total')), ...
            max(cellfun(@(r) gf(r, 'peakMB', NaN), R(ii))), ...
            'VariableNames', {'task', 'config', 'nFolds', 'ReliefF_s', 'GA_s', 'ClassifierTuneFit_s', 'CNNtrain_s', ...
            'PredictPerImage_ms', 'CNNlatency_ms', 'TotalPerOuterFold_s', 'PeakWorkerRAM_MB'})]; %#ok<AGROW>
    end
end
writetable(T11, fullfile(cfg.tabDir, 'E11_training_cost_per_fold.csv')); writetable(T11, xls, 'Sheet', 'E11_cost');
Fx = load(fullfile(cfg.featDir, 'features_octdl.mat'), 'info', 'hw');
Te = table(["DarkNet53"; "DenseNet201"], 1000 * [Fx.info.darknet53.secPerImage; Fx.info.densenet201.secPerImage], ...
    [Fx.info.darknet53.seconds; Fx.info.densenet201.seconds], string({Fx.info.darknet53.env; Fx.info.densenet201.env}), ...
    'VariableNames', {'Network', 'BatchExtraction_ms_per_image', 'Total_s_2064_images', 'Device'});
writetable(Te, fullfile(cfg.tabDir, 'E11_feature_extraction.csv')); writetable(Te, xls, 'Sheet', 'E11_extraction');
say('\nE11 feature extraction (batch %d, two layers per network): DarkNet53 %.1f ms/image, DenseNet201 %.1f ms/image (%s)\n', ...
    cfg.feat.mb, Te.BatchExtraction_ms_per_image(1), Te.BatchExtraction_ms_per_image(2), Te.Device(1));
say('    hardware: %s | %d physical cores | %.0f GB RAM | GPU: %s | MATLAB %s\n', Fx.hw.cpuName, Fx.hw.physicalCores, Fx.hw.ramGB, Fx.hw.gpu, Fx.hw.matlab);
lf = fullfile(cfg.outDir, 'E11_latency.mat');
if isfile(lf)
    Lt = load(lf); writetable(Lt.lat, xls, 'Sheet', 'E11_latency');
    say('    end-to-end single-image latency (median): %.1f ms; peak RAM %.0f MB\n', Lt.lat.MedianMs(end), Lt.peakMB);
end
for k = 1:height(T11)
    say('  T%d %-7s per outer fold: ReliefF %7.1f s, GA %8.1f s, classifier %7.1f s, CNN train %7.1f s, total %8.1f s\n', ...
        T11.task(k), T11.config(k), T11.ReliefF_s(k), T11.GA_s(k), T11.ClassifierTuneFit_s(k), T11.CNNtrain_s(k), T11.TotalPerOuterFold_s(k));
end

%% ---- E12: GA hyper-parameter sensitivity (T2, repeat 1) -------------------------------
T12 = table(); y = yof(I, 2);
for v = 0:numel(cfg.e12.variants)
    ii = pick(meta, 2, "F", "PG", 1, 1, 0, v); if isempty(ii), continue, end
    yp = pooled(R, ii, n, false); if any(isnan(yp)), continue, end
    M = compute_metrics(y, yp, 2);
    if v == 0, tag = "base (MR 0.3, N 100)"; else, tag = string(cfg.e12.variants{v}.tag); end
    T12 = [T12; table(tag, 100 * M.acc, 100 * M.bacc, 100 * M.macroF1, mean(cellfun(@(r) r.nSel, R(ii))), ...
        mean(cellfun(@(r) r.ga.nGen, R(ii))), 100 * mean(cellfun(@(r) 1 - r.ga.bestFit, R(ii))), ...
        'VariableNames', {'variant', 'Acc', 'BAcc', 'MacroF1', 'nSel_mean', 'GA_generations_mean', 'GA_holdout_acc'})]; %#ok<AGROW>
end
if ~isempty(T12)
    writetable(T12, fullfile(cfg.tabDir, 'E12_GA_sensitivity.csv')); writetable(T12, xls, 'Sheet', 'E12_GA_sensitivity');
    say('\nE12 GA sensitivity (T2, repeat 1)\n');
    for k = 1:height(T12)
        say('  %-22s Acc %6.2f  BAcc %6.2f  nSel %6.1f  gens %5.1f\n', T12.variant(k), T12.Acc(k), T12.BAcc(k), T12.nSel_mean(k), T12.GA_generations_mean(k));
    end
end

%% ---- E13: frozen Kermany -> OCTDL transfer -------------------------------------------
ii = find(meta.protocol == "TRANSFER");
if ~isempty(ii)
    y = yof(I, 2); aligned = ismember(I.y7(:), [1 2 4]);     % AMD, DME, Normal
    T13 = table();
    for subset = ["all OCTDL (7 diagnoses)", "aligned (AMD, DME, Normal)"]
        if startsWith(subset, "all"), m = true(n, 1); else, m = aligned; end
        V = [];
        for j = ii(:).'
            M = compute_metrics(y(m), R{j}.yPred(m), 2, R{j}.pPos(m));
            V = [V; 100 * [M.acc M.bacc M.sensitivity M.specificity M.auc]]; %#ok<AGROW>
        end
        ys = y(m); yps = R{ii(1)}.yPred(m);
        ci = cluster_boot_ci(@(idx) 100 * bacc_only(ys(idx), yps(idx)), grp(m), cfg.boot.B, 13);
        T13 = [T13; table(subset, sum(m), size(V, 1), mean(V(:, 1)), std(V(:, 1)), mean(V(:, 2)), std(V(:, 2)), ci(1), ci(2), ...
            mean(V(:, 3)), mean(V(:, 4)), mean(V(:, 5)), 'VariableNames', {'subset', 'nImages', 'nSeeds', 'Acc_mean', 'Acc_sd', ...
            'BAcc_mean', 'BAcc_sd', 'BAcc_seed1_lo', 'BAcc_seed1_hi', 'Sens_mean', 'Spec_mean', 'AUC_mean'})]; %#ok<AGROW>
    end
    writetable(T13, fullfile(cfg.tabDir, 'E13_transfer_Kermany_to_OCTDL.csv')); writetable(T13, xls, 'Sheet', 'E13_transfer');
    say('\nE13 frozen Kermany -> OCTDL transfer (no refitting on OCTDL)\n');
    for k = 1:height(T13)
        say('  %-28s n = %4d  Acc %6.2f +- %4.2f  BAcc %6.2f +- %4.2f  Sens %6.2f  Spec %6.2f  AUC %6.2f\n', T13.subset(k), T13.nImages(k), ...
            T13.Acc_mean(k), T13.Acc_sd(k), T13.BAcc_mean(k), T13.BAcc_sd(k), T13.Sens_mean(k), T13.Spec_mean(k), T13.AUC_mean(k));
    end
end
say('\nDone. Tables: %s\nFigures: %s\n', cfg.tabDir, cfg.figDir);
end

%% ===================================================================================
function fprintf_both(fid, varargin)
fprintf(varargin{:}); fprintf(fid, varargin{:});
end

function v = gf(s, f, d)
if isstruct(s) && isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end

function y = yof(I, task)
if task == 2, y = I.y2(:); else, y = I.y7(:); end
end

function ii = pick(meta, task, conf, prot, rep, lc, perm, var)
ii = find(meta.task == task & meta.config == string(conf) & meta.protocol == string(prot) & meta.rep == rep & ...
    abs(meta.lcFrac - lc) < 1e-9 & meta.permIdx == perm & meta.variant == var);
end

function [yp, pp] = pooled(R, ii, n, useRP)
yp = nan(n, 1); pp = nan(n, 1);
for j = ii(:).'
    r = R{j};
    if useRP
        if ~isfield(r, 'rp') || isempty(r.rp), continue, end
        yp(r.teIdx) = r.rp.yPred;
    else
        yp(r.teIdx) = r.yPred;
        if isfield(r, 'pPos') && ~isempty(r.pPos), pp(r.teIdx) = r.pPos; end
    end
end
end

function [names, scale, bnames] = mnames(task)
if task == 2
    names = {'Acc', 'BAcc', 'MacroF1', 'MCC', 'Kappa', 'Sens', 'Spec', 'Prec', 'F1', 'AUC', 'AP', 'Brier', 'ECE', 'SensAtSpec90', 'SensAtSpec95'};
    scale = [100 100 100 1 1 100 100 100 100 100 100 1 1 100 100];
    bnames = {'Acc', 'BAcc', 'MacroF1', 'MCC', 'Sens', 'Spec'};
else
    names = {'Acc', 'BAcc', 'MacroF1', 'MCC', 'Kappa'};
    scale = [100 100 100 1 1];
    bnames = {'Acc', 'BAcc', 'MacroF1', 'MCC'};
end
end

function v = mvec(M, task)
if task == 2
    v = [M.acc M.bacc M.macroF1 M.mcc M.kappa M.sensitivity M.specificity M.precision M.f1 M.auc M.ap M.brier M.ece M.sensAtSpec90 M.sensAtSpec95];
else
    v = [M.acc M.bacc M.macroF1 M.mcc M.kappa];
end
end

function v = boot_fn(y, YP, idx, task)
nr = size(YP, 2); vv = [];
for r = 1:nr
    M = compute_metrics(y(idx), YP(idx, r), task);
    if task == 2, vv(r, :) = [M.acc M.bacc M.macroF1 M.mcc M.sensitivity M.specificity]; %#ok<AGROW>
    else, vv(r, :) = [M.acc M.bacc M.macroF1 M.mcc]; end %#ok<AGROW>
end
v = mean(vv, 1);
end

function row = summarize(conf, V, ci, scale)
bscale = scale(1:size(ci, 2)); if numel(scale) == 15, bscale = [100 100 100 1 100 100]; end
row = [{conf, size(V, 1)}, num2cell(mean(V, 1, 'omitnan') .* scale), num2cell(std(V, 0, 1, 'omitnan') .* scale), ...
       num2cell(ci(1, :) .* bscale), num2cell(ci(2, :) .* bscale)];
end

function b = bacc_only(y, yp)
b = balanced_acc(y, yp, 2);
end

function out = ternary(c, a, b)
if c, out = a; else, out = b; end
end

function save_fig(fig, cfg, name)
exportgraphics(fig, fullfile(cfg.figDir, [name '.png']), 'Resolution', 600);
close(fig);
end
