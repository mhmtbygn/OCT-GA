function m02_make_folds(cfg)
%M02_MAKE_FOLDS  Outer folds for every repeat:
%   foldsPG  patient-grouped, stratified on the 7-class label (main)
%   foldsIL  image-level, stratified (leakage comparison, E3)
out = fullfile(cfg.outDir, 'folds.mat');
if isfile(out), fprintf('m02: %s exists, skipped.\n', out); return, end
I = load(fullfile(cfg.featDir, 'inventory.mat'), 'y7', 'group', 'T');
n = numel(I.y7);
foldsPG = zeros(n, cfg.R); foldsIL = zeros(n, cfg.R);
for r = 1:cfg.R
    foldsPG(:, r) = strat_group_kfold(I.y7, I.group, cfg.K, cfg.seeds(r));
    foldsIL(:, r) = strat_group_kfold(I.y7, (1:n).', cfg.K, cfg.seeds(r) + 500);
    % integrity check: no patient in more than one fold
    [~, ~, gi] = unique(I.group);
    nf = accumarray(gi, foldsPG(:, r), [], @(v) numel(unique(v)));
    assert(all(nf == 1), 'm02: a patient was split across folds');
end
% per-fold counts for repeat 1 (reported in the manuscript / RR)
rows = {};
for k = 1:cfg.K
    m = foldsPG(:, 1) == k;
    c = accumarray(I.y7(m), 1, [7 1]).';
    rows(end + 1, :) = [{k, sum(m), numel(unique(I.group(m)))}, num2cell(c)]; %#ok<AGROW>
end
foldTab = cell2table(rows, 'VariableNames', [{'Fold', 'Images', 'Patients'}, cfg.classOrder7]);
save(out, 'foldsPG', 'foldsIL', 'foldTab', '-v7');
writetable(foldTab, fullfile(cfg.tabDir, 'E0_PG_fold_composition_rep1.csv'));
disp(foldTab)
end
