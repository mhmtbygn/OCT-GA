function fold = strat_group_kfold(y, g, K, seed)
%STRAT_GROUP_KFOLD  Stratified group K-fold assignment.
%   fold = strat_group_kfold(y, g, K, seed) assigns every observation to
%   one of K folds so that all observations sharing a group id g (patient)
%   fall in the same fold, while the class distribution of y is kept as
%   similar as possible across folds (greedy algorithm of scikit-learn's
%   StratifiedGroupKFold). Use g = (1:n)' for image-level stratified folds.
y = y(:); g = g(:);
[~, ~, gi] = unique(g);
nG = max(gi); C = max(y);
cnt = accumarray([gi y], 1, [nG C]);              % group x class counts
rs = RandStream('mt19937ar', 'Seed', seed);
perm = randperm(rs, nG);
s = std(cnt(perm, :), 0, 2);
[~, ord] = sort(s, 'descend');                     % stable sort
order = perm(ord);
totalC = sum(cnt, 1); totalC(totalC == 0) = 1;
foldCnt = zeros(K, C);
gFold = zeros(nG, 1);
for ii = 1:nG
    gid = order(ii);
    best = inf; bestK = 1; bestN = inf;
    for k = 1:K
        fc = foldCnt;
        fc(k, :) = fc(k, :) + cnt(gid, :);
        ev = mean(std(fc ./ totalC, 0, 1));
        nk = sum(foldCnt(k, :));
        if ev < best - 1e-12 || (abs(ev - best) <= 1e-12 && nk < bestN)
            best = ev; bestK = k; bestN = nk;
        end
    end
    gFold(gid) = bestK;
    foldCnt(bestK, :) = foldCnt(bestK, :) + cnt(gid, :);
end
fold = gFold(gi);
end
