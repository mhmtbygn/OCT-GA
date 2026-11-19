function err = knn_holdout_err(Xt, yt, Xv, yv, mask, k, nCls, rs)
%KNN_HOLDOUT_ERR  Hold-out error of a k-NN classifier (Euclidean, equal
%   weights, ties broken towards the smallest class index, as fitcknn with
%   'NumNeighbors',k) on the feature subset given by the logical mask.
%   Vectorised distance computation; about 10-50x faster than fitcknn.
if ~any(mask)                       % as in the original GA_Fitness
    mask = false(size(mask));
    mask(randi(rs, numel(mask))) = true;
end
A = Xt(:, mask); B = Xv(:, mask);
D2 = sum(B.^2, 2) + sum(A.^2, 2).' - 2 * (B * A.');
[~, nn] = mink(D2, k, 2);
lab = reshape(yt(nn), size(nn));
cnt = zeros(size(lab, 1), nCls);
for c = 1:nCls
    cnt(:, c) = sum(lab == c, 2);
end
[~, pred] = max(cnt, [], 2);
err = mean(pred ~= yv(:));
end
