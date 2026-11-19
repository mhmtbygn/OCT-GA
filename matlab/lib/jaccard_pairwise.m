function J = jaccard_pairwise(sets)
%JACCARD_PAIRWISE  All pairwise Jaccard indices between index sets (cell).
n = numel(sets); J = nan(n * (n - 1) / 2, 1); r = 0;
for i = 1:n - 1
    for j = i + 1:n
        r = r + 1;
        a = sets{i}; b = sets{j};
        u = numel(union(a, b));
        if u == 0, J(r) = NaN; else, J(r) = numel(intersect(a, b)) / u; end
    end
end
end
