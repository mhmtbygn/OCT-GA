function [lo, hi] = wilson_ci(k, n, z)
%WILSON_CI  Wilson score interval for a binomial proportion k/n.
if nargin < 3, z = 1.959964; end
phat = k ./ n;
den = 1 + z.^2 ./ n;
c = (phat + z.^2 ./ (2 * n)) ./ den;
h = z .* sqrt(phat .* (1 - phat) ./ n + z.^2 ./ (4 * n.^2)) ./ den;
lo = c - h; hi = c + h;
lo(n == 0) = NaN; hi(n == 0) = NaN;
end
