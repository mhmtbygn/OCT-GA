function progress_print(msg, nTotal)
%PROGRESS_PRINT  Progress/ETA printer for parallel.pool.DataQueue.
%   progress_print('', N) resets the counter; progress_print(msg) prints.
persistent n N t0
if nargin == 2
    n = 0; N = nTotal; t0 = tic; return
end
n = n + 1; el = toc(t0); eta = el / n * (N - n);
fprintf('[%5d/%5d] %s | elapsed %s | ETA %s\n', n, N, msg, hms(el), hms(eta));
end

function s = hms(x)
s = sprintf('%02d:%02d:%02d', floor(x / 3600), floor(mod(x, 3600) / 60), floor(mod(x, 60)));
end
