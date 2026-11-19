function m07_latency(cfg)
%M07_LATENCY  E11: end-to-end single-image inference latency of the proposed
%   pipeline (read + preprocess + both CNN forward passes + z-score +
%   selection + SVM), median over cfg.lat.nImages OCTDL images, plus peak
%   memory and hardware. Uses the model saved by job T2_F_PG_r1_k1.
%   Can be run alone on another computer (e.g. the laptop named in the
%   manuscript) after copying Revision_Features and the models folder.
dev = 'auto'; if isfield(cfg, 'lat') && isfield(cfg.lat, 'device'), dev = cfg.lat.device; end
out = fullfile(cfg.outDir, 'E11_latency.mat');
if ~strcmp(dev, 'auto'), out = fullfile(cfg.outDir, ['E11_latency_' upper(dev) '.mat']); end
mf = fullfile(cfg.modelDir, 'T2_F_PG_r1_k1.mat');
if ~isfile(mf), warning('m07: %s not found; run m04 first.', mf); return, end
M = load(mf, 'model'); model = M.model;
I = load(fullfile(cfg.featDir, 'inventory.mat'), 'T');
Fx = load(fullfile(cfg.featDir, 'features_octdl.mat'), 'X');
spec = {'darknet53', 'conv53', 'avg1'; 'densenet201', 'fc1000', 'avg_pool'};
useGPU = canUseGPU && ~strcmp(dev, 'cpu');
if useGPU, envArg = {'ExecutionEnvironment', 'gpu'}; else, envArg = {'ExecutionEnvironment', 'cpu'}; end
nets = cell(1, 2); kinds = cell(1, 2);
for b = 1:2
    try
        nets{b} = imagePretrainedNetwork(spec{b, 1}); kinds{b} = 'dlnetwork';
    catch
        if b == 1, nets{b} = darknet53(); else, nets{b} = densenet201(); end
        kinds{b} = 'legacy';
    end
end
nL = min(cfg.lat.nImages, height(I.T));
idx = round(linspace(1, height(I.T), nL));
tRead = nan(nL, 1); tCNN = nan(nL, 2); tSel = nan(nL, 1); tSVM = nan(nL, 1); maxAbsDiff = nan(nL, 1);
for ii = 0:nL                                       % ii = 0 is a warm-up
    i = idx(max(ii, 1));
    t = tic; Iimg = prep_oct_image(char(I.T.file(i))); tr_ = toc(t);
    f = cell(1, 2); tc = zeros(1, 2);
    for b = 1:2
        net = nets{b}; inSz = net.Layers(1).InputSize;
        t = tic;
        Ir = imresize(Iimg, inSz(1:2));
        if strcmp(kinds{b}, 'dlnetwork')
            x = dlarray(single(Ir), 'SSCB'); if useGPU, x = gpuArray(x); end
            [a1, a2] = predict(net, x, 'Outputs', spec(b, 2:3));
            f{b} = [reshape(gather(extractdata(a1)), 1, []), reshape(gather(extractdata(a2)), 1, [])];
        else
            a1 = activations(net, Ir, spec{b, 2}, 'OutputAs', 'rows', envArg{:});
            a2 = activations(net, Ir, spec{b, 3}, 'OutputAs', 'rows', envArg{:});
            f{b} = [a1, a2];
        end
        tc(b) = toc(t);
    end
    t = tic; xz = (single([f{:}]) - model.mu) ./ model.sd; xs = xz(model.sel); ts_ = toc(t);
    t = tic; predict(model.mdl, double(xs)); tv_ = toc(t);
    if ii > 0
        tRead(ii) = tr_; tCNN(ii, :) = tc; tSel(ii) = ts_; tSVM(ii) = tv_;
        maxAbsDiff(ii) = max(abs(single([f{:}]) - Fx.X(i, :)));
    end
end
tTotal = tRead + sum(tCNN, 2) + tSel + tSVM;
lat = table({'read+preprocess'; 'DarkNet53 forward'; 'DenseNet201 forward'; 'z-score+selection'; 'SVM predict'; 'total'}, ...
    1000 * [median(tRead); median(tCNN(:, 1)); median(tCNN(:, 2)); median(tSel); median(tSVM); median(tTotal)], ...
    1000 * [iqr(tRead); iqr(tCNN(:, 1)); iqr(tCNN(:, 2)); iqr(tSel); iqr(tSVM); iqr(tTotal)], ...
    'VariableNames', {'Stage', 'MedianMs', 'IQRMs'});
[peakMB, curMB] = peak_mem_mb(); hw = hardware_info();
gpuPeakMB = NaN;
if useGPU, g = gpuDevice; gpuPeakMB = (g.TotalMemory - g.AvailableMemory) / 2^20; end
featureCheck = struct('maxAbsDiffVsCached', max(maxAbsDiff), 'kinds', {kinds});
save(out, 'lat', 'peakMB', 'curMB', 'gpuPeakMB', 'hw', 'featureCheck', 'nL', 'useGPU', '-v7');
[~, stem] = fileparts(out); writetable(lat, fullfile(cfg.tabDir, [strrep(stem, 'E11_latency', 'E11_end_to_end_latency') '.csv']));
disp(lat); fprintf('m07: peak RAM %.0f MB; max |feature difference| vs cached = %.3g\n', peakMB, max(maxAbsDiff));
end
