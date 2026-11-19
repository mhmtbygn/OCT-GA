function [X, info] = extract_deep_features(files, cfg)
%EXTRACT_DEEP_FEATURES  Frozen ImageNet features, identical to the original code.
%   X is n x 4944 (single). Column blocks, in the order of the original
%   code ([fm fm1] = [Layers(end-2), Layers(end-3)]):
%     1    - 1000  DarkNet53   'conv53'   (1x1 conv acting as the FC layer)
%     1001 - 2024  DarkNet53   'avg1'     (global average pooling)
%     2025 - 3024  DenseNet201 'fc1000'
%     3025 - 4944  DenseNet201 'avg_pool' (global average pooling)
%   Feature extraction uses no labels, so it is done once for all images.
spec = {'darknet53', 'conv53', 'avg1'; 'densenet201', 'fc1000', 'avg_pool'};
blocks = cell(1, 2); bb = []; lay = []; info = struct();
for b = 1:2
    [net, kind] = load_backbone(spec{b, 1});
    inSz = net.Layers(1).InputSize;
    imds = imageDatastore(files, 'ReadFcn', @(f) prep_oct_image(f));
    aug = augmentedImageDatastore(inSz(1:2), imds);
    t = tic;
    if strcmp(kind, 'legacy')
        fc  = activations(net, aug, spec{b, 2}, 'OutputAs', 'rows', 'MiniBatchSize', cfg.feat.mb);
        reset(aug);
        gap = activations(net, aug, spec{b, 3}, 'OutputAs', 'rows', 'MiniBatchSize', cfg.feat.mb);
    else
        [fcA, gapA] = minibatchpredict(net, aug, 'Outputs', spec(b, 2:3), 'MiniBatchSize', cfg.feat.mb);
        fc  = as_rows(gather(fcA),  numel(files));
        gap = as_rows(gather(gapA), numel(files));
    end
    sec = toc(t);
    blocks{b} = single([fc, gap]);
    bb  = [bb,  b * ones(1, size(fc, 2) + size(gap, 2))]; %#ok<AGROW>
    lay = [lay, ones(1, size(fc, 2)), 2 * ones(1, size(gap, 2))]; %#ok<AGROW>
    info.(spec{b, 1}) = struct('kind', kind, 'inputSize', inSz, 'seconds', sec, ...
        'secPerImage', sec / numel(files), 'nFC', size(fc, 2), 'nGAP', size(gap, 2), ...
        'peakMB', peak_mem_mb(), 'env', ternary(canUseGPU, 'gpu', 'cpu'));
end
X = [blocks{:}];
info.bb = bb; info.lay = lay; info.layerSpec = spec; info.nImages = numel(files);
end

function [net, kind] = load_backbone(name)
try
    switch name
        case 'darknet53',   net = darknet53();
        case 'densenet201', net = densenet201();
    end
    kind = 'legacy';
catch
    net = imagePretrainedNetwork(name);
    kind = 'dlnetwork';
end
end

function R = as_rows(A, n)
% Convert a network output (SSCB, CB or BC layout) to an n x C matrix.
if ismatrix(A) && size(A, 1) == n && size(A, 2) ~= n
    R = A;
else
    R = reshape(A, [], n).';
end
end

function out = ternary(c, a, b)
if c, out = a; else, out = b; end
end
