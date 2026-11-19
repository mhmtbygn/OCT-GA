function m05_finetune_cnn(cfg)
%M05_FINETUNE_CNN  E5-iii: end-to-end fine-tuned lightweight CNN (ResNet-18)
%   on exactly the same patient-grouped outer folds. Runs sequentially
%   (training uses all cores / the GPU). Results use the same format as the
%   other jobs (config 'FT').
I  = load(fullfile(cfg.featDir, 'inventory.mat'), 'T', 'y7', 'y2');
Fo = load(fullfile(cfg.outDir, 'folds.mat'), 'foldsPG');
files = cellstr(I.T.file);
for task = cfg.tasks
    nCls = task;
    if nCls == 2, y = I.y2(:); else, y = I.y7(:); end
    for r = 1:cfg.ft.reps
        for k = 1:cfg.K
            id = sprintf('T%d_FT_PG_r%d_k%d', task, r, k);
            out = fullfile(cfg.jobDir, [id '.mat']);
            if isfile(out), continue, end
            fprintf('m05: %s\n', id); T = struct();
            rng(job_seed(id), 'twister');
            te = find(Fo.foldsPG(:, r) == k); tr = find(Fo.foldsPG(:, r) ~= k);
            imdsTr = imageDatastore(files(tr), 'ReadFcn', @(f) prep_oct_image(f), ...
                'Labels', categorical(y(tr), 1:nCls));
            imdsTe = imageDatastore(files(te), 'ReadFcn', @(f) prep_oct_image(f));
            net = imagePretrainedNetwork(cfg.ft.net, 'NumClasses', nCls);
            inSz = net.Layers(1).InputSize;
            augmenter = imageDataAugmenter('RandXReflection', true, ...
                'RandXTranslation', [-10 10], 'RandYTranslation', [-10 10]);
            dsTr = augmentedImageDatastore(inSz(1:2), imdsTr, 'DataAugmentation', augmenter);
            dsTe = augmentedImageDatastore(inSz(1:2), imdsTe);
            opts = trainingOptions('adam', 'InitialLearnRate', cfg.ft.lr, ...
                'MaxEpochs', cfg.ft.epochs, 'MiniBatchSize', cfg.ft.mb, ...
                'Shuffle', 'every-epoch', 'Verbose', false, 'Plots', 'none', ...
                'ExecutionEnvironment', cfg.ft.env);
            t = tic; net = trainnet(dsTr, net, 'crossentropy', opts); T.train = toc(t);
            t = tic; scores = minibatchpredict(net, dsTe, 'MiniBatchSize', cfg.ft.mb); T.predict = toc(t);
            scores = double(gather(scores));
            [~, yPred] = max(scores, [], 2);
            % single-image latency (batch size 1), median over up to nImages test images
            nL = min(cfg.lat.nImages, numel(te)); lat = nan(nL, 1);
            for i = 0:nL
                Ii = imresize(prep_oct_image(files{te(max(i, 1))}), inSz(1:2));
                xi = dlarray(single(Ii), 'SSCB');
                if canUseGPU && ~strcmp(cfg.ft.env, 'cpu'), xi = gpuArray(xi); end
                t = tic; o = predict(net, xi); o = gather(extractdata(o)); tt = toc(t); %#ok<NASGU>
                if i > 0, lat(i) = tt; end          % i = 0 is a warm-up
            end
            T.latencyMed = median(lat); T.total = T.train + T.predict;
            res = struct('id', id, 'task', nCls, 'config', 'FT', 'protocol', 'PG', 'rep', r, ...
                'fold', k, 'lcFrac', 1, 'permIdx', 0, 'variant', 0, 'seed', job_seed(id), ...
                'teIdx', te, 'yTrue', y(te), 'yPred', yPred, 'score', single(scores), ...
                'pPos', [], 'nTrain', numel(tr), 'nTest', numel(te), 'T', T, ...
                'net', cfg.ft.net, 'epochs', cfg.ft.epochs);
            if nCls == 2, res.pPos = scores(:, 1); end
            if canUseGPU && ~strcmp(cfg.ft.env, 'cpu'), res.device = 'gpu'; gd = gpuDevice; res.gpuUsedMB = (gd.TotalMemory - gd.AvailableMemory) / 2^20; else, res.device = 'cpu'; end
            [res.peakMB, res.curMB] = peak_mem_mb();
            parsave(out, res);
            fprintf('m05: %s done | train %.0f s | test acc %.2f%% | latency %.1f ms/image | %s\n', id, T.train, 100 * mean(yPred(:) == y(te)), 1000 * T.latencyMed, res.device);
        end
    end
end
end
