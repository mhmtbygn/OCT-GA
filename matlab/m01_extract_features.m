function m01_extract_features(cfg)
%M01_EXTRACT_FEATURES  E1: frozen DarkNet53 + DenseNet201 features (4944-D)
%   for OCTDL (2064 images) and the local Kermany binary images (E13).
%   Cached in cfg.featDir, shared by the smoke and the full run.
I = load(fullfile(cfg.featDir, 'inventory.mat'), 'T', 'kerm');
f1 = fullfile(cfg.featDir, 'features_octdl.mat');
if ~isfile(f1)
    fprintf('m01: extracting OCTDL features (%d images)...\n', height(I.T));
    [X, info] = extract_deep_features(cellstr(I.T.file), cfg);
    bb = info.bb; lay = info.lay; hw = hardware_info();
    save(f1, 'X', 'bb', 'lay', 'info', 'hw', '-v7.3');
    fprintf('m01: OCTDL done: DarkNet53 %.1f s, DenseNet201 %.1f s.\n', ...
        info.darknet53.seconds, info.densenet201.seconds);
else
    fprintf('m01: %s exists, skipped.\n', f1);
end
f2 = fullfile(cfg.featDir, 'features_kermany.mat');
if ~isfile(f2)
    fprintf('m01: extracting Kermany features (%d images)...\n', height(I.kerm));
    [X, info] = extract_deep_features(cellstr(I.kerm.file), cfg);
    bb = info.bb; lay = info.lay;
    save(f2, 'X', 'bb', 'lay', 'info', '-v7.3');
else
    fprintf('m01: %s exists, skipped.\n', f2);
end
end
