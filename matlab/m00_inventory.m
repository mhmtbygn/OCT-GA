function m00_inventory(cfg)
%M00_INVENTORY  E0: OCTDL inventory (images, patients, eyes per class),
%   exact-duplicate check (MD5) and patient-group construction; Kermany list.
out = fullfile(cfg.featDir, 'inventory.mat');
if isfile(out), fprintf('m00: %s exists, skipped.\n', out); return, end
fprintf('m00: reading %s\n', cfg.octdlCsv);
T = readtable(cfg.octdlCsv, 'TextType', 'string');
n = height(T);
T.file = fullfile(cfg.octdlDir, T.disease, T.file_name + ".jpg");
miss = ~isfile(T.file);
if any(miss), error('m00: %d OCTDL images not found, e.g. %s', sum(miss), T.file(find(miss, 1))); end
[tf, y7] = ismember(T.disease, string(cfg.classOrder7));
assert(all(tf), 'm00: unknown disease label in CSV');
y2 = 1 + double(T.disease == "NO");                % 1 Diseased, 2 Normal
md5 = strings(n, 1);
for i = 1:n, md5(i) = file_md5(T.file(i)); end
group = double(T.patient_id);
[~, ~, hi] = unique(md5); dupCnt = accumarray(hi, 1);
dupH = find(dupCnt > 1).';
changed = true;                                     % merge patients sharing an identical image
while changed
    changed = false;
    for d = dupH
        idx = hi == d; gs = unique(group(idx));
        if numel(gs) > 1, group(ismember(group, gs)) = min(gs); changed = true; end
    end
end
dupRows = ismember(hi, dupH);
dupTab = T(dupRows, {'file_name', 'disease', 'patient_id', 'eye'});
dupTab.md5 = md5(dupRows);
eyeStr = string(T.eye); eyeKnown = eyeStr == "OD" | eyeStr == "OS";
pe = string(T.patient_id) + "_" + eyeStr;
cls = string(cfg.classOrder7).';
nImg = zeros(7, 1); nPat = nImg; nEyeKnown = nImg; nImgEyeKnown = nImg;
for c = 1:7
    m = y7 == c;
    nImg(c) = sum(m); nPat(c) = numel(unique(T.patient_id(m)));
    nEyeKnown(c) = numel(unique(pe(m & eyeKnown))); nImgEyeKnown(c) = sum(m & eyeKnown);
end
inv7 = table(cls, nImg, nPat, nEyeKnown, nImgEyeKnown, 'VariableNames', ...
    {'Class', 'Images', 'Patients', 'EyesKnown', 'ImagesWithEyeInfo'});
tot = {"Total", n, numel(unique(T.patient_id)), numel(unique(pe(eyeKnown))), sum(eyeKnown)};
inv7 = [inv7; cell2table(tot, 'VariableNames', inv7.Properties.VariableNames)];
pd = groupsummary(table(T.patient_id, T.disease, 'VariableNames', {'pid', 'dis'}), 'pid', ...
    @(x) numel(unique(x)), 'dis');
multiDis = pd.pid(pd{:, end} > 1);
inv2 = table(["Diseased"; "Normal"], [sum(y2 == 1); sum(y2 == 2)], ...
    [numel(unique(T.patient_id(y2 == 1))); numel(unique(T.patient_id(y2 == 2)))], ...
    'VariableNames', {'Class', 'Images', 'Patients'});

% Kermany (only for the frozen transfer experiment E13)
kd = [dir(fullfile(cfg.kermanyDir, '*.jpeg')); dir(fullfile(cfg.kermanyDir, '*.jpg'))];
kNames = string({kd.name}).'; kFiles = string(fullfile({kd.folder}, {kd.name})).';
kLab = str2double(extractBefore(kNames, " "));      % '1 (12).jpeg' -> 1 (Normal), 2 (Diseased)
kmd5 = strings(numel(kFiles), 1);
for i = 1:numel(kFiles), kmd5(i) = file_md5(kFiles(i)); end
[~, firstIdx] = unique(kmd5, 'stable');
kKeep = false(numel(kFiles), 1); kKeep(firstIdx) = true;
kerm = table(kFiles(kKeep), kLab(kKeep), 3 - kLab(kKeep), kmd5(kKeep), ...
    'VariableNames', {'file', 'labelOrig', 'y2', 'md5'});   % y2: 1 Diseased, 2 Normal
kermInfo = struct('nFiles', numel(kFiles), 'nUnique', sum(kKeep), ...
    'nDuplicatesRemoved', sum(~kKeep), 'overlapWithOCTDL', sum(ismember(kerm.md5, md5)));
save(out, 'T', 'y7', 'y2', 'group', 'md5', 'dupTab', 'inv7', 'inv2', 'multiDis', 'kerm', 'kermInfo', '-v7');
writetable(inv7, fullfile(cfg.tabDir, 'E0_OCTDL_inventory_7class.csv'));
writetable(inv2, fullfile(cfg.tabDir, 'E0_OCTDL_inventory_binary.csv'));
writetable(dupTab, fullfile(cfg.tabDir, 'E0_OCTDL_exact_duplicates.csv'));
fprintf('m00: OCTDL %d images, %d patients, %d exact-duplicate images, %d multi-disease patients.\n', ...
    n, numel(unique(T.patient_id)), height(dupTab), numel(multiDis));
fprintf('m00: Kermany %d files, %d unique (duplicates removed: %d).\n', ...
    kermInfo.nFiles, kermInfo.nUnique, kermInfo.nDuplicatesRemoved);
disp(inv7)
end
