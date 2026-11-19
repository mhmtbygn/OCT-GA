function cfg = config_revision(smoke)
%CONFIG_REVISION  All settings for the OCT-GA R1 revision experiments.
%   cfg = config_revision()        full run
%   cfg = config_revision(true)    smoke test (tiny GA, 3 folds, 1 repeat)
%
%   Edit the paths in section 1 if the data live elsewhere.
%   Requires MATLAB R2024a or newer (R2025a is the target) with
%     Deep Learning Toolbox, Statistics and Machine Learning Toolbox,
%     Parallel Computing Toolbox, Image Processing Toolbox and the
%     support packages "Deep Learning Toolbox Model for DarkNet-53 Network",
%     "... for DenseNet-201 Network" and "... for ResNet-18 Network".

if nargin < 1, smoke = false; end
cfg.smoke = smoke;

%% 1. Paths ---------------------------------------------------------------
cfg.codeDir    = fileparts(mfilename('fullpath'));
cfg.projDir    = 'D:\MBayrakOCTGA';
cfg.dataDir    = fullfile(cfg.projDir, 'OCT-GA-MAKALE-MERVEBAYRAK', 'Datasets');
cfg.octdlDir   = fullfile(cfg.dataDir, 'OCTDL_full', 'OCTDL');      % <DISEASE>\<file>.jpg
cfg.octdlCsv   = fullfile(cfg.dataDir, 'OCTDL_full', 'OCTDL_labels.csv');
cfg.kermanyDir = fullfile(cfg.dataDir, 'Kermany_5000_2classes');    % '1 (n).jpeg' Normal, '2 (n).jpeg' Diseased
if smoke
    cfg.outDir = fullfile(cfg.projDir, 'Revision_Results_SMOKE');
else
    cfg.outDir = fullfile(cfg.projDir, 'Revision_Results');
end
cfg.featDir  = fullfile(cfg.projDir, 'Revision_Features');   % shared cache (inventory + CNN features)
cfg.jobDir   = fullfile(cfg.outDir, 'jobs');
cfg.modelDir = fullfile(cfg.outDir, 'models');
cfg.figDir   = fullfile(cfg.outDir, 'figures');
cfg.tabDir   = fullfile(cfg.outDir, 'tables');

%% 2. Classes -------------------------------------------------------------
% 7-class coding follows the manuscript (Fig. 12): 1 AMD, 2 DME, 3 ERM,
% 4 Normal, 5 RAO, 6 RVO, 7 VID.  Binary coding follows Fig. 11:
% 1 = Diseased (positive class), 2 = Normal.
cfg.classOrder7 = {'AMD','DME','ERM','NO','RAO','RVO','VID'};
cfg.classNames2 = {'Diseased','Normal'};
cfg.tasks = [2 7];

%% 3. Outer / inner cross-validation ----------------------------------------
cfg.K      = 10;          % outer folds (patient-grouped, stratified on the 7-class label)
cfg.R      = 5;           % repeats
cfg.seeds  = 1:cfg.R;     % repeat r uses seed cfg.seeds(r)
cfg.innerK = 5;           % inner patient-grouped folds inside each outer training set;
                          % inner fold 1 (about 20%) is the GA fitness hold-out

%% 4. Pipeline settings -----------------------------------------------------
cfg.feat.mb       = 32;               % mini-batch for feature extraction
cfg.relieffK      = 10;
cfg.relieffMethod = 'classification'; % the original code ran RReliefF (numeric y default)
cfg.ga = struct('N',100,'maxIter',500,'CR',0.8,'MR',0.3, ...
                'stall',100,'knnK',5,'targetAcc',NaN);
cfg.svm.kernels = {'linear','poly2','poly3','rbf'};
cfg.svm.C       = [0.1 1 10];
cfg.lin.lambda  = logspace(-5,-1,9);

% Replica of the ORIGINAL code (Kod\GA_CNN_FeatureSelection_Code.m):
% global z-score, RReliefF, GA with the fold-1 hold-out of the same
% image-level partition, N = 50, targetAcc = 0.95, cubic SVM (C = 1),
% image-level 10-fold CV.
cfg.orig.ga = struct('N',50,'maxIter',500,'CR',0.8,'MR',0.3, ...
                     'stall',Inf,'knnK',5,'targetAcc',0.95);
cfg.orig.relieffMethod = 'regression';

%% 5. Configurations (E2/E4/E5) -----------------------------------------------
%  F      proposed: per-backbone ReliefF -> concat -> GA -> tuned SVM
%  A      DarkNet53 only (ReliefF + GA)      B  DenseNet201 only (ReliefF + GA)
%  C      concat, no ReliefF, GA             D  per-backbone ReliefF, no GA
%  E      all 4944 features, fixed SVM (cubic, C = 1) = original weak baseline
%  Etuned all 4944 features, inner-tuned SVM
%  G      concat -> single ReliefF -> GA -> tuned SVM   (Reviewer 1, comment 5)
%  L1     L1-regularised logistic regression on all 4944 features (lambda tuned)
%  FZ     single frozen CNN (DenseNet201 GAP, 1920) + linear SVM (C tuned)
cfg.configs   = {'F','A','B','C','D','E','Etuned','G','L1','FZ'};
cfg.gaConfigs = {'F','A','B','C','G'};

%% 6. Additional experiments ----------------------------------------------------
cfg.lc.fracs = [0.25 0.50 0.75];      % E10 learning curve (1.00 = main run)
cfg.lc.R     = 3;
cfg.perm.B   = 100;                   % E8 label permutation (binary task, repeat 1)
cfg.e12.variants = { ...              % E12 GA sensitivity (binary task, repeat 1)
    struct('tag','MR0p1', 'MR',0.1, 'N',100), ...
    struct('tag','MR0p01','MR',0.01,'N',100), ...
    struct('tag','MRinvD','MR',-1,  'N',100), ...   % -1 means 1/D
    struct('tag','N50',   'MR',0.3, 'N',50)};
cfg.ft = struct('net','resnet18','epochs',10,'mb',32,'lr',1e-4, ...
                'reps',1,'env','auto');               % E5-iii
cfg.e13.seeds   = 1:5;                % E13 frozen Kermany -> OCTDL transfer
cfg.boot.B      = 2000;               % patient-cluster bootstrap resamples
cfg.lat.nImages = 100;                % E11 end-to-end latency sample

%% 7. Parallel pool -------------------------------------------------------------
cfg.nWorkers = 16;                    % about 1-1.5 GB RAM per worker; raise to 20-24 if >35 GB RAM is free

%% 8. Smoke-test overrides --------------------------------------------------------
if smoke
    cfg.K = 3; cfg.R = 1; cfg.seeds = 1; cfg.innerK = 3;
    cfg.ga.N = 10; cfg.ga.maxIter = 5; cfg.ga.stall = 3;
    cfg.orig.ga.N = 10; cfg.orig.ga.maxIter = 5;
    cfg.svm.kernels = {'linear','poly3'}; cfg.svm.C = 1;
    cfg.lin.lambda = [1e-4 1e-2];
    cfg.lc.fracs = 0.5; cfg.lc.R = 1;
    cfg.perm.B = 2;
    cfg.e12.variants = cfg.e12.variants(1);
    cfg.ft.epochs = 1;
    cfg.e13.seeds = 1;
    cfg.boot.B = 100;
    cfg.lat.nImages = 5;
end
end
