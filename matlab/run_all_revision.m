%% OCT-GA R1 revision - run every experiment requested by the reviewers
%  1) Set SMOKE = true and run this script once (about 30-60 min; most of it
%     is the one-off CNN feature extraction, which is cached and reused).
%     Send Revision_Results_SMOKE\REPORT.txt and the log file back.
%  2) Set SMOKE = false and run again (long; can be stopped and restarted,
%     finished jobs are skipped).
%  All outputs: D:\MBayrakOCTGA\Revision_Results (tables, figures, REPORT.txt)

clear; clc;
SMOKE = true;                      % <-- first run the smoke test (manual use)
runMode = getenv('OCTGA_MODE');   % set by OCTGA_BASLAT.bat: 'smoke' or 'full'
if ~isempty(runMode), SMOKE = ~strcmpi(runMode, 'full'); end
fprintf('OCT-GA revision run, mode: %s\n', ternary_mode(SMOKE));

here = fileparts(mfilename('fullpath'));
addpath(here, fullfile(here, 'lib'));
cfg = config_revision(SMOKE);
dirs = {cfg.outDir, cfg.featDir, cfg.jobDir, cfg.modelDir, cfg.figDir, cfg.tabDir, fullfile(cfg.jobDir, 'errors')};
for d = 1:numel(dirs), if ~isfolder(dirs{d}), mkdir(dirs{d}); end, end
diary(fullfile(cfg.outDir, ['log_' char(datetime('now', 'Format', 'yyyyMMdd_HHmmss')) '.txt']));
hw = hardware_info(); save(fullfile(cfg.outDir, 'hardware.mat'), 'hw'); disp(hw)
save(fullfile(cfg.outDir, 'config_used.mat'), 'cfg');

tAll = tic;
m00_inventory(cfg);                % E0  inventory, duplicates, patient groups
m01_extract_features(cfg);         % E1  frozen DarkNet53 + DenseNet201 features (cached)
m02_make_folds(cfg);               %     patient-grouped and image-level outer folds
jobs = m03_build_jobs(cfg);
m04_run_jobs(cfg, jobs, 1:2);      % E2/E4/E5(i,ii)/E6/E7/E9 and E3 (protocols)
m05_finetune_cnn(cfg);             % E5-iii fine-tuned ResNet-18 (sequential)
m06_transfer_kermany(cfg);         % E13 frozen Kermany -> OCTDL
m07_latency(cfg);                  % E11 end-to-end latency
m04_run_jobs(cfg, jobs, 3:5);      % E12 GA sensitivity, E10 learning curve, E8 permutation
m08_stats_report(cfg);             % tables, tests, figures, REPORT.txt
fprintf('All done in %.1f h.\n', toc(tAll) / 3600);
diary off

function m = ternary_mode(smoke)
if smoke, m = 'SMOKE TEST'; else, m = 'FULL RUN'; end
end
