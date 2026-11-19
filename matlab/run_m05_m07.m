%% OCT-GA R1: end-to-end latency (m07, CPU and GPU) and fine-tuned ResNet-18 (m05, GPU)
%  Use a FRESH MATLAB session (restart MATLAB first):
%     cd D:\MBayrakOCTGA\Kod\OCTGA_Revision
%     run_m05_m07
%  Keep the computer otherwise idle while the latency is measured (first minutes).
%  Needs the support packages for ResNet-18, DarkNet-53 and DenseNet-201 (Home > Add-Ons).
%  Finished ResNet-18 folds are skipped, so the script can be stopped and restarted.

here = fileparts(mfilename('fullpath'));
addpath(here, fullfile(here, 'lib'));
cfg = config_revision(false);
for d = {cfg.outDir, cfg.jobDir, cfg.tabDir}
    if ~isfolder(d{1}), mkdir(d{1}); end
end
diary(fullfile(cfg.outDir, ['log_m05_m07_' char(datetime('now', 'Format', 'yyyyMMdd_HHmmss')) '.txt']));

% 1) Latency on the CPU (the same hardware without a GPU)
cfg.lat.device = 'cpu';
tic; m07_latency(cfg); fprintf('m07 (CPU) finished in %.1f min\n', toc / 60);

% 2) Enable the RTX 5090 (Blackwell, compute capability 12.0). MATLAB R2025a supports it
%    only in CUDA forward-compatibility mode; the first GPU call compiles kernels (can take a while).
gpuOK = false;
try
    parallel.gpu.enableCUDAForwardCompatibility(true);
    gd = gpuDevice(1);
    fprintf('GPU enabled: %s, %.1f GB, compute capability %s\n', gd.Name, gd.TotalMemory / 2^30, gd.ComputeCapability);
    gpuOK = true;
catch ME
    fprintf(2, 'GPU could not be enabled (%s). Continuing on the CPU.\n', ME.message);
end
hw = hardware_info(); disp(hw); save(fullfile(cfg.outDir, 'hardware_m05_m07.mat'), 'hw');

% 3) Latency on the GPU
if gpuOK
    cfg.lat.device = 'gpu';
    tic; m07_latency(cfg); fprintf('m07 (GPU) finished in %.1f min\n', toc / 60);
end

% 4) Fine-tuned ResNet-18 on the same patient-grouped folds (repeat 1, 10 folds, T2 and T7)
cfg.ft.env = 'auto';                          % GPU if enabled, otherwise CPU
tic; m05_finetune_cnn(cfg); fprintf('m05 finished in %.1f h\n', toc / 3600);

diary off
fprintf('DONE. Results: jobs\\T*_FT_PG_r1_k*.mat, E11_latency_CPU.mat, E11_latency_GPU.mat in %s\n', cfg.outDir);
