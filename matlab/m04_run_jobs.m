function m04_run_jobs(cfg, jobs, tiers)
%M04_RUN_JOBS  Run the job list in parallel (parfor), tier by tier.
%   Each job writes cfg.jobDir\<id>.mat; finished jobs are skipped, so the
%   run can be stopped and resumed at any time. Errors go to jobs\errors.
S  = load(fullfile(cfg.featDir, 'features_octdl.mat'), 'X', 'bb', 'lay');
I  = load(fullfile(cfg.featDir, 'inventory.mat'), 'y7', 'y2', 'group');
Fo = load(fullfile(cfg.outDir, 'folds.mat'), 'foldsPG', 'foldsIL');
D = struct('X', S.X, 'bb', S.bb, 'lay', S.lay, 'y7', I.y7(:), 'y2', I.y2(:), ...
           'group', I.group(:), 'foldsPG', Fo.foldsPG, 'foldsIL', Fo.foldsIL);
pool = gcp('nocreate');
if isempty(pool)
    if isempty(cfg.nWorkers), pool = parpool('Processes');
    else, pool = parpool('Processes', cfg.nWorkers); end
end
fprintf('m04: pool with %d workers.\n', pool.NumWorkers);
Dc = parallel.pool.Constant(D);
errDir = fullfile(cfg.jobDir, 'errors'); if ~isfolder(errDir), mkdir(errDir); end
jobDir = cfg.jobDir;
for tier = tiers
    todo = jobs([jobs.tier] == tier);
    done = arrayfun(@(j) isfile(fullfile(jobDir, [j.id '.mat'])), todo);
    todo = todo(~done);
    fprintf('m04: tier %d, %d jobs to run (%d already done).\n', tier, numel(todo), sum(done));
    if isempty(todo), continue, end
    q = parallel.pool.DataQueue;
    afterEach(q, @progress_print);
    progress_print('', numel(todo));
    parfor j = 1:numel(todo)
        jb = todo(j);
        try
            res = octga_run_job(jb, Dc.Value, cfg);
            parsave(fullfile(jobDir, [jb.id '.mat']), res);
            send(q, sprintf('%-32s %7.1f s (GA %6.1f s, nSel %4d)', jb.id, res.T.total, res.T.ga, res.nSel));
        catch ME
            write_error(fullfile(errDir, [jb.id '.txt']), ME);
            send(q, sprintf('%-32s ERROR: %s', jb.id, ME.message));
        end
    end
end
end
