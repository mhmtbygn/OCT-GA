function jobs = m03_build_jobs(cfg)
%M03_BUILD_JOBS  Flat job list. Tiers are run in order:
%   1 main patient-grouped runs: all configurations x tasks x repeats x folds (E2/E4/E5/E6/E7/E9/E11)
%   2 protocol comparison: image-level nested (IL) and replica of the original code (ORIG) (E3)
%   3 GA sensitivity (E12)       4 learning curve (E10)       5 label permutation (E8)
jobs = struct('id', {}, 'tier', {}, 'task', {}, 'config', {}, 'protocol', {}, 'rep', {}, ...
              'fold', {}, 'lcFrac', {}, 'permIdx', {}, 'variant', {}, 'saveModel', {}, 'isGA', {});
add = @(J, tier, task, conf, prot, r, k, lc, p, v, sm, suf) [J, struct('id', ...
    sprintf('T%d_%s_%s_r%d_k%d%s', task, conf, prot, r, k, suf), 'tier', tier, 'task', task, ...
    'config', conf, 'protocol', prot, 'rep', r, 'fold', k, 'lcFrac', lc, 'permIdx', p, ...
    'variant', v, 'saveModel', sm, 'isGA', ismember(conf, cfg.gaConfigs) || strcmp(prot, 'ORIG'))];
for task = cfg.tasks
    for r = 1:cfg.R
        for k = 1:cfg.K
            for c = 1:numel(cfg.configs)
                conf = cfg.configs{c};
                sm = strcmp(conf, 'F') && r == 1 && k == 1;
                jobs = add(jobs, 1, task, conf, 'PG', r, k, 1, 0, 0, sm, '');
            end
            jobs = add(jobs, 2, task, 'F', 'IL', r, k, 1, 0, 0, false, '');
        end
        jobs = add(jobs, 2, task, 'F', 'ORIG', r, 0, 1, 0, 0, false, '');
    end
    for r = 1:cfg.lc.R
        for k = 1:cfg.K
            for f = cfg.lc.fracs
                jobs = add(jobs, 4, task, 'F', 'PG', r, k, f, 0, 0, false, sprintf('_lc%02d', round(100 * f)));
            end
        end
    end
end
for k = 1:cfg.K
    for v = 1:numel(cfg.e12.variants)
        jobs = add(jobs, 3, 2, 'F', 'PG', 1, k, 1, 0, v, false, ['_' cfg.e12.variants{v}.tag]);
    end
    for p = 1:cfg.perm.B
        jobs = add(jobs, 5, 2, 'F', 'PG', 1, k, 1, p, 0, false, sprintf('_perm%03d', p));
    end
end
% within each tier put GA jobs first (longest) to balance the pool
[~, o] = sortrows([[jobs.tier].', -[jobs.isGA].']);
jobs = jobs(o);
fprintf('m03: %d jobs (tier counts: %s).\n', numel(jobs), mat2str(histcounts([jobs.tier], 0.5:1:5.5)));
end
