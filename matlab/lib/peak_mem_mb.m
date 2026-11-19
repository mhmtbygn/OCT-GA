function [peakMB, curMB] = peak_mem_mb()
%PEAK_MEM_MB  Peak and current working set of this MATLAB process (MB).
%   Uses .NET System.Diagnostics.Process (Windows). Falls back to memory().
peakMB = NaN; curMB = NaN;
try
    p = System.Diagnostics.Process.GetCurrentProcess();
    p.Refresh();
    peakMB = double(p.PeakWorkingSet64) / 2^20;
    curMB  = double(p.WorkingSet64) / 2^20;
catch
    try
        m = memory;
        curMB = m.MemUsedMATLAB / 2^20; peakMB = curMB;
    catch
    end
end
end
