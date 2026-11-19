function hw = hardware_info()
%HARDWARE_INFO  Record the hardware and software used for the timings.
hw = struct();
hw.matlab = version;
hw.date = char(datetime('now'));
hw.cpuIdentifier = strtrim(getenv('PROCESSOR_IDENTIFIER'));
hw.cpuName = '';
try
    [st, out] = system('powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).Name"');
    if st == 0, hw.cpuName = strtrim(out); end
catch
end
hw.physicalCores = feature('numcores');
hw.logicalCores = str2double(getenv('NUMBER_OF_PROCESSORS'));
hw.ramGB = NaN;
try
    [~, s] = memory; hw.ramGB = s.PhysicalMemory.Total / 2^30;
catch
end
hw.gpu = 'none'; hw.gpuMemGB = NaN;
try
    if canUseGPU
        gd = gpuDevice; hw.gpu = gd.Name; hw.gpuMemGB = gd.TotalMemory / 2^30;
    end
catch
end
v = ver; hw.toolboxes = strjoin(strcat({v.Name}, {' '}, {v.Version}), '; ');
end
