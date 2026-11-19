function p = mcnemar_exact(b, c)
%MCNEMAR_EXACT  Two-sided exact McNemar test on the discordant counts b, c.
if b + c == 0, p = 1; return, end
p = min(1, 2 * binocdf(min(b, c), b + c, 0.5));
end
