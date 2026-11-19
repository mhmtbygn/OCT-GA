import pickle, pandas as pd, numpy as np
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
S = pickle.load(open('summary.pkl', 'rb'))
out = '/mnt/user-data/outputs/OCTGA_Python_Results.xlsx'
notes = pd.DataFrame({'Item': [
 'Data', 'Outer CV', 'Inner protocol', 'GA (FAST)', 'GA (ORIG replica)', 'ReliefF', 'SVM', 'Posteriors',
 'Learning curve / permutation', 'Transfer (E13)', 'CI', 'Tests', 'MATLAB reference', 'Not run'],
 'Value': [
 'Revision_Features/features_octdl.mat (2064 x 4944), inventory.mat (y2, y7, patient group)',
 'foldsPG / foldsIL from folds.mat, repeat 1 only (10 folds)',
 'z-score, ReliefF, GA, SVM kernel/C and Platt scaling fitted on the outer training set only; inner 5-fold patient-grouped CV, inner fold 1 = GA hold-out',
 'N 20, maxIter 50, stall 15, CR 0.8, MR 0.3, k-NN k=5 (full config: N 100, maxIter 500, stall 100)',
 'N 20, maxIter 50, targetAcc 0.95, global z-score, RReliefF on all images, cubic SVM C=1, image-level CV',
 'Kononenko ReliefF (k=10, W>0) / RReliefF (sigma 50); range-scaled L1 distance; numba implementation',
 'Kernels linear/poly2/poly3/rbf x C {0.1,1,10}, KernelScale = sqrt(D), selection by inner balanced accuracy; one-vs-one for 7 classes',
 'sklearn SVC Platt scaling (internal 5-fold); L1 logistic posteriors',
 'Learning curve: 1 repeat; permutation: B = 19 patient-level permutations (minimum p = 0.05); libsvm max_iter capped at 300000 from permutation 10 on (non-converging fits on random labels)',
 'Pipeline fitted on Kermany (3097 unique images, image-level inner CV), frozen, applied to all 2064 OCTDL images; 5 seeds',
 'Patient-cluster bootstrap, 2000 resamples, percentile 95%',
 'Corrected resampled t-test (Nadeau-Bengio) on per-fold BAcc, Holm within task',
 'Pooled results of the uploaded MATLAB job files (full GA) where all 10 folds of repeat 1 exist',
 'E5-iii fine-tuned ResNet-18 and E11 end-to-end CNN latency (need images and CNN weights)']})
with pd.ExcelWriter(out, engine='openpyxl') as w:
    notes.to_excel(w, sheet_name='README', index=False)
    for k, v in S.items():
        v.to_excel(w, sheet_name=k[:31], index=k.startswith('CM'))
wb = load_workbook(out)
for ws in wb.worksheets:
    for row in ws.iter_rows():
        for c in row:
            c.font = Font(name='Arial', size=10, bold=(c.row == 1))
            if isinstance(c.value, float): c.number_format = '0.0000'
    for c in ws[1]: c.fill = PatternFill('solid', start_color='DDEBF7'); c.alignment = Alignment(wrap_text=True, vertical='top')
    for i, col in enumerate(ws.columns, 1):
        L = max(len(str(c.value)) if c.value is not None else 0 for c in col)
        ws.column_dimensions[get_column_letter(i)].width = min(max(10, L + 2), 90)
    ws.freeze_panes = 'A2'
wb.save(out); print('saved', out)
