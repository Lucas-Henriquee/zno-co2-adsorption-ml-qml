# Table 2 - QSVM results

| run | dataset | feature_set | reps | entanglement | max_samples | samples | n_features | accuracy | balanced_accuracy | purpose |
|---|---|---|---|---|---|---|---|---|---|---|
| QSVM md3 reps1 90 | Expanded subset | md3 | 1.0 | linear | 90.0 | 90 | 3 | 0.8261 | 0.8214 | Small QSVM pipeline test |
| QSVM md3 reps1 300 | Expanded subset | md3 | 1.0 | linear | 300.0 | 300 | 3 | 1.0000 | 1.0000 | Larger expanded QSVM test |
| QSVM md3 reps2 150 | Expanded subset | md3 | 2.0 | linear | 150.0 | 150 | 3 | 0.9211 | 0.9167 | Deeper QSVM feature map |
| QSVM unique MD reps1 | Unique MD windows | md3 | 1.0 | linear | 0.0 | 30 | 3 | 0.3750 | 0.3333 | Unique-window QSVM reps1 |
| QSVM unique MD reps2 | Unique MD windows | md3 | 2.0 | linear | 0.0 | 30 | 3 | 0.5000 | 0.5000 | Unique-window QSVM reps2 |
| QSVM dynamic4 reps1 300 | Expanded subset | dynamic4 | 1.0 | linear | 300.0 | 300 | 4 | 0.7067 | 0.7067 | Direct comparison with dynamic4 SVM |
