# Submission score
1. submission.csv : 0.43852 = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000, random_state=42)).fit(Xtr, ytr) training embeddings only 1040/1826
2. submissionRF.csv : 0.32240 = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42) training embeddings only 1040/1826
3. submission_q10.csv : 0.54034 = Birdnet v2.4 tf, Logreg C=0.01 Balanced unknown <10th percentile CV max-prob seed 42
4. submission_q20.csv : 0.55533 = Birdnet v2.4 tf, Logreg C=0.01 Balanced unknown <20th percentile CV max-prob seed 42
<!-- 5. submission_kfold.csv : 0.60353 =  best param: {'svc__C': 1, 'svc__gamma': 0.001} | macroF1 CV: 0.7613 best treshold =0.73 (macroF1 sintetis=0.5823) -->
6. submission_svc_q20.csv : 0.50671 = svc-q20: BirdNET2.4 file-mean + SVC RBF C=1 g=.001 platt, Unknown<20th pct CV max-prob, seed42
7. submission_all_unknown.csv : 0.13798 = All are predicted as unknown
<!-- 8. submission_svc3.csv: 0.61719 = 'svc__C': 3, 'svc__gamma': 0.0003 -->
<!-- 9. submission_svc301.csv: 0.57372 = 'svc__C': 3, 'svc__gamma': 0.0003,n_pca=30 unknown_share=0.30 -->
10. submission_lr_rank38.csv : 0.53130 = Birdnetv2.4 lr-rank38: LogReg per window C=.01 balanced, tolak 38% file tes paling tidak yakin, seed42
<!-- 11. submission_knnxmsp: 0.47936 =  'logisticregression__C': 0.03 -->
12. 
