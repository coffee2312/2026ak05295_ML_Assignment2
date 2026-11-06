# Indoor Building and Floor Prediction from WiFi Signal Strength (UJIIndoorLoc)

Machine Learning - Assignment 2 | M.Tech (AIML), BITS Pilani WILP

**Live Streamlit app:** https://2026ak05295mlassignment2-buildingfloorpredictor-usingwifisignal.streamlit.app
**GitHub repository:** https://github.com/coffee2312/2026ak05295_ML_Assignment2

---

## a. Problem statement

### Background
GPS signals are weak or blocked inside buildings, so GPS cannot tell a phone which building or which floor it is on. Indoor positioning systems solve this with **WiFi fingerprinting**. A phone scans for nearby WiFi access points (APs) and records the **received signal strength (RSSI, in dBm)** from each. The set of readings at one place forms a *fingerprint*. Signal strength depends on distance, walls and floors, so fingerprints taken at different places look different, and a model trained on labelled fingerprints can recognise where a new fingerprint was taken. I have taken inspiration to work on this project from my organisation's real world test scenario where we try to use WiFi's RTT and Bluetooth's Channel Sounding feature to predict the distance. We can find the application of location finder in AirTags, etc.

### Task
Given one WiFi fingerprint (RSSI readings from 465 access points), predict the **zone** it was recorded in. A zone is a **building and floor** combination, for example `B1_F2` = building 1, floor 2. There are three buildings and 13 zones in total, so this is a **13-class classification problem**.

### Why building and floor
- Knowing the building and floor is the first step of indoor navigation. Once the floor is known, finer positioning (room or coordinates) only has to search that floor.
- Floor detection is also harder than it looks: WiFi signals pass through ceilings and floors, so adjacent floors of the same building can look similar, and the same access point can be heard on several floors.
- Practical uses include emergency services (telling responders which floor a caller is on), indoor navigation in malls, hospitals and campuses, and asset tracking.

### Objectives
1. Prepare the data by handling the "not detected" placeholder, remove unusable columns and rows, and build the zone label.
2. Train **five classifiers on the same data and split**: Logistic Regression, Decision Tree, K-Nearest Neighbours, Naive Bayes and Support Vector Machine.
3. Evaluate each with six metrics: Accuracy, AUC, Precision, Recall, F1 and MCC.
4. Test the model on the validation data as well
5. Build and deploy a Streamlit app that loads the saved models (without retraining) and shows the results on an uploaded test file.

---

## b. Dataset description

### Source
**UJIIndoorLoc**, UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/310/ujiindoorloc

It is a public, multi-building, multi-floor WiFi fingerprint database from Universitat Jaume I, built for WLAN fingerprint-based indoor localization. The fingerprints were recorded by many users with different phones (the `USERID` and `PHONEID` columns). The data is split by the authors into `trainingData.csv` (used here for training and testing) and `validationData.csv` (recorded later, used here as an extra check).

*Reference:* Torres-Sospedra et al., "UJIIndoorLoc: A new multi-building and multi-floor database for WLAN fingerprint-based indoor localization problems", IPIN 2014. (Citation written from memory, check against the UCI page.)

### Raw file overview (`trainingData.csv`)
| Item | Value |
|---|---|
| Rows (fingerprints) | 19,937 |
| Columns | 529 |
| RSSI columns | 520 (`WAP001` to `WAP520`), one per access point |
| Metadata columns | 9: `LONGITUDE`, `LATITUDE`, `FLOOR`, `BUILDINGID`, `SPACEID`, `RELATIVEPOSITION`, `USERID`, `PHONEID`, `TIMESTAMP` |
| Missing-signal encoding | `100` means "access point not detected" |
| Share of readings that are "not detected" | 96.5% (each fingerprint hears only a few of the 520 access points) |
| Range of detected RSSI | -104 dBm (very weak) to 0 dBm (very strong), mean about -78.5 dBm |

### Features used
- **465 RSSI features**: the access points that are heard at least once in the data.
- **4 engineered features**, computed inside every saved model: number of access points detected, strongest RSSI, mean RSSI of the detected readings, and standard deviation of the detected readings.
- This project uses 465 (469 with the engineered ones) and 19,861.
- **Metadata columns are not used as features.** `LONGITUDE`, `LATITUDE`, `SPACEID` and `RELATIVEPOSITION` describe the location directly and would leak the answer. `USERID`, `PHONEID` and `TIMESTAMP` are not signal information and would not be available when a new fingerprint arrives.

### Engineered features
Besides the 465 raw RSSI columns, every model receives four summary features computed from each fingerprint (function `add_signal_features` in `common.py`). A reading counts as "detected" when it is above -105 dBm. They are calculated inside the saved pipeline, so the app and the validation script only need the raw RSSI values.

| Feature | What it is | What it captures |
|---|---|---|
| Number of access points detected | How many of the 465 access points were heard | How much WiFi coverage there is at that spot (median 17) |
| Strongest RSSI | The highest detected signal, in dBm | How close the nearest access point is |
| Mean RSSI of detected access points | Average of the heard signals only | The overall signal level, without the "not heard" values pulling it down |
| Standard deviation of detected access points | Spread of the heard signals | Whether one access point dominates (large spread) or all signals are similar (small spread) |

If a fingerprint has no detected access point, the mean is set to -105 and the spread to 0. These features describe the overall signal environment of a fingerprint. This project did not test them against a version without them, so no accuracy gain is claimed for them.

### Key facts about the cleaned data
- 19,861 fingerprints, 465 access points.
- A fingerprint hears **a median of 17 access points** (mean 18.1, minimum 1, maximum 51). About 11.6% of fingerprints hear 10 or fewer.
- Rows by building: building 0 = 5,248, building 1 = 5,159, building 2 = 9,454. Building 2 has five floors (0 to 4), the others have four (0 to 3).

### Target: `zone` (13 classes) and its distribution
The target is built as `"B" + BUILDINGID + "_F" + FLOOR`. Counts are after cleaning, with the stratified split described below.

| Zone | Rows | Share | Train | Test |
|---|---|---|---|---|
| B0_F0 | 1,058 | 5.3% | 846 | 212 |
| B0_F1 | 1,356 | 6.8% | 1,085 | 271 |
| B0_F2 | 1,443 | 7.3% | 1,154 | 289 |
| B0_F3 | 1,391 | 7.0% | 1,113 | 278 |
| B1_F0 | 1,368 | 6.9% | 1,094 | 274 |
| B1_F1 | 1,484 | 7.5% | 1,187 | 297 |
| B1_F2 | 1,396 | 7.0% | 1,117 | 279 |
| B1_F3 | 911 | 4.6% | 729 | 182 |
| B2_F0 | 1,906 | 9.6% | 1,525 | 381 |
| B2_F1 | 2,161 | 10.9% | 1,729 | 432 |
| B2_F2 | 1,576 | 7.9% | 1,261 | 315 |
| B2_F3 | 2,709 | 13.6% | 2,167 | 542 |
| B2_F4 | 1,102 | 5.5% | 881 | 221 |
| **Total** | **19,861** | 100% | **15,888** | **3,973** |

The classes are **mildly imbalanced**: the largest zone (B2_F3, 2,709 rows) is about 3 times the size of the smallest (B1_F3, 911 rows). No class is tiny, so no resampling was needed, and that is why the weighted averages are used for precision, recall and F1.

### Preprocessing
1. **Replace the placeholder.** `100` is not a signal level. It is replaced with **-105 dBm**, a value weaker than any real reading, so "not heard" is represented as "very weak signal" and distances between fingerprints stay meaningful for KNN and SVM.
2. **Drop 55 never-detected access points.** These columns are constant (always -105) and carry no information. 465 remain.
3. **Drop 76 empty fingerprints** where no access point was detected at all. They give a model nothing to learn from. They were mostly in B1_F3 (37 rows) and B2_F0 (36 rows). 19,861 rows remain.
4. **Build the target** `zone` from `BUILDINGID` and `FLOOR`.
5. **Feature engineering and scaling inside the model.** Each saved model is a scikit-learn *pipeline*: the four signal features are added, the values are standardised (`StandardScaler`), and then the classifier runs. Because everything is inside the pipeline, the Streamlit app and the validation script feed raw RSSI values and need no separate preprocessing files.

### How the data was split
- **Stratified 80/20 split** with `random_state = 7`. Stratification keeps each zone's share the same in both parts.
- **Training set:** 15,888 rows. **Test set:** 3,973 rows (saved as `test_data.csv`).
- The test rows are never used for fitting. Scaling parameters are learned from the training rows only, because the scaler is inside the pipeline.
- A **second, separate evaluation set** is the authors' `validationData.csv` (1,111 rows, all 13 zones present, none dropped), recorded later than the training data. It is used for evaluation only and never for training or tuning.

---

## c. GitHub repository link

https://github.com/coffee2312/2026ak05295_ML_Assignment2

```
app.py                  Streamlit app (SignalScope), with CSV format checker and chart code
common.py               Shared helpers: signal features, noise simulation, metrics
prepare_data.py         Cleans trainingData.csv into prepared_data.csv
train_models.py         Trains the 5 models; writes model/, test_data.csv, metrics.csv, robustness.csv
evaluate_validation.py  Scores the saved models on validationData.csv (extra check)
requirements.txt        Pinned library versions
test_data.csv           Held-out test split (3,973 rows)
validation_data.csv     Validation data in the app's format (extra)
model/                  5 saved pipelines (.joblib)
metrics.csv, robustness.csv, validation_metrics.csv   Saved results
```

**How to reproduce**
```
pip install -r requirements.txt
python prepare_data.py trainingData.csv
python train_models.py
streamlit run app.py
python evaluate_validation.py validationData.csv    # optional extra check
```
`trainingData.csv` and `validationData.csv` come from the UCI page above and are not stored in the repository because of their size.

---

## d. Models used

### Setup (identical for all five)
Same data, same split, same preprocessing pipeline. No hyperparameter search was done, so models use sensible settings or library defaults:

| Model | Settings |
|---|---|
| Logistic Regression | `max_iter=2000`, other defaults (L2 regularisation, C=1) |
| Decision Tree | `max_depth=15`, `random_state=7` |
| KNN | `k=5`, Euclidean distance, uniform weights |
| Naive Bayes | Gaussian Naive Bayes, defaults |
| Support Vector Machine | RBF kernel, `probability=True`, defaults (C=1, gamma="scale") |

### Metrics
- **Accuracy:** share of fingerprints assigned the correct zone.
- **AUC:** how well the predicted probabilities rank the correct zone (one-vs-rest, weighted over the 13 zones). 1.0 is perfect.
- **Precision, Recall, F1:** weighted averages over the 13 zones. Weighted recall is mathematically equal to accuracy, which is why those two columns are identical in the table.
- **MCC (Matthews correlation coefficient):** a single score from -1 to 1 that accounts for all cells of the confusion matrix. It is stricter than accuracy when classes are unequal.

### Comparison table (test split, 3,973 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9937 | 0.9999 | 0.9937 | 0.9937 | 0.9937 | 0.9931 |
| Decision Tree | 0.8417 | 0.9709 | 0.9074 | 0.8417 | 0.8433 | 0.8351 |
| KNN | 0.9821 | 0.9986 | 0.9821 | 0.9821 | 0.9821 | 0.9805 |
| Naive Bayes | 0.5477 | 0.9449 | 0.7190 | 0.5477 | 0.5101 | 0.5282 |
| Support Vector Machine | 0.9904 | 0.9999 | 0.9907 | 0.9904 | 0.9905 | 0.9896 |

### Building-level and floor-level accuracy (test split)
The app's *Building & floor* tab splits each prediction into its building and floor parts. "Floor" counts a prediction as correct when the floor number matches, whatever the building.

| ML Model Name | Building correct | Floor correct | Exact zone correct |
|---|---|---|---|
| Logistic Regression | 1.0000 | 0.9937 | 0.9937 |
| Support Vector Machine | 0.9962 | 0.9917 | 0.9904 |
| KNN | 0.9997 | 0.9821 | 0.9821 |
| Decision Tree | 0.9554 | 0.8492 | 0.8417 |
| Naive Bayes | 0.9980 | 0.5495 | 0.5477 |

### Prediction speed (from `metrics.csv`, milliseconds per 1,000 rows)
Logistic Regression 17.6 | Decision Tree 17.7 | Naive Bayes 64.9 | KNN 169.9 | SVM 1,122.5. Timings vary slightly from run to run.

### Observations on each model

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best on the test split (accuracy 0.9937, MCC 0.9931). The building is right for every test row and each zone has recall of 0.99 or more. Fast to predict. |
| Decision Tree | Accuracy 0.8417. Precision (0.9074) is higher than recall (0.8417), so when it names a zone it is usually right, but it misses many true rows. B1_F2 gets predicted 757 times although only 279 test rows belong to it. |
| KNN | Accuracy 0.9821. Almost all its mistakes stay inside the right building, mostly between neighbouring floors. About 10 times slower than Logistic Regression, and the saved model has to keep the whole training set. |
| Naive Bayes | Weakest (accuracy 0.5477, MCC 0.5282). It gets the building right (0.9980) but the floor only about half the time (0.5495). |
| Support Vector Machine | Accuracy 0.9904, AUC 0.9999, only 13 rows behind Logistic Regression out of 3,973. The slowest to predict (about 64 times Logistic Regression). |
| **Best for this dataset** | **Logistic Regression.** Best accuracy, F1 and MCC on the test split, one of the fastest, and also best on the validation file. SVM is a close second. |


**Logistic Regression.** The few errors it makes are small and spread out; the most common confusion is only 4 rows (B0_F1 predicted as B0_F0). With 465 inputs and 15,888 training rows a straight-line boundary is enough, and scaling puts weak and strong signals on the same footing.

**Decision Tree.** It has the clearest failure. Building accuracy is 0.9554 and floor accuracy 0.8492, so it mixes up both buildings and floors. B1_F2 acts as a dumping ground: 178 of the 182 B1_F3 rows are predicted as B1_F2, and B1_F1 and B1_F0 rows go there too. A tree splits on one access point at a time, so scans that miss the few access points it relies on end up in the same leaf. Depth was capped at 15 and not tuned, so a tuned tree might do better.

**KNN.** Scans from the same spot look alike, so nearest neighbours work well. Only one test row gets the wrong building. The cost is speed (about 170 ms per 1,000 rows) and memory.

**Naive Bayes.** It assumes the 465 signals are independent and bell-shaped, but nearby access points are correlated and the "not heard" value (-105) is a big spike, not a bell. That fits what we see: building almost perfect, floor poor. It over-predicts the top and bottom floors (for example 381 of the 542 B2_F3 rows are labelled B2_F4). This explanation comes from the pattern in the confusion matrix and was not tested separately.

**Support Vector Machine.** Very close to Logistic Regression. A few of its mistakes cross buildings (some B2 rows predicted as B0_F2), so its building accuracy (0.9962) is lower than KNN's. C and gamma were left at their defaults; tuning them might change the ranking.

### Extra 1: robustness to signal noise (accuracy on the test split)
The saved models are scored on test rows after adding Gaussian noise to the detected RSSI readings, plus 10% random access point dropout whenever the noise is above 0 dB. Models are not retrained. This imitates real conditions, where readings fluctuate because of people, doors and different phones.

| ML Model Name | 0 dB | 3 dB | 6 dB | 9 dB | Drop at 9 dB |
|---|---|---|---|---|---|
| Logistic Regression | 0.9937 | 0.9668 | 0.9587 | 0.9464 | 4.7 points |
| Decision Tree | 0.8417 | 0.7480 | 0.7048 | 0.6595 | 18.2 points |
| KNN | 0.9821 | 0.9711 | 0.9658 | 0.9570 | 2.5 points |
| Naive Bayes | 0.5477 | 0.5326 | 0.5308 | 0.5273 | 2.0 points |
| Support Vector Machine | 0.9904 | 0.9763 | 0.9680 | 0.9552 | 3.5 points |

- KNN, SVM and Logistic Regression all stay above 0.94 at 9 dB. KNN is the most robust of the three: it ends highest (0.9570) with the smallest drop (2.5 points).
- The Decision Tree loses the most (18.2 points), which fits its dependence on a few access points.
- Naive Bayes barely changes, but only because it is already low.
- The Streamlit app has sliders for the same noise and dropout stress test.

### Extra 2: validation on later-recorded data (`validationData.csv`)
1,111 fingerprints recorded after the training data (all 13 zones present, none dropped). The models are not retrained or tuned on this file. Zone counts: B0_F0 78, B0_F1 208, B0_F2 165, B0_F3 85, B1_F0 30, B1_F1 143, B1_F2 87, B1_F3 47, B2_F0 24, B2_F1 111, B2_F2 54, B2_F3 40, B2_F4 39.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8929 | 0.9910 | 0.9085 | 0.8929 | 0.8949 | 0.8814 |
| Support Vector Machine | 0.8470 | 0.9894 | 0.8740 | 0.8470 | 0.8488 | 0.8320 |
| KNN | 0.7921 | 0.9347 | 0.8156 | 0.7921 | 0.7933 | 0.7699 |
| Decision Tree | 0.6508 | 0.8783 | 0.7604 | 0.6508 | 0.6613 | 0.6281 |
| Naive Bayes | 0.4032 | 0.8751 | 0.4864 | 0.4032 | 0.3875 | 0.3503 |

Accuracy change from the test split to validation:

| ML Model Name | Test accuracy | Validation accuracy | Change |
|---|---|---|---|
| Logistic Regression | 0.9937 | 0.8929 | -10.1 points |
| Support Vector Machine | 0.9904 | 0.8470 | -14.3 points |
| Naive Bayes | 0.5477 | 0.4032 | -14.5 points |
| KNN | 0.9821 | 0.7921 | -19.0 points |
| Decision Tree | 0.8417 | 0.6508 | -19.1 points |

- **Every model gets worse.** In a random split, scans from the same spot land in both training and test sets, which makes the test scores look better than they would be in practice. The validation file was recorded later, so it is a harder check. Differences between sessions (devices, people, changes in the building) could be the reason, but this was not tested.
- **Logistic Regression holds up best** (about 10 points lost), then SVM. KNN and the Decision Tree lose about 19 points each.
- **AUC stays high** for Logistic Regression (0.9910) and SVM (0.9894) even though accuracy drops, so the right zone is still usually ranked near the top.
- The order of the models is almost the same as on the test split.

### Limitations and future work
- Everything here is one split with mostly default settings. Tuning (SVM's C and gamma, KNN's k, tree depth) using cross-validation on the training data is the obvious next step.
- The validation result suggests real-world accuracy for the best models is nearer 0.85 to 0.89 than 0.99.
- Room-level or coordinate prediction would be a harder extension on the same data.

---

## e. Streamlit app (SignalScope)

The app is called **SignalScope**, which is a building floor predictor using wifi signal It loads the five saved models and scores them on an uploaded CSV. It does not retrain anything, and nothing is shown until a file has been uploaded and has passed the format check. The header shows one WiFi scan as a strip of bars, one bar per access point.

**Left panel**
- **Your data:** upload a CSV. A format check runs on it: readable CSV, `zone` label present (or built from `BUILDINGID` and `FLOOR` in the raw UCI file), all 465 access point columns present, numeric cells, values between -105 and 0 dBm, only the 13 known zones, row count, and rows with no access point heard. Small problems are fixed automatically (column order, empty cells, the UCI value 100). Real problems stop the run and say what to change.
- **Model:** choose one of the five saved models.
- **Signal conditions (optional):** RSSI noise (dB) and access point dropout (%) applied to the uploaded data before scoring.
- A status list shows whether data, model, signal settings and results are ready.

**Main page**
- A one-paragraph summary of the result, then Accuracy, AUC, Precision, Recall, F1 and MCC for the chosen model, prediction time, and a button to download the predictions as CSV.
- **Confusion matrix** (counts or percentages).
- **Per-zone report:** classification report and a recall chart for each zone. summary of the result, then Accuracy, AUC, Precision, Recall, F1 and MCC for the chosen model
- **Leaderboard:** all five models on the same rows.
- **Building & floor:** building, floor and exact-zone accuracy.
- **One row:** pick one row and compare the predicted and actual building and floor.
