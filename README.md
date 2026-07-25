# Walmart Recruiting - Store Sales Forecasting

პროექტში გამოყენებულია Kaggle-ის კონკურსის **Walmart Recruiting - Store Sales Forecasting** მონაცემები. ამოცანაა ყოველკვირეული გაყიდვების (`Weekly_Sales`) პროგნოზირება თითოეული `(Store, Dept)` წყვილისთვის.

### რეპოზიტორიის სტრუქტურა

```
walmart-sales-forecasting/
├── README.md
├── .gitignore
├── submission.csv
├── submission_arima_order_1_0_0.csv
├── submission_patchtst_x_p8_s8_d64_layers2_exog64_drop0.1_huber_epochs19.csv
├── submission_tft_h64_emb8_drop0.3_huber_epochs13.csv
├── submission_blend_patchtst_x_650_tft_300_arima_050.csv
├── data/
│   └── raw/
│       ├── train.csv
│       ├── test.csv
│       ├── stores.csv
│       └── features.csv
├── notebooks/
│   ├── eda.ipynb
│   ├── feature_engineering_experiment.ipynb
│   ├── test_tabular_feature_engineer.ipynb
│   ├── model_experiment_XGBoost.ipynb
│   ├── model_experiment_LightGBM.ipynb
│   ├── model_experiment_NBEATS.ipynb
│   ├── model_experiment_DLinear.ipynb
│   ├── model_experiment_TFT.ipynb
│   ├── model_experiment_PatchTST.ipynb
│   ├── model_experiment_ARIMA.ipynb
│   ├── model_experiment_SARIMA.ipynb
│   ├── model_experiment_Prophet.ipynb
│   └── model_inference.ipynb
└── src/
    ├── data/
    │   ├── __init__.py
    │   ├── load_data.py
    │   └── splits.py
    ├── features/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── tabular.py
    │   └── neural.py
    └── datasets/
        ├── __init__.py
        └── window_dataset.py           
```

## 1. მონაცემები და შეფასების მეტრიკა

| ფაილი | ზომა | აღწერ|---|---|---|
| `train.csv` | 421,570 row | 2010-02-05 - 2012-10-26 (143 კვირა) |
| `test.csv` | 115,064 row | 2012-11-02 - 2013-07-26 (39 კვირა) |
| `stores.csv` | 45 row | მაღაზიის ტიპი (A/B/C) და ზომა |
| `features.csv` | 8,190 row | ტემპერატურა, საწვავის ფასი, MarkDown1-5, CPI, უმუშევრობა |

შეფასების მეტრიკაა **Weighted Mean Absolute Error (WMAE)**:

$$
WMAE = \frac{\sum_i w_i |y_i - \hat{y_i}|}{\sum_i w_i}
$$

სადაც:

$$
w_i =
\begin{cases}
5, & \text{თუ კვირა არის სადღესასწაულო} \\
1, & \text{სხვა შემთხვევაში}
\end{cases}
$$

ვალიდაციისთვის გამოყენებულია time split-ები:

- **Last-39 split**: `train.csv`-ის ბოლო 39 კვირა გამოყოფილია ვალიდაციად, რადგან `test.csv`-იც ზუსტად 39 კვირიან პროგნოზს მოითხოვს.
- **Calendar-aligned split**: დამატებით გამოიყენება კალენდარულად test-სთან მიახლოებული 39-კვირიანი პერიოდი, რათა უკეთ შეფასდეს მოდელის ქცევა holiday სეზონზე.

შემთხვევითი (`random`) split არ გამოიყენება, რადგან დროითი მწკრივის ამოცანაში ეს გამოიწვევდა leakage-ს ანუ მოდელს წვდომა ექნებოდა მომავლის ინფორმაციაზე.

## 2. EDA - მთავარი დასკვნები

სრული ანალიზი: `notebooks/eda.ipynb`

EDA-ის მიზანი იყო მონაცემების დროითი სტრუქტურის, Store-Dept წყვილების დაფარვის, target-ის განაწილების, holiday ეფექტის და missing values-ის შეფასება.

### 2.1 Store-Dept სტრუქტურა

`train` და `test` ორივე შეიცავს ერთსა და იმავე 45 მაღაზიას და 81 დეპარტამენტს, თუმცა ზუსტი `(Store, Dept)` წყვილების დონეზე დაფარვა სრულად არ ემთხვევა:

- train-ში გვხვდება 3,331 Store-Dept წყვილი
- test-ში გვხვდება 3,169 Store-Dept წყვილი
- 3,158 წყვილი გვხვდება ორივეში
- 11 წყვილი გვხვდება მხოლოდ test-ში
- 173 წყვილი გვხვდება მხოლოდ train-ში

ეს მნიშვნელოვანია რადგან lag/rolling ტიპის feature-ები და univariate time-series მოდელები საჭიროებენ fallback მექანიზმს ისეთ Store-Dept წყვილებზე, რომლებსაც ისტორია არ აქვთ.

<img width="1169" height="505" alt="image" src="https://github.com/user-attachments/assets/bc340308-fdba-46fd-aa80-ea8e31077aa5" />

<img width="687" height="468" alt="image" src="https://github.com/user-attachments/assets/e8da4b30-40a6-4f17-8370-4fefd460d6a2" />

### 2.2 Target (`Weekly_Sales`)

`Weekly_Sales` ძლიერად skewed არის. უმეტესობა დაბალი ან საშუალო გაყიდვების მქონე Store-Dept კვირებია, ხოლო მცირე რაოდენობის დეპარტამენტები ქმნის ძალიან მაღალ პიკებს.

<img width="868" height="468" alt="image" src="https://github.com/user-attachments/assets/87dfddea-8b35-4512-82bd-eafc0cdc6a06" />

ასევე აღმოჩნდა მცირე რაოდენობის უარყოფითი და ნულოვანი გაყიდვები:

- negative sales: 1,285 row
- zero sales: 73 row
- positive sales: 420,212 row

უარყოფითი მნიშვნელობები სავარაუდოდ უკავშირდება დაბრუნებებს, კორექციებს ან ინვენტარის/ბუღალტრული ჩანაწერების სპეციფიკას. მოდელირებისას პროგნოზები საბოლოოდ 0-ზე ქვემოთ იჭრება postprocessing-ის სტრატეგიით (`clip(lower=0)`), რადგან Kaggle submission-ში უარყოფითი გაყიდვები პრაქტიკულად არ არის სასურველი.

### 2.3 Holiday ეფექტი

სადღესასწაულო კვირები train-ში მხოლოდ დაახლოებით 7.04%-ია, ხოლო test-ში დაახლოებით 7.76%. თუმცა WMAE-ში 5x წონის გამო მათი ეფექტური წვლილი საბოლოო შეფასებაში ბევრად უფრო დიდია.

საშუალო გაყიდვები:

- holiday weeks: 17,035.82
- non-holiday weeks: 15,901.45

<img width="558" height="468" alt="image" src="https://github.com/user-attachments/assets/a89e13f5-a371-4b1a-9a8d-9b602c066438" />

საშუალო დონეზე განსხვავება მხოლოდ დაახლოებით 7%-ია, მაგრამ დეპარტამენტების მიხედვით holiday ეფექტი ძალიან არაერთგვაროვანია. ზოგიერთ დეპარტამენტში სადღესასწაულო კვირები მკვეთრ პიკებს იწვევს, ზოგიერთში კი ეფექტი მცირეა ან საერთოდ არ ჩანს.

ასევე ჩანს, რომ ყველაზე დიდი გაყიდვების პიკები ყოველთვის ზუსტად `IsHoliday=True` კვირებზე არ მოდის. ხშირად პიკი შეიძლება holiday-მდე ან მის შემდეგ იყოს, რაც ამართლებს დამატებით calendar feature-ებს: `WeekOfYear`, `Month`, კონკრეტული holiday flag-ები და cyclic week encoding.

<img width="1143" height="505" alt="image" src="https://github.com/user-attachments/assets/4f38704a-1f75-4984-b0e8-f458db5d8f36" />

### 2.4 Missing values

Missing values ძირითადად სტრუქტურულია და არა შემთხვევითი:

- `MarkDown1-5` მასობრივად აკლია 2011-11-11-მდე, რადგან markdown მონაცემები ამ პერიოდამდე უბრალოდ არ არსებობდა
- `CPI` და `Unemployment` აკლია test-ის ბოლო 13 კვირაში ყველა 45 მაღაზიისთვის, სულ 585 მწკრივი
- MarkDown ცვლადები განსხვავებული missingness-ით ხასიათდება, დაახლოებით 50-64% ფარგლებში

ამის გამო გამოყენებულია შემდეგი მიდგომები:

- MarkDown missing values -> 0-fill + missingness indicator
- MarkDown raw ვერსიები შენარჩუნებულია ხის-ტიპის მოდელებისთვის
- CPI/Unemployment -> forward-fill თითოეული Store-ის შიგნით

<img width="1152" height="505" alt="image" src="https://github.com/user-attachments/assets/361c85ba-8fb8-4c8d-9020-3d77340ea841" />

EDA-დან გამომდინარე, საბოლოო pipeline-ს სჭირდება:

- Store/Dept იდენტიფიკატორები
- calendar feature-ები
- კონკრეტული holiday flag-ები
- MarkDown-ის დამუშავებული და raw ვერსიები
- lag/rolling sales history feature-ები tabular მოდელებისთვის
- sequence window-ები neural time-series მოდელებისთვის
- fallback მექანიზმი უცნობი ან არასრული Store-Dept წყვილებისთვის
  
---

## 3. Feature Engineering

სრული ექსპერიმენტები:

- `notebooks/feature_engineering_experiment.ipynb`
- `notebooks/test_tabular_feature_engineer.ipynb`

საბოლოო preprocessing ლოგიკა გატანილია `src/` მოდულებში, რათა იგივე pipeline გამოყენებული იყოს training, validation და inference ეტაპებზე.

### 3.1 `WalmartBasePreprocessor` - საერთო preprocessing

`WalmartBasePreprocessor` გამოიყენება როგორც საერთო საწყისი ფენა tabular და neural მოდელებისთვის.

ძირითადი ნაბიჯები:

- `train`/`test` მონაცემების merge `stores.csv` და `features.csv`-თან `Store`, `Date`, `IsHoliday` სვეტებზე
- `Date` სვეტის datetime ფორმატში გადაყვანა
- CPI/Unemployment missing values-ის forward-fill თითოეული Store-ის შიგნით
- `MarkDown1-5` missing values-ის დამუშავება:
  - `_was_missing` indicator თითოეული MarkDown სვეტისთვის
  - 0-fill neural/tabular-compatible ვერსიისთვის
  - raw NaN-შენარჩუნებული ვერსია ხის-ტიპის მოდელებისთვის
- calendar feature-ები:
  - `Year`
  - `Month`
  - `WeekOfYear`
  - `Week_sin`
  - `Week_cos`
- კონკრეტული holiday flag-ები:
  - `IsSuperBowl`
  - `IsLaborDay`
  - `IsThanksgiving`
  - `IsChristmas`
- Store Type encoding:
  - label encoding (`Type_encoded`)
  - one-hot encoding (`Type_A`, `Type_B`, `Type_C`)
- MarkDown aggregate feature-ები:
  - `total_markdown`
  - `abs_total_markdown`
  - `positive_markdown_sum`
  - `negative_markdown_sum`
  - `has_markdown_signal`
  - `markdown_missing_count`
  - `markdown_available_period`

ეს preprocessing target-ს არ იყენებს, ამიტომ უსაფრთხოა როგორც train/validation/test მონაცემებზე.

### 3.2 `WalmartTabularFeatureEngineer` - ხის-ტიპის მოდელებისთვის

Tabular feature engineering გამოიყენება XGBoost/LightGBM ტიპის მოდელებისთვის, სადაც მიზანია Store-Dept დონეზე გაყიდვების ისტორიის გამოყენება.

დამატებული feature-ები:

- lag feature-ები:
  - `Sales_lag_1`
  - `Sales_lag_4`
  - `Sales_lag_52`
- rolling mean/std feature-ები:
  - `Sales_roll_mean_4`
  - `Sales_roll_std_4`
  - `Sales_roll_mean_12`

ყველა rolling feature ითვლება `shift(1)`-ის შემდეგ, რათა მიმდინარე კვირის target არ მოხვდეს feature-ებში. ეს აუცილებელია leakage-ის თავიდან ასაცილებლად.

`WalmartTabularFeatureEngineer` იყენებს განსხვავებულ მეთოდებს train და future მონაცემებისთვის:

- `fit()` სწავლობს მხოლოდ training history-დან
- `transform_train()` ქმნის lag/rolling feature-ებს train პერიოდში
- `transform_future()` validation/test პერიოდისთვის იყენებს მხოლოდ წარსულის ცნობილ history-ს და არასდროს იყენებს validation/test target მნიშვნელობებს

Fallback:

- უცნობი Store-Dept წყვილებისთვის lag/rolling feature-ები ივსება დეპარტამენტის საშუალო გაყიდვით
- rolling std fallback არის 0

Leakage-safety შემოწმებულია `test_tabular_feature_engineer.ipynb`-ში ხელით გამოთვლილი lag მნიშვნელობის შედარებით კლასის output-თან.

### 3.3 Neural preprocessing და sequence window-ები

Neural მოდელებისთვის (`DLinear`, `TFT`, `PatchTST`, `N-BEATS`) tabular lag feature-ები პირდაპირ არ გამოიყენება. ამის ნაცვლად იქმნება sequence dataset, სადაც თითოეული Store-Dept წყვილი დროით მწკრივად განიხილება.

გამოყენებული სტრუქტურა:

- context length: 52 კვირა
- prediction length: 39 კვირა
- target: `Weekly_Sales_scaled`
- series id: Store-Dept წყვილის უნიკალური identifier
- static categorical feature-ები:
  - `Store`
  - `Dept`
  - `Type`
- static real feature:
  - `Size`
- known future real feature-ები:
  - calendar feature-ები
  - holiday flag-ები
  - MarkDown/economic feature-ები, მოდელის ვარიანტის მიხედვით.

Target scaling ითვლება მხოლოდ training history-ზე. Scaling fallback იერარქიულია:

1. Store-Dept საშუალო/std
2. Dept საშუალო/std
3. Store საშუალო/std
4. Global საშუალო/std

ეს აუცილებელია ისეთი Store-Dept წყვილებისთვის, რომლებსაც არასრული ან არასტაბილური ისტორია აქვთ.

Sequence dataset-ები წინასწარ ინახება tensor-ებად (`WalmartPrecomputedTrainingWindowDataset`, `WalmartPrecomputedForecastWindowDataset`), რაც მნიშვნელოვნად აჩქარებს neural მოდელების training-ს.

### 3.4 Extra Fallbacks (Statistical model)

Fallback იერარქია (ARIMA-ში):

1. Store-Dept ბოლო 13 კვირის საშუალო
2. Dept ბოლო 13 კვირის საშუალო
3. Store ბოლო 13 კვირის საშუალო
4. global ბოლო 13 კვირის საშუალო

---

## 4. მოდელის არქიტექტურები

### 4.1 Tree-Based Models

#### XGBoost - `notebooks/model_experiment_XGBoost.ipynb`
MLflow ექსპერიმენტი: [`XGBoost_Training`](https://dagshub.com/LukaBatilashvili07/walmart-sales-forecasting.mlflow/#/experiments/2/runs?searchFilter=&orderByKey=metrics.%60final_valid_wmae%60&orderByAsc=true&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D)

| Run | აღწერა | შედეგი |
|---|---|---|
| `XGBoost_Cleaning` | `WalmartBasePreprocessor` გამოყენება train/valid-ზე | train_rows/valid_rows/NaN რაოდენობა დალოგილია |
| `XGBoost_Feature_Selection` | baseline (`n_estimators=300, max_depth=6`), feature importance | WMAE = 5587.68 |
| `XGBoost_CV` | Walk-forward CV, 3 fold, 13-კვირიანი ვალიდაცია | ფოლდები: 5170.89 / 4861.34 / 4825.57 - mean 4952.60 ± 155.04 |
| `XGBoost_HPO` | Optuna, 30 trial, პირდაპირ WMAE-ის მინიმიზაცია სამივე fold-ის საშუალოზე | Best WMAE = 4185.08 (trial 13) |
| `XGBoost_Best` | საუკეთესო პარამეტრებით საბოლოო მოდელი, Pipeline-ად რეგისტრირებული | Final validation WMAE = 5017.81 |
| `XGBoost_Final_Refit` | გადატრენინგება მთელ 143-კვირიან train.csv-ზე (421,570 row) | ახალი model ვერსია რეგისტრირებულია |

**საუკეთესო feature-ები** (importance-ის მიხედვით baseline მოდელში): `Sales_roll_mean_4` (0.434), `IsThanksgiving` (0.211), `StoreDept_avg_sales` (0.134), `Sales_lag_1` (0.125), `Sales_diff_1` (0.023) - ანუ ისტორიული გაყიდვების feature-ები (rolling mean, lag) დომინირებს, calendar/holiday feature-ები მეორეხარისხოვანია.

**საუკეთესო ჰიპერპარამეტრები**: `n_estimators=400, max_depth=8, learning_rate=0.1848, subsample=0.7245, colsample_bytree=0.7136, min_child_weight=4, reg_lambda=1.4218`

დაკვირვება: საბოლოო validation WMAE (5017.81) უფრო მაღალია, ვიდრე HPO-ს CV-ის საშუალო (4185.08) - სავარაუდო მიზეზი: CV fold-ები (2010-02 – 2011-10 პერიოდიდან აგებული) არცერთი არ მოიცავს Thanksgiving/Christmas-ის სეზონს, ხოლო საბოლოო ვალიდაციის მონაკვეთი (2011-11-04 – 2012-07-27) სწორედ ამ სადღესასწაულო პიკს მოიცავს - ანუ მოდელს რეალურად უჭირს ამ პერიოდის ზუსტი პროგნოზირება.

#### LightGBM - `notebooks/model_experiment_LightGBM.ipynb`
MLflow ექსპერიმენტი: [`LightGBM_Training`](https://dagshub.com/LukaBatilashvili07/walmart-sales-forecasting.mlflow/#/experiments/3/runs?searchFilter=&orderByKey=metrics.%60best_wmae%60&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D). მეორდება იგივე pipeline რაც XGBoost-ისთვის (იგივე `WalmartBasePreprocessor`/`WalmartTabularFeatureEngineer`, იგივე split-ები, იგივე CV schema), პირდაპირი შედარებისთვის. განსხვავება: LightGBM ტესტდება "raw" (NaN-შენარჩუნებული) MarkDown სვეტებით 0-fill+indicator-ის ნაცვლად, რადგან LightGBM-ს NaN-ის საკუთარი native split-ის ლოგიკა აქვს.

| Run | აღწერა | შედეგი |
|---|---|---|
| `LightGBM_Cleaning` | `WalmartBasePreprocessor` | - |
| `LightGBM_Feature_Selection` | baseline, raw MarkDown ვარიანტი | WMAE = 5307.84 |
| `LightGBM_CV` | Walk-forward CV, 3 fold | ფოლდები: 5177.87 / 4966.51 / 4829.07 - mean 4991.15 ± 143.46 |
| `LightGBM_HPO` | Optuna, 30 trial | Best WMAE = 3948.19 (trial 14) |
| `LightGBM_Best` | საბოლოო მოდელი | Final validation WMAE = 4449.90 |
| `LightGBM_Final_Refit` | გადატრენინგება მთელ train.csv-ზე (421,570 row) | ახალი model ვერსია რეგისტრირებულია |

**საუკეთესო ჰიპერპარამეტრები**: `n_estimators=700, num_leaves=197, max_depth=-1, learning_rate=0.0690, subsample=0.7020, subsample_freq=2, colsample_bytree=0.6004, min_child_samples=36, reg_lambda=0.3755`

**საუკეთესო feature-ები** (split-count მიხედვით): `Sales_diff_1` (1367), `Sales_lag_1` (1316), `StoreDept_avg_sales` (1059), `Sales_seasonal_avg` (585), `Sales_roll_std_4` (455).

**LightGBM vs XGBoost**: LightGBM-ის HPO-ს საუკეთესო CV ქულა (3948.19) და საბოლოო validation WMAE (4449.90) **მკვეთრად სჯობს** XGBoost-ს (4185.08 / 5017.81 შესაბამისად) - ეს მხარს უჭერს ჰიპოთეზას, რომ LightGBM-ის leaf-wise ზრდა და native missing-value handling უფრო კარგად ერგება ამ ამოცანას. ორივე მოდელისთვის CV fold-ების პატერნი იდენტურია - ყველაზე ადრეული fold (2011-02 – 2011-04) ორივესთვის ყველაზე რთულია, დანარჩენი ორი ფოლდი თანდათან უმჯობესდება.

### 4.2 Deep Learning

Deep Learning მოდელებისთვის გამოყენებულია Store-Dept სერიების გლობალური სწავლება: თითოეული `(Store, Dept)` წყვილი განიხილება ცალკე მწკრივად, მაგრამ მოდელი საერთო წონებით ტრენინგდება ყველა სერიაზე ერთად. ეს მიდგომა განსაკუთრებით მნიშვნელოვანია, რადგან ბევრ ინდივიდუალურ სერიას მხოლოდ 143 ისტორიული კვირა აქვს და ცალ-ცალკე ღრმა მოდელის სწავლება არაპრაქტიკულია.

საერთო sequence პარამეტრები:

- `context_length = 52` კვირა
- `prediction_length = 39` კვირა
- target scaling ითვლება მხოლოდ train history-ზე
- გამოყენებულია Store/Dept/Type static identifiers
- known-future covariates მოიცავს calendar/holiday feature-ებს, ხოლო ზოგ ექსპერიმენტში MarkDown/economic covariates-იც დაემატა

#### N-BEATS - `notebooks/model_experiment_NBEATS.ipynb`
MLflow ექსპერიმენტი: [`NBEATS_Training`](https://dagshub.com/LukaBatilashvili07/walmart-sales-forecasting.mlflow/#/experiments/8/runs?searchFilter=&orderByKey=metrics.%60calendar_best_valid_wmae%60&orderByAsc=true&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D). Generic N-BEATS - feed-forward ნეირონული ქსელი, დაფუძნებული backcast/forecast doubly-residual stacking მექანიზმზე, ტრენინგდება **გლობალურად** ყველა Store-Dept სერიაზე ერთად, საერთო წონებით, მაგრამ არის **univariate** - მოდელს მხოლოდ `past_target` (52-კვირიანი ისტორია) მიეწოდება, კოვარიატების გარეშე.

ორი ვალიდაციის სქემა: **last-39-weeks** (ბოლო კვირები) და **calendar-aligned** (2011-11-04 – 2012-07-27, იმავე სეზონურ პერიოდზე, რაც რეალური test.csv, საბოლოო შესარჩევად).

| ეტაპი | აღწერა | შედეგი |
|---|---|---|
| Baselines | Seasonal-naive (t-52) და last-value-naive | Seasonal-naive WMAE = **2064.31**; Last-value-naive WMAE = 3863.15 |
| Last-39 screening grid | 5 არქიტექტურული კონფიგურაცია (2-5 stack, 64-512 hidden units) | საუკეთესო: WMAE ≈ 2333.89 (stacks=3, blocks=3, hidden=128, layers=3) |
| Calendar-aligned დამოწმება | ტოპ-3 კონფიგურაციის გადატესტვა რეალურ test-სეზონთან მიმსგავსებულ პერიოდზე | საუკეთესო: WMAE ≈ 3027.12 (stacks=5, blocks=4, hidden=512, layers=4) |
| Final training | საბოლოო refit calendar-aligned საუკეთესო კონფიგურაციით, 100 epoch | **Final validation WMAE = 3429.57** (epoch 9) |

**ჰიპერპარამეტრები**: `context_length=52, prediction_length=39, num_stacks=5, num_blocks_per_stack=4, hidden_units=512, num_layers=4, lr=0.0001, weight_decay=0.0001, loss=Huber`

**მნიშვნელოვანი დასკვნა**: N-BEATS-ის საბოლოო შედეგი (3429.57) **უფრო ცუდია**, ვიდრე უბრალო seasonal-naive baseline (2064.31).

#### DLinear - `notebooks/model_experiment_DLinear.ipynb`

MLflow ექსპერიმენტი: [`DLinear_Training`](https://dagshub.com/LukaBatilashvili07/walmart-sales-forecasting.mlflow/#/experiments/1/runs?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D).

DLinear გამოყენებულია როგორც მარტივი linear deep time-series baseline. მოდელი შლის სერიას trend და seasonal კომპონენტებად და შემდეგ თითოეულ კომპონენტს linear projection-ით პროგნოზირებს. მისი უპირატესობაა სიმარტივე, სწრაფი სწავლება და ძლიერი baseline-ობა მოკლე სერიებზე.

გაიტესტა ორი ძირითადი ვერსია:

- **DLinear target-only** - მხოლოდ `Weekly_Sales_scaled` history
- **DLinear-X** - target history + known future covariates/static features correction მექანიზმით

| მიდგომა | აღწერა | შედეგი |
|---|---|---:|
| DLinear target-only | მხოლოდ 52-კვირიანი target context | სტაბილური baseline |
| DLinear-X with MarkDown | calendar + holiday + MarkDown covariates | MarkDown feature-ებმა შედეგი გააუარესა |
| DLinear-X stable/no-MarkDown | calendar/holiday/static feature-ები MarkDown-ის გარეშე | საუკეთესო validation შედეგი ≈ **2403** last-39 split-ზე, ≈ **3461** calendar-aligned split - ზე|
| Final DLinear submission | calendar final target-only ვარიანტი | Kaggle Public = **3649.46**, Private = **3789.31** |

**დასკვნა:** DLinear-მა აჩვენა რომ მარტივი linear architecture-ებიც კონკურენტუნარიანია, განსაკუთრებით last-39 validation-ზე. თუმცა Kaggle test-ზე მისი standalone შედეგი PatchTST/TFT-ზე სუსტი იყო. MarkDown covariates-მა DLinear-ისთვის noise დაამატა, ამიტომ უფრო სტაბილური იყო no-MarkDown ვარიანტები.

<img width="1185" height="82" alt="image" src="https://github.com/user-attachments/assets/cf3c71d5-c7cf-457e-87bf-f05c88e807a5" />

#### TFT - `notebooks/model_experiment_TFT.ipynb`

MLflow ექსპერიმენტი: [`TFT_Training`](https://dagshub.com/LukaBatilashvili07/walmart-sales-forecasting.mlflow/#/experiments/6/runs?searchFilter=&orderByKey=metrics.%60calendar_best_valid_wmae%60&orderByAsc=true&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D).

Temporal Fusion Transformer გამოყენებულია როგორც feature-aware sequence model. TFT-ს შეუძლია ერთდროულად გამოიყენოს:

- historical target context
- static categorical embeddings (`Store`, `Dept`, `Type`)
- static real feature (`Size`)
- known future real covariates
- calendar და holiday indicators.

TFT-ის მთავარი იდეა იყო, რომ Store/Dept-ების განსხვავებული ქცევა და მომავალი calendar/holiday ინფორმაცია მოდელში პირდაპირ შესულიყო.

საუკეთესო final submission-ის last-39 split score ≈ **2224**, calendar-aligned split score ≈ **3457**.

| მოდელი | კონფიგურაცია | Kaggle Public | Kaggle Private |
|---|---|---:|---:|
| TFT | `hidden=64, embedding=8, dropout=0.3, loss=Huber, epochs=13` | **3041.21** | **3156.09** |

**დასკვნა:** TFT-მ მნიშვნელოვნად აჯობა DLinear-ს და N-BEATS-ს Kaggle test-ზე. ეს აჩვენებს, რომ static identifiers და known future covariates რეალურად სასარგებლოა Walmart-ის Store-Dept forecasting ამოცანაში.

<img width="1174" height="90" alt="image" src="https://github.com/user-attachments/assets/3b4244d3-088f-4a17-bcdb-4265a4ab0358" />

#### PatchTST - `notebooks/model_experiment_PatchTST.ipynb`

MLflow ექსპერიმენტი: [`PatchTST_Training`](https://dagshub.com/LukaBatilashvili07/walmart-sales-forecasting.mlflow/#/experiments/7/runs?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D).

PatchTST იყო საუკეთესო standalone deep learning მოდელი. მოდელი იყენებს მწკრივის patch-ებად დაყოფას და Transformer encoder-ს, რაც მას საშუალებას აძლევს დაიჭიროს გრძელვადიანი pattern-ები 52-კვირიანი context-იდან.

გაიტესტა ორი ძირითადი ვერსია:

- **PatchTST target-only** - მხოლოდ target history
- **PatchTST-X** - PatchTST backbone + static/covariate correction head

საუკეთესო final submission-ის last-39 split score ≈ **2238**, calendar-aligned split score ≈ **3411**.

| მოდელი | კონფიგურაცია | Kaggle Public | Kaggle Private |
|---|---|---:|---:|
| PatchTST-X | `patch_length=8, stride=8, d_model=64, layers=2, exog_hidden=64, dropout=0.1, loss=Huber, epochs=19` | **2966.15** | **3026.98** |

**დასკვნა:** PatchTST-X გახდა საუკეთესო standalone მოდელი. TFT-სთან შედარებით PatchTST-X უკეთ განზოგადდება test set-ზე, განსაკუთრებით Private score-ზე. სავარაუდო მიზეზია patch-based representation, რომელიც კარგად იჭერს წლიურ/სეზონურ pattern-ებს მოკლე Store-Dept სერიებში.

<img width="1186" height="83" alt="image" src="https://github.com/user-attachments/assets/35fe7f1b-4729-458c-a32e-ad6d24ee57d5" />

#### Deep Learning შეჯამება

| მოდელი | ძირითადი იდეა | საუკეთესო ცნობილი შედეგი |
|---|---|---:|
| N-BEATS | Univariate global feed-forward forecasting | Final validation WMAE = **3429.57** |
| DLinear | Linear decomposition baseline | Kaggle Private = **3789.31** |
| TFT | Feature-aware temporal transformer | Kaggle Private = **3156.09** |
| PatchTST-X | Patch-based transformer + exogenous correction | Kaggle Private = **3026.98** |

Deep Learning ექსპერიმენტებიდან მთავარი დასკვნაა რომ feature-aware neural models აშკარად სჯობს წმინდად univariate neural baseline-ს. საუკეთესო standalone მოდელი გახდა **PatchTST-X**, ხოლო TFT იყო მეორე ყველაზე ძლიერი deep model.

### 4.3 Classical Statistical Time-Series Models

Classical time-series მოდელებში გაიტესტა ორი განსხვავებული მიდგომა:

1. **ARIMA Store-Dept დონეზე** - თითოეული `(Store, Dept)` წყვილისთვის ცალკე univariate ARIMA მოდელი
2. **SARIMA/Prophet აგრეგირებულ დონეზე** - ყველა Store/Dept გაყიდვის ჯამურ weekly სერიაზე.

ეს განსხვავება მნიშვნელოვანია: ARIMA-ის WMAE ითვლება Store-Dept row დონეზე და შედარებადია სხვა Store-Dept მოდელებთან, ხოლო SARIMA/Prophet-ის შედეგები აგრეგირებულ weekly sales დონეზეა და **პირდაპირ შედარებადი არ არის** XGBoost/LightGBM/Deep Learning შედეგებთან.

#### ARIMA - `notebooks/model_experiment_ARIMA.ipynb`

MLflow ექსპერიმენტი: [`ARIMA_Training`](https://dagshub.com/LukaBatilashvili07/walmart-sales-forecasting.mlflow/#/experiments/9/runs?searchFilter=&orderByKey=metrics.%60valid_wmae%60&orderByAsc=true&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D).

ARIMA გამოყენებულია როგორც classical univariate baseline Store-Dept დონეზე. თითოეული `(Store, Dept)` სერია ცალკე მოდელდება თავისი ისტორიული `Weekly_Sales` მნიშვნელობებით.

ARIMA-ს ფორმაა:

`ARIMA(p, d, q)`

სადაც:

- `p` არის autoregressive lag-ების რაოდენობა
- `d` არის differencing-ის რაოდენობა
- `q` არის lagged residual error-ების რაოდენობა moving-average კომპონენტში

გაიტესტა რამდენიმე მარტივი order:

| Order | ინტერპრეტაცია |
|---|---|
| `(0,1,0)` | random walk baseline |
| `(1,0,0)` | AR(1), differencing-ის გარეშე |
| `(1,1,0)` | differenced AR model |
| `(0,1,1)` | differenced MA model |
| `(1,1,1)` | მცირე სრული ARIMA |
| `(2,1,0)` | ოდნავ უფრო დიდი differenced AR model |

Fallback strategy:

1. Store-Dept ბოლო 13 კვირის საშუალო
2. Dept ბოლო 13 კვირის საშუალო
3. Store ბოლო 13 კვირის საშუალო
4. global ბოლო 13 კვირის საშუალო

Last-39 validation-ზე საუკეთესო order აღმოჩნდა `ARIMA(1,0,0)`:

| Order | Validation WMAE | Kaggle Public | Kaggle Private |
|---|---:|
| `ARIMA(1,0,0)` | **2588.45** | **5030.89** | 4796.84 |

**დასკვნა:** `ARIMA(1,0,0)`-ის მოგება ლოგიკურია ამ მონაცემებისთვის. Walmart Store-Dept სერიები ხშირად უფრო level-stationary/noisy სერიებია, ვიდრე მკაფიო trend-ის მქონე სერიები. Differencing (`d=1`) ზოგჯერ აშორებს სასარგებლო sales level ინფორმაციას და პროგნოზს უფრო არასტაბილურს ხდის. AR(1) კი ინარჩუნებს გაყიდვების დონეს და მოქმედებს როგორც მარტივი mean-reverting baseline.

მიუხედავად ძლიერი validation შედეგისა, ARIMA-ს მნიშვნელოვანი შეზღუდვაა ის, რომ თითოეული Store-Dept სერია დამოუკიდებლად მოდელდება და არ ხდება ინფორმაციის გაზიარება მაღაზიებსა და დეპარტამენტებს შორის. ასევე მოდელი პირდაპირ ვერ იყენებს holiday, MarkDown, Store Type და სხვა known future covariates-ს.

<img width="1202" height="87" alt="image" src="https://github.com/user-attachments/assets/726e5cc8-081b-4147-b581-82b475d194f0" />

#### SARIMA - `notebooks/model_experiment_SARIMA.ipynb`
MLflow ექსპერიმენტი: [`SARIMA_Training`](https://dagshub.com/LukaBatilashvili07/walmart-sales-forecasting.mlflow/#/experiments/4/runs?searchFilter=&orderByKey=metrics.%60holdout_roc_auc%60&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D). მოდელი: `SARIMA(1,1,1)(1,1,1,52)` - სეზონურობის პერიოდი `s=52` ირჩევა წლიური (52-კვირიანი) ციკლის ასახვისთვის (მაგ. საშობაო პიკის განმეორება ყოველწლიურად).

| Run | აღწერა | შედეგი |
|---|---|---|
| `SARIMA_Baseline` | `order=(1,1,1)`, `seasonal_order=(1,1,1,52)`, tuning არ ჩატარებულა | WMAE (აგრეგირებულ სერიაზე) = **1,204,244.21**, AIC = 10.0 |

*შენიშვნა*: `enforce_stationarity=False`/`enforce_invertibility=False` საჭირო გახდა, რადგან 104 კვირა არასაკმარისი აღმოჩნდა სეზონური კომპონენტის საწყისი პარამეტრების სანდო შეფასებისთვის (statsmodels-ის warning: "Too few observations to estimate starting parameters for seasonal ARMA").

#### Prophet - `notebooks/model_experiment_Prophet.ipynb`
MLflow ექსპერიმენტი: [`Prophet_Training`](https://dagshub.com/LukaBatilashvili07/walmart-sales-forecasting.mlflow/#/experiments/5/runs?searchFilter=&orderByKey=metrics.%60holdout_roc_auc%60&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D). Prophet ავტორეგრესიის ნაცვლად სერიას შლის კომპონენტებად (trend + yearly seasonality + holiday ეფექტები), რომლებიც ცალ-ცალკე მოდელირდება და ჯამდება.

| Run | აღწერა | შედეგი |
|---|---|---|
| `Prophet_Baseline` | `yearly_seasonality=True`, `weekly_seasonality=False`, სადღესასწაულო კვირები გადაცემულია `holidays` პარამეტრით, default tuning | WMAE (აგრეგირებულ სერიაზე) = **1,728,319.66** |

**SARIMA vs Prophet**: იმავე აგრეგირებულ სერიაზე, იმავე 39-კვირიან ვალიდაციაზე, SARIMA-მ საგრძნობლად აჯობა Prophet-ს (1,204,244.21 vs 1,728,319.66 - დაახლოებით 30%-იანი გაუმჯობესება). შესაძლო მიზეზები:
- SARIMA-ს ავტორეგრესიული მეხსიერება (AR/MA წევრები) პირდაპირ იჭერს ბოლო რამდენიმე კვირის დინამიკას, ხოლო Prophet ამას საერთოდ არ აკეთებს - მხოლოდ გლუვ trend/seasonality დეკომპოზიციაზეა დაფუძნებული.
- Prophet-ის default პარამეტრები (`changepoint_prior_scale`, `seasonality_prior_scale`) საერთოდ არ დარეგულირებულა - შესაძლოა უფრო ფრთხილი tuning-ით შედეგი გაუმჯობესდეს.

#### Classical models შეჯამება

| მოდელი | დონე | შედარებადობა | შედეგი |
|---|---|---|---:|
| ARIMA(1,0,0) | Store-Dept | შედარებადია Store-Dept მოდელებთან | Validation WMAE = **2588.45** |
| SARIMA | Aggregated weekly sales | პირდაპირ არაა შედარებადი | Aggregated WMAE = **1,204,244.21** |
| Prophet | Aggregated weekly sales | პირდაპირ არაა შედარებადი | Aggregated WMAE = **1,728,319.66** |

Classical მოდელებიდან ყველაზე პრაქტიკული აღმოჩნდა **ARIMA(1,0,0)** Store-Dept დონეზე. SARIMA და Prophet სასარგებლო იყო aggregated-level baseline-ისთვის, მაგრამ მათი შედეგები არ უნდა შევადაროთ პირდაპირ Store-Dept forecasting მოდელებს.

---

## 5. მოდელების შედარება

მოდელების შედარებისას მნიშვნელოვანია ორი განსხვავებული შედეგის გამიჯვნა:

- **Validation WMAE** - ლოკალური ვალიდაცია `train.csv`-ის 39-კვირიან holdout-ზე
- **Kaggle Public/Private score** - საბოლოო `test.csv` submission-ის რეალური leaderboard შედეგი.

### [validation შედეგები](https://dagshub.com/LukaBatilashvili07/walmart-sales-forecasting.mlflow/#/compare-experiments/s?experiments=%5B%221%22%2C%222%22%2C%223%22%2C%224%22%2C%225%22%2C%226%22%2C%227%22%2C%228%22%2C%229%22%5D&searchFilter=&orderByKey=metrics.%60calendar_best_valid_wmae%60&orderByAsc=true&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D)

შენიშვნა: SARIMA და Prophet არ არის ამ ცხრილში შეტანილი, რადგან ისინი აგრეგირებულ weekly sales სერიაზე დაიტრენინგდა და მათი WMAE სხვა მასშტაბზეა.

### 5.2 Aggregated classical models

| მოდელი | დონე | WMAE | შედარებადობა |
|---|---|---:|---|
| SARIMA | Aggregated weekly sales | **1,204,244.21** |
| Prophet | Aggregated weekly sales | **1,728,319.66** |

---

## 6. MLflow / DagsHub სტრუქტურა

https://dagshub.com/LukaBatilashvili07/walmart-sales-forecasting.mlflow

ყველა ძირითად მოდელის არქიტექტურას აქვს ცალკე MLflow ექსპერიმენტი. ექსპერიმენტებში დალოგილია preprocessing, validation runs, hyperparameter/configuration runs და final/refit runs.

```text
XGBoost_Training/
  ├── XGBoost_Cleaning
  ├── XGBoost_Feature_Selection
  ├── XGBoost_CV (+ nested fold-run-ები)
  ├── XGBoost_HPO (+ nested trial-run-ები)
  ├── XGBoost_Best
  └── XGBoost_Final_Refit

LightGBM_Training/
  ├── LightGBM_Cleaning
  ├── LightGBM_Feature_Selection
  ├── LightGBM_CV (+ nested fold-run-ები)
  ├── LightGBM_HPO (+ nested trial-run-ები)
  ├── LightGBM_Best
  └── LightGBM_Final_Refit

NBEATS_Training/
  ├── NBEATS_Preprocessing
  ├── NBEATS_Baselines
  ├── NBEATS_Last39_* 
  ├── NBEATS_CalendarAligned_*
  └── NBEATS_CalendarAligned_Final_*

DLinear_Training/
  ├── DLinear_Preprocessing
  ├── DLinear_TargetOnly_*
  ├── DLinear_X_*
  ├── DLinear_Last39_*
  └── DLinear_Final_*

TFT_Training/
  ├── TFT_Preprocessing
  ├── TFT_Last39_*
  ├── TFT_CalendarAligned_*
  └── TFT_Final_*

PatchTST_Training/
  ├── PatchTST_Preprocessing
  ├── PatchTST_TargetOnly_*
  ├── PatchTST_X_*
  ├── PatchTST_Last39_*
  ├── PatchTST_CalendarAligned_*
  ├── PatchTST_Final_*
  └── Register_PatchTST_X_Best

ARIMA_Training/
  ├── ARIMA_Data_Preparation
  ├── ARIMA_Last39_order_*
  ├── ARIMA_Calendar_order_*
  └── ARIMA_Final_Test_order_*

SARIMA_Training/
  └── SARIMA_Baseline

Prophet_Training/
  └── Prophet_Baseline
```

Model Registry-ში დარეგისტრირებულია: `Walmart_XGBoost_Pipeline`, `Walmart_LightGBM_Pipeline`, `Walmart_NBeats_Pipeline`, `Walmart_SARIMA_Pipeline`, `Walmart_Prophet_Pipeline`.
Model Registry-ში საბოლოო inference-ისთვის ასევე დარეგისტრირებულია საუკეთესო standalone მოდელი: `Walmart_PatchTST_X_Best`

---

## 7. საბოლოო შედეგი

[Model Registry](https://dagshub.com/LukaBatilashvili07/walmart-sales-forecasting.mlflow/#/models/Walmart_PatchTST_X_Best) 

<img width="1186" height="83" alt="image" src="https://github.com/user-attachments/assets/35fe7f1b-4729-458c-a32e-ad6d24ee57d5" />

ასევე დამატებით გამოვცადეთ სხვადასხვა blending მიდგომები, როგორც მოდელთა შორის ასევე მოდელშივე, კერძოდ smart blending-ის ტიპის, სადაც Calendar-aligned split-ში საუკეთესო score-ის მქონე მოდელი holiday-ებზე ფოკუსირდებოდა
ხოლო Last-39 split-ის მოდელი არა holiday კვირებზე. ამან კარგი შედეგები მოიტანა, მაგრამ საბოლოოდ მაინც naive blending-ის მიდგომამ `PatchTST-X`-ის, `TFT`-ს და `ARIMA`-ს prediction-ებზე მოგვცა ყველაზე დაბალი WMAE. blending-ის 3 კომპონენტი და თავად შედეგი დართულია.

<img width="1201" height="93" alt="image" src="https://github.com/user-attachments/assets/5208b431-2311-4c6f-9678-06823e5c06ad" />

---
