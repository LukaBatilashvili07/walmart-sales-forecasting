# Walmart Recruiting - Store Sales Forecasting

## 1. მონაცემები და შეფასების მეტრიკა

| ფაილი | ზომა | აღწერა |
|---|---|---|
| `train.csv` | 421,570 row | 2010-02-05 - 2012-10-26 (143 კვირა) |
| `test.csv` | 115,064 row | 2012-11-02 - 2013-07-26 (39 კვირა) |
| `stores.csv` | 45 row | მაღაზიის ტიპი (A/B/C) და ზომა |
| `features.csv` | 8,190 row | ტემპერატურა, საწვავის ფასი, MarkDown1-5, CPI, უმუშევრობა |

შეფასების მეტრიკაა Weighted MAE (WMAE), სადაც სადღესასწაულო კვირებს 5-ჯერ მეტი წონა ენიჭება ჩვეულებრივ კვირებთან შედარებით:

ყველა მოდელის ვალიდაცია სრულდება ერთნაირად: ბოლო **39 კვირა** `train.csv`-დან გამოყოფილია ვალიდაციისთვის (იმეორებს `test.csv`-ის ხანგრძლივობას), ხოლო დანარჩენი 104 კვირა გამოიყენება ტრენინგისთვის. შემთხვევითი (random) split-ი გამორიცხულია, რადგან დროითი მწკრივის შემთხვევაში ეს გამოიწვევდა მონაცემის გაჟონვას (leakage) მომავლიდან წარსულში.

---

## 2. EDA - მთავარი დასკვნები

სრული ანალიზი: `notebooks/eda.ipynb`

- **Store-Dept სტრუქტურა.** train და test შეიცავს ერთსა და იმავე 45 მაღაზიას და 81 დეპარტამენტს, თუმცა ზუსტ `(Store, Dept)` წყვილების დონეზე მხოლოდ 3,158 წყვილია საერთო. 11 წყვილი გვხვდება მხოლოდ test-ში (ისტორიის გარეშე), ხოლო 173 - მხოლოდ train-ში. ეს ნიშნავს, რომ lag/rolling ტიპის feature-ებს სჭირდება fallback მექანიზმი უცნობი წყვილებისთვის.
- **Target (`Weekly_Sales`)** ძლიერ skewed განაწილებისაა - ცალკეული მწკრივები უარყოფითია (სავარაუდოდ დაბრუნებები/ცვლილებები).
- **სადღესასწაულო დღეები.** სადღესასწაულო კვირები საშუალო გაყიდვების მხოლოდ 7.04%-ია train-ში (7.76% test-ში), მაგრამ WMAE-ის 5x წონის გამო ეფექტური წვლილი გაცილებით მეტია. სადღესასწაულო კვირების საშუალო გაყიდვა (17,035.82) მხოლოდ 7%-ით აღემატება ჩვეულებრივი კვირის საშუალოს (15,901.45), თუმცა ეს განსხვავება დეპარტამენტების მიხედვით ძალიან არაერთგვაროვანია - ზოგი დეპარტამენტისთვის სადღესასწაულო ეფექტი გაცილებით ძლიერია. ასევე შენიშნულია, რომ ყველაზე დიდი გაყიდვების პიკები ყოველთვის ზუსტად `IsHoliday=True` დღეებზე არ ხვდება - რაც მიანიშნებს დამატებითი calendar ტიპის feature-ების საჭიროებაზე.
- **Missing values.** `MarkDown1-5` სტრუქტურულად აკლია 2011-11-11-მდე (ანუ არა შემთხვევით), ხოლო `CPI`/`Unemployment` აკლია test-ის ბოლო 13 კვირას ყველა 45 მაღაზიისთვის (585 მწკრივი) - ეს ეკონომიკური ინდიკატორები ნელა იცვლება დროში, ამიტომ ffill/interpolation გონივრული მიდგომაა.
- **დასკვნა მოდელირებისთვის:** საჭიროა Store/Dept იდენტიფიკატორები, calendar feature-ები, სადღესასწაულო flag-ები, MarkDown-ის დამუშავებული ვერსია და sales history (lag/rolling) - ეს გახდა შემდეგი ეტაპის - feature engineering-ის - საფუძველი.

---

## 3. Feature Engineering

სრული ექსპერიმენტი: `notebooks/feature_engineering_experiment.ipynb`, საბოლოო კოდი გატანილია კლასებად `src/features/base.py` (`WalmartBasePreprocessor`) და `src/features/tabular.py` (`WalmartTabularFeatureEngineer`), ორივე გატესტილია `notebooks/test_tabular_feature_engineer.ipynb`-ში.

### 3.1 `WalmartBasePreprocessor` (საერთო ყველა მოდელისთვის)
- **Merge**: `train`/`test` + `stores` + `features` `Store`/`Date`/`IsHoliday`-ზე.
- **CPI/Unemployment**: ffill თითოეული მაღაზიის შიგნით.
- **MarkDown1-5**: დაემატა `_was_missing` indicator თითოეულისთვის, შემდეგ 0-fill. ასევე შენარჩუნებულია "raw" (NaN-შენარჩუნებული) ვერსია ხის-ტიპის მოდელებისთვის, რომლებსაც NaN-ის საკუთარი დამუშავება შეუძლიათ.
- **Calendar**: `Year`, `Month`, `WeekOfYear`, ციკლური `Week_sin`/`Week_cos` კოდირება.
- **სადღესასწაულო flag-ები**: ცალკე ბინარული სვეტები `SuperBowl`/`LaborDay`/`Thanksgiving`/`Christmas`-სთვის (ცალკეული `IsHoliday`-ს დაშლა კონკრეტულ დღესასწაულებად).
- **Store Type**: label-encoding (`Type_encoded`) და one-hot (`Type_A/B/C`).
- **MarkDown აგრეგატები**: `total_markdown`, `abs_total_markdown`, `positive_markdown_sum`, `negative_markdown_sum`, `has_markdown_signal`, `markdown_missing_count`, `markdown_available_period`.

### 3.2 `WalmartTabularFeatureEngineer` (ხის-ტიპის მოდელებისთვის)
- **Lag features**: `Sales_lag_1`, `Sales_lag_4`, `Sales_lag_52` (Store-Dept დონეზე).
- **Rolling features**: `Sales_roll_mean_4`, `Sales_roll_std_4`, `Sales_roll_mean_12`, გამოთვლილი წანაცვლებული (`shift(1)`) მწკრივზე, რომ თავიდან ავირიდოთ leakage.
- **Leakage-safety**: `fit()` მხოლოდ ტრენინგის ისტორიაზე ითვლის, `transform_future()` ვალიდაცია/ტესტ სეტისთვის იყენებს მხოლოდ ამ ისტორიას (არასდროს - თავად ვალიდაციის sales მნიშვნელობებს). ეს დამოწმებულია `test_tabular_feature_engineer.ipynb`-ში ხელით გამოთვლილი lag მნიშვნელობის შედარებით კლასის output-თან - ემთხვევა ზუსტად.
- **Fallback**: Store-Dept წყვილებისთვის, რომლებსაც ისტორია არ აქვთ (ახალი წყვილები test-ში), lag/rolling მნიშვნელობები ივსება დეპარტამენტის საშუალო გაყიდვით; `roll_std`-ის fallback - 0.

### 3.3 CPI/Unemployment imputation - შედარებითი ექსპერიმენტი
რადგან ეს ცვლადები test-ის ბოლო 13 კვირას აკლია ყველა მაღაზიისთვის, გაიტესტა 3 მიდგომა:

| მიდგომა | WMAE |
|---|---|
| Forward-fill (თითო მაღაზიის შიგნით) | 5402.96 |
| extrapolation | 5402.96 |
| ცვლადების მთლიანად გამორიცხვა | 5458.03 |

ორივე imputation მიდგომამ ერთნაირი შედეგი აჩვენა და ორივემ სჯობა ცვლადების უბრალო გამორიცხვას - საბოლოო pipeline-ში აირჩა forward-fill, უფრო მარტივი და საკმარისად ეფექტური.

---

## 4. მოდელის არქიტექტურები

### 4.1 Tree-Based Models

#### XGBoost - `notebooks/model_experiment_XGBoost.ipynb`
MLflow ექსპერიმენტი: `XGBoost_Training`

| Run | აღწერა | შედეგი |
|---|---|---|
| `XGBoost_Cleaning` | `WalmartBasePreprocessor` გამოყენება train/valid-ზე | - |
| `XGBoost_Feature_Selection` | baseline (`n_estimators=300, max_depth=6`), feature importance | WMAE = 6512.08 |
| `XGBoost_CV` | Walk-forward CV, 3 fold, 13-კვირიანი ვალიდაცია | ფოლდები: 6660.82 / 7356.73 / 7966.82 - mean 7328.12 |
| `XGBoost_HPO` | Optuna, 30 trial-ი, პირდაპირ WMAE-ის მინიმიზაცია | Best WMAE = 4638.49 |
| `XGBoost_Best` | საუკეთესო პარამეტრებით საბოლოო მოდელი, Pipeline-ად registrირებული | Final validation WMAE = 4638.49 |
| `XGBoost_Final_Refit` | გადატრენინგება მთელ 143-კვირიან train.csv-ზე | model v2 |

**საუკეთესო feature-ები** (importance-ის მიხედვით): `Sales_lag_1` (0.59), `Sales_roll_mean_4` (0.25), `IsThanksgiving`, `WeekOfYear`, `Sales_lag_52` - ანუ ისტორიული გაყიდვების feature-ები ყველაზე დომინანტურია, calendar/holiday feature-ები მეორეხარისხოვანი, მაგრამ მაინც საყურადღებო.

**საუკეთესო ჰიპერპარამეტრები**: `n_estimators=800, max_depth=7, learning_rate=0.035, subsample=0.80, colsample_bytree=0.88, min_child_weight=3, reg_lambda=1.57`

CV დაკვირვება: Thanksgiving/Christmas პერიოდი (2011-11-04 - 2012-01-27) მნიშვნელოვნად უარესია დანარჩენებზე, რაც აჩვენებს, რომ მოდელს უჭირს სადღესასწაულო სეზონის პროგნოზირება.

#### LightGBM - `notebooks/model_experiment_LightGBM.ipynb`
MLflow ექსპერიმენტი: `LightGBM_Training`. მეორდება იგივე pipeline რაც XGBoost-ისთვის (იგივე `WalmartBasePreprocessor`/`WalmartTabularFeatureEngineer`), პირდაპირი შედარებისთვის. განსხვავება: LightGBM-ს ტესტდება "raw" (NaN-შენარჩუნებული) MarkDown სვეტებით - 0-fill+indicator-ის ნაცვლად - რადგან LightGBM-ს NaN-ის საკუთარი split-ის ლოგიკა აქვს.

| Run | აღწერა | შედეგი |
|---|---|---|
| `LightGBM_Feature_Selection` | baseline, raw MarkDown ვარიანტი |WMAE = 5463.72 |
| `LightGBM_CV` | Walk-forward CV, 3 fold | ფოლდები: 5939.96 / 6075.09 / 8213.54 - mean 6742.86 ± 1041.39 |
| `LightGBM_HPO` | Optuna, 30 trial-ი | `n_estimators=700, num_leaves=138, max_depth=8, learning_rate=0.027, min_child_samples=70` |
| `LightGBM_Best` | საბოლოო მოდელი | Final validation WMAE = 4891.09 |
| `LightGBM_Final_Refit` | გადატრენინგება მთელ train.csv-ზე | model v2 |

**LightGBM vs XGBoost**: LightGBM-ის baseline (5463.72, raw NaN-ით) გაცილებით სჯობს XGBoost-ის baseline-ს (6512.08, 0-fill-ით) - რაც მხარს უჭერს იმ ჰიპოთეზას, რომ LightGBM-ის native missing-value handling უკეთესია ამ ამოცანაზე. თუმცა HPO-ს შემდეგ საბოლოო შედარებით XGBoost სჯობს LightGBM-ს (4638.49 vs 4891.09) - ორივე მოდელისთვის ყველაზე რთული პერიოდი Thanksgiving/Christmas არის.

### 4.2 Classical Statistical Time-Series Models

ორივე მოდელი მორგებულია აგრეგირებულ (ყველა მაღაზია/დეპარტამენტის ჯამურ) კვირეულ სერიაზე - 3,331 ცალკეული Store-Dept სერიისთვის ცალ-ცალკე SARIMA/Prophet-ის დატრენინგება პრაქტიკულად შეუძლებელია დროისა და გამოთვლითი რესურსის თვალსაზრისით. ეს ნიშნავს, რომ მათი WMAE სხვანაირია (მთლიანი კვირეული გაყიდვის დონეზე, არა Store-Dept-ის) და პირდაპირ არაა შედარებადი XGBoost/LightGBM-ის Store-Dept-დონის WMAE-სთან.

#### SARIMA - `notebooks/model_experiment_SARIMA.ipynb`
MLflow ექსპერიმენტი: `SARIMA_Training`. მოდელი: `SARIMA(1,1,1)(1,1,1,52)` - სეზონურობის პერიოდი `s=52` ირჩევა წლიური (52-კვირიანი) ციკლის ასახვისთვის (მაგ. საშობაო პიკის განმეორება ყოველწლიურად).

- Run: `SARIMA_Baseline`
- WMAE (აგრეგირებულ სერიაზე) = 1,204,244.21, AIC = 10.0


#### Prophet - `notebooks/model_experiment_Prophet.ipynb`
MLflow ექსპერიმენტი: `Prophet_Training`. Prophet-ს დამატებით მიდგომა აქვს SARIMA-სთან შედარებით - ავტორეგრესიის ნაცვლად სერიას შლის კომპონენტებად (trend + yearly seasonality + holiday ეფექტები), რომლებიც ცალ-ცალკე მოდელირდება და ჯამდება.

- Run: `Prophet_Baseline` (`yearly_seasonality=True`, `weekly_seasonality=False`, სადღესასწაულო კვირები გადაცემულია `holidays` პარამეტრით)
- WMAE (აგრეგირებულ სერიაზე) = 1,728,319.66

**SARIMA vs Prophet**: იმავე აგრეგირებულ სერიაზე, იგივე 39-კვირიან ვალიდაციაზე, SARIMA-მ საგრძნობლად აჯობა Prophet-ს (1.20M vs 1.73M). შესაძლო მიზეზი: SARIMA-ს ავტორეგრესიული სტრუქტურა უკეთ იჭერს მოკლევადიან დამოკიდებულებებს ამ სერიაში, ხოლო Prophet-ის default trend/seasonality decomposition ნაკლებად მოქნილია სპეციფიკურად ამ დატასეტისთვის დამატებითი tuning-ის გარეშე.

---

## 5. მოდელების შედარება

ყველა შედეგი გაზომილია იმავე 39-კვირიან ვალიდაციაზე (Store-Dept დონეზე, სადაც ეს შესაბამისია):

| არქიტექტურა | ტიპი | Validation WMAE | შენიშვნა |
|---|---|---:|---|
| XGBoost | Tree-Based | 4638.49 | HPO-ს შემდეგ, Optuna 30 trial |
| LightGBM | Tree-Based | 4891.09 | HPO-ს შემდეგ, Optuna 30 trial |
| SARIMA | Classical | 1,204,244.21 | აგრეგირებულ სერიაზე - არაა შედარებადი |
| Prophet | Classical | 1,728,319.66 | აგრეგირებულ სერიაზე - არაა შედარებადი |

**დასკვნა**: XGBoost ოდნავ სჯობს LightGBM-ს. Classical სტატისტიკური მოდელები (SARIMA, Prophet) გაიტესტა მხოლოდ აგრეგირებულ დონეზე გამოთვლითი შეზღუდვების გამო და მათ შორის SARIMA აჩვენებს გაცილებით სტაბილურ და ზუსტ პროგნოზს Prophet-თან შედარებით.

---

## 6. Model Registry, Inference და Kaggle Submission

`notebooks/model_inference.ipynb` კითხულობს ყველა არქიტექტურის საბოლოო validation WMAE-ს MLflow-დან, ირჩევს საუკეთესოს და აგენერირებს submission-ს.

```
Using best AVAILABLE model: XGBoost (Walmart_XGBoost_Pipeline v2)
NOTE: DLinear scored better but has no registered full-history pipeline yet.
```

**საბოლოო submission** - `Walmart_XGBoost_Pipeline` v2 (`XGBoost_Final_Refit`-ში დატრენინგებული მთელ 421,570 მწკრივზე), რომელიც პირდაპირ იტვირთება Model Registry-დან და გაშვებულია დაუმუშავებელ (raw, `test.csv`) მონაცემზე - pipeline-ი შიგნით თავად ასრულებს `WalmartBasePreprocessor` - `WalmartTabularFeatureEngineer` - `predict()` ჯაჭვს.

- `submission.csv`: 115,064 მწკრივი, ფორმატი `{Store}_{Dept}_{Date}, Weekly_Sales`
- პროგნოზები განსაზღვრულია (`clip`) მინიმუმ 0-ზე (უარყოფითი პროგნოზები 0-ის ტოლი ხდება);
- აღწერითი სტატისტიკა: საშუალო ≈ 15,145.72, მედიანა ≈ 6,914.90, მაქსიმუმი ≈ 371,232.16.

---

## 7. MLflow / DagsHub სტრუქტურა

https://dagshub.com/LukaBatilashvili07/walmart-sales-forecasting.mlflow

ყველა მოდელის არქიტექტურას აქვს ცალკე MLflow ექსპერიმენტი, შიგნით კი run-ების სახელები:

```
XGBoost_Training/
  ├── XGBoost_Cleaning
  ├── XGBoost_Feature_Selection
  ├── XGBoost_CV (+ nested fold-run-ები)
  ├── XGBoost_HPO (+ nested trial-run-ები)
  ├── XGBoost_Best
  └── XGBoost_Final_Refit

LightGBM_Training/    (იგივე სტრუქტურა)
SARIMA_Training/
  └── SARIMA_Baseline
Prophet_Training/
  └── Prophet_Baseline
```

Model Registry-ში დარეგისტრირებულია: `Walmart_XGBoost_Pipeline` (v1, v2), `Walmart_LightGBM_Pipeline` (v1, v2), `Walmart_SARIMA_Pipeline` (v1), `Walmart_Prophet_Pipeline` (v1).

---

## 8. რეპოზიტორიის სტრუქტურა

```
walmart-sales-forecasting/
├── README.md
├── submission.csv
├── data/raw/                         # train.csv, test.csv, stores.csv, features.csv
├── notebooks/
│   ├── eda.ipynb
│   ├── feature_engineering_experiment.ipynb
│   ├── test_tabular_feature_engineer.ipynb
│   ├── model_experiment_XGBoost.ipynb
│   ├── model_experiment_LightGBM.ipynb
│   ├── model_experiment_SARIMA.ipynb
│   ├── model_experiment_Prophet.ipynb
│   └── model_inference.ipynb
└── src/
    ├── data/
    │   ├── load_data.py               # load_raw_data()
    │   └── splits.py                  # last_n_weeks_split()
    └── features/
    │   ├── base.py                    # WalmartBasePreprocessor
    │   ├── tabular.py                 # WalmartTabularFeatureEngineer (tree-based models)
    │   └── neural.py                  # WalmartNeuralPreprocessor (DLinear / neural models)
    └── datasets/   
        └── window_dataset.py                
```

---