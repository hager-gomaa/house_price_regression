# Housing Price Regression: Short Report

## 1. Dataset overview
`Housing.csv` has 545 houses and 13 columns (target `price`; 5 numeric features; 6 yes/no features; `furnishingstatus` with 3 levels). Audit: no missing values, no duplicate rows, no formatting problems, no impossible values. The file does not define currency, area unit or the meaning of "preferred area", so none is assumed. `price` and `area` are right-skewed (skewness ≈ 1.2 and 1.3); the high values are plausible and were kept.

## 2. Key insights (associations, not causes)
- Price rises with area (r ≈ 0.54) and bathrooms (r ≈ 0.52), with wide scatter that grows with size. The largest plots are not always the most expensive.
- Median price is higher for houses with air conditioning (5.76M vs 3.89M), in a preferred area (5.58M vs 4.10M), and with a guest room, basement or main-road access. Furnishing shows an ordering furnished > semi-furnished > unfurnished with heavy overlap.
- Some of these gaps partly reflect size: AC and preferred-area houses are also larger on average.
- Several categories are tiny (hot-water heating: 25 houses; 1- and 6-bedroom houses: 2 each), so conclusions about them are weak.

## 3. Preprocessing decisions
`price` separated as target; 80/20 split (`random_state=42`) done **before** any fitting; `StandardScaler` for numeric features (needed because the penalties depend on coefficient size); `OneHotEncoder(drop='first')` for all categorical features (no arbitrary ranks, no dummy-variable trap); target not scaled; no rows removed. Everything sits in a `ColumnTransformer` + `Pipeline` so CV refits preprocessing per fold.

## 4. Validation strategy
5-fold shuffled CV on the training set only, scoring = RMSE. Ridge: alpha in [1e-3, 1e3] (50 values). Lasso: alpha in [1e-1, 1e6] (60). Elastic Net: alpha in [1e-1, 1e6] (40) × l1_ratio {0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0}. Each selected model was refit on the full training set and evaluated once on the test set. Extra experiments were judged by CV only.

## 5. Model comparison (test set, 109 houses)
| Model | CV RMSE | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| Dummy (mean) | 1,755,599 | 1,748,625 | 2,268,298 | -0.018 |
| Linear Regression | 1,023,579 | 970,043 | 1,324,507 | 0.653 |
| Ridge (alpha 6.25) | 1,021,859 | 976,702 | 1,333,729 | 0.648 |
| Lasso (alpha 476) | 1,023,570 | 970,343 | 1,325,067 | 0.653 |
| Elastic Net (alpha 0.35, l1_ratio 0.95) | 1,021,861 | 977,705 | 1,335,451 | 0.647 |

All models clearly beat the mean-only baseline, but they are practically tied: differences are far smaller than the noise of a 109-row test set. Regularisation did not help meaningfully; Lasso and Elastic Net removed no features. Default Elastic Net (alpha=1) was worse in CV (R² 0.59) because scikit-learn's alpha scaling differs from Ridge's, which is why each model was tuned.

**Residuals.** Errors grow with price (funnel shape). The model over-predicts the cheapest quartile and under-predicts the most expensive (mean residual about −0.89M vs +1.33M); the biggest misses are very expensive houses the features cannot explain.

**Coefficients.** Largest weights: air conditioning, preferred area, area, bathrooms, basement/main road, stories; unfurnished is about 0.4M below furnished. `bedrooms` adds little once other layout features are included. The sparse `hotwaterheating` weight is shrunk most by regularisation (≈ 685K → ≈ 483K–509K).

**Extra experiments (CV only).** log(price) with back-transformed predictions: CV RMSE 1.005M vs 1.024M (small gain). MinMaxScaler: worse (1.052M) at an un-retuned alpha. Area × AC interaction: negligible (1.019M vs 1.022M). Dropping bedrooms: no change.

## 6. Limitations
Small data and a noisy test set; undefined units; no location, age or condition information (main reason R² is only ≈ 0.65); observational data (no causal claims); sparse categories; heteroscedastic errors at the high end.

## 7. Recommendation
Use a simple linear model: Ridge (tied for best CV RMSE) or plain Linear Regression are equally defensible, chosen for stability and interpretability, not because they measured better. Most promising next steps: add location/condition data, try non-linear models, and evaluate the log-price target once on the test set.
