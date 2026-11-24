# Housing Price Regression

This project explores how the property features in `Housing.csv` are associated with house prices and compares Linear Regression, Ridge, Lasso, and Elastic Net.

## Files

- `house_price_linear_regression.ipynb` - data audit, exploratory analysis, preprocessing, model tuning and comparison, residual and coefficient analysis, and additional experiments.
- `REPORT.md` - source text for the written summary.
- `Housing_Price_Regression_Report.docx` - formatted Word version of the report.
- `convert_report.py` - converts `REPORT.md` to the Word report.
- `Housing.csv` - the dataset used by the notebook.
- `requirements.txt` - Python packages used by the notebook.

## Run the notebook

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Open `house_price_linear_regression.ipynb` in Jupyter with this folder as the working directory. To execute it from a terminal and save a separate executed copy:

```bash
jupyter nbconvert --to notebook --execute --output executed.ipynb house_price_linear_regression.ipynb
```

The notebook uses a fixed 80/20 train/test split (`random_state=42`) and shuffled five-fold cross-validation on the training set. Its bootstrap stability analysis uses seed 0. The dataset does not define currency, area units, or the geographic meaning of `prefarea`; the project does not infer these details.

## Notebook issue to fix before a clean run

In the model comparison cell, the fitted Linear Regression pipeline is stored as `linear`, but the model dictionary refers to `lin`. Change `"Linear Regression": lin` to `"Linear Regression": linear` before executing the notebook from top to bottom. Saved outputs may be from an earlier version, so rerun the notebook before relying on the reported results.
