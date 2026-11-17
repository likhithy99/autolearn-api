"""Download the public demo datasets used by the examples and benchmarks.

* Adult Census Income (UCI / OpenML 1590, CC BY 4.0): predict whether income > $50K. 48,842 rows,
  mixed numeric + categorical features.
* California Housing (StatLib, 1990 US census): predict median house value. 20,640 rows, numeric.
"""

import os
from pathlib import Path

import certifi

# python.org macOS builds ship without root certificates; use certifi's bundle.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import time

from sklearn.datasets import fetch_california_housing, fetch_openml


def retry(fn, tries=4):
    """OpenML and the StatLib mirror fail intermittently; retry with backoff."""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            print(f"download failed ({e}); retrying in {5 * 2 ** i}s")
            time.sleep(5 * 2 ** i)


OUT = Path(__file__).resolve().parent.parent / "datasets"
OUT.mkdir(exist_ok=True)

adult = retry(lambda: fetch_openml(data_id=1590, as_frame=True).frame)  # by id: no name lookup
adult = adult.rename(columns=lambda c: c.replace("-", "_"))
adult["income"] = (adult["class"].astype(str).str.strip() == ">50K").map({True: ">50K", False: "<=50K"})
adult = adult.drop(columns=["class", "fnlwgt"])
adult.to_csv(OUT / "adult.csv", index=False)
print("adult.csv", adult.shape, adult["income"].value_counts().to_dict())

housing = retry(lambda: fetch_california_housing(as_frame=True).frame)
housing.to_csv(OUT / "california_housing.csv", index=False)
print("california_housing.csv", housing.shape)
