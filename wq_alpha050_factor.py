from __future__ import annotations

import pandas as pd

from wq_alphas import ALPHA_FORMULAS, compute_single_alpha

ALPHA_NUM = 50
FACTOR_NAME = "wq_alpha050"
FACTOR_COLS = (FACTOR_NAME,)
EXPORT_TRIGGERS = False
ALPHA_FORMULA = ALPHA_FORMULAS[ALPHA_NUM]


def compute_factor(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    WorldQuant Alpha#50:

      (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
    """
    return compute_single_alpha(price_df, ALPHA_NUM)
