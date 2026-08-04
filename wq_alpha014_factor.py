from __future__ import annotations

import pandas as pd

from wq_alphas import ALPHA_FORMULAS, compute_single_alpha

ALPHA_NUM = 14
FACTOR_NAME = "wq_alpha014"
FACTOR_COLS = (FACTOR_NAME,)
EXPORT_TRIGGERS = False
ALPHA_FORMULA = ALPHA_FORMULAS[ALPHA_NUM]


def compute_factor(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    WorldQuant Alpha#14:

      ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
    """
    return compute_single_alpha(price_df, ALPHA_NUM)
