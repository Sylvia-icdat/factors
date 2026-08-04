from __future__ import annotations

import pandas as pd

from wq_alphas import ALPHA_FORMULAS, compute_single_alpha

ALPHA_NUM = 20
FACTOR_NAME = "wq_alpha020"
FACTOR_COLS = (FACTOR_NAME,)
EXPORT_TRIGGERS = False
ALPHA_FORMULA = ALPHA_FORMULAS[ALPHA_NUM]


def compute_factor(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    WorldQuant Alpha#20:

      (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))
    """
    return compute_single_alpha(price_df, ALPHA_NUM)
