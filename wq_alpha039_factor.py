from __future__ import annotations

import pandas as pd

from wq_alphas import ALPHA_FORMULAS, compute_single_alpha

ALPHA_NUM = 39
FACTOR_NAME = "wq_alpha039"
FACTOR_COLS = (FACTOR_NAME,)
EXPORT_TRIGGERS = False
ALPHA_FORMULA = ALPHA_FORMULAS[ALPHA_NUM]


def compute_factor(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    WorldQuant Alpha#39:

      ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))
    """
    return compute_single_alpha(price_df, ALPHA_NUM)
