from __future__ import annotations

import pandas as pd

from wq_alphas import ALPHA_FORMULAS, compute_single_alpha

ALPHA_NUM = 38
FACTOR_NAME = "wq_alpha038"
FACTOR_COLS = (FACTOR_NAME,)
EXPORT_TRIGGERS = False
ALPHA_FORMULA = ALPHA_FORMULAS[ALPHA_NUM]


def compute_factor(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    WorldQuant Alpha#38:

      ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
    """
    return compute_single_alpha(price_df, ALPHA_NUM)
