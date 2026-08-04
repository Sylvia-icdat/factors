from __future__ import annotations

import pandas as pd

from wq_alphas import ALPHA_FORMULAS, compute_single_alpha

ALPHA_NUM = 91
FACTOR_NAME = "wq_alpha091"
FACTOR_COLS = (FACTOR_NAME,)
EXPORT_TRIGGERS = False
ALPHA_FORMULA = ALPHA_FORMULAS[ALPHA_NUM]


def compute_factor(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    WorldQuant Alpha#91:

      ((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close, IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) - rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)
    """
    return compute_single_alpha(price_df, ALPHA_NUM)
