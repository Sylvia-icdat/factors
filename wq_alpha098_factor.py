from __future__ import annotations

import pandas as pd

from wq_alphas import ALPHA_FORMULAS, compute_single_alpha

ALPHA_NUM = 98
FACTOR_NAME = "wq_alpha098"
FACTOR_COLS = (FACTOR_NAME,)
EXPORT_TRIGGERS = False
ALPHA_FORMULA = ALPHA_FORMULAS[ALPHA_NUM]


def compute_factor(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    WorldQuant Alpha#98:

      (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) - rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206)))
    """
    return compute_single_alpha(price_df, ALPHA_NUM)
