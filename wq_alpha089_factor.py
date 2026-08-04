from __future__ import annotations

import pandas as pd

from wq_alphas import ALPHA_FORMULAS, compute_single_alpha

ALPHA_NUM = 89
FACTOR_NAME = "wq_alpha089"
FACTOR_COLS = (FACTOR_NAME,)
EXPORT_TRIGGERS = False
ALPHA_FORMULA = ALPHA_FORMULAS[ALPHA_NUM]


def compute_factor(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    WorldQuant Alpha#89:

      (Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10, 6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap, IndClass.industry), 3.48158), 10.1466), 15.3012))
    """
    return compute_single_alpha(price_df, ALPHA_NUM)
