from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

import wq_alpha_ops as op

ALPHA_FORMULAS: dict[int, str] = {
    1: "(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)",
    2: "(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))",
    3: "(-1 * correlation(rank(open), rank(volume), 10))",
    4: "(-1 * Ts_Rank(rank(low), 9))",
    5: "(rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))",
    6: "(-1 * correlation(open, volume, 10))",
    7: "((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))",
    8: "(-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))",
    9: "((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))",
    10: "rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))",
    11: "((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))",
    12: "(sign(delta(volume, 1)) * (-1 * delta(close, 1)))",
    13: "(-1 * rank(covariance(rank(close), rank(volume), 5)))",
    14: "((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))",
    15: "(-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))",
    16: "(-1 * rank(covariance(rank(high), rank(volume), 5)))",
    17: "(((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))",
    18: "(-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))",
    19: "((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))",
    20: "(((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))",
    21: "(((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : (-1 * 1))))",
    22: "(-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))",
    23: "(((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)",
    24: "((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) || ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3)))",
    25: "rank(((((-1 * returns) * adv20) * vwap) * (high - close)))",
    26: "(-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))",
    27: "((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)",
    28: "scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))",
    29: "(min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))",
    30: "(((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))",
    31: "((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 * delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))",
    32: "(scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))",
    33: "rank((-1 * ((1 - (open / close))^1)))",
    34: "rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))",
    35: "((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))",
    36: "((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open - close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap, adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))",
    37: "(rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))",
    38: "((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))",
    39: "((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))",
    40: "((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))",
    41: "(((high * low)^0.5) - vwap)",
    42: "(rank((vwap - close)) / rank((vwap + close)))",
    43: "(ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))",
    44: "(-1 * correlation(high, rank(volume), 5))",
    45: "(-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))",
    46: "((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ? (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 : ((-1 * 1) * (close - delay(close, 1)))))",
    47: "((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))",
    48: "(indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))",
    49: "(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))",
    50: "(-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))",
    51: "(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))",
    52: "(((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))",
    53: "(-1 * delta((((close - low) - (high - close)) / (close - low)), 9))",
    54: "((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))",
    55: "(-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))",
    56: "(0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))",
    57: "(0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))",
    58: "(-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume, 3.92795), 7.89291), 5.50322))",
    59: "(-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap * (1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))",
    60: "(0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10))))))",
    61: "(rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))",
    62: "((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)",
    63: "((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237)) - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180, 37.2467), 13.557), 12.2883))) * -1)",
    64: "((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054), sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 3.69741))) * -1)",
    65: "((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)",
    66: "((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)",
    67: "((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap, IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)",
    68: "((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)",
    69: "((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416), 9.0615)) * -1)",
    70: "((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), adv50, 17.8256), 17.9171)) * -1)",
    71: "max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388))",
    72: "(rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))",
    73: "(max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)), Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)",
    74: "((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) * -1)",
    75: "(rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 12.4413)))",
    76: "(max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)), Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81, 8.14941), 19.569), 17.1543), 19.383)) * -1)",
    77: "min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)), rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))",
    78: "(rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428), sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))",
    79: "(rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))), IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150, 9.18637), 14.6644)))",
    80: "((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))), IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)",
    81: "((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)",
    82: "(min(rank(decay_linear(delta(open, 1.46063), 14.8717)), Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) + (open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)",
    83: "((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))",
    84: "SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))",
    85: "(rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595), 7.11408)))",
    86: "((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open + close) - (vwap + open)))) * -1)",
    87: "(max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))), 1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81, IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)",
    88: "min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957))",
    89: "(Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10, 6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap, IndClass.industry), 3.48158), 10.1466), 15.3012))",
    90: "((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40, IndClass.subindustry), low, 5.38375), 3.21856)) * -1)",
    91: "((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close, IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) - rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)",
    92: "min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221), 18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584))",
    93: "(Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81, 17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 - 0.524434))), 2.77377), 16.2664)))",
    94: "((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap, 19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)",
    95: "(rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low) / 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))",
    96: "(max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404), Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)",
    97: "((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))), IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low, 7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)",
    98: "(rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) - rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206)))",
    99: "((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) < rank(correlation(low, volume, 6.28259))) * -1)",
    100: "(0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high - close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) - scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))), IndClass.subindustry))) * (volume / adv20))))",
    101: "((close - open) / ((high - low) + .001))",
}


def _pivot(price_df: pd.DataFrame, col: str) -> pd.DataFrame:
    wide = price_df.pivot(index="date", columns="stock_code", values=col)
    return wide.sort_index()


def _wide_to_long(wide: pd.DataFrame | pd.Series, factor_name: str) -> pd.DataFrame:
    if isinstance(wide, pd.Series):
        wide = wide.to_frame(factor_name)
    else:
        wide = wide.copy()
        wide.columns.name = "stock_code"
    long = wide.stack(future_stack=True).reset_index()
    long.columns = ["date", "stock_code", factor_name]
    return long


def _row_max(p1: pd.DataFrame | pd.Series, p2: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return pd.concat([p1, p2], axis=1).max(axis=1) if isinstance(p1, pd.Series) else pd.DataFrame(np.maximum(p1.values, p2.values), index=p1.index, columns=p1.columns)


def _row_min(p1: pd.DataFrame | pd.Series, p2: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return pd.concat([p1, p2], axis=1).min(axis=1) if isinstance(p1, pd.Series) else pd.DataFrame(np.minimum(p1.values, p2.values), index=p1.index, columns=p1.columns)


class Alphas:
    def __init__(self, price_df: pd.DataFrame):
        required = {"date", "stock_code", "open", "high", "low", "close", "volume"}
        missing = required - set(price_df.columns)
        if missing:
            raise ValueError(f"缺少列: {sorted(missing)}")

        self.industry = None
        if "industry" in price_df.columns:
            self.industry = _pivot(price_df, "industry")

        self.open = _pivot(price_df, "open")
        self.high = _pivot(price_df, "high")
        self.low = _pivot(price_df, "low")
        self.close = _pivot(price_df, "close")
        self.volume = _pivot(price_df, "volume")
        self.returns = self.close.pct_change(fill_method=None)
        typical = (self.high + self.low + self.close) / 3.0
        amount = self.close * self.volume
        self.vwap = amount / self.volume.replace(0, np.nan)
        self.vwap = self.vwap.fillna(typical)
        self.cap = self.close * self.volume
        self.adv20 = op.sma(self.volume, 20)

    def alpha001(self):
        inner = self.close.copy()
        std_ret = op.stddev(self.returns, 20)
        inner = inner.where(self.returns >= 0, std_ret)
        return op.rank(op.ts_argmax(op.signed_power(inner, 2.0), 5)) - 0.5

    def alpha002(self):
        x = op.rank(op.delta(op.log(self.volume), 2))
        y = op.rank((self.close - self.open) / self.open.replace(0, np.nan))
        return op.clean_corr(-1 * op.correlation(x, y, 6))

    def alpha003(self):
        return op.clean_corr(-1 * op.correlation(op.rank(self.open), op.rank(self.volume), 10))

    def alpha004(self):
        return -1 * op.ts_rank(op.rank(self.low), 9)

    def alpha005(self):
        return op.rank(self.open - op.sma(self.vwap, 10)) * (-1 * op.rank(self.close - self.vwap).abs())

    def alpha006(self):
        return op.clean_corr(-1 * op.correlation(self.open, self.volume, 10))

    def alpha007(self):
        alpha = -1 * op.ts_rank(op.delta(self.close, 7).abs(), 60) * np.sign(op.delta(self.close, 7))
        alpha = alpha.where(self.adv20 < self.volume, -1)
        return alpha

    def alpha008(self):
        x = op.ts_sum(self.open, 5) * op.ts_sum(self.returns, 5)
        return -1 * op.rank(x - op.delay(x, 10))

    def alpha009(self):
        d = op.delta(self.close, 1)
        cond = (op.ts_min(d, 5) > 0) | (op.ts_max(d, 5) < 0)
        alpha = -1 * d
        alpha = alpha.where(~cond, d)
        return alpha

    def alpha010(self):
        d = op.delta(self.close, 1)
        cond = (op.ts_min(d, 4) > 0) | (op.ts_max(d, 4) < 0)
        alpha = -1 * d
        alpha = alpha.where(~cond, d)
        return op.rank(alpha)

    def alpha011(self):
        spread = self.vwap - self.close
        return (op.rank(op.ts_max(spread, 3)) + op.rank(op.ts_min(spread, 3))) * op.rank(op.delta(self.volume, 3))

    def alpha012(self):
        return np.sign(op.delta(self.volume, 1)) * (-1 * op.delta(self.close, 1))

    def alpha013(self):
        return -1 * op.rank(op.covariance(op.rank(self.close), op.rank(self.volume), 5))

    def alpha014(self):
        return -1 * op.rank(op.delta(self.returns, 3)) * op.clean_corr(op.correlation(self.open, self.volume, 10))

    def alpha015(self):
        df = op.clean_corr(op.correlation(op.rank(self.high), op.rank(self.volume), 3))
        return -1 * op.ts_sum(op.rank(df), 3)

    def alpha016(self):
        return -1 * op.rank(op.covariance(op.rank(self.high), op.rank(self.volume), 5))

    def alpha017(self):
        return -1 * op.rank(op.ts_rank(self.close, 10)) * op.rank(op.delta(op.delta(self.close, 1), 1)) * op.rank(
            op.ts_rank(self.volume / self.adv20.replace(0, np.nan), 5)
        )

    def alpha018(self):
        part = op.stddev((self.close - self.open).abs(), 5) + (self.close - self.open)
        corr = op.clean_corr(op.correlation(self.close, self.open, 10))
        return -1 * op.rank(part + corr)

    def alpha019(self):
        return (-1 * np.sign((self.close - op.delay(self.close, 7)) + op.delta(self.close, 7))) * (
            1 + op.rank(1 + op.ts_sum(self.returns, 250))
        )

    def alpha020(self):
        return -1 * op.rank(self.open - op.delay(self.high, 1)) * op.rank(self.open - op.delay(self.close, 1)) * op.rank(
            self.open - op.delay(self.low, 1)
        )

    def alpha021(self):
        cond_1 = op.sma(self.close, 8) + op.stddev(self.close, 8) < op.sma(self.close, 2)
        cond_2 = self.adv20 / self.volume.replace(0, np.nan) < 1
        alpha = pd.DataFrame(1.0, index=self.close.index, columns=self.close.columns)
        alpha = alpha.where(~(cond_1 | cond_2), -1)
        return alpha

    def alpha022(self):
        corr = op.clean_corr(op.correlation(self.high, self.volume, 5))
        return -1 * op.delta(corr, 5) * op.rank(op.stddev(self.close, 20))

    def alpha023(self):
        cond = op.sma(self.high, 20) < self.high
        alpha = pd.DataFrame(0.0, index=self.close.index, columns=self.close.columns)
        alpha = alpha.where(~cond, -1 * op.delta(self.high, 2).fillna(0))
        return alpha

    def alpha024(self):
        cond = op.delta(op.sma(self.close, 100), 100) / op.delay(self.close, 100).replace(0, np.nan) <= 0.05
        alpha = -1 * op.delta(self.close, 3)
        alpha = alpha.where(~cond, -1 * (self.close - op.ts_min(self.close, 100)))
        return alpha

    def alpha025(self):
        return op.rank(((-1 * self.returns) * self.adv20) * self.vwap * (self.high - self.close))

    def alpha026(self):
        df = op.clean_corr(op.correlation(op.ts_rank(self.volume, 5), op.ts_rank(self.high, 5), 5))
        return -1 * op.ts_max(df, 3)

    def alpha027(self):
        inner = op.rank(op.sma(op.correlation(op.rank(self.volume), op.rank(self.vwap), 6), 2) / 2.0)
        alpha = pd.DataFrame(1.0, index=self.close.index, columns=self.close.columns)
        alpha = alpha.where(~(inner > 0.5), -1)
        return alpha

    def alpha028(self):
        corr = op.clean_corr(op.correlation(self.adv20, self.low, 5))
        return op.scale(corr + (self.high + self.low) / 2.0 - self.close)

    def alpha029(self):
        inner = op.rank(op.rank(op.scale(op.log(op.ts_sum(op.rank(op.rank(-1 * op.rank(op.delta(self.close - 1, 5)))), 2)))))
        return op.ts_min(inner, 5) + op.ts_rank(op.delay(-1 * self.returns, 6), 5)

    def alpha030(self):
        d = op.delta(self.close, 1)
        inner = np.sign(d) + np.sign(op.delay(d, 1)) + np.sign(op.delay(d, 2))
        return ((1.0 - op.rank(inner)) * op.ts_sum(self.volume, 5)) / op.ts_sum(self.volume, 20).replace(0, np.nan)

    def alpha031(self):
        corr = op.clean_corr(op.correlation(self.adv20, self.low, 12))
        p1 = op.rank(op.rank(op.rank(op.decay_linear(-1 * op.rank(op.rank(op.delta(self.close, 10))), 10))))
        p2 = op.rank(-1 * op.delta(self.close, 3))
        p3 = np.sign(op.scale(corr))
        return p1 + p2 + p3

    def alpha032(self):
        return op.scale(op.sma(self.close, 7) - self.close) + 20 * op.scale(
            op.correlation(self.vwap, op.delay(self.close, 5), 230)
        )

    def alpha033(self):
        return op.rank(-1 + self.open / self.close.replace(0, np.nan))

    def alpha034(self):
        inner = op.stddev(self.returns, 2) / op.stddev(self.returns, 5).replace(0, np.nan)
        inner = inner.replace([np.inf, -np.inf], 1).fillna(1)
        return op.rank(2 - op.rank(inner) - op.rank(op.delta(self.close, 1)))

    def alpha035(self):
        return op.ts_rank(self.volume, 32) * (1 - op.ts_rank(self.close + self.high - self.low, 16)) * (
            1 - op.ts_rank(self.returns, 32)
        )

    def alpha036(self):
        adv20 = self.adv20
        return (
            2.21 * op.rank(op.correlation(self.close - self.open, op.delay(self.volume, 1), 15))
            + 0.7 * op.rank(self.open - self.close)
            + 0.73 * op.rank(op.ts_rank(op.delay(-1 * self.returns, 6), 5))
            + op.rank(op.correlation(self.vwap, adv20, 6).abs())
            + 0.6 * op.rank((op.sma(self.close, 200) - self.open) * (self.close - self.open))
        )

    def alpha037(self):
        return op.rank(op.correlation(op.delay(self.open - self.close, 1), self.close, 200)) + op.rank(
            self.open - self.close
        )

    def alpha038(self):
        ratio = (self.close / self.open.replace(0, np.nan)).replace([np.inf, -np.inf], 1).fillna(1)
        return -1 * op.rank(op.ts_rank(self.close, 10)) * op.rank(ratio)

    def alpha039(self):
        vol_ratio = self.volume / self.adv20.replace(0, np.nan)
        decay = op.decay_linear(vol_ratio, 9)
        return (-1 * op.rank(op.delta(self.close, 7) * (1 - op.rank(decay)))) * (1 + op.rank(op.sma(self.returns, 250)))

    def alpha040(self):
        return -1 * op.rank(op.stddev(self.high, 10)) * op.correlation(self.high, self.volume, 10)

    def alpha041(self):
        return (self.high * self.low) ** 0.5 - self.vwap

    def alpha042(self):
        return op.rank(self.vwap - self.close) / op.rank(self.vwap + self.close).replace(0, np.nan)

    def alpha043(self):
        return op.ts_rank(self.volume / self.adv20.replace(0, np.nan), 20) * op.ts_rank(-1 * op.delta(self.close, 7), 8)

    def alpha044(self):
        return op.clean_corr(-1 * op.correlation(self.high, op.rank(self.volume), 5))

    def alpha045(self):
        corr1 = op.clean_corr(op.correlation(self.close, self.volume, 2))
        corr2 = op.correlation(op.ts_sum(self.close, 5), op.ts_sum(self.close, 20), 2)
        return -1 * op.rank(op.sma(op.delay(self.close, 5), 20)) * corr1 * op.rank(corr2)

    def alpha046(self):
        inner = (op.delay(self.close, 20) - op.delay(self.close, 10)) / 10 - (
            op.delay(self.close, 10) - self.close
        ) / 10
        alpha = -1 * op.delta(self.close, 1)
        alpha = alpha.where(~(inner < 0), 1)
        alpha = alpha.where(~(inner > 0.25), -1)
        return alpha

    def alpha047(self):
        return (
            (op.rank(1 / self.close.replace(0, np.nan)) * self.volume)
            / self.adv20.replace(0, np.nan)
            * (self.high * op.rank(self.high - self.close))
            / op.sma(self.high, 5).replace(0, np.nan)
            - op.rank(self.vwap - op.delay(self.vwap, 5))
        )

    def alpha048(self):
        num = op.correlation(op.delta(self.close, 1), op.delta(op.delay(self.close, 1), 1), 250) * op.delta(
            self.close, 1
        )
        num = op.indneutralize(num / self.close.replace(0, np.nan), self.industry)
        den = op.ts_sum((op.delta(self.close, 1) / op.delay(self.close, 1).replace(0, np.nan)) ** 2, 250)
        return num / den.replace(0, np.nan)

    def alpha049(self):
        inner = (op.delay(self.close, 20) - op.delay(self.close, 10)) / 10 - (
            op.delay(self.close, 10) - self.close
        ) / 10
        alpha = -1 * op.delta(self.close, 1)
        alpha = alpha.where(~(inner < -0.1), 1)
        return alpha

    def alpha050(self):
        return -1 * op.ts_max(op.rank(op.correlation(op.rank(self.volume), op.rank(self.vwap), 5)), 5)

    def alpha051(self):
        inner = (op.delay(self.close, 20) - op.delay(self.close, 10)) / 10 - (
            op.delay(self.close, 10) - self.close
        ) / 10
        alpha = -1 * op.delta(self.close, 1)
        alpha = alpha.where(~(inner < -0.05), 1)
        return alpha

    def alpha052(self):
        return (
            (-1 * op.delta(op.ts_min(self.low, 5), 5))
            * op.rank((op.ts_sum(self.returns, 240) - op.ts_sum(self.returns, 20)) / 220)
            * op.ts_rank(self.volume, 5)
        )

    def alpha053(self):
        denom = (self.close - self.low).replace(0, 0.0001)
        inner = ((self.close - self.low) - (self.high - self.close)) / denom
        return -1 * op.delta(inner, 9)

    def alpha054(self):
        denom = (self.low - self.high).replace(0, -0.0001)
        return -1 * (self.low - self.close) * (self.open**5) / (denom * (self.close**5))

    def alpha055(self):
        span = (op.ts_max(self.high, 12) - op.ts_min(self.low, 12)).replace(0, 0.0001)
        inner = (self.close - op.ts_min(self.low, 12)) / span
        return op.clean_corr(-1 * op.correlation(op.rank(inner), op.rank(self.volume), 6))

    def alpha056(self):
        num = op.sma(self.returns, 10) / op.sma(op.sma(self.returns, 2), 3).replace(0, np.nan)
        return -(op.rank(num) * op.rank(self.returns * self.cap))

    def alpha057(self):
        return -(self.close - self.vwap) / op.decay_linear(op.rank(op.ts_argmax(self.close, 30)), 2).replace(0, np.nan)

    def alpha058(self):
        v = op.indneutralize(self.vwap, self.industry)
        corr = op.correlation(v, self.volume, 4)
        return -1 * op.ts_rank(op.decay_linear(corr, 8), 6)

    def alpha059(self):
        v = op.indneutralize(self.vwap, self.industry)
        corr = op.correlation(v, self.volume, 4)
        return -1 * op.ts_rank(op.decay_linear(corr, 16), 8)

    def alpha060(self):
        span = (self.high - self.low).replace(0, 0.0001)
        inner = ((self.close - self.low) - (self.high - self.close)) * self.volume / span
        return -(2 * op.scale(op.rank(inner)) - op.scale(op.rank(op.ts_argmax(self.close, 10))))

    def alpha061(self):
        adv180 = op.sma(self.volume, 180)
        return (op.rank(self.vwap - op.ts_min(self.vwap, 16)) < op.rank(op.correlation(self.vwap, adv180, 18))).astype(float)

    def alpha062(self):
        adv20 = self.adv20
        left = op.rank(op.correlation(self.vwap, op.sma(adv20, 22), 10))
        right = op.rank((op.rank(self.open) + op.rank(self.open)) < (op.rank((self.high + self.low) / 2) + op.rank(self.high)))
        return (left < right) * -1

    def alpha063(self):
        d = op.indneutralize(op.delta(self.close, 2), self.industry)
        p1 = op.rank(op.decay_linear(d, 8))
        blend = 0.318108 * self.vwap + (1 - 0.318108) * self.open
        p2 = op.rank(op.decay_linear(op.correlation(blend, op.sma(op.sma(self.volume, 180), 37), 14), 12))
        return (p1 - p2) * -1

    def alpha064(self):
        adv120 = op.sma(self.volume, 120)
        blend = 0.178404 * self.open + (1 - 0.178404) * self.low
        left = op.rank(op.correlation(op.sma(blend, 13), op.sma(adv120, 13), 17))
        blend2 = 0.178404 * (self.high + self.low) / 2 + (1 - 0.178404) * self.vwap
        right = op.rank(op.delta(blend2, 4))
        return (left < right) * -1

    def alpha065(self):
        adv60 = op.sma(self.volume, 60)
        blend = 0.00817205 * self.open + (1 - 0.00817205) * self.vwap
        left = op.rank(op.correlation(blend, op.sma(adv60, 9), 6))
        right = op.rank(self.open - op.ts_min(self.open, 14))
        return (left < right) * -1

    def alpha066(self):
        p1 = op.rank(op.decay_linear(op.delta(self.vwap, 4), 7))
        blend = self.low - self.vwap
        denom = self.open - (self.high + self.low) / 2
        p2 = op.ts_rank(op.decay_linear(blend / denom.replace(0, np.nan), 11), 7)
        return (p1 + p2) * -1

    def alpha067(self):
        v = op.indneutralize(self.vwap, self.industry)
        a = op.indneutralize(self.adv20, self.industry)
        corr = op.correlation(v, a, 6)
        base = op.rank(self.high - op.ts_min(self.high, 2))
        return (base ** op.rank(corr)) * -1

    def alpha068(self):
        adv15 = op.sma(self.volume, 15)
        left = op.ts_rank(op.correlation(op.rank(self.high), op.rank(adv15), 9), 14)
        blend = 0.518371 * self.close + (1 - 0.518371) * self.low
        right = op.rank(op.delta(blend, 1))
        return (left < right) * -1

    def alpha069(self):
        v = op.indneutralize(self.vwap, self.industry)
        left = op.rank(op.ts_max(op.delta(v, 3), 5))
        blend = 0.490655 * self.close + (1 - 0.490655) * self.vwap
        right = op.ts_rank(op.correlation(blend, self.adv20, 5), 9)
        return (left ** right) * -1

    def alpha070(self):
        c = op.indneutralize(self.close, self.industry)
        left = op.rank(op.delta(self.vwap, 1))
        adv50 = op.sma(self.volume, 50)
        right = op.ts_rank(op.correlation(c, adv50, 18), 18)
        return (left ** right) * -1

    def alpha071(self):
        adv180 = op.sma(self.volume, 180)
        p1 = op.ts_rank(op.decay_linear(op.correlation(op.ts_rank(self.close, 3), op.ts_rank(adv180, 12), 18), 4), 16)
        p2 = op.ts_rank(op.decay_linear(op.rank((self.low + self.open - self.vwap - self.vwap) ** 2), 16), 4)
        return _row_max(p1, p2)

    def alpha072(self):
        adv40 = op.sma(self.volume, 40)
        num = op.rank(op.decay_linear(op.correlation((self.high + self.low) / 2, adv40, 9), 10))
        den = op.rank(op.decay_linear(op.correlation(op.ts_rank(self.vwap, 4), op.ts_rank(self.volume, 19), 7), 3))
        return num / den.replace(0, np.nan)

    def alpha073(self):
        p1 = op.rank(op.decay_linear(op.delta(self.vwap, 5), 3))
        blend = 0.147155 * self.open + (1 - 0.147155) * self.low
        ratio = op.delta(blend, 2) / blend.replace(0, np.nan) * -1
        p2 = op.ts_rank(op.decay_linear(ratio, 3), 17)
        return -1 * _row_max(p1, p2)

    def alpha074(self):
        adv30 = op.sma(self.volume, 30)
        left = op.rank(op.correlation(self.close, op.sma(adv30, 37), 15))
        blend = 0.0261661 * self.high + (1 - 0.0261661) * self.vwap
        right = op.rank(op.correlation(op.rank(blend), op.rank(self.volume), 11))
        return (left < right) * -1

    def alpha075(self):
        adv50 = op.sma(self.volume, 50)
        left = op.rank(op.correlation(self.vwap, self.volume, 4))
        right = op.rank(op.correlation(op.rank(self.low), op.rank(adv50), 12))
        return (left < right).astype(float)

    def alpha076(self):
        low_n = op.indneutralize(self.low, self.industry)
        adv81 = op.sma(self.volume, 81)
        p1 = op.rank(op.decay_linear(op.delta(self.vwap, 1), 12))
        inner = op.ts_rank(op.correlation(low_n, adv81, 8), 20)
        p2 = op.ts_rank(op.decay_linear(inner, 17), 19)
        return -1 * _row_max(p1, p2)

    def alpha077(self):
        adv40 = op.sma(self.volume, 40)
        p1 = op.rank(op.decay_linear((self.high + self.low) / 2 + self.high - self.vwap - self.high, 20))
        p2 = op.rank(op.decay_linear(op.correlation((self.high + self.low) / 2, adv40, 3), 6))
        return _row_min(p1, p2)

    def alpha078(self):
        adv40 = op.sma(self.volume, 40)
        blend = 0.352233 * self.low + (1 - 0.352233) * self.vwap
        left = op.rank(op.correlation(op.ts_sum(blend, 20), op.ts_sum(adv40, 20), 7))
        right = op.rank(op.correlation(op.rank(self.vwap), op.rank(self.volume), 6))
        return left ** right

    def alpha079(self):
        blend = 0.60733 * self.close + (1 - 0.60733) * self.open
        left = op.rank(op.delta(op.indneutralize(blend, self.industry), 1))
        adv150 = op.sma(self.volume, 150)
        right = op.rank(op.correlation(op.ts_rank(self.vwap, 4), op.ts_rank(adv150, 9), 15))
        return (left < right).astype(float)

    def alpha080(self):
        blend = 0.868128 * self.open + (1 - 0.868128) * self.high
        left = op.rank(np.sign(op.delta(op.indneutralize(blend, self.industry), 4)))
        adv10 = op.sma(self.volume, 10)
        right = op.ts_rank(op.correlation(self.high, adv10, 5), 6)
        return (left ** right) * -1

    def alpha081(self):
        adv10 = op.sma(self.volume, 10)
        inner = op.rank(op.correlation(self.vwap, op.ts_sum(adv10, 50), 8)) ** 4
        left = op.rank(op.log(op.product(op.rank(inner), 15)))
        right = op.rank(op.correlation(op.rank(self.vwap), op.rank(self.volume), 5))
        return (left < right) * -1

    def alpha082(self):
        vol_n = op.indneutralize(self.volume, self.industry)
        p1 = op.rank(op.decay_linear(op.delta(self.open, 1), 15))
        corr = op.correlation(vol_n, self.open, 17)
        p2 = op.ts_rank(op.decay_linear(corr, 7), 13)
        return -1 * _row_min(p1, p2)

    def alpha083(self):
        rng = (self.high - self.low) / op.sma(self.close, 5).replace(0, np.nan)
        return (op.rank(op.delay(rng, 2)) * op.rank(op.rank(self.volume))) / (rng / (self.vwap - self.close).replace(0, np.nan))

    def alpha084(self):
        return op.signed_power(op.ts_rank(self.vwap - op.ts_max(self.vwap, 15), 21), op.delta(self.close, 5))

    def alpha085(self):
        adv30 = op.sma(self.volume, 30)
        blend = 0.876703 * self.high + (1 - 0.876703) * self.close
        left = op.rank(op.correlation(blend, adv30, 10))
        right = op.rank(op.correlation(op.ts_rank((self.high + self.low) / 2, 4), op.ts_rank(self.volume, 10), 7))
        return left ** right

    def alpha086(self):
        left = op.ts_rank(op.correlation(self.close, op.sma(self.adv20, 15), 6), 20)
        right = op.rank(self.open + self.close - self.vwap - self.open)
        return (left < right) * -1

    def alpha087(self):
        blend = 0.369701 * self.close + (1 - 0.369701) * self.vwap
        p1 = op.rank(op.decay_linear(op.delta(blend, 2), 3))
        adv81 = op.sma(self.volume, 81)
        adv_n = op.indneutralize(adv81, self.industry)
        corr = op.correlation(adv_n, self.close, 13).abs()
        p2 = op.ts_rank(op.decay_linear(corr, 5), 14)
        return -1 * _row_max(p1, p2)

    def alpha088(self):
        adv60 = op.sma(self.volume, 60)
        p1 = op.rank(op.decay_linear(op.rank(self.open) + op.rank(self.low) - op.rank(self.high) - op.rank(self.close), 8))
        corr = op.correlation(op.ts_rank(self.close, 8), op.ts_rank(adv60, 21), 8)
        p2 = op.ts_rank(op.decay_linear(corr, 7), 3)
        return _row_min(p1, p2)

    def alpha089(self):
        adv10 = op.sma(self.volume, 10)
        p1 = op.ts_rank(op.decay_linear(op.correlation(self.low, adv10, 7), 6), 4)
        p2 = op.ts_rank(op.decay_linear(op.delta(op.indneutralize(self.vwap, self.industry), 3), 10), 15)
        return p1 - p2

    def alpha090(self):
        adv40 = op.sma(self.volume, 40)
        adv_n = op.indneutralize(adv40, self.industry)
        left = op.rank(self.close - op.ts_max(self.close, 5))
        right = op.ts_rank(op.correlation(adv_n, self.low, 5), 3)
        return (left ** right) * -1

    def alpha091(self):
        c = op.indneutralize(self.close, self.industry)
        inner = op.decay_linear(op.correlation(c, self.volume, 10), 16)
        p1 = op.ts_rank(op.decay_linear(inner, 4), 5)
        adv30 = op.sma(self.volume, 30)
        p2 = op.rank(op.decay_linear(op.correlation(self.vwap, adv30, 4), 3))
        return (p1 - p2) * -1

    def alpha092(self):
        adv30 = op.sma(self.volume, 30)
        cond = ((self.high + self.low) / 2 + self.close) < (self.low + self.open)
        p1 = op.ts_rank(op.decay_linear(cond.astype(float), 15), 19)
        p2 = op.ts_rank(op.decay_linear(op.correlation(op.rank(self.low), op.rank(adv30), 8), 7), 7)
        return _row_min(p1, p2)

    def alpha093(self):
        adv81 = op.sma(self.volume, 81)
        v = op.indneutralize(self.vwap, self.industry)
        p1 = op.ts_rank(op.decay_linear(op.correlation(v, adv81, 17), 20), 8)
        blend = 0.524434 * self.close + (1 - 0.524434) * self.vwap
        p2 = op.rank(op.decay_linear(op.delta(blend, 3), 16))
        return p1 / p2.replace(0, np.nan)

    def alpha094(self):
        adv60 = op.sma(self.volume, 60)
        left = op.rank(self.vwap - op.ts_min(self.vwap, 12))
        right = op.ts_rank(op.correlation(op.ts_rank(self.vwap, 20), op.ts_rank(adv60, 4), 18), 3)
        return (left ** right) * -1

    def alpha095(self):
        adv40 = op.sma(self.volume, 40)
        left = op.rank(self.open - op.ts_min(self.open, 12))
        corr = op.rank(op.correlation(op.sma((self.high + self.low) / 2, 19), op.sma(adv40, 19), 13)) ** 5
        right = op.ts_rank(corr, 12)
        return (left < right).astype(float)

    def alpha096(self):
        adv60 = op.sma(self.volume, 60)
        p1 = op.ts_rank(op.decay_linear(op.correlation(op.rank(self.vwap), op.rank(self.volume), 4), 4), 8)
        corr = op.correlation(op.ts_rank(self.close, 7), op.ts_rank(adv60, 4), 4)
        p2 = op.ts_rank(op.decay_linear(op.ts_argmax(corr, 13), 14), 13)
        return -1 * _row_max(p1, p2)

    def alpha097(self):
        blend = 0.721001 * self.low + (1 - 0.721001) * self.vwap
        p1 = op.rank(op.decay_linear(op.delta(op.indneutralize(blend, self.industry), 3), 20))
        adv60 = op.sma(self.volume, 60)
        inner = op.ts_rank(op.correlation(op.ts_rank(self.low, 8), op.ts_rank(adv60, 17), 5), 19)
        p2 = op.ts_rank(op.decay_linear(inner, 16), 7)
        return (p1 - p2) * -1

    def alpha098(self):
        adv5 = op.sma(self.volume, 5)
        adv15 = op.sma(self.volume, 15)
        p1 = op.rank(op.decay_linear(op.correlation(self.vwap, op.sma(adv5, 26), 5), 7))
        corr = op.correlation(op.rank(self.open), op.rank(adv15), 21)
        p2 = op.rank(op.decay_linear(op.ts_rank(op.ts_argmin(corr, 9), 7), 8))
        return p1 - p2

    def alpha099(self):
        adv60 = op.sma(self.volume, 60)
        left = op.rank(op.correlation(op.ts_sum((self.high + self.low) / 2, 20), op.ts_sum(adv60, 20), 9))
        right = op.rank(op.correlation(self.low, self.volume, 6))
        return (left < right) * -1

    def alpha100(self):
        span = (self.high - self.low).replace(0, 0.0001)
        pos = ((self.close - self.low) - (self.high - self.close)) / span * self.volume
        p1 = op.scale(op.indneutralize(op.indneutralize(op.rank(pos), self.industry), self.industry))
        corr = op.correlation(self.close, op.rank(self.adv20), 5) - op.rank(op.ts_argmin(self.close, 30))
        p2 = op.scale(op.indneutralize(corr, self.industry))
        return -(1.5 * p1 - p2) * (self.volume / self.adv20.replace(0, np.nan))

    def alpha101(self):
        return (self.close - self.open) / ((self.high - self.low) + 0.001)


def _alpha_method_name(num: int) -> str:
    return f"alpha{num:03d}"


def compute_single_alpha(price_df: pd.DataFrame, alpha_num: int) -> pd.DataFrame:
    if alpha_num < 1 or alpha_num > 101:
        raise ValueError("alpha_num must be between 1 and 101")

    stock = Alphas(price_df)
    method: Callable[[], pd.DataFrame | pd.Series] = getattr(stock, _alpha_method_name(alpha_num))
    wide = method()
    factor_name = f"wq_alpha{alpha_num:03d}"
    return _wide_to_long(wide, factor_name)


def compute_all_alphas(price_df: pd.DataFrame) -> pd.DataFrame:
    base = price_df[["date", "stock_code"]].drop_duplicates()
    for n in range(1, 102):
        part = compute_single_alpha(price_df, n)
        base = base.merge(part, on=["date", "stock_code"], how="left")
    return base
