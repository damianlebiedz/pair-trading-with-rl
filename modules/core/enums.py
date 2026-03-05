from enum import Enum


class BetaHedge(str, Enum):
    NO_HEDGE = "no_hedge"
    STATIC = "static"
    ROLLING = "rolling"


class Interval(str, Enum):
    D1 = "1d"
    H4 = "4h"
    H1 = "1h"
    M30 = "30m"
    M15 = "15m"
    M5 = "5m"
    M3 = "3m"
    M1 = "1m"


class Source(str, Enum):
    LOG = "log"


class ObsSpaceType(str, Enum):
    MINIMAL = "minimal"
    STANDARD = "standard"
    FULL = "full"


class RLModelName(str, Enum):
    RECURRENT_PPO = "recurrent_ppo"
    A2C_BASELINE = "a2c_baseline"


class RLRewards(str, Enum):
    PNL = "pnl"
    PNL_SIGNAL = "pnl_signal"


class RLPolicyType(str, Enum):
    MLP_POLICY = "MlpPolicy"
    MLP_LSTM_POLICY = "MlpLstmPolicy"


class WandbMode(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
