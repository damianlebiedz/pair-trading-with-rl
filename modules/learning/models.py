from dataclasses import dataclass, fields
import numpy as np

from modules.core.enums import ObsSpaceType


@dataclass(slots=True)
class AgentState:
    """
    Encapsulates the observable state of the trading environment for the RL agent.

    Attributes:
        z_score (float): Z-Score based on entry beta-hedge and std (when in-position), equals to market_z_score when out-of-position.
        hurst (float): Hurst Exponent value.
        position (float): Current position in the strategy (-1, 0, 1 scaled by capital).
        signal (float): Position (signal) from non-RL backtest.
        norm_time_in_pos (float): Normalized time in current position (0.0–1.0), where 1.0 means equal to window length.

    Methods:
        get_state_arr() -> np.ndarray:
            Converts the dataclass into a fixed-size float32 array suitable as
            input to RL agents.
    """

    z_score: float
    position: float
    norm_time_in_pos: float
    signal: float
    hurst: float

    @classmethod
    def get_obs_shape(cls, obs_space_type: ObsSpaceType) -> tuple[int]:
        """
        Dynamically calculates the shape of the observation array.

        It creates a dummy state instance with zeroed fields to evaluate
        the array length for a given observation space type, preventing
        the need for hardcoded dimensions.
        """
        dummy_kwargs = {f.name: 0 if f.type is int else 0.0 for f in fields(cls)}
        dummy_state = cls(**dummy_kwargs)
        return dummy_state.get_state_arr(obs_space_type).shape

    def get_state_arr(
        self,
        obs_space_type: ObsSpaceType,
    ) -> np.ndarray:
        """
        obs_space_type == "autonomous":
            - state_arr: z_score, position, norm_time_in_pos
        obs_space_type == "standard":
            - state_arr: z_score, position, norm_time_in_pos, signal
        obs_space_type == "full":
            - state_arr: z_score, position, norm_time_in_pos, signal, hurst
        """
        if obs_space_type == ObsSpaceType.AUTONOMOUS:
            return np.array(
                [
                    self.z_score,
                    self.position,
                    self.norm_time_in_pos,
                ],
                dtype=np.float32,
            )

        elif obs_space_type == ObsSpaceType.STANDARD:
            return np.array(
                [
                    self.z_score,
                    self.position,
                    self.norm_time_in_pos,
                    self.signal,
                ],
                dtype=np.float32,
            )

        else:
            return np.array(
                [
                    self.z_score,
                    self.position,
                    self.norm_time_in_pos,
                    self.signal,
                    self.hurst,
                ],
                dtype=np.float32,
            )
