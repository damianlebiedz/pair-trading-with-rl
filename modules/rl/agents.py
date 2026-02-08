import numpy as np


class RLAgentAdapter:
    def __init__(self, model, training_mode=False):
        self.model = model
        self.training_mode = training_mode
        self.last_state = None
        self.last_action = None

    def get_action(self, state_dict: dict) -> float:
        # the same obs vector as in _get_observation (env)
        obs = np.array(
            [
                state_dict.get("z_score", 0),
                state_dict.get("spread", 0),
                state_dict.get("portfolio_value", 1000) / 1000.0,  # normalization
                # other features
            ]
        )

        self.last_state = obs

        action, _ = self.model.predict(obs, deterministic=not self.training_mode)

        self.last_action = action
        return float(action)
