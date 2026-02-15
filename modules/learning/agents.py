import numpy as np

from modules.learning.models import AgentState


class RLAgentAdapter:
    def __init__(self, model, training_mode=False):
        self.model = model
        self.training_mode = training_mode
        self._lstm_states = None
        self._last_episode_start = np.ones((1,), dtype=bool)

    def get_action(self, state: AgentState) -> float:
        obs = state.get_state_arr(normalize=True).reshape(1, -1)
        is_recurrent = hasattr(self.model, "policy") and "LstmPolicy" in str(type(self.model.policy))

        if is_recurrent:
            action, self._lstm_states = self.model.predict(
                obs,
                state=self._lstm_states,
                episode_start=self._last_episode_start,
                deterministic=not self.training_mode
            )
            self._last_episode_start = np.array([False])
        else:
            action, _ = self.model.predict(obs, deterministic=not self.training_mode)

        return float(action[0])

    def reset_agent(self):
        self._lstm_states = None
        self._last_episode_start = np.ones((1,), dtype=bool)
