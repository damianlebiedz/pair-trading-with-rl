from modules.rl.models import AgentState


class RLAgentAdapter:
    def __init__(self, model, training_mode=False):
        self.model = model
        self.training_mode = training_mode
        self.last_state = None
        self.last_action = None

    def get_action(self, state: AgentState) -> float:
        # the same obs vector as in _get_observation (env)

        # obs = state.get_state_arr().reshape(1, -1)
        obs = state.get_state_arr()
        self.last_state = obs

        action, _ = self.model.predict(obs, deterministic=not self.training_mode)

        self.last_action = action
        return float(action)
