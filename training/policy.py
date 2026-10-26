"""PPO policy with charging removed from the categorical action choices.

State[8] is battery fraction; state[0:4] are current/start coordinates.
The same mask is used for rollout sampling, PPO log probabilities and prediction.
"""
import torch
from stable_baselines3.common.policies import MultiInputActorCriticPolicy


class ChargeMaskedPolicy(MultiInputActorCriticPolicy):
    def _latents(self, obs):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            return self.mlp_extractor(features)
        pi, vf = features
        return self.mlp_extractor.forward_actor(pi), self.mlp_extractor.forward_critic(vf)

    def _distribution(self, obs, latent):
        state = obs['state']
        available = (state[:, 8] <= 0.5) & (state[:, 0] == state[:, 2]) & (state[:, 1] == state[:, 3])
        logits = self.action_net(latent)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[:, 3] = available
        logits = logits.masked_fill(~mask, -torch.inf)
        return self.action_dist.proba_distribution(action_logits=logits)

    def forward(self, obs, deterministic=False):
        pi, vf = self._latents(obs)
        dist = self._distribution(obs, pi)
        actions = dist.get_actions(deterministic=deterministic)
        return actions.reshape((-1, *self.action_space.shape)), self.value_net(vf), dist.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        pi, vf = self._latents(obs)
        dist = self._distribution(obs, pi)
        return self.value_net(vf), dist.log_prob(actions), dist.entropy()

    def get_distribution(self, obs):
        pi, _ = self._latents(obs)
        return self._distribution(obs, pi)
