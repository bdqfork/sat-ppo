from itertools import chain

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.nn.modules.loss import _Loss
from torch.optim import Adam
from torch.utils.data import DataLoader

from agents.agent import Agent
from curiosity import CuriosityFactory
from envs import MultiEnv
from models import ModelFactory
from reporters import Reporter, NoReporter
from rewards import Reward, Advantage
import os


class PPOLoss(_Loss):
    r"""
    Calculates the PPO loss given by equation:

    .. math:: L_t^{CLIP+VF+S}(\theta) = \mathbb{E} \left [L_t^{CLIP}(\theta) - c_v * L_t^{VF}(\theta)
                                        + c_e S[\pi_\theta](s_t) \right ]

    where:

    .. math:: L_t^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left [\text{min}(r_t(\theta)\hat{A}_t,
                                  \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon)\hat{A}_t )\right ]

    .. math:: r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\hat{A}_t

    .. math:: \L_t^{VF}(\theta) = (V_\theta(s_t) - V_t^{targ})^2

    and :math:`S[\pi_\theta](s_t)` is an entropy

    """

    def __init__(self, clip_range: float, v_clip_range: float, c_entropy: float, c_value: float, reporter: Reporter):
        """

        :param clip_range: clip range for surrogate function clipping
        :param v_clip_range: clip range for value function clipping
        :param c_entropy: entropy coefficient constant
        :param c_value: value coefficient constant
        :param reporter: reporter to be used to report loss scalars
        """
        super().__init__()
        self.clip_range = clip_range
        self.v_clip_range = v_clip_range
        self.c_entropy = c_entropy
        self.c_value = c_value
        self.reporter = reporter

    def forward(self, distribution_old: Distribution, value_old: Tensor, distribution: Distribution,
                value: Tensor, action: Tensor, reward: Tensor, advantage: Tensor):
        # Value loss
        value_old_clipped = value_old + \
            (value - value_old).clamp(-self.v_clip_range, self.v_clip_range)
        v_old_loss_clipped = (reward - value_old_clipped).pow(2)
        v_loss = (reward - value).pow(2)
        value_loss = torch.min(v_old_loss_clipped, v_loss).mean()

        # Policy loss
        advantage = (advantage - advantage.mean()) / \
            (advantage.std(unbiased=False) + 1e-8)
        advantage.detach_()
        log_prob = distribution.log_prob(action)
        log_prob_old = distribution_old.log_prob(action)
        ratio = (log_prob - log_prob_old).exp().view(-1)

        surrogate = advantage * ratio
        surrogate_clipped = advantage * \
            ratio.clamp(1 - self.clip_range, 1 + self.clip_range)
        policy_loss = torch.min(surrogate, surrogate_clipped).mean()

        # Entropy
        entropy = distribution.entropy().mean()

        # Total loss
        losses = policy_loss + self.c_entropy * entropy - self.c_value * value_loss
        total_loss = -losses
        self.reporter.scalar('ppo_loss/policy', -policy_loss.item())
        self.reporter.scalar('ppo_loss/entropy', -entropy.item())
        self.reporter.scalar('ppo_loss/value_loss', value_loss.item())
        self.reporter.scalar('ppo_loss/total', total_loss)
        return total_loss


class PPO(Agent):
    """
    Implementation of PPO algorithm described in paper: https://arxiv.org/pdf/1707.06347.pdf
    """

    def __init__(self, env: MultiEnv, model_factory: ModelFactory, curiosity_factory: CuriosityFactory,
                 reward: Reward, advantage: Advantage, learning_rate: float, clip_range: float, v_clip_range: float,
                 c_entropy: float, c_value: float, n_mini_batches: int, n_optimization_epochs: int,
                 clip_grad_norm: float, normalize_state: bool, normalize_reward: bool, save_path: str, save_freq: str,
                 model_path: str = None, reporter: Reporter = NoReporter()) -> None:
        """

        :param env: environment to train on
        :param model_factory: factory to construct the model used as the brain of the agent
        :param curiosity_factory: factory to construct curiosity object
        :param reward: reward function to use for discounted reward calculation
        :param advantage: advantage function to use for advantage calculation
        :param learning_rate: learning rate
        :param clip_range: clip range for surrogate function and value clipping
        :param v_clip_range: clip range for value function clipping
        :param c_entropy: entropy coefficient constant
        :param c_value: value coefficient constant
        :param n_mini_batches: number of mini batches to devide experience into for optimization
        :param n_optimization_epochs number of optimization epochs on same experience. This value is called ``K``
               in paper
        :param clip_grad_norm: value used to clip gradient by norm
        :param normalize_state whether to normalize the observations or not
        :param normalize_reward whether to normalize rewards or not
        :param reporter: reporter to be used for reporting learning statistics
        """
        super().__init__(env, model_factory, curiosity_factory,
                         normalize_state, normalize_reward, save_freq, reporter)
        self.reward = reward
        self.advantage = advantage
        self.n_mini_batches = n_mini_batches
        self.n_optimization_epochs = n_optimization_epochs
        self.clip_grad_norm = clip_grad_norm
        self.optimizer = Adam(
            chain(self.model.parameters(), self.curiosity.parameters()), learning_rate)
        self.loss = PPOLoss(clip_range, v_clip_range,
                            c_entropy, c_value, reporter)
        self.save_path = save_path
        self.model_path = model_path

    def _train(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray):
        policy_old, values_old = self.model(self.state_converter.reshape_as_input(
            states, self.model.recurrent, self.device))
        policy_old = policy_old.detach().view(*states.shape[:2], -1)
        values_old = values_old.detach().view(*states.shape[:2])
        values_old_numpy = values_old.cpu().detach().numpy()
        discounted_rewards = self.reward.discounted(
            rewards, values_old_numpy, dones)
        advantages = self.advantage.discounted(
            rewards, values_old_numpy, dones)
        dataset = self.model.dataset(policy_old[:, :-1], values_old[:, :-1], states[:, :-1], states[:, 1:], actions,
                                     discounted_rewards, advantages)
        loader = DataLoader(dataset, batch_size=len(
            dataset) // self.n_mini_batches, shuffle=True, collate_fn=collate)
        # with torch.autograd.detect_anomaly():
        scaler = torch.cuda.amp.GradScaler()
        for _ in range(self.n_optimization_epochs):
            for tuple_of_batches in loader:
                (batch_policy_old, batch_values_old, batch_states, batch_next_states,
                    batch_actions, batch_rewards, batch_advantages) = tuple_of_batches
                batch_policy_old = batch_policy_old.to(self.device)
                batch_values_old = batch_values_old.to(self.device)
                batch_states = [
                    [i.to(self.device) for i in el] for el in batch_states]
                batch_next_states = [[i.to(self.device) for i in el]
                                     for el in batch_next_states]

                batch_actions = batch_actions.to(self.device)
                batch_rewards = batch_rewards.to(self.device)
                batch_advantages = batch_advantages.to(self.device)
                with torch.cuda.amp.autocast():
                    batch_policy, batch_values = self.model(batch_states)
                    batch_values = batch_values.squeeze()
                    distribution_old = self.action_converter.distribution(
                        batch_policy_old)
                    distribution = self.action_converter.distribution(
                        batch_policy)
                    loss: Tensor = self.loss(distribution_old, batch_values_old, distribution, batch_values,
                                             batch_actions, batch_rewards, batch_advantages)
                    loss = self.curiosity.loss(
                        loss, batch_states, batch_next_states, batch_actions)
                # print('loss:', loss)
                self.optimizer.zero_grad()
                scaler.scale(loss).backward(retain_graph=True)
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip_grad_norm)
                scaler.step(self.optimizer)
                scaler.update()

    def _save(self, epoch):
        print('Saving......')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        model_path = os.path.join(self.save_path, f'model-{epoch}.pt')
        state = {'net': self.model.state_dict(
        ), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, model_path)

    def _load(self):
        if self.model_path is not None:
            print(f'Loading {self.model_path}')
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint['net'],)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch'] + 1


def collate(samples):
    from models import utils
    (batch_policy_old, batch_values_old, batch_states, batch_next_states,
     batch_actions, batch_rewards, batch_advantages) = map(
        list, zip(*samples))
    batch_policy_old = torch.tensor(
        np.array([[el.cpu().numpy()] for el in batch_policy_old]))
    batch_values_old = torch.tensor(
        np.array([el.cpu().numpy() for el in batch_values_old]))

    batch_states = np.array([utils.batch_graphs([[torch.tensor(i) for i in el]])
                             for el in batch_states], dtype=np.object)
    batch_next_states = np.array([utils.batch_graphs([[torch.tensor(i) for i in el]])
                                  for el in batch_next_states], dtype=np.object)

    batch_actions = torch.tensor(np.array(batch_actions, dtype=np.float))

    batch_rewards = torch.tensor(batch_rewards)

    batch_advantages = torch.tensor(batch_advantages)

    return (batch_policy_old, batch_values_old, batch_states, batch_next_states,
            batch_actions, batch_rewards, batch_advantages)
