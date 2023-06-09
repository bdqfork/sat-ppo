from typing import List, Union

import numpy as np
import torch

from curiosity import CuriosityFactory
from envs import Runner, MultiEnv, Converter, RandomRunner
from models import ModelFactory
from normalizers import StandardNormalizer, NoNormalizer
from reporters import Reporter, NoReporter


class Agent:
    """
    Base interface for agents
    """

    def __init__(self, env: MultiEnv, model_factory: ModelFactory, curiosity_factory: CuriosityFactory,
                 normalize_state: bool, normalize_reward: bool, save_freq: int = None,
                 reporter: Reporter = NoReporter()) -> None:
        self.env = env
        self.reporter = reporter
        self.state_converter = Converter.for_space(self.env.observation_space)
        self.action_converter = Converter.for_space(self.env.action_space)
        self.model = model_factory.create(
            self.state_converter, self.action_converter)
        self.curiosity = curiosity_factory.create(
            self.state_converter, self.action_converter)
        self.reward_normalizer = StandardNormalizer(
        ) if normalize_reward else NoNormalizer()
        self.state_normalizer = self.state_converter.state_normalizer(
        ) if normalize_state else NoNormalizer()
        self.normalize_state = normalize_state
        self.device: torch.device = None
        self.dtype: torch.dtype = None
        self.numpy_dtype: object = None
        self.start_epoch = 0
        self.save_freq = save_freq

    def act(self, state: np.ndarray) -> np.ndarray:
        """
        Acts in the environment. Returns the action for the given state

        Note: ``N`` in the dimensions stands for number of parallel environments being explored

        :param state: state of shape N * (state space shape) that we want to know the action for
        :return: the action which is array of shape N * (action space shape)
        """
        state = self.state_normalizer.transform(state[:, np.newaxis])
        reshaped_states = self.state_converter.reshape_as_input(
            state, self.model.recurrent, device=self.device)
        logits = self.model.policy_logits(reshaped_states)
        return self.action_converter.action(logits)

    def _train(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray):
        """
        Trains the agent using previous experience.
        Legend for the dimensions of input arrays:
         * ``N`` - number of parallel environments being explored(see ``MultiEnv``)
         * ``T`` - number of time steps that were run on the environment

        :param states: of shape N * T * (state space shape)
        :param actions: of shape N * T * (action space shape)
        :param rewards: of shape N * T * 1
        :param dones: of shape N * T * 1
        """
        raise NotImplementedError('Implement me')

    def _save(self, epoch: int):
        raise NotImplementedError('Implement me')

    def _load(self):
        raise NotImplementedError('Implement me')

    def learn(self, epochs: int, n_steps: int, initialization_steps: int = 1000, render: bool = False):
        """
        Trains the agent for ``epochs`` number of times by running simulation on the environment for ``n_steps``

        :param epochs: number of epochs of training
        :param n_steps: number of steps made in the environment each epoch
        :param initialization_steps: number of steps made on the environment to gather the states then used for
               initialization of the state normalizer
        :param render: whether to render the environment during learning
        """
        if initialization_steps and self.normalize_state:
            s, _, _, _ = RandomRunner(self.env).run(initialization_steps)
            self.state_normalizer.partial_fit(s)

        self._load()
        for epoch in range(self.start_epoch, epochs):
            self.model.eval()
            states, actions, rewards, dones = Runner(
                self.env, self).run(n_steps, render)
            states = self.state_normalizer.partial_fit_transform(states)
            rewards = self.curiosity.reward(rewards, states, actions)
            rewards = self.reward_normalizer.partial_fit_transform(rewards)
            self.model.train()
            self._train(states, actions, rewards, dones)
            print(f'Epoch: {epoch} done')
            if (self.save_freq is not None) and not (epoch + 1) % self.save_freq:
                self._save(epoch)

    def eval(self, n_steps: int, render: bool = False):
        self.model.eval()
        self._load()
        Runner(self.env, self).run(n_steps, render)

    def to(self, device: torch.device, dtype: torch.dtype, numpy_dtype: Union[object, str]) -> None:
        """
        Transfers the agent's model to device
        :param device: device to transfer agent to
        :param dtype: dtype to which cast the model parameters
        :param numpy_dtype: dtype to use for the environment. *Must* be the same as ``dtype`` parameter
        :return:
        """
        self.device = device
        self.dtype = dtype
        self.numpy_dtype = numpy_dtype
        self.model.to(device, dtype)
        self.curiosity.to(device, dtype)
        self.env.astype(numpy_dtype)

    def _tensors_to_device(self, *tensors: torch.Tensor) -> List[torch.Tensor]:
        return [tensor.to(self.device, self.dtype) for tensor in tensors]

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.tensor(array, device=self.device, dtype=self.dtype)
