import numpy as np
import torch

from agents import PPO
from curiosity import NoCuriosity
from envs import MultiEnv
from models import GNN
from reporters import TensorBoardReporter
from rewards import GeneralizedAdvantageEstimation, GeneralizedRewardEstimation


def main(args):
    from gym.envs.registration import register
    register(
        id="sat-v0", entry_point="sat.env:Sat", kwargs={'config_path': args.config_path})

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reporter = TensorBoardReporter()

    agent = PPO(MultiEnv('sat-v0', args.env_num, reporter),
                reporter=reporter,
                normalize_state=False,
                normalize_reward=False,
                model_factory=GNN.factory(),
                curiosity_factory=NoCuriosity.factory(),
                reward=GeneralizedRewardEstimation(gamma=0.99, lam=0.95),
                advantage=GeneralizedAdvantageEstimation(gamma=0.99, lam=0.95),
                learning_rate=args.learning_rate,
                clip_range=0.2,
                v_clip_range=0.3,
                c_entropy=1e-2,
                c_value=0.5,
                n_mini_batches=args.n_mini_batches,
                n_optimization_epochs=args.n_optimization_epochs,
                clip_grad_norm=0.5,
                save_path=args.save_path,
                save_freq=args.save_freq,
                model_path=args.model_path)
    agent.to(device, torch.float32, np.object)

    agent.learn(epochs=args.epochs, n_steps=args.n_steps)
    agent.eval(n_steps=args.n_steps, render=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--save_freq', type=int, default=10,
                        help='save_freq (default: 10)')
    parser.add_argument('-a', '--save_path', type=str, default='model/sat/100',
                        help='where you store model (default: model/sat/100)')
    parser.add_argument('-p', '--model_path', type=str,
                        help='the model path')
    parser.add_argument('-e', '--epochs', type=int, default=1000,
                        help='epochs (default: 1000)')
    parser.add_argument('-c', '--n_optimization_epochs', type=int, default=5,
                        help='n_optimization_epochs (default: 5)')
    parser.add_argument('-n', '--n_steps', type=int, default=4,
                        help='n_steps (default: 32)')
    parser.add_argument('-b', '--n_mini_batches', type=int, default=4,
                        help='n_mini_batches (default: 4)')
    parser.add_argument('-v', '--env_num', type=int, default=4,
                        help='env_num (default: 4)')
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-4,
                        help='learning_rate (default: 1e-4)')
    parser.add_argument('-k', '--config_path', type=str, default='sat.json',
                        help='config_path (default: sat.json)')
    args = parser.parse_args()

    print(args)
    main(args)
