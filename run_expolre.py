import numpy as np
from gym.envs.registration import make, register


def main():
    register(
        id="sat-v0", entry_point="sat.minisat:minisat_env", kwargs={'config_path': 'sat.json'})

    env = make("sat-v0")
    epochs = 10000
    epoch = 1
    while epoch <= epochs:
        _ = env.reset()
        done = False
        step = 0
        while not done:
            action = np.array([np.random.choice(100)])
            _, reward, done, cur_lv = env.step(action)
            print(
                f'epoch: {epoch}, step: {step}, reward: {reward}, done: {done}, cur_lv:{cur_lv}')
            step += 1
        epoch += 1


if __name__ == '__main__':
    main()
