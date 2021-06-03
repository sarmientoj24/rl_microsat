import argparse
import time
import gym
import numpy as np
import random
import torch
from omegaconf import OmegaConf
from src.sac import Agent 
from src.commons.utils import (
    plot_learning_curve, NormalizedActions, set_seed_everywhere, WandbLogger,
)
from gym import wrappers
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.environment import UnityEnvironment
from src.commons.utils import get_remaining_time


def main(train=True, wandb_log=False):
    METHOD = 'sac'

    conf = OmegaConf.load(f'./config/{METHOD}.yaml')
    
    # Folders
    env_folder = conf.folder.environment
    plot_folder = conf.folder.plot
    save_folder = conf.folder.models

    # Description
    description = conf.extras.description

    # Parameters
    seed = conf.seed

    # wandb credentials
    proj = conf.wandb.proj
    name = conf.wandb.name
    entity = conf.wandb.entity

    # Training Parameters
    save_frequency = conf.train.save_frequency
    fast_forward = conf.train.fast_forward
    episodes = conf.train.episodes
    train_interval = conf.train.train_interval 

    # Environment Name
    environment_name = conf.environment.name
    no_graphics = conf.environment.no_graphics
    action_dim = conf.environment.action_dim
    state_dim = conf.environment.state_dim
    action_range = conf.environment.action_range

    # Hyperparameters
    batch_size = conf.hparams.batch_size
    hidden_dim = conf.hparams.hidden_dim
    reward_scale = conf.hparams.reward_scale
    replay_buffer_size = conf.hparams.replay_buffer_size
    alpha = conf.hparams.alpha

    # Evaluation
    deterministic = conf.eval.deterministic

    ################# ENVIRONMENT ###################
    if not train:
        fast_forward = 1.0
    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(time_scale = fast_forward)
    unity_env = UnityEnvironment(
        f'./{env_folder}/{environment_name}/{environment_name}', 
        side_channels=[channel], no_graphics=no_graphics, seed=seed)
    env = UnityToGymWrapper(unity_env)

    set_seed_everywhere(seed)

    agent = Agent(
        alpha=alpha,
        state_dim=state_dim, 
        env=env, 
        batch_size=batch_size, 
        hidden_dim=hidden_dim, 
        action_dim=action_dim, 
        action_range=action_range, 
        max_size=replay_buffer_size, 
        reward_scale=reward_scale
    )

    best_score = env.reward_range[0]

    ########### Logger
    if wandb_log:
        LOGGER = WandbLogger(
                    project=proj,
                    name=name,
                    entity=entity
                )


    if train:
        start_train_time = time.time()

        kill_training = False
        reward_history = []
        for episode in range(episodes):
            state = env.reset()
            done = False

            cumulative_reward = 0
            per_ep_time = time.time()

            step = 0
            q1_loss, q2_loss, p1_loss, v1_loss = [], [], [], []

            while not done:
                try:
                    action = agent.choose_action(
                        state,
                        deterministic=deterministic
                    )

                    next_state, reward, done, _ = env.step(action)
                    agent.remember(state, action, reward, next_state, done)

                    ### Training update on given interval
                    if  (done and  agent.memory.get_length() > agent.batch_size) or \
                        (   
                            step > 0
                            and
                            step % train_interval == 0
                            and 
                            agent.memory.get_length() > agent.batch_size
                        ):
                        v1, q1, q2, p1 = agent.learn(
                            debug=False,
                            deterministic=deterministic,
                        )

                        q1_loss.append(q1.detach().cpu().numpy())
                        q2_loss.append(q2.detach().cpu().numpy())
                        p1_loss.append(p1.detach().cpu().numpy())
                        v1_loss.append(v1.detach().cpu().numpy())

                    step += 1
                    cumulative_reward += reward

                except KeyboardInterrupt:
                    print("Training aborted...")
                    agent.save_models()
                    kill_training = True
                    break
            
            # Gracefully end training
            if kill_training:
                break

            reward_history.append(cumulative_reward)
            avg_reward = np.mean(reward_history[-100:])
        
            ### Plot to wandb
            if wandb_log:
                LOGGER.plot_metrics('avg_reward', avg_reward)
                LOGGER.plot_epoch_loss('q1_ave_loss', q1_loss)
                LOGGER.plot_epoch_loss('q2_ave_loss', q2_loss)
                LOGGER.plot_epoch_loss('p1_ave_loss', p1_loss)
                LOGGER.plot_epoch_loss('v1_ave_loss', p1_loss)

            ### Save models
            if  (
                    episode > 0 and
                    episode % save_frequency == 0 and
                    avg_reward > best_score
                ):
                agent.save_models()
            
            ### Print Information
            print(f"############### Episode {episode} ###############")
            print('Score %.1f' % cumulative_reward, 'Average score %.1f' % avg_reward)
            get_remaining_time(episodes, start_train_time, episode)
            epoch_time = time.time() - per_ep_time
            elapsed_time = time.time() - start_train_time
            print("Elapsed time: ", elapsed_time)
            print("Epoch time: ", epoch_time)
            print("")

        # Plot scores
        figure_file = f'./{plot_folder}/{METHOD}_{description}.png'
        x = [i + 1 for i in range(episode)]
        plot_learning_curve(x, reward_history, figure_file)

    else:
        print("Testing...")

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test")
    parser.add_argument("--wandb")
    args = parser.parse_args()

    train = False if args.test else True
    wandb_log = True if args.wandb else False

    main(train=train, wandb_log=wandb_log)
