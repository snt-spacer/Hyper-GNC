# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--runs_per_env",
    type=int,
    default=1,
    help="The number of runs to be performed for each environment.",
)
parser.add_argument(
    "--overload-experiment-cfg", 
    action="store_true", 
    default=True, help="Overload experiment config. If set to True, it will load the cfg of the model that was used for training."
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    help="The RL algorithm used for training the rsl-rl agent.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import copy
import pandas as pd
import wandb  # Add this import at the top with other imports

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config


from isaaclab_tasks.rans.utils import EvalMetrics

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "rsl_rl_cfg_entry_point" if algorithm in ["ppo"] else f"rsl_rl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    
    if args_cli.overload_experiment_cfg:
        if args_cli.checkpoint:
            agent_cfg = cli_args.load_rsl_rl_cfg(args_cli.checkpoint, args_cli.task)
        else:
            raise ValueError("Missing checkpoint path for loading the experiment config.")
    else:
        # parse configuration
        env_cfg = parse_env_cfg(
            args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        )
        agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        from gymnasium.wrappers.record_video import RecordVideo
        env = RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # Handle multitask vs single task environments
    if "MultiTask" in args_cli.task:
        robot_name = env.env.get_wrapper_attr('robot_api')._robot_cfg.robot_name
        tasks_apis = env.env.get_wrapper_attr('tasks_apis')
        num_tasks = len(tasks_apis)
        task_names = [task_api.__class__.__name__[:-4] for task_api in tasks_apis]  # remove "Task" suffix
        task_chunks = torch.chunk(env.env.env.scene.env_origins, num_tasks)
        task_chunk_sizes = [len(chunk) for chunk in task_chunks]
        
        print(f"[INFO] MultiTask environment with {num_tasks} tasks:")
        for i, task_name in enumerate(task_names):
            print(f"  Task {i}: {task_name} ({task_chunk_sizes[i]} environments)")
        
        # Create separate evaluation metrics for each task (including robot data)
        eval_metrics_list = []
        for i, task_name in enumerate(task_names):
            # Use the main log directory for all tasks instead of creating separate task directories
            eval_metrics = EvalMetrics(
                env=env,
                robot_name=robot_name,
                task_name=task_name,
                folder_path=log_dir,  # Use main log directory
                device=args_cli.device,
                num_runs_per_env=args_cli.runs_per_env,
                task_index=i
            )
            eval_metrics_list.append(eval_metrics)
        
    else:
        robot_name = env.env.get_wrapper_attr('robot_api')._robot_cfg.robot_name
        task_name = env.env.get_wrapper_attr('task_api').__class__.__name__[:-4] # remove "Task" suffix
        
        metrics_dir = os.path.join(log_dir, "metrics")
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
        plots_dir = os.path.join(log_dir, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        eval_metrics = EvalMetrics(
            env=env, 
            robot_name=robot_name, 
            task_name=task_name, 
            folder_path=log_dir, 
            device=env.unwrapped.device,
            num_runs_per_env=args_cli.runs_per_env,
        )

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_jit(
    #     ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    # )
    # export_policy_as_onnx(
    #     ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    # )

    dt = env.unwrapped.physics_dt

    # Initialize data collection structures
    if "MultiTask" in args_cli.task:
        # For multitask: collect data for each task separately (including robot data)
        tasks_data = []
        for i in range(num_tasks):
            # Combine task and robot data keys
            task_data_keys = tasks_apis[i].eval_data_keys
            robot_data_keys = env.env.get_wrapper_attr('robot_api').eval_data_keys
            all_data_keys = task_data_keys + robot_data_keys
            
            task_data = {k: [] for k in all_data_keys}
            task_data["dones"] = []
            tasks_data.append(task_data)
        
        # Track completion for each task
        task_completion_counts = [torch.zeros(task_chunk_sizes[i], device=env.unwrapped.device) for i in range(num_tasks)]
        
    else:
        # For single task: use the original approach
        data = {k: [] for k in env.env.get_wrapper_attr('eval_data_keys')}
        data["dones"] = []

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # Initialize wandb if requested
    use_wandb = False #(hasattr(agent_cfg, 'logger') and agent_cfg.logger == 'wandb') or (hasattr(args_cli, 'logger') and args_cli.logger == 'wandb')
    if use_wandb:
        wandb_kwargs = getattr(agent_cfg, 'wandb_kwargs', {"project": "isaaclab", "entity": "isaaclab"})
        wandb.init(
            project=wandb_kwargs.get("project", "isaaclab"),
            entity=wandb_kwargs.get("entity", "isaaclab"),
            name=f"eval_{agent_cfg.experiment_name}",
            group=agent_cfg.experiment_name,
            config=vars(args_cli),
            dir=log_dir,
            reinit=True,
        )
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)

            if "MultiTask" in args_cli.task:
                # Collect data from all tasks (including robot data for each task)
                for i, task_api in enumerate(tasks_apis):
                    # Get task-specific data
                    task_eval_data = task_api.eval_data
                    
                    # Get robot data for this task's environments
                    robot_eval_data = env.env.get_wrapper_attr('robot_api').eval_data
                    task_start_idx = sum(task_chunk_sizes[:i])
                    task_end_idx = task_start_idx + task_chunk_sizes[i]
                    task_robot_data = {k: v[task_start_idx:task_end_idx] for k, v in robot_eval_data.items()}
                    
                    # Combine task and robot data
                    combined_data = {**task_eval_data, **task_robot_data}
                    
                    # if i == 2:
                    #     print(f"Ang vel err: {combined_data['error_angular_velocity'][0].item()}, Target ang vel: {combined_data['angular_velocity_target'][0].item()} actual: {combined_data['angular_velocity'][0][2].item()}")
                    

                    for k, v in combined_data.items():
                        tasks_data[i][k].append(v.clone())
                    
                    # Get task-specific dones
                    task_dones = dones[task_start_idx:task_end_idx]
                    tasks_data[i]["dones"].append(task_dones)
                    
                    # Update completion counts
                    if torch.any(task_dones == 1):
                        completed_indices = torch.where(task_dones == 1)[0]
                        task_completion_counts[i][completed_indices] += 1
                        # Log to wandb at episode end for these envs
                        if use_wandb:
                            # Log task logs
                            if hasattr(task_api, 'scalar_logger'):
                                log_dict = task_api.scalar_logger.compute_extras()
                                wandb.log({f"task_{i}/" + k: v.item() if hasattr(v, 'item') else v for k, v in log_dict.items()})
                            # Log robot logs (for this chunk)
                            robot_logger = getattr(env.env.get_wrapper_attr('robot_api'), 'scalar_logger', None)
                            if robot_logger is not None:
                                # Only log for the relevant chunk
                                robot_log_dict = robot_logger.compute_extras()
                                wandb.log({f"robot_{i}/" + k: v.item() if hasattr(v, 'item') else v for k, v in robot_log_dict.items()})
                
                # Check if all tasks have completed their required runs
                all_tasks_completed = True
                for i in range(num_tasks):
                    if not torch.all(task_completion_counts[i] >= args_cli.runs_per_env).item():
                        all_tasks_completed = False
                        break
                
                if all_tasks_completed:
                    # print(f"[INFO] All tasks completed {args_cli.runs_per_env} runs per environment.")
                    break
                    
            else:
                # Single task evaluation (original logic)
                new_data = copy.deepcopy(env.env.unwrapped.eval_data)
                for k, v in new_data.items():
                    data[k].append(v)

                data["dones"].append(dones)

                # # Log to wandb at episode end for these envs
                # if use_wandb and torch.any(dones == 1):
                #     # Log task logs
                #     task_logger = getattr(env.env.get_wrapper_attr('task_api'), 'scalar_logger', None)
                #     if task_logger is not None:
                #         log_dict = task_logger.compute_extras()
                #         wandb.log({"task/" + k: v.item() if hasattr(v, 'item') else v for k, v in log_dict.items()})
                #     # Log robot logs
                #     robot_logger = getattr(env.env.get_wrapper_attr('robot_api'), 'scalar_logger', None)
                #     if robot_logger is not None:
                #         robot_log_dict = robot_logger.compute_extras()
                #         wandb.log({"robot/" + k: v.item() if hasattr(v, 'item') else v for k, v in robot_log_dict.items()})

                # Check if the number of runs per env is reached
                if torch.all(torch.sum(torch.cat(data["dones"], dim=-1).view(-1, env_cfg.scene.num_envs), dim=0) >= args_cli.runs_per_env).item():
                    # print(f"[INFO] Collected {args_cli.runs_per_env} runs per env.")
                    break

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


    # Calculate metrics
    if "MultiTask" in args_cli.task:
        # Calculate metrics for each task
        for i, (task_name, eval_metrics) in enumerate(zip(task_names, eval_metrics_list)):
            # print(f"[INFO] Calculating metrics for task {i}: {task_name}")
            task_data_processed = {k: torch.stack(v, dim=0) for k, v in tasks_data[i].items()}
            eval_metrics.calculate_metrics(data=task_data_processed)
            # Save extracted trajectories
            eval_metrics.save_extracted_trajectories_to_csv()

            if use_wandb:
                df = eval_metrics.convert_metrics_to_pd()
                wandb.log({f"metrics/task_{i}_{task_name}": wandb.Table(dataframe=df)})
    else:
        # Single task metrics calculation
        data = {k: torch.stack(v, dim=0) for k, v in data.items()}
        eval_metrics.calculate_metrics(data=data)
        # Save extracted trajectories
        eval_metrics.save_extracted_trajectories_to_csv()
        # Log processed metrics to wandb after metrics calculation
        if use_wandb:
            df = eval_metrics.convert_metrics_to_pd()
            wandb.log({f"metrics/{task_name}": wandb.Table(dataframe=df)})

    

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
