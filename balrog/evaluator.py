import copy
import csv
import json
import logging
import multiprocessing
import os
import random
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from balrog.agents.few_shot import FewShotAgent
from balrog.dataset import InContextDataset
from balrog.environments import make_env
from balrog.utils import get_episode_seed, get_unique_seed

logger = logging.getLogger(__name__)


class EvaluatorManager:
    """Manages evaluation of agents across multiple environments and tasks.

    The EvaluatorManager initializes evaluators for each specified environment and handles the execution
    of evaluation tasks either sequentially or in parallel using multiple workers.
    """

    def __init__(self, config, original_cwd="", output_dir="."):
        """Initialize the EvaluatorManager.

        Args:
            config (omegaconf.DictConfig): Configuration object containing evaluation settings.
            original_cwd (str, optional): Original current working directory. Defaults to "".
            output_dir (str, optional): Directory to save evaluation outputs. Defaults to ".".
        
        This initialization sets up evaluators for each environment specified in the configuration, 
        queues up tasks for evaluation, and prepares for either sequential or parallel execution based 
        on the number of workers specified in the configuration.
        No agent or env is created at this point.
        """
        self.config = config
        self.original_cwd = original_cwd
        self.output_dir = output_dir
        if int(config.eval.num_trials) < 0:
            raise ValueError("eval.num_trials must be non-negative")

        # By default: babyai-babaisai-textworld-crafter-nle-minihack
        self.env_names = config.envs.names.split("-")
        self.env_evaluators = {}
        self.tasks = []
        for env_name in self.env_names:
            # Create an Evaluator for each environment
            evaluator = Evaluator(env_name, config, original_cwd=original_cwd, output_dir=self.output_dir)
            self.env_evaluators[env_name] = evaluator
            # Queue up tasks for each environment and episode index: (env_name, task, episode_idx)
            for task in evaluator.tasks:
                for episode_idx in range(evaluator.num_episodes):
                    # If every configured trial exists, or an earlier trial
                    # succeeded under stop_on_success, this episode is complete.
                    if self._episode_is_complete(env_name, task, episode_idx):
                        logging.info(f"Skipping completed task: {env_name}, {task}, episode {episode_idx}")
                    else:
                        self.tasks.append((env_name, task, episode_idx))
        self.num_workers = config.eval.num_workers

    def _episode_is_complete(self, env_name, task, episode_idx):
        """Return whether an episode bundle is complete for resume purposes."""
        configured_trials = int(self.config.eval.num_trials)
        base_dir = os.path.join(self.output_dir, env_name, task)

        if configured_trials == 0:
            legacy_path = os.path.join(base_dir, f"{task}_run_{episode_idx:02d}.json")
            return os.path.exists(legacy_path)

        trial_paths = [
            os.path.join(
                base_dir,
                f"{task}_run_{episode_idx:02d}_trial_{trial_idx:02d}.json",
            )
            for trial_idx in range(1, configured_trials + 1)
        ]
        if all(os.path.exists(path) for path in trial_paths):
            return True

        if bool(self.config.eval.stop_on_success):
            for path in trial_paths:
                if not os.path.exists(path):
                    continue
                try:
                    with open(path, encoding="utf-8") as file:
                        trial_log = json.load(file)
                except (OSError, json.JSONDecodeError):
                    continue
                if trial_log.get("progression", 0.0) >= 1.0:
                    return True

        return False

    def run(self, agent_factory):
        """Run the evaluation using the specified agent factory.

        Args:
            agent_factory (AgentFactory): Factory object to create agents for evaluation.

        Returns:
            dict: Results of the evaluation aggregated by environment name.
        """
        if self.num_workers > 1:
            # Multiple workers, each with one agent, in parallel
            results = self._run_parallel(agent_factory)
        else:
            # One worker, one episode, one agent, one env at a time, sequentially
            # Avoid agent state contamination between different episodes
            results = self._run_sequential(agent_factory)
        return results

    def _run_sequential(self, agent_factory):
        """Run the evaluation sequentially.

        Args:
            agent_factory (AgentFactory): Factory object to create agents for evaluation.

        Returns:
            dict: Results of the evaluation aggregated by environment name.
        """
        results = defaultdict(list)
        total_episodes = len(self.tasks) # tasks to run, excluding already completed tasks
        with tqdm(total=total_episodes, desc="Evaluating Episodes", position=0) as pbar:
            for env_name, task, episode_idx in self.tasks:
                # Example format: 
                # self.env_evaluators = {
                #     "babyai": Evaluator(env_name="babyai", ...),
                #     "textworld": Evaluator(env_name="textworld", ...),
                # }
                evaluator = self.env_evaluators[env_name]
                # Create a new agent for each episode to ensure a fresh state
                agent = agent_factory.create_agent()
                episode_logs = evaluator.run_episode(task, agent, position=1, episode_idx=episode_idx)
                results[env_name].extend(episode_logs)
                pbar.update(1)
        return results

    def _run_parallel(self, agent_factory):
        """Run the evaluation in parallel using multiple workers.

        Args:
            agent_factory (AgentFactory): Factory object to create agents for evaluation.

        Returns:
            dict: Results of the evaluation aggregated by environment name.
        """
        # task_queue: distribute tasks to workers
        task_queue = multiprocessing.Queue()
        # results_queue: collect results from workers
        results_queue = multiprocessing.Queue()

        # fork will copy the parent process's memory space
        # child processes will inherit the parent's state, including the agent factory and environment evaluators
        ctx = multiprocessing.get_context("fork")

        # Initially fill the task queue with tasks up to the number of workers
        for item in self.tasks[: self.num_workers]:
            task_queue.put(item)

        # Create a master progress bar
        pbar = tqdm(total=len(self.tasks), position=0, leave=True)

        # Assign unique positions for progress bars
        positions = list(range(self.num_workers))

        # Create and start worker processes
        processes = []
        for idx in range(self.num_workers):
            position = positions[idx]
            p = ctx.Process(
                target=self._worker,
                args=(task_queue, results_queue, agent_factory, position),
            )
            processes.append(p)
            p.start()

        # Initialize counters
        results = defaultdict(list)
        tasks_completed = 0
        tasks_queued = self.num_workers

        total_tasks = len(self.tasks)

        while tasks_completed < total_tasks:
            result = results_queue.get()
            # task ends no matter success or failure, failed tasks will be added to task list when using resume_from
            if "error" in result:
                logging.error(f"Error in task {result['task']} processed by {result['process_num']}: {result['error']}")
                logging.error(f"Traceback:\n{result['traceback']}")
            else:
                results[result["env_name"]].extend(result["trial_logs"])
            tasks_completed += 1

            # Update progress bar
            pbar.update(1)
            pbar.set_description(f"Last task: {result['task']}, Process: {result.get('process_num', 'N/A')}")

            # Queue another task if there are any left
            if tasks_queued < total_tasks:
                task_queue.put(self.tasks[tasks_queued])
                tasks_queued += 1

        # Signal all workers to stop when all tasks are done (tasks_completed == total_tasks)
        for _ in range(self.num_workers):
            task_queue.put(None)

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Close the master bar when done
        pbar.close()

        return results

    def _worker(self, task_queue, results_queue, agent_factory, position):
        """Worker process for parallel evaluation.

        Args:
            task_queue (multiprocessing.Queue): Queue containing tasks to process.
            results_queue (multiprocessing.Queue): Queue to put the results.
            agent_factory (AgentFactory): Factory object to create agents.
            position (int): Position index for the progress bar.
        """
        # Make sure different workers have different random seeds to avoid identical behavior
        # SHA256 hash of the concatenated string of PID, worker index, episode index, and current time
        # then take the first 8 characters of the hex digest and convert to an integer
        seed = get_unique_seed(process_num=position)
        random.seed(seed)
        np.random.seed(seed)

        process_num = multiprocessing.current_process().name
        while True:
            # Get a task from the queue; if None, exit the loop
            item = task_queue.get()
            if item is None:
                break
            try:
                env_name, task, episode_idx = item
                evaluator = self.env_evaluators[env_name]
                # One agent owns all trials of this episode
                # A new agent is created for the next episode to prevent memory leakage
                agent = agent_factory.create_agent()
                trial_logs = evaluator.run_episode(
                    task,
                    agent,
                    process_num=process_num,
                    position=position + 1,
                    episode_idx=episode_idx,
                )
                # Send one result per episode back to the manager
                # Trials stay grouped and are never independently scheduled by workers
                results_queue.put(
                    {
                        "env_name": env_name,
                        "task": task,
                        "episode_idx": episode_idx,
                        "process_num": process_num,
                        "trial_logs": trial_logs,
                    }
                )
            except Exception as e:
                tb = traceback.format_exc()
                logging.error(f"Error in worker processing task {task}: {e}\n{tb}")
                results_queue.put(
                    {
                        "env_name": env_name,
                        "task": task,
                        "error": str(e),
                        "traceback": tb,
                        "process_num": process_num,
                    }
                )


class Evaluator:
    """Evaluator for a single environment and task.

    The Evaluator handles the execution of evaluation episodes for a specific environment and task,
    including loading in-context learning episodes and running episodes with the agent.
    """

    def __init__(self, env_name, config, original_cwd="", output_dir="."):
        """Initialize the Evaluator.

        Args:
            env_name (str): Name of the environment to evaluate.
            config (omegaconf.DictConfig): Configuration object containing evaluation settings.
            original_cwd (str, optional): Original current working directory. Defaults to "".
            output_dir (str, optional): Directory to save evaluation outputs. Defaults to ".".
        """
        self.env_name = env_name.strip()
        self.config = config
        self.output_dir = output_dir
        self.tasks = config.tasks[f"{self.env_name}_tasks"] # tasks defined from config.yaml

        self.num_episodes = config.eval.num_episodes[self.env_name]
        self.num_workers = config.eval.num_workers
        self.max_steps_per_episode = config.eval.max_steps_per_episode

        # Load the demonstration for FewShotAgent in the future
        self.dataset = InContextDataset(self.config, self.env_name, original_cwd=original_cwd)

    def run_episode(self, task, agent, process_num=None, position=0, episode_idx=0):
        """Run every trial for one episode sequentially with one agent.
        The episode is the unit scheduled by EvaluatorManager.
        All trials reuse the same environment seed and agent, so a reflective agent can carry verbal memory from trial n to trial n + 1.

        Args:
            task (str): Task name.
            agent (Agent): Agent to evaluate.
            process_num (str, optional): Identifier of the process running the episode. Defaults to None.
            position (int, optional): Position index for the progress bar. Defaults to 0.
            episode_idx (int, optional): Index of the episode. Defaults to 0.

        Returns:
            list[dict]: One result log for each trial.
        """
        configured_trials = int(self.config.eval.num_trials)
        # If num_trials is 0, we run one trial and save one JSON file per episode --> as before trial_idx = 0
        # If num_trials > 0, we run multiple trials and save one JSON file for each trial --> trial_idx = 1, 2, ..., n
        trial_indices = range(1, configured_trials + 1) if configured_trials > 0 else [0]
        # trials in one episode share the same seed, so that the agent can reflect on previous trials of the same episode
        seed = self._get_episode_seed(task, episode_idx, process_num)

        # Create the environment only once for this episode. Reusing the same
        # environment object prevents factories such as TextWorld from selecting
        # a different game between trials. The derived episode seed is copied into
        # the environment config so environment selection is also episode-specific.
        episode_config = copy.deepcopy(self.config)
        episode_config.envs.env_kwargs.seed = seed
        env = make_env(self.env_name, task, episode_config)

        agent.reset()
        trial_logs = []

        try:
            # Run every trial of this episode sequentially. With four trials,
            # Trial 1 starts without memory and Trials 2-4 receive reflections
            # accumulated from their preceding trials (bounded by max_reflections).
            for trial_idx in trial_indices:
                # The full reset above handles the legacy Trial 0 and Trial 1.
                # Later trials clear only short-term state so reflection memory survives.
                if trial_idx > 1:
                    if hasattr(agent, "reset_trial"):
                        agent.reset_trial()
                    else:
                        agent.reset()

                reflections_used = list(getattr(agent, "reflections", []))
                trial_log = self._run_trial(
                    env=env,
                    task=task,
                    agent=agent,
                    seed=seed,
                    process_num=process_num,
                    position=position,
                    episode_idx=episode_idx,
                    trial_idx=trial_idx,
                )

                success = trial_log.get("progression", 0.0) >= 1.0

                # If trial is set and the agent has a reflect method, call it to allow the agent to reflect on the completed trial
                if configured_trials > 0 and hasattr(agent, "reflect"):
                    trial_log["reflections_used"] = reflections_used
                    # Record the input and output tokens used for reflection separately, so we can analyze the cost of reflection
                    input_tokens_before = getattr(agent, "reflection_input_tokens", 0)
                    output_tokens_before = getattr(agent, "reflection_output_tokens", 0)
                    # Store the reflection result to trial log
                    reflection_kwargs = {
                        "success": success,
                        "reward": trial_log["episode_return"],
                        "stats": {
                            "progression": trial_log.get("progression", 0.0),
                            "num_steps": trial_log["num_steps"],
                            "failed_candidates": trial_log["failed_candidates"],
                        },
                    }
                    # If the agent has a reflect_on_trajectory method, use it to reflect on the trial's trajectory
                    if hasattr(agent, "reflect_on_trajectory"):
                        reflection = agent.reflect_on_trajectory(
                            trajectory=trial_log.get("trajectory", []),
                            **reflection_kwargs,
                        )
                    # otherwise, use the reflect method
                    else:
                        reflection = agent.reflect(**reflection_kwargs)
                    
                    reflection_input_tokens = (
                        getattr(agent, "reflection_input_tokens", 0) - input_tokens_before
                    )
                    reflection_output_tokens = (
                        getattr(agent, "reflection_output_tokens", 0) - output_tokens_before
                    )
                    trial_log["reflection"] = reflection
                    trial_log["reflection_memory_after_trial"] = list(agent.reflections)
                    trial_log["reflection_input_tokens"] = reflection_input_tokens
                    trial_log["reflection_output_tokens"] = reflection_output_tokens
                    trial_log["input_tokens"] += reflection_input_tokens
                    trial_log["output_tokens"] += reflection_output_tokens

                self._save_trial_log(task, episode_idx, trial_idx, trial_log)
                trial_logs.append(trial_log)

                # If stop_on_success is set and the trial succeeded, break the loop to stop further trials for this episode
                if success and bool(self.config.eval.stop_on_success):
                    logging.info(
                        "Stopping episode %s after successful trial %s",
                        episode_idx,
                        trial_idx,
                    )
                    break
        finally:
            # A worker may process many episodes, so always release resources
            # even when an action or reflection call raises an exception.
            env.close()

        return trial_logs

    def _get_episode_seed(self, task, episode_idx, process_num):
        """Return one seed shared by all trials of an episode."""
        base_seed = self.config.envs.env_kwargs.seed
        if base_seed is None:
            return get_unique_seed(process_num=process_num, episode_idx=episode_idx)
        return get_episode_seed(
            base_seed=base_seed,
            env_name=self.env_name,
            task=task,
            episode_idx=episode_idx,
        )

    def _run_trial(self, env, task, agent, seed, process_num, position, episode_idx, trial_idx):
        """Run one trial without resetting episode-level reflection memory."""
        random.seed(seed)
        np.random.seed(seed)
        # EnvWrapper stores invalid candidates outside the underlying environment,
        # so reset this list explicitly when reusing the object for another trial.
        env.failed_candidates.clear()
        obs, info = env.reset(seed=seed)

        episode_log = {
            "task": task,
            "action_frequency": defaultdict(int),
            "input_tokens": 0,
            "output_tokens": 0,
            "episode_idx": episode_idx,
            "trial_idx": trial_idx,
            "trajectory": [], # store the full trajectory of the trial for reflection and analysis
        }

        instructions = obs["mission"] if self.env_name == "babyai" else None
        agent.prompt_builder.update_instruction_prompt(
            env.get_instruction_prompt(instructions=instructions)
        )

        episode_return = 0.0
        max_steps = env.max_steps if self.max_steps_per_episode is None else self.max_steps_per_episode
        csv_filename = self._trial_path(task, episode_idx, trial_idx, "csv")
        Path(csv_filename).parent.mkdir(exist_ok=True, parents=True)

        with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file, escapechar="˘", quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(
                [
                    "Step",
                    "Action",
                    "Executed Action",
                    "Plan",
                    "Raw Completion",
                    "Reasoning",
                    "Observation",
                    "Reward",
                    "Done",
                ]
            )

            if isinstance(agent, FewShotAgent):
                self.dataset.load_in_context_learning_episodes(
                    self.config.eval.icl_episodes, task, agent
                )
                if self.config.agent.cache_icl and self.config.client.client_name == "gemini":
                    agent.cache_icl()

            pbar = tqdm(
                total=max_steps,
                desc=f"Task: {task}, Trial: {trial_idx}, Proc: {process_num}",
                position=position,
                leave=False,
                dynamic_ncols=True,
            )

            action = None
            last_valid_action = None
            # Run the trial for a maximum number of steps, or until the environment signals termination
            for step in range(max_steps):
                response = agent.act(obs, prev_action=action)
                valid_action = env.check_action_validity(response.completion)
                last_valid_action = valid_action
                reasoning = response.reasoning if hasattr(response, "reasoning") else ""

                episode_log["action_frequency"][valid_action] += 1
                episode_log["input_tokens"] += response.input_tokens
                episode_log["output_tokens"] += response.output_tokens

                obs, reward, terminated, truncated, info = env.step(valid_action)
                done = terminated or truncated
                episode_return += reward

                obs["text"]["long_term_context"] = (
                    f"\n\nYour previous output did not contain a valid action. "
                    f"Defaulted to action: {valid_action}\n\nObservation:\n"
                    + obs["text"]["long_term_context"]
                    if valid_action != response.completion
                    and self.config.eval.feedback_on_invalid_action
                    else obs["text"]["long_term_context"]
                )
                action = response.completion
                long_term_observation = obs["text"].get("long_term_context", "")
                short_term_observation = obs["text"].get("short_term_context", "")
                observation_text = "\n".join(
                    part for part in [long_term_observation, short_term_observation] if part
                )
                plan = getattr(agent, "last_generated_plan", None)
                raw_completion = getattr(agent, "last_raw_completion", None)
                episode_log["trajectory"].append(
                    {
                        "step": step,
                        "requested_action": action,
                        "action": valid_action,
                        "plan": plan,
                        "raw_completion": raw_completion,
                        "reasoning": reasoning,
                        "observation": observation_text,
                        "reward": float(reward),
                        "done": bool(done),
                    }
                )
                csv_writer.writerow(
                    [
                        step,
                        action,
                        valid_action,
                        plan,
                        raw_completion,
                        reasoning,
                        observation_text,
                        reward,
                        done,
                    ]
                )
                pbar.update(1)

                if self.config.eval.save_images and obs["image"]:
                    images_dir = os.path.join(
                        self.output_dir,
                        self.env_name,
                        task,
                        f"episode_{episode_idx:02d}_trial_{trial_idx:02d}",
                    )
                    Path(images_dir).mkdir(exist_ok=True, parents=True)
                    obs["image"].save(os.path.join(images_dir, f"step_{step:04d}.png"))

                if done:
                    logging.info(
                        "Episode %s trial %s done with reward: %s",
                        episode_idx,
                        trial_idx,
                        episode_return,
                    )
                    episode_log["done"] = True
                    break

            if pbar.n < pbar.total:
                pbar.update(pbar.total - pbar.n)
            pbar.set_postfix_str("DONE")
            pbar.close()

        # CustomAgent normally records an action and its resulting observation at the beginning of the next act() call
        # There is no next call after a terminal step, so append the final transition here
        # This lets reflect() inspect the action and observation that actually caused success/failure
        if hasattr(agent, "reflect") and last_valid_action is not None:
            agent.prompt_builder.update_action(last_valid_action)
            agent.prompt_builder.update_observation(obs)

        episode_log["episode_return"] = episode_return
        episode_log["num_steps"] = step + 1
        # Copy the list because the same wrapper is reused and cleared before
        # the next trial; otherwise older in-memory logs would be mutated too.
        episode_log["failed_candidates"] = list(env.failed_candidates)
        episode_log.update(env.get_stats())
        episode_log["process_num"] = process_num
        episode_log["seed"] = seed
        episode_log["agent"] = OmegaConf.to_container(self.config.agent, resolve=True)
        episode_log["client"] = OmegaConf.to_container(self.config.client, resolve=True)
        return episode_log

    def _trial_path(self, task, episode_idx, trial_idx, extension):
        """Build a legacy or trial-specific output path."""
        if int(self.config.eval.num_trials) == 0:
            filename = f"{task}_run_{episode_idx:02d}.{extension}"
        else:
            filename = f"{task}_run_{episode_idx:02d}_trial_{trial_idx:02d}.{extension}"
        return os.path.join(self.output_dir, self.env_name, task, filename)

    def _save_trial_log(self, task, episode_idx, trial_idx, trial_log):
        json_filename = self._trial_path(task, episode_idx, trial_idx, "json")
        Path(json_filename).parent.mkdir(exist_ok=True, parents=True)
        with open(json_filename, "w") as file:
            json.dump(trial_log, file, indent=4)
