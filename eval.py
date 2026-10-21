import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from balrog.agents import AgentFactory
from balrog.evaluator import EvaluatorManager
from balrog.utils import (
    collect_and_summarize_results,
    collect_and_summarize_results_by_trial,
    print_summary_table,
    print_trial_summary_table,
    setup_environment,
)


@contextmanager
def redirect_to_file(filepath):
    original = sys.stdout
    with open(filepath, "w") as file:
        sys.stdout = file
        try:
            yield
        finally:
            sys.stdout = original


@hydra.main(config_path="balrog/config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    original_cwd = get_original_cwd()
    setup_environment(original_cwd=original_cwd)

    # Determine output directory
    if config.eval.resume_from is not None:
        output_dir = config.eval.resume_from
    else:
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{timestamp}_{config.agent.type}_{config.client.model_id.replace('/', '_')}"
        output_dir = os.path.join(config.eval.output_dir, run_name)

        # Create the directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Setup logger
    log_filename = os.path.join(output_dir, "eval.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename)],
        force=True,
    )

    # Create an EvaluatorManager and run evaluation
    evaluator_manager = EvaluatorManager(config, original_cwd=original_cwd, output_dir=output_dir)
    agent_factory = AgentFactory(config)
    with redirect_to_file(log_filename):
        evaluator_manager.run(agent_factory)

    # Always keep the original aggregate summary.
    summary = collect_and_summarize_results(output_dir)
    print_summary_table(summary)

    # Trial mode adds a second summary without changing the original one.
    if int(config.eval.num_trials) > 0:
        trial_summary = collect_and_summarize_results_by_trial(output_dir)
        print_trial_summary_table(trial_summary)


if __name__ == "__main__":
    main()
