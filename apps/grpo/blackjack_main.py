# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Usage: python -m apps.grpo.blackjack_main --config apps/grpo/blackjack.yaml

"""
BlackJack GRPO Training with OpenEnv

This demonstrates integrating OpenSpiel's BlackJack environment with GRPO training
using the OpenEnv framework and Forge.

Tutorial Highlights:
- Native Python OpenSpiel server (no Docker)
- Text-based prompts for game state
- LLM learns to play BlackJack via GRPO
- Action history tracking for better learning
"""

import asyncio
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
import torchstore as ts
from forge.actors._torchstore_utils import (
    get_dcp_whole_state_dict_key,
    get_param_prefix,
)
from forge.actors.generator import Generator
from forge.actors.reference_model import ReferenceModel
from forge.actors.replay_buffer import ReplayBuffer
from forge.actors.trainer import RLTrainer
from forge.cli.config import parse
from forge.controller.actor import ForgeActor
from forge.controller.provisioner import init_provisioner, shutdown
from forge.data_models.completion import Completion
from forge.observability.metric_actors import get_or_create_metric_logger
from forge.observability.metrics import record_metric, Reduce
from forge.observability.perf_tracker import Tracer
from forge.types import LauncherConfig, ProvisionerConfig
from forge.util.ops import compute_logprobs
from monarch.actor import endpoint
from omegaconf import DictConfig
from vllm.transformers_utils.tokenizer import get_tokenizer

# Add OpenEnv to path
openenv_path = Path("/Users/sanyambhutani/OpenEnv/OpenEnv/src")
if openenv_path.exists():
    sys.path.insert(0, str(openenv_path))

from envs.openspiel_env import OpenSpielEnv, OpenSpielAction


@dataclass
class Episode:
    """Episode data for BlackJack game."""

    episode_id: str
    pad_id: int
    request_len: int
    response_len: int
    game_id: str  # Track which game this episode belongs to
    step_in_game: int  # Which step in the game (0, 1, 2, ...)
    # Processed data
    completion: Completion | None = None
    ref_logprobs: torch.Tensor | None = None
    reward: float | None = None
    advantage: float | None = None

    @property
    def policy_version(self) -> int | None:
        return self.completion.generator_version

    @property
    def request_tensor(self) -> torch.Tensor:
        request_tokens: torch.Tensor = self.completion.prompt_ids
        tensor = torch.tensor(request_tokens, dtype=torch.long)
        if tensor.shape[0] < self.request_len:  # left pad
            diff = self.request_len - tensor.shape[0]
            tensor = F.pad(tensor, (diff, 0), value=self.pad_id)
        return tensor

    @property
    def response_tensor(self) -> torch.Tensor:
        response_tokens: torch.Tensor = self.completion.token_ids
        tensor = torch.tensor(response_tokens, dtype=torch.long)
        if tensor.shape[0] < self.response_len:  # right pad
            diff = self.response_len - tensor.shape[0]
            tensor = F.pad(tensor, (0, diff), value=self.pad_id)
        return tensor


# Represents the group (G) of episodes in GRPO
Group = list[Episode]

# Represents the Policy Model to collect data from
Policy = Generator


def collate(
    batches: list[Group],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collates a list of batches into a single batch of inputs and targets."""
    inputs = []
    targets = []
    for batch in batches:
        request = [e.request_tensor for e in batch]
        request = torch.stack(request)  # [b x s]

        response = [e.response_tensor for e in batch]
        response = torch.stack(response)  # [b x s]

        ref_logprobs = [e.ref_logprobs for e in batch]
        ref_logprobs = torch.stack(ref_logprobs).squeeze()  # [b x s]

        advantages = [e.advantage for e in batch]
        advantages = torch.tensor(advantages).unsqueeze(-1)  # [b x 1]

        pad_id = batch[0].pad_id
        mask = response != pad_id

        input = {"tokens": torch.cat([request, response], dim=1)}
        target = {
            "response": response,
            "ref_logprobs": ref_logprobs,
            "advantages": advantages,
            "padding_mask": mask,
        }
        inputs.append(input)
        targets.append(target)
    return inputs, targets


def simple_grpo_loss(
    logits: torch.Tensor,
    response: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    padding_mask: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """GRPO Loss Function."""
    logprobs: torch.Tensor = compute_logprobs(logits, response)
    kl = torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1
    per_token_policy_loss = torch.exp(logprobs - logprobs.detach()) * advantages
    per_token_loss = -(per_token_policy_loss - beta * kl)
    loss = (
        ((per_token_loss * padding_mask).sum(dim=1))
        / (padding_mask.sum(dim=1).clamp(min=1.0))
    ).mean()
    return loss


@dataclass
class BlackJackReward(ForgeActor):
    """Reward actor for BlackJack that uses game outcomes."""

    @endpoint
    async def evaluate_response(
        self, prompt: str, response: str, game_reward: float
    ) -> float:
        """
        Evaluate BlackJack response based on game outcome.

        Args:
            prompt: The prompt sent to the model
            response: The model's response (HIT/STAND)
            game_reward: The reward from the game (+1 win, -1 loss, 0 push)

        Returns:
            Reward value for GRPO training
        """
        # For now, use game outcome directly
        # Could add shaped rewards later (e.g., penalty for busting)
        reward = float(game_reward)

        # Log metrics
        record_metric("reward/evaluate_response/avg_reward", reward, Reduce.MEAN)
        record_metric("reward/evaluate_response/sum_reward", reward, Reduce.SUM)
        record_metric("reward/evaluate_response/count_calls", 1, Reduce.SUM)

        return reward


@dataclass
class ComputeAdvantages(ForgeActor):
    """Compute advantages for GRPO using reward signals."""

    @endpoint
    async def compute(self, group: Group) -> list[float]:
        """Compute group-relative advantages."""
        rewards = torch.tensor([[e.reward for e in group]])
        mean = rewards.mean(1, keepdim=True)
        std = rewards.std(1, keepdim=True)
        advantages = (rewards - mean) / (std + 1e-4)
        return advantages.squeeze(0).tolist()


def parse_action_from_text(response_text: str, legal_actions: list[int]) -> int:
    """
    Parse LLM output to BlackJack action.

    Args:
        response_text: Model's generated text
        legal_actions: List of legal action IDs

    Returns:
        Action ID (0=HIT, 1=STAND)
    """
    text_lower = response_text.lower().strip()

    # Look for action keywords
    if "hit" in text_lower:
        action_id = 0
    elif "stand" in text_lower:
        action_id = 1
    else:
        # Default to STAND if unclear (safer strategy)
        action_id = 1

    # Validate action is legal
    if action_id not in legal_actions:
        action_id = legal_actions[0]  # Fallback to first legal action

    return action_id


def format_blackjack_prompt(
    step_num: int,
    info_state: list[float],
    legal_actions: list[int],
    action_history: list[tuple[int, str]],
    tokenizer,
) -> str:
    """
    Format BlackJack game state as a prompt for the LLM.

    Args:
        step_num: Current step number in the game
        info_state: 189-dim state vector from OpenSpiel
        legal_actions: Legal action IDs
        action_history: List of (action_id, action_name) tuples
        tokenizer: Tokenizer for chat template

    Returns:
        Formatted prompt string
    """
    # Create system prompt
    system_prompt = """You are an expert BlackJack player. Your goal is to beat the dealer by getting as close to 21 as possible without going over (busting).

Rules:
- HIT: Take another card (increases your total)
- STAND: Keep your current hand (dealer plays next)

Strategy tips:
- Generally HIT if you have 11 or less
- Generally STAND if you have 17 or more
- Be careful between 12-16 (depends on dealer's card)

You must output ONLY 'HIT' or 'STAND'. Nothing else."""

    # Build game state description
    state_description = f"=== BlackJack Game (Step {step_num + 1}) ===\n\n"

    # Add action history if available
    if action_history:
        state_description += "Your previous actions:\n"
        for i, (action_id, action_name) in enumerate(action_history):
            state_description += f"  {i + 1}. {action_name}\n"
        state_description += "\n"

    # Add current decision point
    state_description += "Current situation:\n"
    state_description += f"- Step: {step_num + 1}\n"
    state_description += f"- Legal actions: {['HIT' if a == 0 else 'STAND' for a in legal_actions]}\n"
    state_description += "\n"

    # Add state info (simplified - parsing 189-dim vector is complex)
    # For now, just indicate game is in progress
    state_description += "Game Status: In Progress\n"
    state_description += "\n"
    state_description += "What is your action? (Output only 'HIT' or 'STAND')"

    # Format as chat
    as_chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state_description},
    ]

    # Apply chat template
    formatted_prompt = tokenizer.apply_chat_template(
        as_chat, tokenize=False, add_generation_prompt=True
    )

    return formatted_prompt


@dataclass
class BlackJackEnvironmentActor(ForgeActor):
    """Actor that manages BlackJack game environment via OpenEnv."""

    server_url: str = "http://localhost:8000"
    model: str = "Qwen/Qwen3-1.7B"

    @endpoint
    def setup(self):
        """Initialize tokenizer and environment client."""
        self._tokenizer = get_tokenizer(self.model)
        # Note: We'll create new env connections per game to avoid state issues
        print(f"BlackJackEnvironmentActor initialized (server: {self.server_url})")

    @endpoint
    async def play_game(self, group_size: int, policy) -> list[dict[str, Any]]:
        """
        Play a full BlackJack game for each member of the group.

        Args:
            group_size: Number of parallel games to play
            policy: Policy actor to generate actions

        Returns:
            List of game results with prompts, responses, and rewards
        """
        game_results = []

        for game_idx in range(group_size):
            game_id = str(uuid.uuid4())

            # Create fresh environment connection
            env = OpenSpielEnv(base_url=self.server_url)

            try:
                # Reset environment
                result = env.reset()
                obs = result.observation

                # Track game state
                done = False
                step_num = 0
                action_history = []
                step_results = []

                while not done and step_num < 10:  # Max 10 steps per game
                    # Format prompt
                    prompt = format_blackjack_prompt(
                        step_num=step_num,
                        info_state=obs.info_state,
                        legal_actions=obs.legal_actions,
                        action_history=action_history,
                        tokenizer=self._tokenizer,
                    )

                    # Get policy response (single completion for this step)
                    responses: list[Completion] = await policy.generate.route(prompt)
                    response = responses[0]  # Take first completion

                    # Parse action
                    action_id = parse_action_from_text(
                        response.text, obs.legal_actions
                    )
                    action_name = "HIT" if action_id == 0 else "STAND"

                    # Record action
                    action_history.append((action_id, action_name))

                    # Take step in environment
                    result = env.step(
                        OpenSpielAction(action_id=action_id, game_name="blackjack")
                    )
                    obs = result.observation
                    done = result.done

                    # Store step result
                    step_result = {
                        "game_id": game_id,
                        "step_num": step_num,
                        "prompt": prompt,
                        "response": response,
                        "action_id": action_id,
                        "action_name": action_name,
                        "done": done,
                        "reward": result.reward if done else 0.0,
                    }
                    step_results.append(step_result)

                    step_num += 1

                # Game finished
                final_reward = result.reward if result.reward is not None else 0.0

                # Assign reward to all steps in the game
                for step_result in step_results:
                    step_result["final_reward"] = final_reward

                game_results.extend(step_results)

            finally:
                env.close()

            # Log game metrics
            record_metric("blackjack/play_game/count_games_played", 1, Reduce.SUM)
            record_metric(
                "blackjack/play_game/avg_game_length", step_num, Reduce.MEAN
            )
            record_metric(
                "blackjack/play_game/avg_game_reward", final_reward, Reduce.MEAN
            )

        return game_results

    @endpoint
    async def pad_token(self):
        """Return pad token ID."""
        return self._tokenizer.pad_token_id


async def drop_weights(version: int):
    """Drop old model weights from torchstore."""
    print(f"Dropping weights @ version {version}")
    start_time = time.perf_counter()
    prefix = get_param_prefix(version)
    matching_keys = await ts.keys(prefix)
    dcp_key = get_dcp_whole_state_dict_key(version)
    if dcp_key in matching_keys:
        dcp_handle = await ts.get(dcp_key)
        dcp_handle.drop()
    for key in matching_keys:
        await ts.delete(key)
    elapsed = time.perf_counter() - start_time
    print(f"Dropped weights @ version {version}, took {elapsed:.2f} seconds")


async def main(cfg: DictConfig):
    """Main GRPO training loop for BlackJack."""
    group_size = cfg.group_size
    max_req_tokens = cfg.max_req_tokens
    max_res_tokens = cfg.max_res_tokens

    # ---- Global setups ---- #
    provisioner = None
    if cfg.get("provisioner", None) is not None:
        provisioner = await init_provisioner(
            ProvisionerConfig(launcher_config=LauncherConfig(**cfg.provisioner))
        )
    else:
        provisioner = await init_provisioner()

    metric_logging_cfg = cfg.get(
        "metric_logging", {"console": {"log_per_rank": False}}
    )
    mlogger = await get_or_create_metric_logger()
    await mlogger.init_backends.call_one(metric_logging_cfg)

    # ---- Setup services ---- #
    print("Initializing services...")

    (
        blackjack_env,
        policy,
        trainer,
        replay_buffer,
        compute_advantages,
        ref_model,
        reward_actor,
    ) = await asyncio.gather(
        BlackJackEnvironmentActor.options(**cfg.actors.blackjack_env).as_actor(
            **cfg.blackjack_env
        ),
        Policy.options(**cfg.services.policy).as_service(**cfg.policy),
        RLTrainer.options(**cfg.actors.trainer).as_actor(
            **cfg.trainer, loss=simple_grpo_loss
        ),
        ReplayBuffer.options(**cfg.actors.replay_buffer).as_actor(
            **cfg.replay_buffer, collate=collate
        ),
        ComputeAdvantages.options(**cfg.actors.compute_advantages).as_actor(),
        ReferenceModel.options(**cfg.services.ref_model).as_service(**cfg.ref_model),
        BlackJackReward.options(**cfg.services.reward_actor).as_service(),
    )

    # Set max_steps
    max_steps = cfg.trainer.training.steps or -1

    print("All services initialized successfully!")
    shutdown_event = asyncio.Event()

    # Initialize torchstore
    trainer_num_procs = cfg.actors.trainer["procs"]
    trainer_host_mesh_name = cfg.actors.trainer["mesh_name"]
    trainer_hosts = provisioner.get_host_mesh(trainer_host_mesh_name)
    await ts.initialize(
        mesh=trainer_hosts.spawn_procs(per_host={"procs": trainer_num_procs}),
        strategy=ts.LocalRankStrategy(),
    )
    print("Torchstore successfully initialized with local rank strategy")

    # ---- Core RL loops ---- #
    async def continuous_rollouts():
        """Collect BlackJack game episodes."""
        rollout_count = 0
        pad_id = await blackjack_env.pad_token.call_one()

        while not shutdown_event.is_set():
            t = Tracer("main_perf/continuous_rollouts")
            t.start()

            # Play group_size games
            game_results = await blackjack_env.play_game.call_one(group_size, policy)
            t.step("play_games")

            # Process each step result into episodes
            episodes = []
            input_ids = torch.ones(
                (len(game_results), max_req_tokens + max_res_tokens),
                dtype=torch.long,
            )

            for i, step_result in enumerate(game_results):
                episode = Episode(
                    episode_id=str(uuid.uuid4()),
                    pad_id=pad_id,
                    request_len=max_req_tokens,
                    response_len=max_res_tokens,
                    game_id=step_result["game_id"],
                    step_in_game=step_result["step_num"],
                    completion=step_result["response"],
                )

                # Evaluate reward
                episode.reward = await reward_actor.evaluate_response.route(
                    prompt=step_result["prompt"],
                    response=step_result["response"].text,
                    game_reward=step_result["final_reward"],
                )

                episodes.append(episode)

                # Build input_ids for reference logprobs
                input_ids[i, :max_req_tokens] = episode.request_tensor
                input_ids[i, max_req_tokens:] = episode.response_tensor

            t.step("reward_evaluation")

            # Get reference logprobs
            ref_logprobs = await ref_model.forward.route(
                input_ids, max_req_tokens, return_logprobs=True
            )
            t.step("reference_model_calculate_logprobs")

            for i, episode in enumerate(episodes):
                episode.ref_logprobs = ref_logprobs[i]
            del ref_logprobs, input_ids

            # Calculate advantages and add to replay buffer
            advantages = await compute_advantages.compute.call_one(episodes)
            for episode, advantage in zip(episodes, advantages):
                episode.advantage = advantage
                await replay_buffer.add.call_one(episode)

            # Log metrics
            rollout_count += 1
            record_metric(
                "main/continuous_rollouts/count_rollout_iterations", 1, Reduce.SUM
            )
            t.stop()

    async def continuous_training():
        """Training loop."""
        training_step = 0
        restart_tracer = True

        while max_steps == -1 or training_step < max_steps:
            if restart_tracer:
                t = Tracer("main_perf/continuous_training")
                t.start()
                restart_tracer = False

            batch = await replay_buffer.sample.call_one(
                curr_policy_version=training_step
            )
            if batch is None:
                await asyncio.sleep(0.1)
            else:
                t.step("waiting_for_buffer")

                inputs, targets = batch
                await trainer.train_step.call(inputs, targets)
                training_step += 1
                t.step("train_step")

                await trainer.push_weights.call(training_step)
                t.step("push_weights")

                await policy.update_weights.fanout(training_step)
                t.step("update_weights")

                if training_step >= 2:
                    await drop_weights(training_step - 1)
                    t.step("drop_weights")

                t.stop()
                restart_tracer = True

                # Flush metrics
                await mlogger.flush.call_one(training_step)

        print(
            f"Reached training limit ({max_steps} steps). Exiting continuous_training loop."
        )

    num_rollout_threads = cfg.get("rollout_threads", 1)
    num_training_threads = cfg.get("training_threads", 1)
    print(
        f"Starting BlackJack GRPO with {num_rollout_threads} rollout threads, {num_training_threads} training threads"
    )

    rollout_tasks = [
        asyncio.create_task(continuous_rollouts()) for _ in range(num_rollout_threads)
    ]
    training_task = asyncio.create_task(continuous_training())

    try:
        await training_task
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        print("Shutting down...")
        shutdown_event.set()

        try:
            await asyncio.wait_for(
                asyncio.gather(*rollout_tasks, return_exceptions=True),
                timeout=5,
            )
        except asyncio.TimeoutError:
            print("Timeout waiting for rollouts; forcing cancellation...")
            for t in rollout_tasks:
                t.cancel()
            await asyncio.gather(*rollout_tasks, return_exceptions=True)

        training_task.cancel()
        await shutdown()


if __name__ == "__main__":

    @parse
    def _main(cfg):
        asyncio.run(main(cfg))

    _main()  # @parse grabs the cfg from CLI
