"""
GRPO Utilities for BlackJack Training

This module contains reusable components extracted from the production GRPO implementation.
Used by both the tutorial notebook and the full training script.
"""

import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchstore as ts

from envs.openspiel_env import OpenSpielAction, OpenSpielEnv
from forge.actors._torchstore_utils import (
    get_dcp_whole_state_dict_key,
    get_param_prefix,
)
from forge.controller.actor import ForgeActor
from forge.data_models.completion import Completion
from forge.observability.metrics import Reduce, record_metric
from forge.util.ops import compute_logprobs
from monarch.actor import endpoint
from vllm.transformers_utils.tokenizer import get_tokenizer


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class Episode:
    """Episode data for BlackJack game."""

    episode_id: str
    pad_id: int
    request_len: int
    response_len: int
    game_id: str
    step_in_game: int
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
        if tensor.shape[0] < self.request_len:
            diff = self.request_len - tensor.shape[0]
            tensor = F.pad(tensor, (diff, 0), value=self.pad_id)
        return tensor

    @property
    def response_tensor(self) -> torch.Tensor:
        response_tokens: torch.Tensor = self.completion.token_ids
        tensor = torch.tensor(response_tokens, dtype=torch.long)
        if tensor.shape[0] < self.response_len:
            diff = self.response_len - tensor.shape[0]
            tensor = F.pad(tensor, (0, diff), value=self.pad_id)
        return tensor


Group = list[Episode]


# ============================================================================
# GRPO Loss and Collation
# ============================================================================


def collate(batches: list[Group]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Collate batches of episodes into model inputs and targets.

    Args:
        batches: List of episode groups

    Returns:
        Tuple of (inputs, targets) for training
    """
    inputs = []
    targets = []
    for batch in batches:
        request = torch.stack([e.request_tensor for e in batch])
        response = torch.stack([e.response_tensor for e in batch])
        ref_logprobs = torch.stack([e.ref_logprobs for e in batch]).squeeze()
        advantages = torch.tensor([e.advantage for e in batch]).unsqueeze(-1)
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
    """
    GRPO loss with KL penalty.

    Args:
        logits: Model logits
        response: Response tokens
        ref_logprobs: Reference model log probabilities
        advantages: Normalized advantages (group-relative)
        padding_mask: Mask for padded tokens
        beta: KL penalty coefficient

    Returns:
        Scalar loss value
    """
    logprobs: torch.Tensor = compute_logprobs(logits, response)

    # KL divergence: KL(ref || policy) in closed form
    kl = torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1

    # Policy gradient term with importance weight
    per_token_policy_loss = torch.exp(logprobs - logprobs.detach()) * advantages

    # Combined loss: maximize policy improvement, minimize KL
    per_token_loss = -(per_token_policy_loss - beta * kl)

    # Average over valid tokens
    loss = (
        ((per_token_loss * padding_mask).sum(dim=1))
        / (padding_mask.sum(dim=1).clamp(min=1.0))
    ).mean()

    return loss


# ============================================================================
# Prompt Formatting and Action Parsing
# ============================================================================


def format_prompt(step_num: int, action_history: list, tokenizer) -> str:
    """
    Format BlackJack game state as text prompt for LLM.

    Args:
        step_num: Current step number in game
        action_history: List of (action_id, action_name) tuples
        tokenizer: HuggingFace tokenizer with chat template

    Returns:
        Formatted prompt string
    """
    system = """You are an expert BlackJack player. Output only 'HIT' or 'STAND'."""

    state_desc = f"=== BlackJack Game (Step {step_num + 1}) ===\n\n"
    if action_history:
        state_desc += "Previous actions:\n"
        for i, (_, name) in enumerate(action_history):
            state_desc += f"  {i + 1}. {name}\n"
        state_desc += "\n"

    state_desc += "What do you do? (Output only 'HIT' or 'STAND')"

    chat = [
        {"role": "system", "content": system},
        {"role": "user", "content": state_desc},
    ]

    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )


def parse_action(response_text: str, legal_actions: list[int]) -> int:
    """
    Parse action from model's text response.

    Args:
        response_text: Model's generated text
        legal_actions: List of legal action IDs

    Returns:
        Action ID (0=HIT, 1=STAND)
    """
    text_lower = response_text.lower().strip()

    if "hit" in text_lower:
        action_id = 0
    elif "stand" in text_lower:
        action_id = 1
    else:
        action_id = 1  # Default: STAND

    # Ensure action is legal
    if action_id not in legal_actions:
        action_id = legal_actions[0]

    return action_id


# ============================================================================
# Forge Actors
# ============================================================================


@dataclass
class BlackJackReward(ForgeActor):
    """Reward actor for evaluating BlackJack game outcomes."""

    @endpoint
    async def evaluate_response(
        self, prompt: str, response: str, game_reward: float
    ) -> float:
        """
        Evaluate episode reward with optional shaping.

        Args:
            prompt: Game state prompt
            response: Model's action
            game_reward: Raw game outcome (+1/-1/0)

        Returns:
            Shaped reward value
        """
        # Base reward from game outcome
        reward = float(game_reward)

        # Optional reward shaping: Scale up wins
        if game_reward > 0:
            reward = 2.0  # Make wins more valuable
        elif game_reward == 0:
            reward = 0.5  # Pushes better than losses

        record_metric("reward/evaluate_response/avg_reward", reward, Reduce.MEAN)
        record_metric("reward/evaluate_response/sum_reward", reward, Reduce.SUM)

        return reward


@dataclass
class ComputeAdvantages(ForgeActor):
    """Actor for computing group-relative advantages."""

    @endpoint
    async def compute(self, group: Group) -> list[float]:
        """
        Compute advantages normalized by group statistics.

        Args:
            group: List of episodes from same rollout

        Returns:
            List of advantage values
        """
        rewards = torch.tensor([[e.reward for e in group]])
        mean = rewards.mean(1, keepdim=True)
        std = rewards.std(1, keepdim=True)
        advantages = (rewards - mean) / (std + 1e-4)
        return advantages.squeeze(0).tolist()


@dataclass
class BlackJackEnvActor(ForgeActor):
    """Actor that manages OpenEnv connections and tokenizer."""

    server_url: str = "http://localhost:8004"
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    @endpoint
    def setup(self):
        """Initialize tokenizer."""
        self._tokenizer = get_tokenizer(self.model)
        print(f"BlackJackEnvActor initialized (server: {self.server_url})")

    @endpoint
    async def get_tokenizer(self):
        """Get tokenizer instance."""
        return self._tokenizer

    @endpoint
    async def pad_token(self):
        """Get padding token ID."""
        return self._tokenizer.pad_token_id


# ============================================================================
# Logging and Utilities
# ============================================================================


def setup_game_logger(log_dir: str = "blackjack_logs"):
    """
    Setup detailed game logging to file.

    Args:
        log_dir: Directory for log files

    Returns:
        Logging function
    """
    Path(log_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"blackjack_games_{timestamp}.log"

    def log(message: str):
        """Write message to log file and console."""
        with open(log_file, "a") as f:
            f.write(f"{message}\n")
        print(message)

    log("=" * 80)
    log(f"BlackJack GRPO Training - Game Log Started at {datetime.now()}")
    log("=" * 80)
    log("")

    return log


async def drop_weights(version: int):
    """
    Drop old model weights from torchstore.

    Args:
        version: Weight version to drop
    """
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


# ============================================================================
# Game Playing Logic
# ============================================================================


async def play_blackjack_game(
    game_idx: int,
    game_id: str,
    server_url: str,
    policy,
    tokenizer,
    game_log,
    rollout_count: int = 0
):
    """
    Play a single BlackJack game and collect episode data.

    Args:
        game_idx: Index of this game in the rollout
        game_id: Unique game identifier
        server_url: OpenEnv server URL
        policy: Policy (Generator) for action selection
        tokenizer: Tokenizer for prompt formatting
        game_log: Logging function
        rollout_count: Current rollout iteration

    Returns:
        List of step results with prompts, responses, and final reward
    """
    env = OpenSpielEnv(base_url=server_url)

    game_log("")
    game_log("=" * 80)
    game_log(f"🎰 GAME {game_idx + 1} (Rollout #{rollout_count + 1}) - ID: {game_id}")
    game_log("=" * 80)

    try:
        result = env.reset()
        obs = result.observation
        done = False
        step_num = 0
        action_history = []
        game_steps = []

        while not done and step_num < 10:  # Max 10 steps per game
            # Format prompt
            prompt = format_prompt(step_num, action_history, tokenizer)

            game_log(f"\n--- Step {step_num + 1} ---")
            game_log(f"Legal actions: {obs.legal_actions}")
            game_log(f"\nPrompt sent to model:")
            game_log("-" * 40)
            game_log(prompt)
            game_log("-" * 40)

            # Generate action with policy
            responses = await policy.generate.route(prompt)
            response = responses[0]

            game_log(f"\n🤖 Model response: '{response.text}'")

            # Parse and execute action
            action_id = parse_action(response.text, obs.legal_actions)
            action_name = "HIT" if action_id == 0 else "STAND"
            action_history.append((action_id, action_name))

            game_log(f"➡️  Parsed action: {action_name} (action_id={action_id})")

            # Store step data (reward assigned later)
            game_steps.append({
                "step_num": step_num,
                "prompt": prompt,
                "response": response,
            })

            # Take action in environment
            result = env.step(OpenSpielAction(action_id=action_id, game_name="blackjack"))
            obs = result.observation
            done = result.done

            if done:
                game_log(f"🏁 Game ended!")

            step_num += 1

        # Get final game outcome
        final_game_reward = result.reward  # +1 (win), -1 (loss), or 0 (push)

        outcome_emoji = "🏆" if final_game_reward > 0 else ("💀" if final_game_reward < 0 else "🤝")
        outcome_text = "WIN" if final_game_reward > 0 else ("LOSS" if final_game_reward < 0 else "PUSH")

        game_log("")
        game_log(f"{outcome_emoji} FINAL OUTCOME: {outcome_text} (reward={final_game_reward})")
        game_log(f"📊 Game length: {len(game_steps)} steps")
        game_log(f"🎲 Action sequence: {' → '.join([name for _, name in action_history])}")

        # Assign final reward to all steps
        all_step_results = []
        for step_data in game_steps:
            all_step_results.append({
                "game_id": game_id,
                "final_reward": final_game_reward,
                **step_data,
            })

        # Record metrics
        record_metric("blackjack/count_games_played", 1, Reduce.SUM)
        record_metric("blackjack/avg_game_length", len(game_steps), Reduce.MEAN)
        record_metric("blackjack/game_outcome", final_game_reward, Reduce.MEAN)

        return all_step_results

    finally:
        env.close()
