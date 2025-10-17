# Copyright (c) Meta Platforms, Inc. and affiliates.
# Usage: python -m apps.grpo.blackjack_main_fixed --config apps/grpo/blackjack.yaml

"""
BlackJack GRPO Training - FIXED VERSION

Changes:
- Moved policy calls out of actor to avoid RPC timeouts
- Fixed reward assignment: ALL steps in game get final outcome
- Added detailed game logging to blackjack_logs/blackjack_games_<timestamp>.log

Logs include:
- Every prompt sent to model
- Every model response
- Actions taken
- Game outcomes (win/loss/push)
- Rollout summaries
"""

import asyncio
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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

from envs.openspiel_env import OpenSpielEnv, OpenSpielAction


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
Policy = Generator


def collate(batches: list[Group]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
    @endpoint
    async def evaluate_response(self, prompt: str, response: str, game_reward: float) -> float:
        reward = float(game_reward)
        record_metric("reward/evaluate_response/avg_reward", reward, Reduce.MEAN)
        record_metric("reward/evaluate_response/sum_reward", reward, Reduce.SUM)
        return reward


@dataclass
class ComputeAdvantages(ForgeActor):
    @endpoint
    async def compute(self, group: Group) -> list[float]:
        rewards = torch.tensor([[e.reward for e in group]])
        mean = rewards.mean(1, keepdim=True)
        std = rewards.std(1, keepdim=True)
        advantages = (rewards - mean) / (std + 1e-4)
        return advantages.squeeze(0).tolist()


def parse_action(response_text: str, legal_actions: list[int]) -> int:
    text_lower = response_text.lower().strip()
    if "hit" in text_lower:
        action_id = 0
    elif "stand" in text_lower:
        action_id = 1
    else:
        action_id = 1  # Default STAND

    if action_id not in legal_actions:
        action_id = legal_actions[0]
    return action_id


def format_prompt(step_num: int, action_history: list, tokenizer) -> str:
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
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


@dataclass
class BlackJackEnvActor(ForgeActor):
    """Simple actor that just manages OpenEnv connections."""
    server_url: str = "http://localhost:8000"
    model: str = "Qwen/Qwen3-1.7B"

    @endpoint
    def setup(self):
        self._tokenizer = get_tokenizer(self.model)
        print(f"BlackJackEnvActor initialized (server: {self.server_url})")

    @endpoint
    async def get_tokenizer(self):
        return self._tokenizer

    @endpoint
    async def pad_token(self):
        return self._tokenizer.pad_token_id


def setup_game_logger(log_dir: str = "blackjack_logs"):
    """Setup detailed game logging to file."""
    Path(log_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"blackjack_games_{timestamp}.log"

    def log(message: str):
        with open(log_file, "a") as f:
            f.write(f"{message}\n")
        print(message)  # Also print to console

    log("=" * 80)
    log(f"BlackJack GRPO Training - Game Log Started at {datetime.now()}")
    log("=" * 80)
    log("")

    return log


async def drop_weights(version: int):
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
    server_url = cfg.blackjack_env.server_url

    # Global setups
    provisioner = None
    if cfg.get("provisioner", None) is not None:
        provisioner = await init_provisioner(
            ProvisionerConfig(launcher_config=LauncherConfig(**cfg.provisioner))
        )
    else:
        provisioner = await init_provisioner()

    metric_logging_cfg = cfg.get("metric_logging", {"console": {"log_per_rank": False}})
    mlogger = await get_or_create_metric_logger()
    await mlogger.init_backends.call_one(metric_logging_cfg)

    # Setup services
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
        BlackJackEnvActor.options(**cfg.actors.blackjack_env).as_actor(**cfg.blackjack_env),
        Policy.options(**cfg.services.policy).as_service(**cfg.policy),
        RLTrainer.options(**cfg.actors.trainer).as_actor(**cfg.trainer, loss=simple_grpo_loss),
        ReplayBuffer.options(**cfg.actors.replay_buffer).as_actor(**cfg.replay_buffer, collate=collate),
        ComputeAdvantages.options(**cfg.actors.compute_advantages).as_actor(),
        ReferenceModel.options(**cfg.services.ref_model).as_service(**cfg.ref_model),
        BlackJackReward.options(**cfg.services.reward_actor).as_service(),
    )

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
    print("Torchstore successfully initialized")

    # Get tokenizer
    tokenizer = await blackjack_env.get_tokenizer.call_one()
    pad_id = await blackjack_env.pad_token.call_one()

    # Setup game logger
    game_log = setup_game_logger()

    # Core RL loops
    async def continuous_rollouts():
        """Collect BlackJack episodes - POLICY CALLS IN MAIN LOOP"""
        rollout_count = 0

        while not shutdown_event.is_set():
            t = Tracer("main_perf/continuous_rollouts")
            t.start()

            # Play games - NO POLICY CALLS in actor, done here instead
            all_step_results = []

            for game_idx in range(group_size):
                game_id = str(uuid.uuid4())[:8]  # Short ID for readability
                env = OpenSpielEnv(base_url=server_url)

                game_log("")
                game_log("=" * 80)
                game_log(f"🎰 GAME {game_idx + 1}/{group_size} (Rollout #{rollout_count + 1}) - ID: {game_id}")
                game_log("=" * 80)

                try:
                    result = env.reset()
                    obs = result.observation
                    done = False
                    step_num = 0
                    action_history = []
                    game_steps = []  # ✅ Collect steps WITHOUT reward first

                    while not done and step_num < 10:
                        # Format prompt
                        prompt = format_prompt(step_num, action_history, tokenizer)

                        game_log(f"\n--- Step {step_num + 1} ---")
                        game_log(f"Legal actions: {obs.legal_actions}")
                        game_log(f"\nPrompt sent to model:")
                        game_log("-" * 40)
                        game_log(prompt)
                        game_log("-" * 40)

                        # Call policy HERE (not in actor) ✅
                        responses: list[Completion] = await policy.generate.route(prompt)
                        response = responses[0]

                        game_log(f"\n🤖 Model response: '{response.text}'")

                        # Parse and take action
                        action_id = parse_action(response.text, obs.legal_actions)
                        action_name = "HIT" if action_id == 0 else "STAND"
                        action_history.append((action_id, action_name))

                        game_log(f"➡️  Parsed action: {action_name} (action_id={action_id})")

                        # Store step data WITHOUT reward yet
                        game_steps.append({
                            "step_num": step_num,
                            "prompt": prompt,
                            "response": response,
                        })

                        result = env.step(OpenSpielAction(action_id=action_id, game_name="blackjack"))
                        obs = result.observation
                        done = result.done

                        if done:
                            game_log(f"🏁 Game ended!")

                        step_num += 1

                    # ✅ Game finished - get final outcome
                    final_game_reward = result.reward  # +1 (win), -1 (loss), or 0 (push)

                    outcome_emoji = "🏆" if final_game_reward > 0 else ("💀" if final_game_reward < 0 else "🤝")
                    outcome_text = "WIN" if final_game_reward > 0 else ("LOSS" if final_game_reward < 0 else "PUSH")

                    game_log("")
                    game_log(f"{outcome_emoji} FINAL OUTCOME: {outcome_text} (reward={final_game_reward})")
                    game_log(f"📊 Game length: {len(game_steps)} steps")
                    game_log(f"🎲 Action sequence: {' → '.join([name for _, name in action_history])}")

                    # ✅ Assign final reward to ALL steps in this game
                    for step_data in game_steps:
                        all_step_results.append({
                            "game_id": game_id,
                            "final_reward": final_game_reward,  # Same reward for all steps
                            **step_data,
                        })

                    # Log game metrics
                    record_metric("blackjack/count_games_played", 1, Reduce.SUM)
                    record_metric("blackjack/avg_game_length", len(game_steps), Reduce.MEAN)
                    record_metric("blackjack/game_outcome", final_game_reward, Reduce.MEAN)

                finally:
                    env.close()

            t.step("play_games")

            # Process episodes
            episodes = []
            input_ids = torch.ones((len(all_step_results), max_req_tokens + max_res_tokens), dtype=torch.long)

            for i, step_result in enumerate(all_step_results):
                episode = Episode(
                    episode_id=str(uuid.uuid4()),
                    pad_id=pad_id,
                    request_len=max_req_tokens,
                    response_len=max_res_tokens,
                    game_id=step_result["game_id"],
                    step_in_game=step_result["step_num"],
                    completion=step_result["response"],
                )

                episode.reward = await reward_actor.evaluate_response.route(
                    prompt=step_result["prompt"],
                    response=step_result["response"].text,
                    game_reward=step_result["final_reward"],
                )

                episodes.append(episode)
                input_ids[i, :max_req_tokens] = episode.request_tensor
                input_ids[i, max_req_tokens:] = episode.response_tensor  # Fixed: should be max_req_tokens not max_res_tokens

            t.step("reward_evaluation")

            # Get reference logprobs
            ref_logprobs = await ref_model.forward.route(input_ids, max_req_tokens, return_logprobs=True)
            t.step("reference_model")

            for i, episode in enumerate(episodes):
                episode.ref_logprobs = ref_logprobs[i]
            del ref_logprobs, input_ids

            # Calculate advantages
            advantages = await compute_advantages.compute.call_one(episodes)
            for episode, advantage in zip(episodes, advantages):
                episode.advantage = advantage
                await replay_buffer.add.call_one(episode)

            rollout_count += 1
            record_metric("main/rollout_iterations", 1, Reduce.SUM)
            t.stop()

            # Log rollout summary
            wins = sum(1 for e in episodes if e.reward > 0)
            losses = sum(1 for e in episodes if e.reward < 0)
            pushes = sum(1 for e in episodes if e.reward == 0)
            avg_reward = sum(e.reward for e in episodes) / len(episodes) if episodes else 0

            game_log("")
            game_log("=" * 80)
            game_log(f"📈 ROLLOUT #{rollout_count} SUMMARY")
            game_log("=" * 80)
            game_log(f"Total episodes collected: {len(episodes)}")
            game_log(f"Win/Loss/Push: {wins}/{losses}/{pushes}")
            game_log(f"Win rate: {wins / len(episodes) * 100:.1f}%")
            game_log(f"Average reward: {avg_reward:.3f}")
            game_log("=" * 80)
            game_log("")

            print(f"Rollout {rollout_count} complete - collected {len(episodes)} episodes (W/L/P: {wins}/{losses}/{pushes})")

    async def continuous_training():
        """Training loop."""
        training_step = 0
        restart_tracer = True

        while max_steps == -1 or training_step < max_steps:
            if restart_tracer:
                t = Tracer("main_perf/continuous_training")
                t.start()
                restart_tracer = False

            batch = await replay_buffer.sample.call_one(curr_policy_version=training_step)
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
                await mlogger.flush.call_one(training_step)

                print(f"Training step {training_step} complete")

        print(f"Reached training limit ({max_steps} steps)")

    num_rollout_threads = cfg.get("rollout_threads", 1)
    print(f"Starting BlackJack GRPO with {num_rollout_threads} rollout threads")

    rollout_tasks = [asyncio.create_task(continuous_rollouts()) for _ in range(num_rollout_threads)]
    training_task = asyncio.create_task(continuous_training())

    try:
        await training_task
    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        print("Shutting down...")
        shutdown_event.set()

        try:
            await asyncio.wait_for(asyncio.gather(*rollout_tasks, return_exceptions=True), timeout=5)
        except asyncio.TimeoutError:
            for t in rollout_tasks:
                t.cancel()
            await asyncio.gather(*rollout_tasks, return_exceptions=True)

        training_task.cancel()
        await shutdown()


if __name__ == "__main__":
    @parse
    def _main(cfg):
        asyncio.run(main(cfg))

    _main()
