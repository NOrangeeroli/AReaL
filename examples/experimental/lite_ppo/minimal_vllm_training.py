"""Minimal vLLM-powered AReaL-lite training example.

This script exposes :func:`run_minimal_areal_lite_training`, a compact helper
that wires up the AReaL-lite PPO stack with a vLLM inference backend.  The
function accepts basic configuration primitives (model identifier, dataset
path, RL hyper-parameters, GPU visibility and checkpoint directory) and
performs an RL training loop that periodically saves HuggingFace checkpoints.

The example assumes that a vLLM server compatible with
``areal.thirdparty.vllm.areal_vllm_server`` is already running.  You can either
register its address via ``rl_config["inference_server_addrs"]`` or export the
``AREAL_LLM_SERVER_ADDRS`` environment variable before calling the function.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Mapping, MutableMapping, Sequence

import torch.distributed as dist

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import (
    ClusterSpecConfig,
    DatasetConfig,
    EvaluatorConfig,
    GenerationHyperparameters,
    GRPOConfig,
    InferenceEngineConfig,
    LauncherConfig,
    MicroBatchSpec,
    NameResolveConfig,
    NormConfig,
    OptimizerConfig,
    PPOActorConfig,
    RecoverConfig,
    SaverConfig,
    StatsLoggerConfig,
    vLLMConfig,
)
from areal.api.io_struct import FinetuneSpec, WeightUpdateMeta
from areal.dataset import get_custom_dataset
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.vllm_remote import RemotevLLMEngine
from areal.utils import logging, seeding, stats_tracker
from areal.utils.data import cycle_dataloader
from areal.utils.dataloader import create_dataloader
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.workflow.rlvr import RLVRWorkflow

LOGGER = logging.getLogger("minimal_areal_lite")


def _ensure_single_process_env(visible_gpu_ids: Sequence[str | int]) -> None:
    """Configure minimal distributed environment variables.

    The helper defaults to a single-process launch to make it easy to call the
    training loop from a notebook or script.  For multi-process FSDP runs you
    should prefer the standard launchers (``torchrun``/``areal.launcher``).
    """

    visible = ",".join(str(gid) for gid in visible_gpu_ids)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", visible)
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")


def _gsm8k_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids,
    completion_ids,
    answer: str | Sequence[str] | None = None,
    solutions: Sequence[str] | None = None,
    **_: Any,
) -> float:
    """Reward function that checks GSM8K style answers.

    The dataset loader keeps the gold solution in either ``answer`` or
    ``solutions``.  We reuse the built-in math parser to score completions.
    """

    from areal.reward.math_parser import process_results

    references: Sequence[str] | None
    if solutions is not None:
        references = list(solutions)
    elif answer is None:
        references = None
    elif isinstance(answer, str):
        references = [answer]
    else:
        references = list(answer)

    if not references:
        return 0.0

    parsed = process_results(completions, references[0])
    return float(parsed[0])


def _build_config(
    *,
    model_id: str,
    dataset_path: str,
    rl_config: Mapping[str, Any],
    save_dir: str,
) -> GRPOConfig:
    """Create a ``GRPOConfig`` populated with sensible lightweight defaults."""

    experiment_name = rl_config.get("experiment_name", "minimal-lite-rl")
    trial_name = rl_config.get("trial_name", "trial0")
    seed = int(rl_config.get("seed", 1))
    epochs = int(rl_config.get("epochs", 1))
    allocation_mode = rl_config.get("allocation_mode", "vllm.d1+fsdp.d1")
    batch_size = int(rl_config.get("batch_size", 2))
    n_samples = int(rl_config.get("n_samples", 2))
    max_new_tokens = int(rl_config.get("max_new_tokens", 256))
    temperature = float(rl_config.get("temperature", 1.0))
    lr = float(rl_config.get("learning_rate", 1.0e-5))
    weight_decay = float(rl_config.get("weight_decay", 0.01))
    dtype = rl_config.get("dtype", "bfloat16")
    max_tokens_per_mb = int(rl_config.get("max_tokens_per_mb", 4096))
    saver_freq_epochs = rl_config.get("checkpoint_freq_epochs", 1)
    dataset_type = rl_config.get("dataset_type", "rl")
    max_prompt_length = rl_config.get("max_prompt_length")
    async_training = bool(rl_config.get("async_training", False))
    kl_coef = float(rl_config.get("kl_coef", 0.0))
    reward_scaling = float(rl_config.get("reward_scaling", 1.0))
    reward_bias = float(rl_config.get("reward_bias", 0.0))

    fileroot = os.path.abspath(save_dir)
    name_resolve_root = os.path.join(fileroot, "name_resolve")

    cluster = ClusterSpecConfig(
        fileroot=fileroot,
        n_nodes=1,
        n_gpus_per_node=int(rl_config.get("n_gpus_per_node", 1)),
        name_resolve=NameResolveConfig(nfs_record_root=name_resolve_root),
    )
    stats_logger_cfg = StatsLoggerConfig(
        experiment_name=experiment_name,
        trial_name=trial_name,
        fileroot=fileroot,
    )
    saver_cfg = SaverConfig(
        experiment_name=experiment_name,
        trial_name=trial_name,
        fileroot=fileroot,
        freq_epochs=saver_freq_epochs,
        freq_steps=None,
        freq_secs=None,
    )
    evaluator_cfg = EvaluatorConfig(
        experiment_name=experiment_name,
        trial_name=trial_name,
        fileroot=fileroot,
        freq_epochs=None,
        freq_steps=None,
        freq_secs=None,
    )
    recover_cfg = RecoverConfig(
        experiment_name=experiment_name,
        trial_name=trial_name,
        fileroot=fileroot,
        mode="disabled",
        freq_epochs=None,
        freq_steps=None,
        freq_secs=3600,
    )

    train_dataset = DatasetConfig(
        path=dataset_path,
        type=dataset_type,
        batch_size=batch_size,
        shuffle=bool(rl_config.get("shuffle", True)),
        pin_memory=bool(rl_config.get("pin_memory", True)),
        num_workers=int(rl_config.get("num_workers", 0)),
        drop_last=True,
        max_length=max_prompt_length,
    )

    gconfig = GenerationHyperparameters(
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=float(rl_config.get("top_p", 1.0)),
        top_k=int(rl_config.get("top_k", int(1e8))),
    )

    optimizer = OptimizerConfig(
        type=str(rl_config.get("optimizer", "adam")),
        lr=lr,
        weight_decay=weight_decay,
        beta1=float(rl_config.get("beta1", 0.9)),
        beta2=float(rl_config.get("beta2", 0.999)),
        eps=float(rl_config.get("eps", 1.0e-8)),
        lr_scheduler_type=str(rl_config.get("lr_scheduler_type", "constant")),
        gradient_clipping=float(rl_config.get("gradient_clipping", 1.0)),
        warmup_steps_proportion=float(rl_config.get("warmup_steps_ratio", 0.0)),
    )

    actor_config = PPOActorConfig(
        experiment_name=experiment_name,
        trial_name=trial_name,
        path=model_id,
        dtype=dtype,
        backend="fsdp",
        group_size=n_samples,
        optimizer=optimizer,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=max_tokens_per_mb),
        disable_dropout=bool(rl_config.get("disable_dropout", True)),
        gradient_checkpointing=bool(rl_config.get("gradient_checkpointing", False)),
        recompute_logprob=True,
        use_decoupled_loss=True,
        kl_ctl=kl_coef,
        reward_scaling=reward_scaling,
        reward_bias=reward_bias,
        reward_norm=NormConfig(mean_level="group", std_level="batch", group_size=n_samples),
        adv_norm=NormConfig(mean_level="batch", std_level="batch", group_size=n_samples),
    )

    rollout_cfg = InferenceEngineConfig(
        experiment_name=experiment_name,
        trial_name=trial_name,
        consumer_batch_size=batch_size,
        max_head_offpolicyness=int(rl_config.get("max_head_offpolicyness", 2)),
        pause_grace_period=float(rl_config.get("pause_grace_period", 0.0)),
    )

    vllm_cfg = vLLMConfig(
        model=model_id,
        seed=seed,
        dtype=dtype,
        max_model_len=int(rl_config.get("max_model_len", 32768)),
        gpu_memory_utilization=float(rl_config.get("gpu_memory_utilization", 0.9)),
    )

    base_cfg = GRPOConfig(
        experiment_name=experiment_name,
        trial_name=trial_name,
        seed=seed,
        total_train_epochs=epochs,
        tokenizer_path=rl_config.get("tokenizer_path", model_id),
        async_training=async_training,
        allocation_mode=allocation_mode,
        cluster=cluster,
        train_dataset=train_dataset,
        valid_dataset=None,
        saver=saver_cfg,
        evaluator=replace(evaluator_cfg),
        stats_logger=stats_logger_cfg,
        recover=recover_cfg,
        gconfig=gconfig,
        rollout=rollout_cfg,
        actor=actor_config,
        ref=None,
        vllm=vllm_cfg,
        launcher=LauncherConfig(),
    )

    return base_cfg


def run_minimal_areal_lite_training(
    model_id: str,
    dataset_path: str,
    rl_config: MutableMapping[str, Any] | None = None,
    visible_gpu_ids: Sequence[str | int] | None = None,
    save_dir: str = "/tmp/areal-lite",
) -> None:
    """Train a model with AReaL-lite PPO using a vLLM inference backend.

    Parameters
    ----------
    model_id:
        HuggingFace model identifier or local checkpoint directory.
    dataset_path:
        HuggingFace dataset name or local dataset path.  The example expects a
        GSM8K-style math dataset (e.g. ``"openai/gsm8k"``).
    rl_config:
        Dictionary with optional overrides (batch size, epochs, sampling
        hyper-parameters, etc.).  Provide
        ``{"inference_server_addrs": ["host:port", ...]}`` if you want to skip
        name resolution and connect directly to an existing vLLM server.
    visible_gpu_ids:
        Sequence of GPU identifiers to expose via ``CUDA_VISIBLE_DEVICES``.
        Defaults to the first GPU.
    save_dir:
        Root directory used for checkpoints, logs and name-resolution records.
    """

    rl_config = dict(rl_config or {})
    visible_gpu_ids = tuple(visible_gpu_ids or (0,))

    os.makedirs(save_dir, exist_ok=True)
    _ensure_single_process_env(visible_gpu_ids)

    config = _build_config(
        model_id=model_id, dataset_path=dataset_path, rl_config=rl_config, save_dir=save_dir
    )

    if config.train_dataset.batch_size < 1:
        raise ValueError("Batch size must be >= 1 for minimal training.")

    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    seeding.set_random_seed(config.seed, key="trainer0")

    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    if parallel_strategy is None:
        raise ValueError(
            "Allocation mode must specify a training backend (e.g. 'vllm.d1+fsdp.d1')."
        )

    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    train_dataset = get_custom_dataset(
        split=rl_config.get("dataset_split", "train"),
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    train_loader = create_dataloader(
        train_dataset,
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
        dataset_config=config.train_dataset,
    )

    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise RuntimeError("The training dataloader is empty; check dataset configuration.")

    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=steps_per_epoch * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    rollout_engine = RemotevLLMEngine(config.rollout)
    rollout_addrs = rl_config.get("inference_server_addrs")
    rollout_engine.initialize(
        addr=rollout_addrs,
        train_data_parallel_size=parallel_strategy.dp_size,
    )

    weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(allocation_mode)

    actor.initialize(None, ft_spec)
    actor.connect_engine(rollout_engine, weight_update_meta)

    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)

    dump_dir = os.path.join(save_dir, "generated", config.experiment_name, config.trial_name)
    workflow = RLVRWorkflow(
        reward_fn=_gsm8k_reward_fn,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        enable_thinking=bool(rl_config.get("enable_thinking", False)),
        dump_dir=dump_dir,
    )

    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config, ft_spec)

    max_steps = rl_config.get("max_steps")
    if max_steps is None:
        max_steps = config.total_train_epochs * steps_per_epoch
    else:
        max_steps = int(max_steps)

    data_stream = cycle_dataloader(train_loader)
    LOGGER.info(
        "Starting training: epochs=%s, steps_per_epoch=%s, total_steps=%s",
        config.total_train_epochs,
        steps_per_epoch,
        max_steps,
    )

    global_step = 0
    try:
        while global_step < max_steps:
            epoch = global_step // steps_per_epoch
            epoch_step = global_step % steps_per_epoch
            batch_samples = next(data_stream)

            batch = actor.rollout_batch(
                batch_samples,
                granularity=actor.config.group_size,
                workflow=workflow,
            )

            if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
                batch["prox_logp"] = actor.compute_logp(batch)

            actor.compute_advantages(batch)
            stats = actor.ppo_update(batch)
            actor.step_lr_scheduler()

            rollout_engine.pause()
            actor.update_weights(weight_update_meta)
            actor.set_version(global_step + 1)
            rollout_engine.set_version(global_step + 1)

            saver.save(
                actor,
                epoch=epoch,
                step=epoch_step,
                global_step=global_step,
                tokenizer=tokenizer,
            )

            if dist.is_initialized():
                dist.barrier()

            stats[0].update(
                stats_tracker.export_all(reduce_group=actor.data_parallel_group)
            )
            stats_logger.commit(epoch, epoch_step, global_step, stats)

            global_step += 1
    finally:
        LOGGER.info("Training loop finished at global_step=%s", global_step)
        try:
            stats_logger.close()
        finally:
            try:
                rollout_engine.destroy()
            finally:
                actor.destroy()


if __name__ == "__main__":
    run_minimal_areal_lite_training(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        dataset_path="openai/gsm8k",
        rl_config={
            "epochs": 1,
            "batch_size": 2,
            "n_samples": 2,
            "allocation_mode": "vllm.d1+fsdp.d1",
        },
        visible_gpu_ids=(0,),
        save_dir="/tmp/areal-lite-example",
    )
