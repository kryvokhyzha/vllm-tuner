from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


# ──────────────────────────────────────
# Enums
# ──────────────────────────────────────


class TrialStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PRUNED = "pruned"


class Direction(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class AcceleratorType(str, Enum):
    GPU = "gpu"
    TPU = "tpu"
    CPU = "cpu"


class BackendType(str, Enum):
    LOCAL = "local"
    RAY = "ray"


class BenchmarkProviderType(str, Enum):
    GUIDELLM = "guidellm"
    VLLM_BENCHMARK = "vllm_benchmark"
    HTTP = "http"


class SamplerType(str, Enum):
    TPE = "tpe"
    NSGA2 = "nsga2"
    RANDOM = "random"
    GRID = "grid"
    BOTORCH = "botorch"


# ──────────────────────────────────────
# Hardware models
# ──────────────────────────────────────


class HardwareStats(BaseModel):
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    memory_utilization: float = 0.0
    accelerator_utilization: float = 0.0
    temperature_c: float = 0.0
    power_usage_w: float = 0.0
    power_limit_w: float = 0.0
    clock_sm_mhz: float = 0.0
    clock_memory_mhz: float = 0.0


class AggregateHardwareStats(BaseModel):
    peak_memory_used_mb: float = 0.0
    avg_memory_used_mb: float = 0.0
    avg_utilization: float = 0.0
    max_temperature_c: float = 0.0
    total_power_w: float = 0.0
    num_samples: int = 0


class AcceleratorInfo(BaseModel):
    accelerator_type: AcceleratorType = AcceleratorType.GPU
    name: str = "unknown"
    count: int = 1
    memory_per_device_mb: float = 0.0
    total_memory_mb: float = 0.0


# ──────────────────────────────────────
# Benchmark models
# ──────────────────────────────────────


class BenchmarkConfig(BaseModel):
    provider: BenchmarkProviderType = BenchmarkProviderType.GUIDELLM
    prompt_tokens: int = 1500
    output_tokens: int = 200
    prompt_tokens_stdev: int = 128
    dataset: str | None = None
    max_seconds: int = 300
    concurrent_requests: int = 50
    warmup_requests: int = 10


class BenchmarkResult(BaseModel):
    throughput_req_per_sec: float = 0.0
    output_tokens_per_sec: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    ttft_ms: float = 0.0
    itl_ms: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    duration_seconds: float = 0.0
    avg_output_tokens_per_request: float = 0.0
    min_output_tokens_per_request: int = 0
    max_output_tokens_per_request: int = 0
    expected_output_tokens: int = 0
    expected_prompt_tokens: int = 0


# ──────────────────────────────────────
# Parameter & optimization models
# ──────────────────────────────────────


class ParameterSpec(BaseModel):
    name: str
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list[str] | None = None
    log_scale: bool = False


class ObjectiveSpec(BaseModel):
    metric: str
    direction: Direction = Direction.MAXIMIZE
    percentile: str | None = None
    weight: float = 1.0


class ConstraintSpec(BaseModel):
    expression: str


# ──────────────────────────────────────
# Config models
# ──────────────────────────────────────


class StudySettings(BaseModel):
    name: str = "vllm-tuning-study"
    storage: str = "sqlite:///study.db"


class OptimizationConfig(BaseModel):
    preset: str | None = None
    objectives: list[ObjectiveSpec] = Field(
        default_factory=lambda: [
            ObjectiveSpec(metric="output_tokens_per_sec", direction=Direction.MAXIMIZE),
        ]
    )
    sampler: SamplerType = SamplerType.TPE
    n_trials: int = 50
    max_concurrent_trials: int = 1
    n_startup_trials: int = 10


class HardwareConfig(BaseModel):
    accelerator: AcceleratorType = AcceleratorType.GPU
    device_ids: list[int] = Field(default_factory=lambda: [0])
    tpu_type: str | None = None


class ExecutionConfig(BaseModel):
    backend: BackendType = BackendType.LOCAL
    ray_address: str = "auto"
    runtime_env: dict | None = None


class BaselineConfig(BaseModel):
    enabled: bool = True
    num_requests: int = 100


class CostConfig(BaseModel):
    target_throughput: float | None = None
    cloud_provider: str = "gcp"
    instance_type: str = ""
    pricing_mode: str = "spot"
    price_per_hour: float | None = None


class OutputConfig(BaseModel):
    directory: Path = Path("./results")
    reports: list[str] = Field(default_factory=lambda: ["html", "json", "yaml"])
    export_helm_values: bool = True


class StudyConfig(BaseModel):
    """Top-level configuration combining all sub-configs."""

    study: StudySettings = Field(default_factory=StudySettings)
    model: str = ""
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    parameters: list[ParameterSpec] = Field(default_factory=list)
    static_parameters: dict[str, str | int | float | bool] = Field(default_factory=dict)
    constraints: list[ConstraintSpec] = Field(default_factory=list)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


# ──────────────────────────────────────
# Trial models
# ──────────────────────────────────────


class TrialConfig(BaseModel):
    trial_number: int
    parameters: dict[str, str | int | float | bool] = Field(default_factory=dict)
    static_parameters: dict[str, str | int | float | bool] = Field(default_factory=dict)


class VLLMTelemetry(BaseModel):
    kv_cache_usage: float | None = None
    num_preemptions: int = 0
    num_swaps: int = 0
    oom_detected: bool = False


class TrialResult(BaseModel):
    trial_number: int
    status: TrialStatus = TrialStatus.PENDING
    config: TrialConfig | None = None
    benchmark: BenchmarkResult | None = None
    hardware: AggregateHardwareStats | None = None
    telemetry: VLLMTelemetry | None = None
    duration_seconds: float = 0.0
    error_message: str | None = None


# ──────────────────────────────────────
# Cost models
# ──────────────────────────────────────


class CostReport(BaseModel):
    instances_needed: int = 0
    hourly_cost: float = 0.0
    monthly_cost: float = 0.0
    perf_per_dollar: float = 0.0
    cloud_provider: str = ""
    instance_type: str = ""
    pricing_mode: str = ""
