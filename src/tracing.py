"""OpenTelemetry tracing setup with Langfuse OTLP export.

Provides distributed tracing across the DriftLog pipeline:
  ingestion → chunking → embedding
  query → dense search → sparse search → fusion → reranking → generation

Exports spans to Langfuse via their OpenTelemetry-compatible OTLP endpoint.
Each LLM call records span attributes for model name, token counts, and latency.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Generator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

logger = logging.getLogger(__name__)

_initialized = False


def init_tracing() -> None:
    """Initialize OpenTelemetry with Langfuse OTLP exporter.

    Required env vars:
      LANGFUSE_PUBLIC_KEY  — Langfuse project public key
      LANGFUSE_SECRET_KEY  — Langfuse project secret key
      LANGFUSE_HOST        — Langfuse host (default: https://cloud.langfuse.com)
    """
    global _initialized
    if _initialized:
        return

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        logger.warning(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set — "
            "tracing will use a no-op provider (spans are still created but not exported)"
        )
        _initialized = True
        return

    # Langfuse OTLP endpoint expects Basic auth: base64(public_key:secret_key)
    auth_token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()

    exporter = OTLPSpanExporter(
        endpoint=f"{host.rstrip('/')}/api/public/otel/v1/traces",
        headers={
            "Authorization": f"Basic {auth_token}",
        },
    )

    resource = Resource.create(
        {
            "service.name": "driftlog",
            "service.version": "0.1.0",
            "deployment.environment": os.environ.get("APP_ENV", "development"),
        }
    )

    provider = TracerProvider(resource=resource)

    # Use BatchSpanProcessor in production, SimpleSpanProcessor in dev for immediate visibility
    env = os.environ.get("APP_ENV", "development")
    if env == "production":
        provider.add_span_processor(BatchSpanProcessor(exporter))
    else:
        provider.add_span_processor(SimpleSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    _initialized = True
    logger.info("OpenTelemetry tracing initialized — exporting to Langfuse at %s", host)


def get_tracer(name: str = "driftlog") -> trace.Tracer:
    """Return a named tracer from the global provider."""
    return trace.get_tracer(name)


@contextmanager
def timed_span(
    tracer: trace.Tracer,
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[trace.Span, None, None]:
    """Context manager that creates a span and automatically records latency_ms on exit."""
    with tracer.start_as_current_span(name, attributes=attributes or {}) as span:
        start = time.perf_counter()
        try:
            yield span
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            span.set_attribute("latency_ms", round(latency_ms, 2))


def set_llm_attributes(
    span: trace.Span,
    *,
    model: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> None:
    """Set standard LLM span attributes following OpenTelemetry GenAI semantic conventions."""
    span.set_attribute("gen_ai.system", "anthropic" if "claude" in model.lower() else "openai")
    span.set_attribute("gen_ai.request.model", model)
    if input_tokens is not None:
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
    if output_tokens is not None:
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
    if total_tokens is not None:
        span.set_attribute("gen_ai.usage.total_tokens", total_tokens)
    if max_tokens is not None:
        span.set_attribute("gen_ai.request.max_tokens", max_tokens)
    if temperature is not None:
        span.set_attribute("gen_ai.request.temperature", temperature)
