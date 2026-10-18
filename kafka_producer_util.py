"""
Utility for producing light-curve data points to Kafka.
Used by streaming generators to emit points as they're computed.
"""

from __future__ import annotations

import json
import logging
from typing import Callable
from kafka import KafkaProducer
from kafka.errors import KafkaError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightCurveProducer:
    """Produces individual light-curve points to Kafka."""

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "lightcurve-points",
        batch_size: int = 1000,
    ):
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.batch_size = batch_size
        
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

    def emit_point(
        self,
        source_id: str,
        time: float,
        magnitude: float,
        band: str,
        sequence: int,
        total_points: int,
        callback: Callable | None = None,
    ):
        """
        Emit a single light-curve point to Kafka.
        
        Args:
            source_id: Unique identifier for this light curve (e.g., "sample_0001")
            time: Time value for this point
            magnitude: Magnitude value for this point
            band: Band name (e.g., "g_ps1_mag", "r_ps1_mag")
            sequence: 0-based index of this point in the full sequence
            total_points: Total number of points for this source
            callback: Optional callback(future) when message is sent
        """
        message = {
            "source_id": source_id,
            "time": float(time),
            "magnitude": float(magnitude),
            "band": band,
            "sequence": sequence,
            "total_points": total_points,
        }
        
        def default_callback(record_metadata):
            logger.debug(f"Message sent to {record_metadata.topic} "
                       f"partition {record_metadata.partition} "
                       f"offset {record_metadata.offset}")
        
        self.producer.send(
            self.topic,
            value=message,
            key=source_id.encode("utf-8"),
        ).add_callback(callback or default_callback)

    def flush(self, timeout: int = 30):
        """Wait for all pending messages to be sent."""
        self.producer.flush(timeout=timeout)

    def close(self):
        """Close the producer."""
        self.producer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test producer: emit sample light-curve points"
    )
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic", default="lightcurve-points")
    parser.add_argument("--n-samples", type=int, default=3,
                        help="Number of samples to generate")
    parser.add_argument("--n-points", type=int, default=10,
                        help="Points per sample")
    args = parser.parse_args()
    
    with LightCurveProducer(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
    ) as producer:
        for sample_idx in range(args.n_samples):
            source_id = f"sample_{sample_idx+1:04d}"
            for pt_idx in range(args.n_points):
                producer.emit_point(
                    source_id=source_id,
                    time=0.1 + pt_idx * 0.1,
                    magnitude=15.0 + pt_idx * 0.01,
                    band="g_ps1_mag",
                    sequence=pt_idx,
                    total_points=args.n_points,
                )
        producer.flush()
        print(f"Produced {args.n_samples * args.n_points} messages")
