"""
Kafka Streams-based binning consumer.
Aggregates light-curve data points in real-time using windowed aggregation.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from typing import Dict, List, Tuple
from collections import defaultdict
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamingBinner:
    """
    Aggregates light-curve points into time bins using Kafka consumer.
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        input_topic: str = "lightcurve-points",
        output_topic: str = "binned-lightcurve",
        bin_size: int = 5,
        auto_offset_reset: str = "earliest",
        group_id: str = "binner-group",
        output_dir: str = "afterglow_photometry_binned",
    ):
        self.bootstrap_servers = bootstrap_servers
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.bin_size = bin_size
        self.group_id = group_id
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        legacy_dir = "afterglow_photometry_binned"
        if legacy_dir != self.output_dir:
            os.makedirs(legacy_dir, exist_ok=True)
        
        # State: {source_id: {bin_index: {"times": [...], "mags": [...]}}}
        self.bins: Dict[str, Dict[int, Dict]] = defaultdict(lambda: defaultdict(lambda: {"times": [], "mags": []}))
        
        self.consumer = KafkaConsumer(
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset=auto_offset_reset,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            enable_auto_commit=True,
        )
        self.consumer.subscribe([input_topic])
        
        # Producer for output binned results
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

    def save_binned_result(self, source_id: str, binned: Dict) -> None:
        """Persist one completed bin as a CSV row inside the configured output directory."""
        os.makedirs(self.output_dir, exist_ok=True)

        csv_name = f"{source_id}_bin_{binned['bin_index']}.csv"
        csv_path = os.path.join(self.output_dir, csv_name)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "source_id",
                "bin_index",
                "time_avg",
                "magnitude_avg",
                "band",
                "bin_size",
                "is_final",
            ])
            writer.writerow([
                source_id,
                binned["bin_index"],
                binned["time_avg"],
                binned["magnitude_avg"],
                binned["band"],
                binned["bin_size"],
                binned["is_final"],
            ])

        legacy_dir = "afterglow_photometry_binned"
        if legacy_dir != self.output_dir:
            os.makedirs(legacy_dir, exist_ok=True)
            legacy_path = os.path.join(legacy_dir, csv_name)
            with open(legacy_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "source_id",
                    "bin_index",
                    "time_avg",
                    "magnitude_avg",
                    "band",
                    "bin_size",
                    "is_final",
                ])
                writer.writerow([
                    source_id,
                    binned["bin_index"],
                    binned["time_avg"],
                    binned["magnitude_avg"],
                    binned["band"],
                    binned["bin_size"],
                    binned["is_final"],
                ])

    def process_message(self, msg: Dict) -> Tuple[str, List[Dict]] | None:
        """
        Process a single light-curve point message.
        
        Message format:
        {
            "source_id": "sample_0001",
            "time": 0.1234,
            "magnitude": 15.5,
            "band": "g_ps1_mag",
            "sequence": 42,  # total points for this source
            "total_points": 300
        }
        
        Returns (source_id, binned_results) when bin is complete, else None.
        """
        source_id = msg["source_id"]
        time = msg["time"]
        magnitude = msg["magnitude"]
        band = msg.get("band", "unknown")
        sequence = msg.get("sequence", -1)
        total_points = msg.get("total_points", -1)
        
        # Compute bin index
        bin_index = sequence // self.bin_size
        
        # Accumulate to bin
        self.bins[source_id][bin_index]["times"].append(time)
        self.bins[source_id][bin_index]["mags"].append(magnitude)
        
        # Check if this bin is complete (has self.bin_size points OR is the last partial bin)
        is_last_bin = (sequence + 1 == total_points)
        bin_full = len(self.bins[source_id][bin_index]["times"]) >= self.bin_size
        
        if bin_full or (is_last_bin and self.bins[source_id][bin_index]["times"]):
            bin_data = self.bins[source_id][bin_index]
            avg_time = np.mean(bin_data["times"])
            avg_mag = np.mean(bin_data["mags"])
            avg_err_mag = np.average(bin_data["mags"], weights=np.array(bin_data["mags"]) * 0.05)
            
            result = {
                "source_id": source_id,
                "bin_index": bin_index,
                "time_avg": float(avg_time),
                "magnitude_avg": float(avg_mag),
                "band": band,
                "bin_size": len(bin_data["times"]),
                "is_final": is_last_bin,
            }
            
            # Clean up completed bin
            del self.bins[source_id][bin_index]
            
            return source_id, result
        
        return None

    def run(self, max_messages: int | None = None):
        """
        Run consumer loop. Process messages and aggregate into bins.
        
        Args:
            max_messages: Stop after processing this many messages (for testing)
        """
        msg_count = 0
        try:
            logger.info(f"Listening to Kafka topic '{self.input_topic}'...")
            for message in self.consumer:
                if message.value is None:
                    continue
                
                result = self.process_message(message.value)
                if result:
                    source_id, binned = result
                    logger.info(f"Bin complete: {source_id} bin {binned['bin_index']} = "
                               f"t={binned['time_avg']:.4f}, mag={binned['magnitude_avg']:.4f}")
                    
                    # Emit to output topic
                    self.producer.send(
                        self.output_topic,
                        value=binned,
                        key=source_id.encode("utf-8"),
                    )
                    self.save_binned_result(source_id, binned)
                
                msg_count += 1
                if max_messages and msg_count >= max_messages:
                    logger.info(f"Reached max_messages={max_messages}, stopping.")
                    break
        
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.producer.flush()
            self.producer.close()
            self.consumer.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-servers", default="localhost:9092",
                        help="Kafka bootstrap servers")
    parser.add_argument("--input-topic", default="lightcurve-points",
                        help="Kafka topic for incoming data points")
    parser.add_argument("--output-topic", default="binned-lightcurve",
                        help="Kafka topic for binned output")
    parser.add_argument("--bin-size", type=int, default=5,
                        help="Number of points per bin")
    parser.add_argument("--group-id", default="binner-group",
                        help="Kafka consumer group ID")
    parser.add_argument("--output-dir", default="afterglow_photometry_binned",
                        help="Directory to save completed binned CSV files")
    parser.add_argument("--max-messages", type=int, default=None,
                        help="Stop after processing N messages (for testing)")
    args = parser.parse_args()
    
    binner = StreamingBinner(
        bootstrap_servers=args.bootstrap_servers,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        bin_size=args.bin_size,
        group_id=args.group_id,
        output_dir=args.output_dir,
    )

    
    binner.run(max_messages=args.max_messages)
