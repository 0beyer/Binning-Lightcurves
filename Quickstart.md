# Installation

1. **Install dependencies:**

2. **Start Kafka cluster:**
   ```bash
   docker-compose up -d
   ```
   
   This starts:
   - Zookeeper (port 2181)
   - Kafka broker (port 9092)

3. **Verify Kafka is running:**
   ```bash
   docker-compose ps
   ```

## Scripts

Run generators without Kafka for the original behavior:

```bash
# With Binning
conda run -n fiesta python streaming_binner.py --bin-size 5
conda run -n fiesta python generate_afterglow_samples.py --n-samples 100 --m-bins 5

# Without Binning
conda run -n fiesta python generate_afterglow_samples.py --n-samples 100
```