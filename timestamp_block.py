#!/usr/bin/env python3

from datetime import datetime, timezone

def get_nearest_block_for_timestamp(subtensor, target_time_utc: datetime, 
                                    start_block: int = 1, 
                                    end_block: int | None = None, 
                                    max_iterations: int = 25) -> int | None:
    """
    Given a Bittensor subtensor and UTC timestamp, return the nearest block number.

    Uses binary search on block timestamps for efficiency.

    Args:
        subtensor: bittensor subtensor instance (e.g., bt.subtensor("finney"))
        target_time_utc (datetime): UTC timestamp to find nearest block for.
        start_block (int): Lowest block to consider (default: 1)
        end_block (int | None): Highest block (default: current chain head)
        max_iterations (int): Maximum binary search steps.

    Returns:
        int | None: Closest block number to the target timestamp, or None if unavailable.
    """
    # Ensure timezone awareness
    if target_time_utc.tzinfo is None:
        target_time_utc = target_time_utc.replace(tzinfo=timezone.utc)

    # Get latest block number if not given
    if end_block is None:
        end_block = subtensor.get_current_block()

    def get_block_time(block_num: int):
        block_hash = subtensor.substrate.get_block_hash(block_num)
        if not block_hash:
            return None
        data = subtensor.substrate.query('Timestamp', 'Now', block_hash=block_hash)
        return datetime.utcfromtimestamp(data.value / 1000) if data and data.value else None

    # Binary search for nearest timestamp
    low, high = start_block, end_block
    nearest_block = None
    nearest_diff = float("inf")

    for _ in range(max_iterations):
        mid = (low + high) // 2
        mid_time = get_block_time(mid)
        if not mid_time:
            break

        diff = abs((mid_time - target_time_utc).total_seconds())
        if diff < nearest_diff:
            nearest_block, nearest_diff = mid, diff

        if mid_time < target_time_utc:
            low = mid + 1
        else:
            high = mid - 1

        # Early stop if difference is very small (< block time)
        if nearest_diff < 6:
            break

    return nearest_block

if __name__ == "__main__":
    import bittensor as bt
    import argparse

    parser = argparse.ArgumentParser(description="Get nearest Bittensor block for a UTC timestamp.")
    parser.add_argument("--network", type=str, default="finney", help="Bittensor network (default: finney)")
    parser.add_argument("--timestamp", type=str, required=True, help="UTC timestamp (ISO format, e.g., 2023-10-01T12:00:00Z)")
    parser.add_argument("--start-block", type=int, default=1, help="Lowest block to consider (default: 1)")
    parser.add_argument("--end-block", type=int, default=None, help="Highest block to consider (default: current chain head)")
    args = parser.parse_args()

    try:
        target_time = datetime.fromisoformat(args.timestamp.replace("Z", "+00:00"))
    except ValueError:
        print("Invalid timestamp format. Use ISO format like '2023-10-01T12:00:00Z'.")
        exit(1)

    subtensor = bt.subtensor(network=args.network)
    nearest_block = get_nearest_block_for_timestamp(
        subtensor, 
        target_time_utc=target_time, 
        start_block=args.start_block, 
        end_block=args.end_block
    )

    if nearest_block is not None:
        print(f"Nearest block to {target_time.isoformat()}Z is block number: {nearest_block}")
    else:
        print("Could not determine nearest block.")
