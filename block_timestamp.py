#!/usr/bin/env python3

from datetime import datetime

def get_block_timestamp_utc(subtensor, block_number: int) -> datetime | None:
    """
    Given a Bittensor subtensor instance and block number,
    return the block's UTC timestamp as a datetime object.

    Args:
        subtensor: bittensor subtensor instance (e.g., bt.subtensor("finney"))
        block_number (int): Target block number

    Returns:
        datetime | None: UTC timestamp of the block, or None if unavailable.
    """
    # Get block hash for the block number
    block_hash = subtensor.substrate.get_block_hash(block_number)
    if not block_hash:
        return None  # block doesn't exist

    # Query timestamp pallet
    timestamp_data = subtensor.substrate.query(
        module='Timestamp',
        storage_function='Now',
        block_hash=block_hash
    )

    if not timestamp_data or not timestamp_data.value:
        return None

    # Convert milliseconds to seconds, then to UTC datetime
    timestamp_ms = timestamp_data.value
    return datetime.utcfromtimestamp(timestamp_ms / 1000)

if __name__ == "__main__":
    import bittensor as bt
    import argparse

    parser = argparse.ArgumentParser(description="Get UTC timestamp of a Bittensor block.")
    parser.add_argument("--network", type=str, default="finney", help="Bittensor network (default: finney)")
    parser.add_argument("--block", type=int, required=True, help="Block number to query")
    args = parser.parse_args()

    subtensor = bt.subtensor(network=args.network)
    timestamp = get_block_timestamp_utc(subtensor, args.block)
    if timestamp:
        print(f"Block {args.block} on network '{args.network}' has UTC timestamp: {timestamp.isoformat()}Z")
    else:
        print(f"Could not retrieve timestamp for block {args.block} on network '{args.network}'.")
        