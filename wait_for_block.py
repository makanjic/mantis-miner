#!/usr/bin/env python3
import argparse
import sys
import bittensor as bt

def wait_until_remainder(n: int = 1, remainder: int = 0, network: str = "finney"):
    # Validate inputs
    if n <= 0:
        bt.logging.error(f"Invalid divisor n={n}. Must be a positive integer greater than zero.")
        sys.exit(1)
    if not (0 <= remainder < n):
        bt.logging.error(f"Invalid remainder {remainder}. Must satisfy 0 <= remainder < n ({n}).")
        sys.exit(1)

    # Connect to the specified network
    subtensor = bt.subtensor(network)
    bt.logging.info(f"Connected to Bittensor network: {network}")

    # Get current block number
    current_block = subtensor.get_current_block()
    bt.logging.info(f"Current block: {current_block}")

    # Compute the next block satisfying (block % n == remainder)
    current_mod = current_block % n
    if current_mod <= remainder:
        next_target = current_block + (remainder - current_mod)
    else:
        next_target = current_block + (n - current_mod) + remainder

    bt.logging.info(f"Waiting for block % {n} == {remainder}. Target block: {next_target}")

    # Wait for the target block
    subtensor.wait_for_block(next_target)

    bt.logging.success(f"Reached block {next_target} (block % {n} == {remainder})!")
    return next_target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wait until the Bittensor block number modulo N equals a specified remainder."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Divisor for block number (must be > 0, default: 1).",
    )
    parser.add_argument(
        "--remainder", "-r",
        type=int,
        default=0,
        help="Desired remainder (0 <= remainder < n, default: 0).",
    )
    parser.add_argument(
        "--network",
        type=str,
        default="finney",
        help="Bittensor network to connect to (default: finney).",
    )

    args = parser.parse_args()
    wait_until_remainder(args.n, args.remainder, args.network)
