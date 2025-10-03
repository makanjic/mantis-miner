#!/usr/bin/env python3

import sys, os
import bittensor as bt
from urllib.parse import quote


def commit_to_chain(subtensor, wallet, netuid, bucket, obj_key):
    # public_url = quote(os.path.join(bucket, obj_key))
    public_url = os.path.join(bucket, obj_key).replace(" ", "%20")
    print(f"Public URL: {public_url}")
    subtensor.commit(wallet=wallet, netuid=netuid, data=public_url)


if __name__ == "__main__":
    import argparse
    import config
    from dotenv import load_dotenv
    
    load_dotenv()

    p = argparse.ArgumentParser()
    p.add_argument("--wallet.path", default=None)
    p.add_argument("--wallet.name", required=True)
    p.add_argument("--wallet.hotkey", required=True)
    p.add_argument("--network", default="finney")
    p.add_argument("--netuid", type=int, default=config.NETUID)
    p.add_argument("--bucket", default=os.environ.get("R2_BUCKET_PUBLIC_URL"))
    args = p.parse_args()

    while True:
        try:
            subtensor = bt.subtensor(network=args.network)
            wallet = bt.wallet(
                path=getattr(args, "wallet.path"),
                name=getattr(args, "wallet.name"),
                hotkey=getattr(args, "wallet.hotkey"),
            )
            break
        except Exception:
            logging.exception("Subtensor connect failed")
            time.sleep(30)

    if not args.bucket:
        print("No R2 bucket specified, set R2_BUCKET_PUBLIC_URL env var or pass --bucket")
        sys.exit(1)

    commit_to_chain(subtensor, wallet, args.netuid, args.bucket, wallet.hotkey.ss58_address)
