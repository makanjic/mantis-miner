# run_mantis_btc.py
import time, os
import pandas as pd
from btc_embedding import build_btc_embedding_at, EncoderConfig

def current_block():
    try:
        import bittensor as bt
        st = bt.subtensor("finney")
        return int(st.get_current_block())
    except Exception:
        return None

def main():
    last_fired = None
    while True:
        blk = current_block()
        now = pd.Timestamp.utcnow().tz_localize("UTC").floor("T")

        should_fire = False
        if blk is not None:
            if blk % 5 == 0 and blk != last_fired:
                should_fire = True
        else:
            # fallback: fire once per new minute
            should_fire = True if last_fired is None else False

        if should_fire:
            ts = now
            emb, feat_names = build_btc_embedding_at(
                ts,
                encoder_cfg=EncoderConfig(weights_path=os.getenv("BTC_ENCODER_WEIGHTS"))
            )
            # TODO: submit `emb` to MANTIS miner interface here
            print(f"Built BTC embedding at {ts.isoformat()} (dim={len(emb)})")
            last_fired = blk if blk is not None else last_fired
        time.sleep(2)

if __name__ == "__main__":
    main()
