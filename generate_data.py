"""
gen dataset of simulated bank shots.

    python generate_data.py              # 200k attempts -> shots.csv
    python generate_data.py --n 50000    # smaller/faster

only shots whose FIRST contact is the backboard are kept (true bank shots).

columns
    release_x, release_y      where the ball left the shooter's hands (m)
    release_vx, release_vy    release velocity (m/s)
    release_spin              release spin (rad/s, + = backspin)
    impact_y                  height of ball center when it hits the board (m)
    impact_vx, impact_vy      velocity just before hitting the board (m/s)
    impact_spin               spin just before hitting the board (rad/s)
    made                      1 = swish/roll in, 0 = miss
"""
import argparse
import time
import numpy as np
import pandas as pd
from hoopsim import Court, simulate, random_bank_shots, FIRST_BOARD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000, help="shot attempts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="shots.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    court = Court()
    chunks = []
    t0 = time.time()
    chunk = 25_000
    for start in range(0, args.n, chunk):
        m = min(chunk, args.n - start)
        x0, y0, vx0, vy0, w0 = random_bank_shots(m, rng, court)
        out = simulate(x0, y0, vx0, vy0, w0, court)
        bank = out["first_contact"] == FIRST_BOARD
        chunks.append(pd.DataFrame({
            "release_x": x0[bank], "release_y": y0[bank],
            "release_vx": vx0[bank], "release_vy": vy0[bank],
            "release_spin": w0[bank],
            "impact_y": out["impact_y"][bank],
            "impact_vx": out["impact_vx"][bank],
            "impact_vy": out["impact_vy"][bank],
            "impact_spin": out["impact_w"][bank],
            "made": out["made"][bank],
        }))
        print(f"  simulated {start + m:,}/{args.n:,} attempts "
              f"({time.time() - t0:.0f}s)")

    df = pd.concat(chunks, ignore_index=True).round(5)
    df.to_csv(args.out, index=False)
    print(f"saved {len(df):,} bank shots to {args.out} "
          f"(make rate {df.made.mean():.1%})")


if __name__ == "__main__":
    main()
