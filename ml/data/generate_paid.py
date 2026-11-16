"""Paid image generation (hosted APIs). Deliberately inert: it checks the spend guard, prints the plan and
stops. Wire a provider into `generate_one` before first use. Never run from cron or the nightly loop."""
import argparse

PRICE_PER_IMAGE = {"example_provider": 0.04}  # placeholder; fill with the provider's list price


def generate_one(provider: str, prompt: str) -> bytes:
    raise NotImplementedError("no paid provider is wired up")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--approve-spend", type=float, required=True, metavar="DOLLARS",
                    help="hard cap in dollars; required, 0 refuses")
    ap.add_argument("--provider", default="example_provider")
    ap.add_argument("--limit", type=int, default=10)
    a = ap.parse_args()
    if a.approve_spend <= 0:
        raise SystemExit("refusing: --approve-spend must be > 0")
    est = a.limit * PRICE_PER_IMAGE.get(a.provider, float("inf"))
    if est > a.approve_spend:
        raise SystemExit(f"refusing: estimated ${est:.2f} exceeds approved ${a.approve_spend:.2f}")
    raise SystemExit(f"plan ok (${est:.2f} of ${a.approve_spend:.2f}) but no provider is wired up; nothing was generated")


if __name__ == "__main__":
    main()
