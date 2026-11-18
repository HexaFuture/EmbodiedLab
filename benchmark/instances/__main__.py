"""Instance CLI (trusted operator side).

  python -m benchmark.instances materialize --exp hooke_k --split dev --index 0 --store DIR
  python -m benchmark.instances table --exp hooke_k --split dev --count 20 [--start 0] --out FILE.jsonl

`materialize` prints only the opaque instance_id (and the skipped indices, if any).
"""
import argparse
import json

from .service import TABLE_VERSION, InstanceRejected, derive, issue_table, load_key, materialize


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("materialize", "table"):
        p = sub.add_parser(name)
        p.add_argument("--exp", required=True)
        p.add_argument("--split", default="dev")
        p.add_argument("--key-file", help="required for test/train; never printed")
        if name == "materialize":
            p.add_argument("--index", type=int, required=True)
            p.add_argument("--store", required=True)
            p.add_argument("--next-valid", action="store_true", help="skip rejected indices forward")
        else:
            p.add_argument("--count", type=int, required=True)
            p.add_argument("--start", type=int, default=0)
            p.add_argument("--segment", default="formal", choices=("formal", "canary", "reserve", "dev"))
            p.add_argument("--store", help="also materialize every accepted instance here")
            p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    key = load_key(args.key_file) if args.key_file else None
    if args.command == "materialize":
        index, skipped = args.index, []
        while True:
            try:
                instance = derive(args.exp, args.split, index, key)
                break
            except InstanceRejected as exc:
                if not args.next_valid:
                    raise
                skipped.append({"index": index, "failed_constraints": exc.failed})
                index += 1
        materialize(instance, args.store)
        print(json.dumps({"instance_id": instance.instance_id, "exp_id": args.exp,
                          "split": args.split, "skipped": len(skipped)}))
        return
    accepted, skipped = issue_table(args.exp, args.split, args.count, args.start, key, args.segment)
    with open(args.out, "w") as handle:
        for instance in accepted:
            row = {"instance_id": instance.instance_id, "exp_id": args.exp, "split": args.split,
                   "segment": args.segment, "index": instance.index, "key_id": instance.key_id,
                   "table_version": TABLE_VERSION, "spec_digest": instance.spec_digest}
            handle.write(json.dumps(row) + "\n")
            if args.store:
                materialize(instance, args.store)
    print(json.dumps({"accepted": len(accepted), "skipped": skipped}))


if __name__ == "__main__":
    main()
