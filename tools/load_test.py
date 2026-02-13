#!/usr/bin/env python3
import argparse
import socket
import struct
import threading


def roundtrip(host: str, port: int, payload: bytes, loops: int, out: dict):
    ok = busy = bad = 0
    s = socket.create_connection((host, port))
    for i in range(loops):
        req = struct.pack("<IHHQI", 0x53464954, 1, 1, i, len(payload)) + payload
        s.sendall(req)
        hdr = s.recv(22)
        if len(hdr) < 22:
            break
        _, _, _, _, status, result_len = struct.unpack("<IHHQHI", hdr)
        body = b""
        while len(body) < result_len:
            body += s.recv(result_len - len(body))
        if status == 0:
            ok += 1
        elif status == 2:
            busy += 1
        else:
            bad += 1
    s.close()
    out[threading.get_ident()] = (ok, busy, bad)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--threads", type=int, default=20)
    p.add_argument("--loops", type=int, default=500)
    p.add_argument("--payload", default="hello")
    args = p.parse_args()

    results = {}
    ts = []
    for _ in range(args.threads):
        t = threading.Thread(target=roundtrip, args=(args.host, args.port, args.payload.encode(), args.loops, results))
        ts.append(t)
        t.start()
    for t in ts:
        t.join()

    ok = sum(v[0] for v in results.values())
    busy = sum(v[1] for v in results.values())
    bad = sum(v[2] for v in results.values())
    print(f"done ok={ok} busy={busy} bad={bad} total={ok+busy+bad}")


if __name__ == "__main__":
    main()
