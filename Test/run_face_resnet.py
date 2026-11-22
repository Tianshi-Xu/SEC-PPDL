#!/usr/bin/env python3
"""Run both parties of the FaceResNet64 CPU benchmark on one host."""

import argparse
import asyncio
from contextlib import suppress
import json
import math
from pathlib import Path
import re
import shutil
import tempfile


CONNECTIONS = 3  # The face benchmark uses three OT connections, starting at p+1.
MIB = 1 << 20


class Direction:
    """Bandwidth scheduler shared by all connections in one direction."""

    def __init__(self, mbps, rtt_ms):
        self.bytes_per_second = mbps * 1_000_000 / 8
        self.propagation_seconds = rtt_ms / 2000
        self.available_at = 0.0

    def delivery_time(self, now, size):
        self.available_at = max(now, self.available_at) + size / self.bytes_per_second
        return self.available_at + self.propagation_seconds


async def relay(reader, writer, direction):
    queue = asyncio.Queue(maxsize=32)
    clock = asyncio.get_running_loop().time

    async def receive():
        while data := await reader.read(64 * 1024):
            await queue.put((direction.delivery_time(clock(), len(data)), data))
        await queue.put(None)

    async def send():
        while entry := await queue.get():
            delivery, data = entry
            if delivery > clock():
                await asyncio.sleep(delivery - clock())
            writer.write(data)
            await writer.drain()
        writer.write_eof()

    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(receive())
        tasks.create_task(send())


async def start_wan_proxy(server_port, proxy_port, mbps, rtt_ms, timeout, tasks):
    upload = Direction(mbps, rtt_ms)
    download = Direction(mbps, rtt_ms)

    async def proxy(client_reader, client_writer, port):
        server_writer = None
        try:
            # Each party initializes OT before opening its next connection.
            async with asyncio.timeout(timeout):
                while True:
                    try:
                        server_reader, server_writer = await asyncio.open_connection("127.0.0.1", port)
                        break
                    except ConnectionRefusedError:
                        await asyncio.sleep(0.02)
            async with asyncio.TaskGroup() as streams:
                streams.create_task(relay(client_reader, server_writer, upload))
                streams.create_task(relay(server_reader, client_writer, download))
        finally:
            client_writer.close()
            if server_writer is not None:
                server_writer.close()

    def handler(port):
        def accept(reader, writer):
            tasks.create_task(proxy(reader, writer, port))
        return accept

    listeners = []
    try:
        for offset in range(1, CONNECTIONS + 1):
            listener = await asyncio.start_server(
                handler(server_port + offset), "127.0.0.1", proxy_port + offset)
            listeners.append(listener)
        return listeners
    except BaseException:
        for listener in listeners:
            listener.close()
        raise


def metrics(directory):
    logs = [(directory / f"{role}.log").read_text() for role in ("server", "client")]

    def numbers(label, cast):
        matches = [re.search(rf"^{re.escape(label)}: (\S+)$", log, re.MULTILINE) for log in logs]
        if not all(matches):
            raise ValueError(f"Missing benchmark metric {label!r} in {directory}")
        return [cast(match[1]) for match in matches]

    def traffic(label, requests=1):
        return sum(numbers(label, int)) / (requests * MIB)

    pair_times = [max(numbers(f"{ordinal} verification seconds", float))
                  for ordinal in ("First", "Second")]
    convolution = traffic("Convolution bytes sent (two pairs)", 2)
    linear = traffic("Online linear-layer bytes sent (two pairs)", 2)
    online = traffic("Online verification bytes sent (two pairs)", 2)
    return {
        "seconds_per_image": max(numbers("Average forward seconds per image", float)),
        "seconds_per_pair": sum(pair_times) / len(pair_times),
        "session_setup_seconds": max(numbers("Session setup seconds", float)),
        "server_model_packing_seconds": numbers("Local model packing seconds", float)[0],
        "online_mib_per_pair": online,
        "convolution_mib_per_pair": convolution,
        "online_conversion_mib_per_pair": linear - convolution,
        "other_online_mib_per_pair": online - linear,
        "ot_refill_mib_per_pair": traffic("OT refill bytes sent (two pairs)", 2),
        "ot_setup_mib_per_session": traffic("OT setup and warm-up bytes sent (one session)"),
        "other_setup_mib_per_session": traffic("Other setup bytes sent (one session)"),
    }


async def run(args, protocol, directory, server_port):
    directory.mkdir()
    print(f"Running ResNet{args.backbone} {protocol}. Logs in {directory}", flush=True)
    baseline = protocol == "baseline"
    flags = [f"backbone={args.backbone}", "full=1", "dense=1", "vole=1", "checks=0",
             f"baseline={int(baseline)}", f"mm={int(not baseline)}", f"he_threads={args.he_threads}"]
    processes = []

    async def party(role, port, name, cpu_list):
        command = [str(args.binary), f"r={role}", f"p={port}", *flags]
        if cpu_list:
            command = ["taskset", "-c", cpu_list, *command]
        with (directory / f"{name}.log").open("w") as log:
            process = await asyncio.create_subprocess_exec(
                *command, cwd=directory, stdout=log, stderr=asyncio.subprocess.STDOUT)
        processes.append(process)
        status = await process.wait()
        if status != 0:
            raise RuntimeError(f"{name} exited with {status}. See {directory / (name + '.log')}")

    listeners = []
    client_port = server_port + 10 if args.wan else server_port
    try:
        async with asyncio.timeout(args.timeout):
            async with asyncio.TaskGroup() as proxy_tasks:
                if args.wan:
                    listeners = await start_wan_proxy(
                        server_port, client_port, args.mbps, args.rtt_ms, args.timeout, proxy_tasks)
                try:
                    async with asyncio.TaskGroup() as parties:
                        parties.create_task(party(1, server_port, "server", args.server_cpus))
                        parties.create_task(party(2, client_port, "client", args.client_cpus))
                finally:
                    for listener in listeners:
                        listener.close()
    finally:
        for process in processes:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
        await asyncio.gather(*(process.wait() for process in processes))
        await asyncio.gather(*(listener.wait_closed() for listener in listeners))
    result = metrics(directory)
    result["he_threads"] = 1 if baseline else args.he_threads
    return result


async def benchmark(args, directory):
    protocols = ("baseline", "hybrid") if args.protocol == "both" else (args.protocol,)
    results = {}
    for index, protocol in enumerate(protocols):
        result = await run(args, protocol, directory / protocol, args.port + index * 20)
        results[protocol] = result
        print(f"{protocol}: {result['seconds_per_image']:.2f} s/image, "
              f"{result['seconds_per_pair']:.2f} s/pair, {result['online_mib_per_pair']:.2f} online MiB/pair",
              flush=True)
    summary = {"backbone": args.backbone,
               "server_cpus": args.server_cpus, "client_cpus": args.client_cpus,
               "network": {"mbps_per_direction": args.mbps, "rtt_ms": args.rtt_ms} if args.wan else "loopback",
               "results": results}
    if args.protocol == "both":
        ratio = results["baseline"]["online_mib_per_pair"] / results["hybrid"]["online_mib_per_pair"]
        summary["online_communication_ratio"] = ratio
        print(f"Online communication improvement: {ratio:.2f}x", flush=True)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=int, choices=(18, 20, 32, 34), required=True)
    parser.add_argument("--protocol", choices=("baseline", "hybrid", "both"), default="both")
    parser.add_argument("--binary", type=Path,
                        default=Path(__file__).resolve().parents[1] / "build/Test/test_face_resnet")
    parser.add_argument("--he-threads", type=int, default=16, help="HE workers in hybrid mode")
    parser.add_argument("--wan", action="store_true", help="simulate a WAN through a local TCP proxy")
    parser.add_argument("--mbps", type=float, default=400, help="WAN bandwidth per direction")
    parser.add_argument("--rtt-ms", type=float, default=20, help="WAN round-trip time")
    parser.add_argument("--port", type=int, default=23000, help="base port for the first session")
    parser.add_argument("--timeout", type=float, default=1800, help="timeout in seconds per session")
    parser.add_argument("--server-cpus", help="optional taskset CPU list for the server")
    parser.add_argument("--client-cpus", help="optional taskset CPU list for the client")
    parser.add_argument("--output-dir", type=Path, help="parent directory for the logs and summary")
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    if not args.binary.is_file():
        parser.error(f"benchmark executable not found: {args.binary}")
    if not 1 <= args.port <= 65502:
        parser.error("--port must be in [1, 65502]")
    if (args.he_threads < 1 or args.timeout <= 0 or args.mbps <= 0 or args.rtt_ms < 0
            or not all(map(math.isfinite, (args.timeout, args.mbps, args.rtt_ms)))):
        parser.error("threads, timeout, and bandwidth must be positive and RTT must be nonnegative")
    if (args.server_cpus or args.client_cpus) and shutil.which("taskset") is None:
        parser.error("--server-cpus and --client-cpus require taskset")
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"face-resnet{args.backbone}-", dir=args.output_dir)).resolve()
    asyncio.run(benchmark(args, directory))
    print(f"Results saved to {directory / 'summary.json'}")


if __name__ == "__main__":
    main()
