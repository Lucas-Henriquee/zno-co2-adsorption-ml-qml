import time
from contextlib import contextmanager

try:
    from gpaw.mpi import world
    RANK = world.rank
except ImportError:
    RANK = 0

def print0(*args, **kwargs):
    if RANK == 0:
        print(*args, **kwargs)

@contextmanager
def timer(name: str):
    start = time.perf_counter()

    if RANK == 0:
        print(f"\n[START] {name}")

    try:
        yield
    finally:
        end = time.perf_counter()

        if RANK == 0:
            print(f"[END] {name} → {end - start:.4f} s")
