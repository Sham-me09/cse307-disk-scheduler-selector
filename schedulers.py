"""Classical disk scheduling algorithms.

Every function takes the list of pending track requests, the starting head
position and the number of cylinders, and returns the total head movement
(seek distance in tracks). Seek time is taken as proportional to distance.
"""

def fcfs(reqs, head, ncyl=200):
    total, pos = 0, head
    for r in reqs:
        total += abs(r - pos)
        pos = r
    return total


def sstf(reqs, head, ncyl=200):
    pending = list(reqs)
    total, pos = 0, head
    while pending:
        # closest pending request; ties go to the lower track number
        nxt = min(pending, key=lambda r: (abs(r - pos), r))
        pending.remove(nxt)
        total += abs(nxt - pos)
        pos = nxt
    return total


def scan(reqs, head, ncyl=200):
    """Elevator, textbook version: sweep up to the last cylinder, then reverse."""
    up = sorted(r for r in reqs if r >= head)
    down = sorted((r for r in reqs if r < head), reverse=True)
    total, pos = 0, head
    if up:
        total += (ncyl - 1) - pos      # run to the end of the disk
        pos = ncyl - 1
    if down:
        total += pos - down[-1]        # come back to the lowest request
    return total


def cscan(reqs, head, ncyl=200):
    """Circular SCAN: sweep up to the end, jump to 0, keep sweeping up.
    The return jump is counted as head movement (as in Silberschatz)."""
    up = sorted(r for r in reqs if r >= head)
    low = sorted(r for r in reqs if r < head)
    total, pos = 0, head
    if low:
        total += (ncyl - 1) - pos      # to the end
        total += (ncyl - 1)            # jump back to track 0
        total += low[-1]               # sweep up to the highest low request
    elif up:
        total += up[-1] - pos
    return total


SCHEDULERS = {"FCFS": fcfs, "SCAN": scan, "C-SCAN": cscan, "SSTF": sstf}
NAMES = list(SCHEDULERS)
