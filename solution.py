## Student Name: Daniel Ferlisi
## Student ID: 218714923

"""
Task A: Appointment Timeslot Recommender (Stub)

In this lab, you will design and implement an Appointment Slot Recommender using an LLM assistant
as your primary programming collaborator.

You are asked to implement a Python module that recommends available meeting slots within a
defined working window.

The system must:
  • Accept working hours (start and end time).
  • Accept a list of existing busy intervals.
  • Accept a required meeting duration.
  • Accept an optional buffer time between meetings.
  • Optionally restrict suggestions to a candidate time window.
  • Return chronologically ordered appointment slots that satisfy all constraints.

The system must ensure that:
  • Suggested slots fall within working hours.
  • Suggested slots do not overlap busy intervals.
  • Buffer time is respected when evaluating availability.
  • Output ordering is deterministic under identical inputs.

The module must preserve the following invariants:
  • Returned slots must be at least as long as the required duration.
  • No returned slot may violate buffer constraints.
  • The returned list must reflect the current system state.

The system must correctly handle non-trivial scenarios such as:
  • Adjacent busy intervals.
  • Very small gaps between meetings.
  • Buffers eliminating otherwise valid availability.
  • Overlapping or unsorted busy intervals.
  • A meeting duration longer than any available gap.
  • No availability within the working window.

Output:
  The output consists of the next N valid appointment suggestions in chronological order.
  Behavior must be deterministic under ties (if any).

See the lab handout for full requirements.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta, time
from typing import List, Optional, Tuple


# ---------------- Data Models ----------------

@dataclass(frozen=True)
class TimeWindow:
    """
    A daily time window.
    Assumption (unless stated otherwise in handout): non-wrapping window where start < end.
    """
    start: time
    end: time


@dataclass(frozen=True)
class BusyInterval:
    """
    A busy interval on the given day.
    Invariant: start < end
    """
    start: time
    end: time


@dataclass(frozen=True)
class Slot:
    """
    A recommended appointment slot.

    start_time is a time-of-day within the working window.
    Deterministic ordering: sort by start_time ascending.
    """
    start_time: time


class InfeasibleSchedule(Exception):
    """Raised when no valid slots can be produced (if required by handout)."""
    pass


# ---------------- Core Function ----------------

def suggest_slots(
    day: date,
    working_hours: TimeWindow,
    busy_intervals: List[BusyInterval],
    duration: timedelta,
    n: int,
    buffer: timedelta = timedelta(0),
    candidate_window: Optional[TimeWindow] = None
) -> List[Slot]:

    # ---------------- Input Validation ----------------
    if duration <= timedelta(0):
        raise ValueError("Meeting duration must be positive.")

    if buffer < timedelta(0):
        raise ValueError("Buffer time cannot be negative.")

    if n < 0:
        raise ValueError("Number of suggestions (n) cannot be negative.")

    if working_hours.start >= working_hours.end:
        raise ValueError("Working hours start must be before end.")

    for b in busy_intervals:
        if b.start >= b.end:
            raise ValueError("Busy interval start must be before end.")

    if candidate_window:
        if candidate_window.start >= candidate_window.end:
            raise ValueError("Candidate window start must be before end.")

        if candidate_window.start < working_hours.start or candidate_window.end > working_hours.end:
            raise ValueError("Candidate window must lie within working hours.")

    # ---------------- Determine Effective Window ----------------
    start_time = working_hours.start
    end_time = working_hours.end

    if candidate_window:
        start_time = max(start_time, candidate_window.start)
        end_time = min(end_time, candidate_window.end)

    if start_time >= end_time:
        raise InfeasibleSchedule("No valid appointment slots: candidate window invalid.")

    window_start = datetime.combine(day, start_time)
    window_end = datetime.combine(day, end_time)

    # ---------------- Normalize Busy Intervals (Revision 10) ----------------
    busy = []

    for b in busy_intervals:
        start = datetime.combine(day, b.start)
        end = datetime.combine(day, b.end)

        # Apply buffer (Revision 6)
        start -= buffer
        end += buffer

        busy.append((start, end))

    # Sort intervals (deterministic order)
    busy.sort()

    # Merge overlapping / adjacent intervals
    merged_busy = []

    for interval in busy:
        if not merged_busy:
            merged_busy.append(interval)
            continue

        prev_start, prev_end = merged_busy[-1]
        cur_start, cur_end = interval

        if cur_start <= prev_end:  # overlap or adjacency
            merged_busy[-1] = (prev_start, max(prev_end, cur_end))
        else:
            merged_busy.append(interval)

    # ---------------- Slot Generation ----------------

    step = timedelta(minutes=5)  # deterministic slot granularity
    current = window_start
    results: List[Slot] = []

    while current + duration <= window_end:

        slot_start = current
        slot_end = slot_start + duration

        conflict = False

        for busy_start, busy_end in merged_busy:
            if slot_start < busy_end and busy_start < slot_end:
                conflict = True
                break

        if not conflict:
            slot = Slot(start_time=slot_start.time())

            # Prevent duplicate slots (Revision 2)
            if slot not in results:
                results.append(slot)

            if len(results) == n:
                break

        current += step

    # ---------------- No Availability Handling ----------------
    if not results:
        raise InfeasibleSchedule(
            "No valid appointment slots available within the given constraints."
        )

    # ---------------- Deterministic Ordering ----------------
    results.sort(key=lambda s: s.start_time)

    return results