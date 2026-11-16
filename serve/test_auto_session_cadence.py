"""The auto-session save cadence (AUTO_SESSION_GROWTH / AUTO_SESSION_MAX_AGE_S) on a fake clock: the
real idle loop, a fake engine, no sleeping, no GPU and no model.

    python -m pytest serve/test_auto_session_cadence.py -q
"""
from __future__ import annotations

import io
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from serve import server as server_mod
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service
from serve.test_slots import FakeEngine

NOW = 1_780_000_000.0        # an ordinary epoch second: sess_started_at is made with the real clock


class EndLoop(Exception):
    """Raised out of the idle loop's own sleep. The loop sleeps first and then checks, so a sleep that
    ends it closes that round: the test drives the real loop body without waiting on wall-clock seconds."""


class Clock:
    """`serve.server`'s `time` on demand. Only the driving thread's sleeps count rounds and raise; a
    background thread gets the real time module, so the fake cannot end a loop it never started."""

    def __init__(self, now=NOW):
        self.now = now
        self.sleeps = 0
        self.stop_after = 2                    # 2 = the body runs one round, then the sleep ends the loop
        self.owner = threading.current_thread()

    def time(self):
        return self.now

    def sleep(self, seconds):
        if threading.current_thread() is not self.owner:
            return time.sleep(seconds)
        self.now += seconds
        self.sleeps += 1
        if self.sleeps >= self.stop_after:
            raise EndLoop()

    def __getattr__(self, name):
        return getattr(time, name)             # monotonic(), strftime() ... for anything else in the server


class Cadence(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.engine = FakeEngine()
        self.svc = Service(self.engine, ByteTokenizer(),
                           ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.svc.slot_save_path = self.dir.name
        self.clock = Clock()
        self.stdout = mock.patch("sys.stdout", new_callable=io.StringIO)
        self.stdout.start()
        self.addCleanup(self.stdout.stop)

    def arm(self, live, saved_tokens=0, saved_at=0.0, quiet=20.0, started=60.0):
        """One settled request past: the engine idled `quiet` s (over AUTO_SESSION_QUIET_S) and the request
        finished after this engine start, so the idle loop reaches the cadence test. `saved_at` 0 means the
        file holds nothing since this start: a start that restored nothing, or a refused restore."""
        now = self.clock.now
        self.svc.auto_session = "auto-session"
        self.svc.sess_needs_restore = False
        self.svc.started_at = now - 3600.0
        self.svc.sess_started_at = now - started
        self.svc.last_request_at = now - quiet
        self.svc.sess_tried_at = 0.0
        self.svc.sess_saved_at = saved_at
        self.svc.sess_saved_tokens = saved_tokens
        self.svc.sess_live_tokens = live
        self.svc.status.update(busy=False, queued=0)

    def rounds(self, rounds=1):
        """The real _auto_session_loop for `rounds` idle rounds on the fake clock (its own sleeps end it),
        with auto_session_save replaced by what the loop records. -> that recorder."""
        self.clock.sleeps = 0
        self.clock.stop_after = rounds + 1
        with mock.patch.object(server_mod, "time", self.clock), \
                mock.patch.object(self.svc, "auto_session_save") as save:
            try:
                self.svc._auto_session_loop()
            except EndLoop:
                pass
        return save


    def test_small_growth_and_a_fresh_file_skip_the_save(self):
        g = Service.AUTO_SESSION_GROWTH
        self.arm(live=56_000 + g - 1, saved_tokens=56_000, saved_at=self.clock.now - 100.0)
        self.assertEqual(self.rounds(3).call_count, 0)   # under the growth gate, and 100 s is young

    def test_growth_over_the_gate_saves(self):
        g = Service.AUTO_SESSION_GROWTH
        self.arm(live=56_000 + g, saved_tokens=56_000, saved_at=self.clock.now - 100.0)
        self.rounds().assert_called_once_with("idle")

    def test_a_stale_file_saves_without_growth(self):
        self.arm(live=56_000, saved_tokens=56_000,
                 saved_at=self.clock.now - Service.AUTO_SESSION_MAX_AGE_S - 1.0, started=90.0)
        self.rounds().assert_called_once_with("idle")

    def test_the_first_save_after_a_start_that_restored_nothing_saves(self):
        self.arm(live=120, saved_tokens=0, saved_at=0.0)  # sess_saved_at 0: the age gate is open
        self.rounds().assert_called_once_with("idle")

    def test_the_gate_follows_the_live_conversation(self):
        self.arm(live=56_000, saved_tokens=56_000, saved_at=self.clock.now - 100.0)
        self.assertEqual(self.rounds(2).call_count, 0)
        self.svc.sess_live_tokens = 56_000 + Service.AUTO_SESSION_GROWTH
        self.rounds().assert_called_once_with("idle")


if __name__ == "__main__":
    unittest.main()
