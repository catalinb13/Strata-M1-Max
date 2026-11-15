"""A missing wrapper closer ends before a following call, but prose alone does not confirm a call.

A call's own </function> separates it from the next wrapper without merging their arguments.
Parameter values may still contain call-format tags; recovery batches stay opt-in.

    python -m unittest serve.test_tool_call_unclosed -v
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import CALL_START, OutputParser  # noqa: E402

TOOLS = [{"name": "read", "parameters": {"properties": {"path": {"type": "string"}}}},
         {"name": "write", "parameters": {"properties": {"path": {"type": "string"}, "content": {"type": "string"}}}}]
READ = "<function=read>\n<parameter=path>\na.py\n</parameter>\n</function>"
WRITE = "<function=write>\n<parameter=path>\nb.py\n</parameter>\n<parameter=content>\nx = 1\n</parameter>\n</function>"


def run(text, stream_tools, recover, step):
    p = OutputParser(thinking=False, tools=TOOLS, stream_tools=stream_tools, recover=recover)
    evs = []
    for i in range(0, len(text), step):                # fed in small pieces, as a stream arrives
        evs += p.feed(text[i:i + step])
    evs += p.finish("stop")
    calls = [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]
    if stream_tools and not calls:
        arguments = "".join(e.text for e in evs if e.kind == "tool_args")
        if arguments:
            try:
                json.loads(arguments)
            except ValueError:
                pass
            else:
                raise AssertionError("an unconfirmed call has complete streamed arguments")
    return calls, "".join(e.text for e in evs if e.kind == "content")


class UnclosedCall(unittest.TestCase):
    def check(self, text, want_calls, want_content=""):
        for stream_tools in (False, True):
            for recover in (False, True):
                for step in (1, 3, 7, 4096):
                    with self.subTest(stream_tools=stream_tools, recover=recover, step=step):
                        calls, content = run(text, stream_tools, recover, step)
                        self.assertEqual(calls, want_calls)
                        self.assertEqual(content.strip(), want_content)

    def test_closed_calls_are_unchanged(self):
        self.check("<tool_call>\n" + READ + "\n</tool_call>\n<tool_call>\n" + WRITE + "\n</tool_call>",
                   [("read", {"path": "a.py"}), ("write", {"path": "b.py", "content": "x = 1"})])

    def test_a_second_call_after_an_unclosed_one_is_its_own_call(self):
        self.check("<tool_call>\n" + READ + "\n<tool_call>\n" + WRITE + "\n</tool_call>",
                   [("read", {"path": "a.py"}), ("write", {"path": "b.py", "content": "x = 1"})])

    def test_quoted_incomplete_wrapper_stays_literal(self):
        text = "Format: <tool_call>\n" + READ + "\nand then I explain it."
        self.check(text, [], text)

    def test_following_literal_marker_does_not_complete_an_unclosed_call(self):
        text = "<tool_call>\n" + READ + "\n<tool_call> is a format marker."
        self.check(text, [], text)

    def test_omitted_wrapper_recovers_only_on_a_natural_stop(self):
        for closed in (True, False):
            text = "<tool_call>\n" + READ + ("\n</tool_call>" if closed else "")
            for reason in (None, "stop", "length", "cancel", "disconnect"):
                for step in (1, 7, 4096):
                    with self.subTest(closed=closed, reason=reason, step=step):
                        p = OutputParser(thinking=False, tools=TOOLS, stream_tools=True)
                        events = []
                        for offset in range(0, len(text), step):
                            events += p.feed(text[offset:offset + step])
                        events += p.finish(reason)
                        calls = [e for e in events if e.kind == "tool_call"]
                        complete = closed or reason in (None, "stop")
                        self.assertEqual(len(calls), int(complete))
                        if not complete:
                            with self.assertRaises(ValueError):
                                json.loads("".join(e.text for e in events if e.kind == "tool_args"))

    def test_a_value_that_holds_the_tags_still_ends_at_the_calls_own_closer(self):
        value = "a </function> and a <tool_call> in a string"
        call = ("<tool_call>\n<function=write>\n<parameter=path>\nc.py\n</parameter>\n<parameter=content>\n" + value +
                "\n</parameter>\n</function>\n</tool_call>")
        self.check(call, [("write", {"path": "c.py", "content": value})])


# The shapes a model drifts into when its turn ends inside a call it opened (one agent session on this box, 2026-11-15: 6
# of 135 turns).  Not one of them is the template's call form: there is no
# `<function=NAME>` at all, so no client can run them, and the parser returns them as the answer's text (#211).
DRIFTED = [
    '<tool_call>\n<<parameter name="command">cd ~/x && y --help 2>&1 | tail -60\n</parameter>\n<parameter=i>\nReading help\n</parameter>\n</function>\n</tool_call>',
    "<tool_call>\n<functiontodo>\n</function>\n</tool_call>",
    "<tool_call>item>",
    '<tool_call>\ned"</parameter>\n<parameter=task>\nPrune caches\n</parameter>\n</function>\n</tool_call>',
    "<tool_call>\n",
]


class UnfinishedCallText(unittest.TestCase):
    """`drop_unfinished_call_text`: off by default (the text is the answer, #211), on for these shapes the reply shows no
    call markup - in the answer and in the reasoning, which a client displays - and the parser says what it held back."""

    def events(self, text, drop, step=7, thinking=False):
        p = OutputParser(thinking=thinking, tools=TOOLS, stream_tools=True)
        p.drop_unfinished_call_text = drop
        out = []
        for i in range(0, len(text), step):
            out += p.feed(text[i:i + step])
        out += p.finish("stop")
        return p, out

    def reasoning(self, text, drop, tail="\n</think>\n\nDone."):
        return self.events("<think>\n" + text + tail, drop, thinking=True)

    def test_the_default_keeps_the_text(self):
        for text in DRIFTED:
            with self.subTest(text=text[:24]):
                p, events = self.events(text, drop=False)
                self.assertEqual("".join(e.text for e in events if e.kind == "content").strip(), text.strip())
                self.assertEqual(p.dropped_unfinished, "")

    def test_drop_keeps_the_markup_out_of_the_answer(self):
        for text in DRIFTED:
            with self.subTest(text=text[:24]):
                p, events = self.events(text, drop=True)
                self.assertEqual([e for e in events if e.kind == "content" and e.text.strip()], [])
                self.assertEqual([e for e in events if e.kind == "tool_call"], [])
                self.assertEqual(p.dropped_unfinished, text)   # the server logs it

    def test_drop_keeps_the_markup_out_of_the_reasoning(self):
        """The reasoning is what an agent session shows while the model thinks, so the same block there is the same
        leak.  Held back whole, and the answer after `</think>` still arrives."""
        for text in DRIFTED:
            with self.subTest(text=text[:24]):
                p, events = self.reasoning(text, drop=True)
                self.assertEqual([e for e in events if e.kind == "reasoning" and CALL_START in (e.text or "")], [])
                self.assertEqual([e for e in events if e.kind == "tool_call"], [])
                self.assertEqual("".join(e.text for e in events if e.kind == "content").strip(), "Done.")
                self.assertEqual(p.dropped_unfinished_blocks, 1)
                self.assertIn(text.split("\n")[0], p.dropped_unfinished)

    def test_drop_keeps_a_reasoning_block_the_think_tag_cut_short(self):
        """The thinking ends inside the block (no `</tool_call>`): there is nothing after it to settle it, and with
        the option on it is the same leak."""
        p, events = self.reasoning("<tool_call>\n<functiontodo>\n</function>", drop=True)
        self.assertEqual([e for e in events if e.kind == "reasoning" and CALL_START in (e.text or "")], [])
        self.assertEqual(p.dropped_unfinished_blocks, 1)

    def test_drop_leaves_a_call_of_the_template_form_alone(self):
        """A body that names its function is not this option's business: a call of an undeclared tool stays text
        ("undeclared_tool_calls" decides that one), and a whole call is still a call."""
        drifted_name = '<tool_call>\n<function=bash:hermes>\n{"command": "x"}\n</parameter>\n</function>\n</tool_call>'
        p, events = self.events(drifted_name, drop=True)
        self.assertIn(drifted_name.lstrip("<tool_call>").strip(), "".join(e.text for e in events if e.kind == "content"))
        self.assertEqual(p.dropped_unfinished, "")
        whole = "<tool_call>\n" + READ + "\n</tool_call>"
        p, events = self.events(whole, drop=True)
        self.assertEqual([(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"],
                         [("read", {"path": "a.py"})])
        self.assertEqual("".join(e.text for e in events if e.kind == "content").strip(), "")
        p, events = self.reasoning(whole, drop=True)     # a real call made while thinking is still a call
        self.assertEqual([(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"],
                         [("read", {"path": "a.py"})])
        self.assertEqual(p.dropped_unfinished, "")
        p, events = self.reasoning(drifted_name, drop=True)   # an undeclared name stays reasoning text
        self.assertIn('<function=bash:hermes>', "".join(e.text for e in events if e.kind == "reasoning"))
        self.assertEqual(p.dropped_unfinished, "")



if __name__ == "__main__":
    unittest.main()
