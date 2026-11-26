import string
import unittest

from tokenizers import AddedToken, Tokenizer, models, processors
from transformers import PreTrainedTokenizerFast

from sglang.srt.entrypoints.openai import encoding_dsv41 as dsv41
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDSV41SegmentedEncoding(unittest.TestCase):
    def setUp(self):
        try:
            from fastokens import Tokenizer as FastTokenizer
            from fastokens import _ConfiguredTokenizerShim
        except ImportError:
            self.skipTest("fastokens is not installed")
        if not hasattr(FastTokenizer, "encode_segments"):
            self.skipTest("fastokens does not support segmented encoding")
        self.shim = _ConfiguredTokenizerShim
        alphabet = sorted(set(string.printable + "中文👩🏽‍💻"))
        vocab = {char: index for index, char in enumerate(alphabet)}
        vocab["ab"] = len(vocab)
        self.backend = Tokenizer(models.BPE(vocab=vocab, merges=[("a", "b")]))
        self.backend.add_special_tokens(
            [
                dsv41.bos_token,
                dsv41.eos_token,
                dsv41.USER_SP_TOKEN,
                dsv41.SYSTEM_SP_TOKEN,
                dsv41.ASSISTANT_SP_TOKEN,
                dsv41.LATEST_REMINDER_SP_TOKEN,
                dsv41.thinking_start_token,
                dsv41.thinking_end_token,
                dsv41.DS_TASK_SP_TOKENS["action"],
                "<image>",
            ]
        )

    def make_tokenizers(self):
        reference = PreTrainedTokenizerFast(tokenizer_object=self.backend)
        fast = PreTrainedTokenizerFast(tokenizer_object=self.backend)
        fast._tokenizer = self.shim(fast.backend_tokenizer)
        return reference, fast

    def test_rendered_prompt_parity(self):
        reference, fast = self.make_tokenizers()
        segmented = dsv41.load_segmented_tokenizer(fast)
        self.assertIsNotNone(segmented)
        messages = [
            {"role": "system", "content": "a"},
            {"role": "assistant", "content": "b", "wo_eos": True},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "中文 👩🏽‍💻 <think>ab</think>"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/i"},
                    },
                ],
                "task": "action",
            },
        ]
        for mode in ("chat", "thinking"):
            with self.subTest(mode=mode):
                parts, media = dsv41.encode_messages(
                    messages,
                    thinking_mode=mode,
                    reasoning_effort=83,
                    return_multi_modal_data=True,
                    return_prompt_parts=True,
                )
                prompt = dsv41.encode_messages(
                    messages, thinking_mode=mode, reasoning_effort=83
                )
                self.assertEqual("".join(parts), prompt)
                self.assertEqual(media["images"], [{"url": "https://example.com/i"}])
                expected = reference.encode(
                    prompt.replace(dsv41.IMAGE_PLACEHOLDER, "<image>")
                )
                for backend in (None, segmented):
                    self.assertEqual(
                        dsv41.tokenize_prompt_parts(
                            parts, fast, backend, image_token="<image>"
                        ),
                        expected,
                    )

    def test_falls_back_for_context_dependent_added_tokens(self):
        self.backend.add_special_tokens([AddedToken("<trim>", lstrip=True)])
        reference, fast = self.make_tokenizers()
        self.assertIsNone(dsv41.load_segmented_tokenizer(fast))
        self.assertIsNone(dsv41.load_segmented_tokenizer(reference))

    def test_falls_back_when_post_processor_inserts_tokens(self):
        bos = dsv41.bos_token
        self.backend.post_processor = processors.TemplateProcessing(
            single=f"{bos} $A", special_tokens=[(bos, self.backend.token_to_id(bos))]
        )
        _, fast = self.make_tokenizers()
        self.assertIsNone(dsv41.load_segmented_tokenizer(fast))


if __name__ == "__main__":
    unittest.main()
