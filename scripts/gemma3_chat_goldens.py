"""Record Gemma 3 chat goldens for the restricted renderer (ADR-012, Gemma).

Prompts are rendered with Jinja2 configured the way Hugging Face applies
chat templates (sandboxed, trim_blocks + lstrip_blocks, loopcontrols,
`raise_exception`), from the template bytes declared by each pinned GGUF.
Token ids come from the pinned llama.cpp `llama-tokenize` tool, not from
Chine's encoder, so the goldens are independent of the code they test.
Gemma's tokenizer adds the model-default BOS itself, and the template
emits the `<bos>` spelling, so the spelling is stripped before
tokenizing — every golden carries exactly one BOS (id 2), which is what
the served prompt must contain too.

Usage (no network; run in the mlx venv, which provides jinja2):
  .deps/mlx-venv/bin/python -I scripts/gemma3_chat_goldens.py \
      artifacts-local/gemma-3-1b-it-Q8_0.gguf artifacts-local/gemma-3-1b-it-Q4_0.gguf \
      > crates/supervisor/tests/data/gemma3_chat_goldens.json
"""

import json
import os
import struct
import subprocess
import sys
import tempfile

from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZE = os.path.join(ROOT, ".deps", "llama", "llama-tokenize")
BOS = "<bos>"


def chat_template(path):
    """The declared tokenizer.chat_template bytes of a GGUF file."""
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with open(path, "rb") as f:
        read = f.read
        assert read(4) == b"GGUF"
        read(4)
        _, nkv = struct.unpack("<QQ", read(16))

        def string():
            (n,) = struct.unpack("<Q", read(8))
            return read(n)

        for _ in range(nkv):
            key = string().decode()
            (t,) = struct.unpack("<I", read(4))
            if t == 8:
                value = string()
                if key == "tokenizer.chat_template":
                    return value
            elif t == 9:
                (et,) = struct.unpack("<I", read(4))
                (n,) = struct.unpack("<Q", read(8))
                if et == 8:
                    for _ in range(n):
                        string()
                else:
                    f.seek(n * sizes[et], 1)
            else:
                f.seek(sizes[t], 1)
    raise SystemExit(f"{path}: no tokenizer.chat_template")


class TemplateError(Exception):
    pass


def raise_exception(message):
    raise TemplateError(message)


def render(template_bytes, messages):
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
    )
    env.globals["raise_exception"] = raise_exception
    template = env.from_string(template_bytes.decode("utf-8"))
    try:
        return (
            template.render(
                messages=messages, bos_token=BOS, add_generation_prompt=True
            ),
            None,
        )
    except TemplateError as e:
        return None, str(e)
    except Exception as e:  # messages[0] on an empty list, as the template does
        return None, str(e)


def tokenize(gguf, prompt):
    # llama-tokenize adds the model-default BOS itself; the template emits
    # the <bos> spelling, so strip it and let the reference supply id 2.
    assert prompt.startswith(BOS), prompt[:32]
    body = prompt[len(BOS):]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(body)
        name = f.name
    try:
        out = subprocess.run(
            [TOKENIZE, "-m", gguf, "-f", name],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    finally:
        os.unlink(name)
    ids = []
    for line in out.splitlines():
        line = line.strip()
        if " -> " in line and line.split(" -> ")[0].strip().isdigit():
            ids.append(int(line.split(" -> ")[0].strip()))
    return ids


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    template = None
    for path in argv[1:]:
        declared = chat_template(path)
        if template is None:
            template = declared
        elif declared != template:
            raise SystemExit(
                f"{path}: template bytes differ from the first artifact; "
                "the pins must be byte-identical"
            )
    gguf = argv[1]

    conversations = [
        ("single_user", [
            {"role": "user", "content": "Hello, world!"}]),
        ("with_system", [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"}]),
        ("capital_of_france", [
            {"role": "user", "content": "What is the capital of France? Answer in one word."}]),
        ("multi_turn_with_system", [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Name a red fruit."},
            {"role": "assistant", "content": "Apple."},
            {"role": "user", "content": "And a green one?"}]),
        ("multi_turn_no_system", [
            {"role": "user", "content": "One."},
            {"role": "assistant", "content": "Two."},
            {"role": "user", "content": "Three?"}]),
        ("trailing_assistant", [
            {"role": "user", "content": "Say hi."},
            {"role": "assistant", "content": "Hi there!"}]),
        ("content_trim", [
            {"role": "user", "content": "   padded content \n"}]),
        ("system_prefix_join", [
            {"role": "system", "content": "Be brief.\n"},
            {"role": "user", "content": "  2+2? "}]),
        ("unicode_content", [
            {"role": "user", "content": "Café in Zürich?"}]),
        ("empty_assistant_turn", [
            {"role": "user", "content": "Ping"},
            {"role": "assistant", "content": ""}]),
        ("long_multi_turn", [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
            {"role": "user", "content": "e"}]),
        ("system_only", [
            {"role": "system", "content": "Be brief."}]),
        ("error_empty_messages", []),
        ("error_assistant_first", [
            {"role": "assistant", "content": "Hi"}]),
        ("error_consecutive_users", [
            {"role": "user", "content": "One"},
            {"role": "user", "content": "Two"}]),
        ("error_system_mid_conversation", [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "system", "content": "Be brief."}]),
    ]

    out = {
        "source": "pinned Gemma 3 1B GGUF chat template "
        "(both pins carry byte-identical template bytes)",
        "templates": [
            {
                "artifact": "gemma-3-1b-it-Q8_0.gguf / gemma-3-1b-it-Q4_0.gguf",
                "bytes": len(template),
                "sha256": __import__("hashlib").sha256(template).hexdigest(),
            }
        ],
        "method": (
            "prompts rendered with Jinja2 3.1.6 as Hugging Face applies chat "
            "templates (ImmutableSandboxedEnvironment, trim_blocks, "
            "lstrip_blocks, loopcontrols, raise_exception), bos_token='<bos>', "
            "add_generation_prompt=True; token ids from pinned llama.cpp "
            "b11695 llama-tokenize, which adds the model-default BOS itself, "
            "so the template's leading <bos> spelling is stripped before "
            "tokenizing (exactly one id-2 BOS per prompt); refusals record "
            "the template's own raise message"
        ),
        "conversations": [],
    }

    for cid, messages in conversations:
        entry = {"id": cid, "messages": messages}
        prompt, error = render(template, messages)
        if error is not None:
            entry["error"] = error
        else:
            entry["prompt"] = prompt
            entry["tokens"] = tokenize(gguf, prompt)
        out["conversations"].append(entry)
        kind = "error" if error is not None else f"ok tokens={len(entry['tokens'])}"
        print(f"{cid} {kind}", file=sys.stderr)

    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main(sys.argv)
