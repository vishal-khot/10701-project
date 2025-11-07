import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class Task:
    task_id: str
    prompt: str
    test: Optional[str]
    language: Optional[str]
    stop_tokens: List[str]

def batched(iterable, n: int):
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]

def cut_at_first_stop(text: str, stop_strs: List[str]) -> str:
    if not stop_strs:
        return text
    first = None
    for s in stop_strs:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1:
            first = idx if (first is None or idx < first) else first
    return text if first is None else text[:first]

_FENCE_OPEN = re.compile(r"^```(?:ocaml|ml)?\s*$", re.IGNORECASE | re.MULTILINE)
_FENCE_CLOSE = re.compile(r"^```\s*$", re.IGNORECASE | re.MULTILINE)

def clean_completion(text: str) -> str:
    """Remove markdown fences / leading prose the model might emit."""
    if not text:
        return ""
    t = text.strip()
    # Drop fenced code blocks if present
    t = _FENCE_OPEN.sub("", t)
    t = _FENCE_CLOSE.sub("", t)
    # If there's leading explanation, drop until first OCaml code-ish keyword
    m = re.search(r"(?m)^(let|type|module|open)\b", t)
    if m:
        t = t[m.start():]
    return t.strip()

def has_toplevel_directives(src: str) -> bool:
    return bool(re.search(r"(?m)^\s*#(use|load|directory)\b", src))

def run_ocaml_code(joined_source: str, timeout_sec: int = 20) -> Tuple[bool, str, int, str]:
    """
    If no toplevel directives and ocamlc exists -> compile+run (reliable exit codes).
    Else -> run in REPL; treat as FAIL if stderr shows failure markers.
    Returns (passed, exec_mode, return_code, stderr_head).
    """
    stderr_head = ""
    exec_mode = "repl"
    rc = 0

    if not has_toplevel_directives(joined_source) and shutil.which("ocamlc"):
        exec_mode = "compile"
        with tempfile.TemporaryDirectory() as td:
            main_ml = os.path.join(td, "main.ml")
            exe = os.path.join(td, "main.byte")
            with open(main_ml, "w", encoding="utf-8") as f:
                f.write(joined_source)

            comp = subprocess.run(["ocamlc", "-g", main_ml, "-o", exe],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_sec)
            if comp.returncode != 0:
                rc = comp.returncode
                stderr_head = (comp.stderr or b"").decode(errors="ignore").splitlines()[:1]
                stderr_head = stderr_head[0] if stderr_head else ""
                return False, exec_mode, rc, stderr_head

            runp = subprocess.run([exe], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_sec)
            rc = runp.returncode
            stderr_head = (runp.stderr or b"").decode(errors="ignore").splitlines()[:1]
            stderr_head = stderr_head[0] if stderr_head else ""
            return (rc == 0), exec_mode, rc, stderr_head

    # REPL path: exit code is 0 even on failure; infer from stderr
    p = subprocess.run(["ocaml"],
                       input=joined_source.encode("utf-8"),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_sec)
    err = (p.stderr or b"").decode(errors="ignore")
    stderr_head = err.splitlines()[:1]
    stderr_head = stderr_head[0] if stderr_head else ""
    failed = bool(re.search(r"Exception:|Assert_failure|Failure|Invalid_argument|Not_found|Match_failure", err))
    return (not failed), exec_mode, 0, stderr_head

def load_multipl_e_humaneval_ocaml(split: str = "test", revision: Optional[str] = None) -> List[Task]:
    ds = load_dataset("nuprl/MultiPL-E", name="humaneval-ml", split=split, revision=revision)
    tasks: List[Task] = []
    for ex in ds:
        task_id = str(ex.get("task_id", ex.get("name", "")))
        prompt = str(ex.get("prompt", ""))
        test = ex.get("test", ex.get("tests", None))
        language = ex.get("language", "ml")
        stop_tokens = ex.get("stop_tokens", []) or []
        tasks.append(Task(task_id, prompt, test, language, stop_tokens))
    return tasks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    torch.manual_seed(10701)

    # Load dataset
    tasks = load_multipl_e_humaneval_ocaml(split="test")
    total = len(tasks)

    print(f"Loading model: " + args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                dtype=torch.float16,
                device_map=device,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
    
    model = model.to(device)

    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    # ----- Generation -----
    print(f"Generating for {total} tasks...")
    completions: List[str] = [""] * total

    def encode_batch(prompts: List[str]):
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
        return {k: v.to(model.device) for k, v in enc.items()}

    for batch_idx, chunk in enumerate(batched(list(enumerate(tasks)), args.batch_size), 1):
        idxs = [i for i, _ in chunk]
        prompts = [t.prompt for _, t in chunk]
        inputs = encode_batch(prompts)

        with torch.inference_mode():
            gen = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for (i, t), full in zip(chunk, decoded):
            # Extract completion suffix relative to prompt
            if full.startswith(t.prompt):
                completion = full[len(t.prompt):]
            else:
                parts = full.split(t.prompt, 1)
                completion = parts[1] if len(parts) == 2 else full

            completion = cut_at_first_stop(completion, t.stop_tokens)
            completion = clean_completion(completion)
            completions[i] = completion

        done = idxs[-1] + 1
        print(f"[gen] batch {batch_idx}  items {idxs[0]}-{idxs[-1]}  ({done}/{total})")

    # ----- Testing -----
    print("Testing")
    records: List[dict] = []
    num_tests_passed = 0
    num_total_tests = 0
    with open('outputs/qwen3_humaneval_ocaml.jsonl', 'w') as f:
        for i, t in enumerate(tasks):
            completion = completions[i]
            # Append prompt + completion
            full_source = (t.prompt or "") + (("\n" + completion) if completion else "")
            tests_passed = None
            exec_mode = None
            return_code = None
            stderr_head = None
            ran_tests = False

            if t.test:
                ran_tests = True
                # Append tests
                joined = (full_source.strip() + "\n\n" + (t.test or "").strip() + "\n")
                passed, mode, rc, err_head = run_ocaml_code(joined)
                tests_passed = passed
                exec_mode = mode
                return_code = rc
                stderr_head = err_head

            rec = {
                "idx": i,
                "task_id": t.task_id,
                "language": t.language,
                "stop_tokens": t.stop_tokens,
                "prompt": t.prompt,
                "completion": completion,
                "full_source": full_source,
                "ran_tests": ran_tests,
                "tests_passed": tests_passed,
                "exec_mode": exec_mode,
                "return_code": return_code,
                "stderr_head": stderr_head,
            }

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records.append(rec)

            status = "PASS" if tests_passed else ("FAIL" if tests_passed is False else "GEN")
            print(f"[{i+1}/{total}] {t.task_id}  {status}")

            if tests_passed is True:
                num_tests_passed += 1
            num_total_tests += 1
    
    print("{} / {} tests passed.".format(num_tests_passed, num_total_tests))


if __name__ == "__main__":
    main()
