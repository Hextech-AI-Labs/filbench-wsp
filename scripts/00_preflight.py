"""Pre-experiment preflight: verify implementation completeness and safety invariants.

Loads no model weights, runs no evaluation, writes no real checkpoints
(only tiny random tensors in a temp dir that is deleted immediately).
Usage (repo root):  python scripts/00_preflight.py
Exit code 0 = all checks passed (warnings allowed), 1 = at least one failure.
"""
from __future__ import annotations

import inspect
import json
import py_compile
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS: list[tuple[str, str, str]] = []


def check(name: str, fn) -> None:
    try:
        detail = fn()
        RESULTS.append(("PASS", name, str(detail) if detail else ""))
    except Exception as exc:
        RESULTS.append(("FAIL", name, f"{type(exc).__name__}: {exc}"))


def warn(name: str, detail: str) -> None:
    RESULTS.append(("WARN", name, detail))


def main() -> int:
    src = REPO / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    required = [
        "scripts/00_preflight.py",
        "scripts/06_language_diagnostics.py",
        "scripts/07_counterfactual_controls.py",
        "scripts/08_screen_plans.py",
        "tests/test_tied_rows.py",
        "src/filbench_wsp/editing/schema.py",
        "src/filbench_wsp/editing/applier.py",
        "src/filbench_wsp/editing/random_control.py",
        "src/filbench_wsp/proposer.py",
        "src/filbench_wsp/eval_adapter.py",
        "configs/eval_frozen.yaml",
        "docs/04_pilot_retrospective_and_pivot.md",
        "docs/05_implementation_status.md",
    ]

    def files_present():
        missing = [p for p in required if not (REPO / p).is_file()]
        if missing:
            raise FileNotFoundError(f"missing {missing}")
        return f"{len(required)} files present"

    check("required files present", files_present)

    def compile_all():
        files = [p for d in ("src", "scripts", "tests")
                 for p in (REPO / d).rglob("*.py")
                 if "__pycache__" not in str(p) and not p.name.endswith(".bak")]
        for p in files:
            py_compile.compile(str(p), doraise=True)
        return f"{len(files)} python files compile"

    check("all python sources compile", compile_all)

    def env():
        import torch, safetensors, transformers, yaml  # noqa: F401
        return (f"torch {torch.__version__}, transformers {transformers.__version__}, "
                f"cuda_available={torch.cuda.is_available()}")

    check("runtime deps importable", env)

    def op_registered():
        from filbench_wsp.editing.schema import VALID_OP_TYPES, EditBudget
        assert "scale_tied_rows" in VALID_OP_TYPES, "op not in VALID_OP_TYPES"
        b = EditBudget()
        assert b.allow_tied_embedding_rows is False, "tied op must default OFF"
        assert b.max_tied_rows == 512
        return "registered; default OFF; max_tied_rows=512"

    check("new operator registered", op_registered)

    def api_surface():
        from filbench_wsp.editing.applier import (  # noqa: F401
            compute_and_validate_tied_pair_edit, compute_and_validate_tensor_edit,
            apply_edit_plan_to_checkpoint,
        )
        from filbench_wsp.proposer import ingest_chatbot_response_and_materialize
        params = inspect.signature(ingest_chatbot_response_and_materialize).parameters
        assert "allow_tied_rows" in params, "ingest missing allow_tied_rows kwarg"
        assert params["allow_tied_rows"].default is False
        return "tied-pair applier + ingest kwarg present"

    check("new API surface present", api_surface)

    def backward_compat():
        from filbench_wsp.editing.schema import EditPlan
        bundles = sorted((REPO / "configs" / "plans").glob("*/edit_bundle.json"))
        if not bundles:
            raise FileNotFoundError("no committed edit bundles found")
        ids = []
        for b in bundles:
            d = json.loads(b.read_text(encoding="utf-8"))
            if d.get("status") != "accepted":
                continue
            plan = EditPlan.from_dict(d["plan"])
            assert plan.budget.allow_tied_embedding_rows is False
            ids.append(plan.plan_id)
        if not ids:
            raise ValueError("no accepted bundles to re-validate")
        return f"{len(ids)} old plans still validate"

    check("old plans still validate", backward_compat)

    def tied_smoke():
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
        from filbench_wsp.editing.applier import (
            apply_edit_plan_to_checkpoint, compute_and_validate_tied_pair_edit,
        )
        from filbench_wsp.editing.schema import EditBudget, EditOp, EditPlan, EditValidationError

        g = torch.Generator().manual_seed(0)
        embed = torch.randn(64, 16, generator=g).to(torch.bfloat16)
        op = EditOp(op_type="scale_tied_rows", tensor_name="model.embed_tokens.weight",
                    delta_scale=0.05, row_indices=[1, 5, 9])
        budget = EditBudget(allow_tied_embedding_rows=True)

        e, _, l, _ = compute_and_validate_tied_pair_edit(
            embed, embed.clone(), [op], budget, "llm_edited")
        assert torch.equal(e, l), "tie broken (llm_edited)"
        assert not torch.equal(e, embed), "edit was a no-op"
        er, _, lr, _ = compute_and_validate_tied_pair_edit(
            embed, embed.clone(), [op], budget, "random_control", random_seed=7)
        assert torch.equal(er, lr), "tie broken (random_control)"

        try:
            EditPlan.from_dict({"plan_id": "x", "target_model": "t", "hypothesis": "h",
                                "ops": [op.to_dict()]})
        except EditValidationError:
            pass
        else:
            raise AssertionError("tied op accepted while flag OFF")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = root / "base"
            base.mkdir()
            save_file({"model.embed_tokens.weight": embed.clone(),
                       "lm_head.weight": embed.clone()},
                      str(base / "model.safetensors"))
            (base / "config.json").write_text(
                json.dumps({"tie_word_embeddings": True, "num_hidden_layers": 0}),
                encoding="utf-8")

            ordinary = EditPlan.from_dict({
                "plan_id": "x", "target_model": "t", "hypothesis": "h",
                "ops": [{"op_type": "scale_tensor",
                         "tensor_name": "model.embed_tokens.weight", "delta_scale": 0.02}],
                "budget": {"allow_embeddings_and_head": True}})
            try:
                apply_edit_plan_to_checkpoint(ordinary, base, root / "out_bad")
            except EditValidationError:
                pass
            else:
                raise AssertionError("ordinary embedding edit accepted on a tied checkpoint")

            tied = EditPlan.from_dict({
                "plan_id": "ok", "target_model": "t", "hypothesis": "h",
                "ops": [op.to_dict()],
                "budget": {"allow_tied_embedding_rows": True}})
            out = root / "out_ok"
            rep = apply_edit_plan_to_checkpoint(tied, base, out)
            assert rep.num_edited_tensors == 2, f"expected 2 edited tensors, got {rep.num_edited_tensors}"
            with safe_open(out / "model.safetensors", framework="pt", device="cpu") as f:
                e2, l2 = f.get_tensor("model.embed_tokens.weight"), f.get_tensor("lm_head.weight")
            assert torch.equal(e2, l2), "saved copies diverged"
            assert not torch.equal(e2, embed), "saved edit was a no-op"
        return "tie preserved (edited+random); fail-closed rejects; end-to-end apply OK"

    check("tied-row safety smoke test", tied_smoke)

    def cli_flags():
        s05 = (REPO / "scripts/05_propose_and_apply_edit.py").read_text(encoding="utf-8")
        s06 = (REPO / "scripts/06_language_diagnostics.py").read_text(encoding="utf-8")
        s08 = (REPO / "scripts/08_screen_plans.py").read_text(encoding="utf-8")
        for label, text in (("05", s05), ("06", s06), ("08", s08)):
            assert "--allow-tied-rows" in text, f"script {label} missing --allow-tied-rows"
        assert "allow_tied_rows=args.allow_tied_rows" in s05, "05 does not forward the flag"
        assert "allow_tied_rows=args.allow_tied_rows" in s06, "06 does not forward the flag"
        return "05/06/08 expose and forward the flag"

    check("CLI flags wired", cli_flags)

    def batch_patch():
        start = re.compile(r"^def build_lighteval_model_args\(", flags=re.MULTILINE)
        hosts = []
        for d in ("scripts", "src"):
            root = REPO / d
            if not root.is_dir():
                continue
            for p in sorted(root.rglob("*.py")):
                if "__pycache__" in str(p) or p.name.endswith(".bak"):
                    continue
                try:
                    text = p.read_text(encoding="utf-8")
                except Exception:
                    continue
                if start.search(text):
                    hosts.append((p, text))
        if not hosts:
            raise AssertionError("build_lighteval_model_args not found in scripts/ or src/")
        patched = [p for p, t in hosts
                   if "batch_size={batch_size}" in t
                   and re.search(r"model_args\.get\(\s*['\"]batch_size['\"]", t)]
        if not patched:
            where = ", ".join(p.relative_to(REPO).as_posix() for p, _ in hosts)
            raise AssertionError(
                f"batch_size patch missing in {where} -> run: python scripts/00_patch_eval_batch_size.py")
        return "configurable batch size in " + ", ".join(
            p.relative_to(REPO).as_posix() for p in patched)

    check("eval batch-size patch", batch_patch)

    def batch_yaml():
        import yaml
        cfg = yaml.safe_load((REPO / "configs/eval_frozen.yaml").read_text(encoding="utf-8"))
        bs = (cfg.get("model_args") or {}).get("batch_size")
        if bs is None:
            warn("eval_frozen batch_size", "not set; defaults to 1 (slow). Add batch_size: 4 under model_args.")
            return "not set (default 1)"
        if int(bs) < 1:
            raise ValueError(f"batch_size must be >= 1, got {bs}")
        return f"batch_size={bs}"

    check("eval_frozen batch_size value", batch_yaml)

    if not (REPO / "external" / "filbench-eval" / "lighteval").is_dir():
        warn("upstream eval checkout", "external/filbench-eval/lighteval missing; needed for scripts/03 later")
    base_dirs = sorted(p for p in (REPO / "checkpoints" / "base").glob("*") if p.is_dir()) \
        if (REPO / "checkpoints" / "base").is_dir() else []
    if not base_dirs:
        warn("base checkpoints", "none under checkpoints/base; restore from your archive before diagnostics/eval")
    else:
        RESULTS.append(("PASS", "base checkpoints present", ", ".join(p.name for p in base_dirs)))

    width = max(len(n) for _, n, _ in RESULTS)
    for status, name, detail in RESULTS:
        print(f"[{status}] {name.ljust(width)}  {detail}")
    n_pass = sum(1 for s, _, _ in RESULTS if s == "PASS")
    n_warn = sum(1 for s, _, _ in RESULTS if s == "WARN")
    n_fail = sum(1 for s, _, _ in RESULTS if s == "FAIL")
    print(f"\nSummary: {n_pass} passed, {n_warn} warnings, {n_fail} failed")
    if n_fail:
        print("Fix the FAIL lines before running any experiment.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())