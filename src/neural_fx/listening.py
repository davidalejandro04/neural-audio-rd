"""Offline ABX stimuli and descriptive, listener-level response analysis."""
import csv
import random
import shutil
from pathlib import Path

from scipy.stats import binomtest

from .utils import read_json, sha256, write_json


def package_abx(evaluation, output, seed=1337):
    evaluation, output = Path(evaluation), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"ABX directory already exists: {output}")
    examples = sorted((evaluation / "audio").glob("*"))
    if not examples:
        raise ValueError("No evaluated audio examples found")
    rng = random.Random(seed)
    rng.shuffle(examples)
    answers = []
    for index, example in enumerate(examples):
        trial = f"trial_{index + 1:03d}"
        destination = output / "public" / trial
        destination.mkdir(parents=True)
        chosen = rng.choice(("A", "B"))
        for name, source in {"A": example / "target.wav", "B": example / "model.wav",
                             "X": example / ("target.wav" if chosen == "A" else "model.wav")}.items():
            shutil.copyfile(source, destination / f"{name}.wav")
        answers.append({"trial_id": trial, "answer": chosen, "pair_id": example.name,
                        "target_sha256": sha256(example / "target.wav"), "model_sha256": sha256(example / "model.wav")})
    write_json(output / "private" / "answers.json", {"seed": seed, "condition": "original_level", "trials": answers})
    (output / "public" / "INSTRUCTIONS.txt").write_text(
        "Listen to A, B and X using the same playback gain. Decide whether X matches A or B.\n"
        "Record one answer per trial in responses.csv. Do not inspect private answers.\n"
        "This package preserves original levels; it is not a loudness-matched condition.\n", encoding="utf-8")
    with (output / "public" / "responses.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("listener_id", "trial_id", "choice"))
        writer.writerows(("", item["trial_id"], "") for item in answers)
    return {"trials": len(answers), "public": str(output / "public"), "private": str(output / "private")}


def score_abx(answers_path, responses, output):
    key = {trial["trial_id"]: trial["answer"] for trial in read_json(answers_path)["trials"]}
    listeners, seen = {}, set()
    with Path(responses).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            listener, trial, choice = row["listener_id"].strip(), row["trial_id"].strip(), row["choice"].strip().upper()
            if not listener or trial not in key or choice not in ("A", "B"):
                raise ValueError("Each response needs a listener_id, known trial_id, and choice A or B")
            if (listener, trial) in seen:
                raise ValueError("Duplicate response from one listener on a trial")
            seen.add((listener, trial))
            listeners.setdefault(listener, []).append(choice == key[trial])
    if not listeners:
        raise ValueError("No responses")
    scores = []
    for listener, correctness in sorted(listeners.items()):
        successes, total = sum(correctness), len(correctness)
        test = binomtest(successes, total, 0.5, alternative="two-sided")
        interval = test.proportion_ci(confidence_level=0.95, method="wilson")
        scores.append({"listener_id": listener, "correct": successes, "trials": total,
                       "accuracy": successes / total, "binomial_p": test.pvalue,
                       "wilson_95": [interval.low, interval.high]})
    result = {"listeners": scores, "analysis": "descriptive per-listener exact binomial and Wilson interval; no pooled repeated-measures inference",
              "interpretation": "Failure to reject chance does not establish indistinguishability. Confirmatory group analysis requires the preregistered listener/source clustered method."}
    write_json(output, result)
    return result
