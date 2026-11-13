# FramedFacts

Code and data for *Research Agents Feed Doubts, Not Beliefs: Auditing Belief Framing in
Live-Web Search*, by Siddharth Vohra and Min Xu. Published as a Spotlight paper in the
Proceedings of the Second Workshop for Research on Agent Language Models (REALM 2026) at
EMNLP 2026.

- Paper: https://aclanthology.org/2026.realm-1.45/
- Project page: https://siddvoh.com/framedfacts/

Forty-eight expert-verified health claims went to four search-equipped agents. The only
thing that changed between conditions was one sentence in the request: whether the user
said they believed the claim, doubted it, or said nothing. The search tool ran from a
frozen cache, so the same query always came back with the same eight results.

Saying you believe a claim changed nothing measurable. Saying you doubt it moved the
agent's conclusion 0.26 points toward rejection on a five-point scale, and it more than
doubled the share of searches that went looking for reasons to deny the claim. A separate
crossed-replay arm splits the believer-to-skeptic gap into how the agent reads what came
back and which results came back at all, and puts about three quarters of it on the
reading. The paper has the design, the intervals and the arms this summary leaves out.

## Rebuilding the analysis

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r code/requirements.txt
.venv/bin/python code/analysis.py --audit-record data/audits/latest.json \
    --luna-call-slots data/luna_call_slots.json --output rebuilt
.venv/bin/python code/figures.py --analysis rebuilt --output-dir rebuilt/figures
```

`code/analysis.py` reads the 2,756 run manifests in `data/runs/` and writes `values.json`,
`diagnostics.json` and the 29 tables of `data/analysis/` to the output directory, in about
a minute. `code/figures.py` draws the three figures from them.

The proportional-odds fit and the bootstrap resamples go through scipy and statsmodels,
whose iterative solvers do not land in exactly the same place from release to release or
from machine to machine. Rebuilt elsewhere, around a hundred of the 10,663 values move.
Nearly all of them sit inside `models.proportional_odds_terms`, where the largest is a
p-value that shifts by about one part in ten thousand; the rest differ only in the last
few digits a double can hold. None of them reaches a number the paper prints.

## What is here

```
code/                              the harness, the analysis, and the figure script
code/requirements.txt              pinned versions
data/claims/framedfacts48.json     the 48 claims with verdicts, tiers and verification sources
data/claims/selection_manifest.json  how the 48 were drawn from the two source datasets
data/claims/source_metadata.json   the two upstream datasets, their licenses and label maps
data/prompts.json                  every prompt template and judge rubric, verbatim
data/parity_table.txt              sampling settings per model and where they differed
data/model_params.json             the requested sampling parameters
data/model_params_v5.json          the same, for the fifth pilot generation onward
data/schedule.json                 the registered schedule, pilot cohorts included
data/schedule_phase2.json          the five arms added after the first review round
data/schedule_trims/step_1.json    the one recorded change to the schedule
data/pilot_gate.json               the gate decision that authorised that change
data/runs/                         one manifest per run, 2,756 of them
data/analysis/                     the measured tables, values.json and diagnostics.json
data/figures/                      the three figures
data/audits/latest.json            the audit record the analysis binds to
data/luna_sample.json              the frozen cross-family judge sample
data/luna_call_slots.json          the terminal status of the two sample members never judged
data/query_sample.json             the frozen query sample the second judge relabelled
data/model_catalog_2026-08-01.json the six catalog entries the parity table is read from
data/source_manifest.json          the upstream dataset releases and their hashes
```

A run manifest records the arm, framing, model and claim; the queries the agent issued in
order and each query's stance label; every result it was shown as a title and a URL; the
report it wrote; the URLs it cited; and the judges' labels. The 112 runs in the retrieved
evidence audit also carry a stance label per result.

The schedule holds 3,314 rows against 2,756 manifests. Three hundred and twenty-two of the
missing rows are pilot runs, which were scheduled and are not released. The other 236 sit
in four arms that stopped short: two hit a hard run cap, and two were the scholarly repair
windows, abandoned once the failure was diagnosed. The paper's run accounting reports
planned against attempted for every arm.

`data/schedule_trims/step_1.json` is the one change made to the frozen schedule, recorded
before any analysis: it dropped GLM's second repeat from the core grid and added the 144
Kimi probe runs, on the authority of the pilot gate. That gate decision is released as
`data/pilot_gate.json`: eight checks, whether each passed, and the decision to include the
Kimi probe. The hash the trim record cites is of the gate file
as it stood a few minutes earlier, before the gate rewrote itself to record that the trim
had happened, so the file shows the decision but does not match that hash.

Two things are not here. Retrieved snippets are third-party page text, so a result ships as
a title and a URL and nothing else. Provider call records, the search cache and the
per-call ledger are not released either: they carry credentials and local paths, and no
number in the paper rests on them. Two members of the frozen cross-family judge sample were
never judged, because their call failed in transport; `data/luna_call_slots.json` records
that outcome so the analysis can close the sample without reading the ledger.

## Running the experiment rather than the analysis

`code/run_experiments.py` drives the agents, `code/search.py` the cached search tool,
`code/judge.py` the labeling, and `code/schedule.py` builds the frozen schedule. They read
and write an `archive/` and a `working/` directory that are not part of this release, and
they need a `.env`. `code/config.py` refuses to start without the model gateway's API key;
the search backends read their own keys where they use them. These scripts are here because
they are how the data in `data/runs/` was produced, not because they will run from a fresh
clone.

A released run manifest is a projection of the working one. It keeps the arm, framing,
model, claim, repeat and status; the queries and their stance labels; the retrieved
results as a title and a URL; the report; the cited URLs; the judges' labels; and the
structured failure record where a run failed, along with when the run started and
finished. It drops the provider call records, and with them every per-call timing and usage
figure, the retrieved snippets and the judge call payloads.

## Licenses

The code is MIT (see `LICENSE`).

The claims come from two public datasets and keep their own terms. PUBHEALTH is MIT
(`PUBHEALTH_LICENSE.txt`). HealthFC is CC BY-NC-ND 4.0
(`HEALTHFC_LICENSE_AND_ATTRIBUTION.txt`); we redistribute claim text and labels only, with
attribution, and every claim in `framedfacts48.json` carries its source, its license and a
current verification URL.

Agent outputs in `data/runs/` were generated for research. They are not health advice, and
some of them confidently contradict a verified verdict. That is what the paper is about.

## Citing

```bibtex
@inproceedings{vohra-xu-2026-research,
  title     = {Research Agents Feed Doubts, Not Beliefs: Auditing Belief Framing in Live-Web Search},
  author    = {Vohra, Siddharth and Xu, Min},
  booktitle = {Proceedings of the Second Workshop for {RE}search on Agent Language Models ({REALM} 2026)},
  month     = oct,
  year      = {2026},
  address   = {Budapest, Hungary},
  publisher = {Association for Computational Linguistics},
  pages     = {664--674},
  doi       = {10.18653/2026.realm-1.45},
  url       = {https://aclanthology.org/2026.realm-1.45/}
}
```

Siddharth Vohra, svohra@andrew.cmu.edu
