# Inbound Pharmacy Sales Agent

**Muhammad Tameem Rafay** · rafaytariq369616@gmail.com

A text-based simulation of an AI-powered inbound sales agent for pharmacies
calling **TJM Labs**. On "call start" it identifies the pharmacy from the
caller's phone number, greets them, understands intent, pitches how TJM Labs
supports high-Rx-volume pharmacies, and arranges a follow-up (email or callback).

Built with **LangGraph** and **Claude**, with a **deterministic fallback** so it
runs even without an API key.

> See [DESIGN.md](DESIGN.md) for the design write-up, tradeoffs, and the
> "3 more hours" answer.

---

## Features

- **Pharmacy identification** via the directory API, matched on the caller's
  phone number (formatting-insensitive).
- **Recognized callers** are greeted by name with their location and Rx volume.
- **Unrecognized callers** are asked for their pharmacy name and Rx volume
  conversationally; the name is used throughout once known.
- **Value pitch** tailored to Rx volume (high-volume pharmacies get the
  high-volume benefits).
- **Safe by default** — out-of-scope questions are declined without hallucinating.
- **Cache-friendly prompt** — static, XML-structured instructions form a cached
  prefix (Anthropic `cache_control`); only the per-call caller context varies.
- **Mocked follow-up tools** — `send_email_followup` and `schedule_callback` log
  and print (no real emails/calls).
- **Logging** throughout, and a **pytest** suite.

---

## Requirements

- Python 3.9+
- Internet access (to reach the pharmacy directory API)
- *(Optional)* an `ANTHROPIC_API_KEY` to enable the Claude-powered path

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # optional; edit to add your API key
```

Without an API key the agent runs in **deterministic fallback mode** (fully
functional, no LLM). Add `ANTHROPIC_API_KEY` to `.env` to enable Claude.

---

## Run

```bash
python main.py
```

Simulate a **recognized** caller (default) vs. an **unrecognized** one:

```bash
# Recognized (on file)
python main.py --phone "+1-555-123-4567"

# Unrecognized (not on file → collects name + Rx volume)
python main.py --phone "+1-555-000-0000"
```

Type replies at the `You:` prompt. Type `quit` (or `exit`/`bye`) to hang up.

### Example (recognized caller)

```
Agent: Hi HealthFirst Pharmacy, thanks for calling TJM Labs! Great to hear
          from you over in New York, NY. I see you're filling around 100
          prescriptions a month. How can I help you today?
You: tell me more
Agent: HealthFirst Pharmacy, since you're running a high prescription volume,
          TJM Labs is a strong fit... Would you like me to email you the details
          or schedule a callback?
You: please email me the details

[MOCK EMAIL SENT]
    To:       contact@healthfirst.com
    Pharmacy: HealthFirst Pharmacy
    Summary:  Overview of how TJM Labs supports high-Rx-volume pharmacies.
```

---

## Configuration

All settings are environment variables (see [.env.example](.env.example)):

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(unset)* | Enables the Claude path; unset → fallback |
| `MODEL_NAME` | `claude-sonnet-5` | Claude model id |
| `PHARMACY_API_URL` | mockapi.io endpoint | Directory API |
| `HIGH_RX_VOLUME_THRESHOLD` | `100` | Monthly Rx count considered "high volume" |
| `MOCK_CALLER_PHONE` | `+1-555-123-4567` | Default caller ID for the simulation |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Tests

```bash
pytest -q
```

Covers Rx-volume logic, phone normalization/identification (incl. graceful API
failure), the mock tools, and end-to-end graph behavior on the fallback path.

---

## Project layout

```
pharmacy_agent/
├── config.py          # settings from env
├── logging_config.py  # logging setup
├── models.py          # Pharmacy / Prescription + Rx-volume logic
├── pharmacy_api.py     # directory API client + identification
├── llm.py             # Claude accessor + availability check
├── prompts.py         # grounded system prompt + TJM value prop
├── tools.py           # mocked email / callback tools
├── fallback.py        # deterministic no-LLM engine
├── state.py           # LangGraph state
├── graph.py           # the conversation graph
└── cli.py             # text REPL
main.py                # entry point: `python main.py`
conftest.py            # pytest path setup
tests/                 # pytest suite
```
