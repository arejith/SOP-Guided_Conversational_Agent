# SOP-guided insurance claims support

A Python, LangGraph, OpenAI, and Streamlit demo using fictional insurance fixtures.
Python controls identity checks, representative authorization, claim ownership,
and transitions through `VERIFY_ID → RESOLVE_INTENT → PROCESS_CASE → POST_PROCESS`.
The model interprets customer language and writes answers using authorized records.

## Run locally

Use Python 3.12 from the repository directory:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Then install and launch:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py --browser.gatherUsageStats=false
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).
Enter an OpenAI API key into the password field in the sidebar, confirm the model,
and select **Connect**. `gpt-4.1-mini` remains the default. The selected model must
support the Responses API and strict structured outputs.

Alternatively, copy `.env.example` to `.env` and edit it locally, or set
`OPENAI_API_KEY` and `OPENAI_MODEL` in the server environment. Never paste a real
key into chat, source code, or a commit. Environment variables take precedence
over `.env`; a session-entered key overrides the server default for that session.
`.env` is read without modifying the process environment.

Each Streamlit session owns its client, graph, authorization simulator, and
conversation. Changing the key or model closes the old client and clears the
conversation; select **Connect** again. **New conversation** closes the client
and clears the conversation, while keeping the sidebar inputs available.
No credential is written to a file by the UI. Session data is not persisted by
this application across server restarts.

## Docker setup

Docker is an optional alternative; a Docker engine must be installed and running.

```bash
docker build -t sop-claims-demo .
docker run --rm -p 127.0.0.1:8501:8501 sop-claims-demo
```

Enter the key in the sidebar, or provide a local environment file at runtime:

```bash
docker run --rm -p 127.0.0.1:8501:8501 --env-file .env sop-claims-demo
```

The image runs as a non-root user and copies only application code and fixtures.
Keys, `.env`, tests, and the local virtual environment are excluded from the image.
The supplied Dockerfile was not built in the review environment because Docker
was unavailable. The local Python environment, application tests, and dependency
consistency were checked instead.

## Demonstrate all four phases

Use fictional Margaret Chen fixture details in the UI:

1. **VERIFY_ID:** “I'm the policyholder, Margaret Chen. My DOB is March 15, 1985.”
   The agent requests another permitted field and discloses no claim information.
2. **RESOLVE_INTENT:** “My SSN last four is 4472.” With three matching fields,
   the agent asks what help is needed.
3. **PROCESS_CASE:** “Why was my January 2026 healthcare claim denied?”
   The record lists a missing pathology report and treating provider office note.
   Ask follow-ups, such as “Can you confirm the documents arrived?” The fixture
   cannot establish a new receipt. The discussion stays in this phase.
4. **POST_PROCESS:** “That's everything, thanks.” The agent offers a summary.
   Choose “Yes, send the email summary” or “Skip the summary.” A send choice
   generates a labelled preview; no email is sent.

For an acceptance-style combined opening, use:

> I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling
> about my denied healthcare claim from January. DOB is 1985-03-15, SSN last
> four is 4472.

The first two phases can complete within one turn when enough information is
supplied. The recorded appeal deadline is March 18, 2026. Once that date has
passed, neither a new deadline nor a rule forbidding late appeals may be inferred.

For the representative extension, start a new conversation:

> I am David Chen, Margaret Chen's son, calling for her about her denied January
> healthcare claim. Her policy is POL-9921. Her DOB is March 15, 1985, and her
> SSN last four is 4472.

The default simulated authorization is pending on the first check, approved on
the next. Say “Check authorization again.” The remembered claim request should
then continue. The fixture scenarios also include `denied` and `timeout`, tested
deterministically. A customer's statement of permission never supplies approval.
Scenario selection is server-controlled, not a customer command.

An offline controller demonstration is available without any API credential:

```bash
python -m tests.demo_workflow
```

It runs the actual graph and fixtures with explicitly scripted model decisions.
It demonstrates all phases and preview delivery, but is not evidence of live
language understanding. See [the recorded transcript](demo-transcript.txt).

## Tests and review

```bash
python -m pytest -q
python -m ruff check agent services app.py tests
python -m ruff format --check agent services app.py tests
python -m pip check
```

The default suite excludes all live-model tests. It exercises the actual graph
using scripted semantic decisions and private-service spies, plus extraction
validation, identity matching, authorization outcomes, rollback, consent, and
Streamlit session lifecycle checks. Scripted tests deliberately do not pretend
to verify the model's interpretation or generated prose.

Optional live tests incur OpenAI API usage and read the configured key locally:

```bash
python -m pytest -m live tests/test_live_semantics.py -q
python -m pytest -m live tests/test_existing.py -q
```

They skip when no key is configured. No API credential was available during this
review, so live interpretation, answer quality, and API connectivity remain to
be checked before presenting the live demo. See [review and validation notes](docs/review.md).

## Scope and limitations

- Verification requires three distinct matching fields from name, DOB, phone,
  email, and SSN last four, belonging to one record. Policy numbers only narrow
  lookup; the fixtures' national-ID fields do not count as SSN.
- General verification and document-preparation guidance is available before
  verification through an explicit allowlist. Private claim service calls are gated.
- Identity changes invalidate access. Changed names or policy references require
  collecting the new identity; formatting-only changes do not. Ambiguous caller
  roles require clarification.
- Summary consent and preview status are separate. Supplying an email address
  is not consent. Email, authorization, and human transfer are simulated; the app
  cannot upload documents, edit claims, contact a policyholder, or send email.
- This is a fixture-based demonstration, not production identity proofing or
  a claim of regulatory compliance. Extraction and generated answers still
  depend on model interpretation. Use only fictional customer information.
- Runtime API requests use `store=False`; this does not imply that provider-side
  retention is universally disabled. Review provider data controls for any real deployment.
