# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types
from google.adk.workflow import Workflow, START, node
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.agents.context import Context
from pydantic import BaseModel, Field

import os
import google.auth

try:
    _, project_id = google.auth.default()
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
except Exception:
    os.environ["GOOGLE_CLOUD_PROJECT"] = "mock-project-id"

os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"


class ExpenseClaim(BaseModel):
    item: str = Field(description="The name or description of the item being claimed.")
    amount: float = Field(description="The total amount of the expense claim.")
    employee_id: str = Field(description="The unique identifier of the employee submitting the claim.")


@node
def classify_expense(ctx: Context, node_input: types.Content | dict | ExpenseClaim) -> Event:
    """Classifies an expense claim based on the amount."""
    if "claim" in ctx.state:
        claim = ExpenseClaim(**ctx.state["claim"])
    else:
        if isinstance(node_input, dict):
            claim = ExpenseClaim(**node_input)
        elif isinstance(node_input, ExpenseClaim):
            claim = node_input
        else:
            text = node_input.parts[0].text.strip()
            data = json.loads(text)
            claim = ExpenseClaim(**data)

    route = "auto" if claim.amount < 100.0 else "review"
    return Event(output=claim.model_dump(), route=route, state={"claim": claim.model_dump()})


@node
def auto_approve(node_input: dict) -> Event:
    """Automatically approves claims under $100."""
    claim = ExpenseClaim(**node_input)
    msg = f"Expense of ${claim.amount:.2f} for '{claim.item}' has been automatically approved."
    return Event(
        output={"status": "approved", "reason": "Under $100 auto-approval threshold", "amount": claim.amount},
        content=types.Content(role='model', parts=[types.Part.from_text(text=msg)])
    )


@node(rerun_on_resume=True)
async def review_agent(ctx: Context, node_input: dict):
    """Triggers a human-in-the-loop pause for manual review of expenses $100 or more."""
    claim = ExpenseClaim(**node_input)
    interrupt_id = f"review_expense_{ctx.session.id}"
    if not ctx.resume_inputs or interrupt_id not in ctx.resume_inputs:
        msg = f"Expense of ${claim.amount:.2f} for '{claim.item}' requires manual approval. Please approve or deny this expense."
        yield RequestInput(interrupt_id=interrupt_id, message=msg)
        return

    decision = ctx.resume_inputs[interrupt_id]
    if isinstance(decision, dict) and "response" in decision:
        decision_str = str(decision["response"]).lower().strip()
    else:
        decision_str = str(decision).lower().strip()

    if "approve" in decision_str or "yes" in decision_str:
        status = "approved"
        msg = f"Expense of ${claim.amount:.2f} for '{claim.item}' has been manually approved."
    else:
        status = "denied"
        msg = f"Expense of ${claim.amount:.2f} for '{claim.item}' has been denied."

    yield Event(
        output={"status": status, "reason": f"Manual review: {decision_str}", "amount": claim.amount},
        content=types.Content(role='model', parts=[types.Part.from_text(text=msg)])
    )


root_agent = Workflow(
    name="expense_approver",
    edges=[
        (START, classify_expense),
        (classify_expense, {"auto": auto_approve, "review": review_agent}),
    ],
)

app = App(
    root_agent=root_agent,
    name="app",
)
