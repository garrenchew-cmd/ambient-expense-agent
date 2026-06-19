import asyncio
from app.agent import app
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

async def run_tests():
    # Initialize session service
    session_service = InMemorySessionService()
    
    # Test Case 1: Under $100 (Auto Approval)
    print("=== Test Case 1: Standard Meal Expense of $50 (Auto-Approval) ===")
    runner = Runner(agent=app.root_agent, session_service=session_service, app_name="app")
    session = session_service.create_session_sync(user_id="user_1", app_name="app")
    
    claim_msg = types.Content(
        role="user",
        parts=[types.Part.from_text(text='{"item": "standard meal expense", "amount": 50.0, "employee_id": "emp_001"}')]
    )
    
    events = []
    async for event in runner.run_async(user_id="user_1", session_id=session.id, new_message=claim_msg):
        events.append(event)
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"Agent Output: {part.text}")
                    
    last_event = events[-1]
    print(f"Final Event Output: {last_event.output}\n")
    assert last_event.output["status"] == "approved"
    assert "Under $100 auto-approval threshold" in last_event.output["reason"]

    # Test Case 2: >= $100 (Manual Approval - Pausing and Resuming)
    print("=== Test Case 2: Client Dinner Expense of $150 (Pause & Resume) ===")
    session2 = session_service.create_session_sync(user_id="user_2", app_name="app")
    claim_msg2 = types.Content(
        role="user",
        parts=[types.Part.from_text(text='{"item": "client dinner", "amount": 150.0, "employee_id": "emp_001"}')]
    )
    
    print("Submitting claim of $150...")
    interrupt_id = None
    async for event in runner.run_async(user_id="user_2", session_id=session2.id, new_message=claim_msg2):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.function_call and part.function_call.name == "adk_request_input":
                    interrupt_id = part.function_call.args["interruptId"]
                    print(f"Triggered Pause. Interrupt ID: {interrupt_id}")
                    print(f"Message shown to user: {part.function_call.args['message']}")

    # Re-run / resume the runner with a function response
    print("\nResuming session with manual approval ('approve')...")
    resume_msg = types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    name="adk_request_input",
                    response={"response": "approve"},
                    id=interrupt_id
                )
            )
        ]
    )
    
    final_events = []
    async for event in runner.run_async(user_id="user_2", session_id=session2.id, new_message=resume_msg):
        final_events.append(event)
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"Agent Output on Resume: {part.text}")

    last_final_event = final_events[-1]
    print(f"Final Event Output after Resume: {last_final_event.output}\n")
    assert last_final_event.output["status"] == "approved"
    assert "Manual review" in last_final_event.output["reason"]

if __name__ == "__main__":
    asyncio.run(run_tests())

