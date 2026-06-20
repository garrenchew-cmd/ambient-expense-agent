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
import logging
import os
from typing import Any

import vertexai
from dotenv import load_dotenv
from google.adk.artifacts import GcsArtifactService, InMemoryArtifactService
from google.cloud import logging as google_cloud_logging
from vertexai.agent_engines.templates.adk import AdkApp

from app.agent import app as adk_app
from app.app_utils.telemetry import setup_telemetry
from app.app_utils.typing import Feedback

# Load environment variables from .env file at runtime
load_dotenv()

# Safely initialize Vertex AI to avoid import-time crashes when default credentials are missing
import google.auth
try:
    _, project_id = google.auth.default()
    if project_id:
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
except Exception:
    pass

project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or "mock-project-id"
location = os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"
vertexai.init(project=project_id, location=location)


class AgentEngineApp(AdkApp):
    def project_id(self) -> str:
        """Returns the project ID."""
        return os.environ.get("GOOGLE_CLOUD_PROJECT") or "mock-project-id"

    def query(
        self,
        *,
        message: Any,
        user_id: str,
        session_id: Optional[str] = None,
        run_config: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Queries the ADK application synchronously."""
        events = list(
            self.stream_query(
                message=message,
                user_id=user_id,
                session_id=session_id,
                run_config=run_config,
                **kwargs,
            )
        )
        final_output = {}
        msg_text = ""
        for ev in events:
            if ev.get("output"):
                final_output = ev["output"]
            if ev.get("content") and ev["content"].get("parts"):
                for part in ev["content"]["parts"]:
                    if part.get("text"):
                        msg_text = part["text"]
                        
        return {
            "response": final_output,
            "message": msg_text,
        }

    def _tracing_enabled(self) -> bool:
        """Disable tracing if default credentials are not present to avoid crashes."""
        import google.auth
        try:
            google.auth.default()
            return super()._tracing_enabled()
        except Exception:
            return False

    def set_up(self) -> None:
        """Initialize the agent engine app with logging and telemetry."""
        vertexai.init()
        setup_telemetry()
        super().set_up()
        logging.basicConfig(level=logging.INFO)
        try:
            logging_client = google_cloud_logging.Client()
            self.logger = logging_client.logger(__name__)
        except Exception:
            self.logger = logging.getLogger(__name__)
            if not hasattr(self.logger, "log_struct"):
                def log_struct(info_dict, severity="INFO"):
                    self.logger.info(f"[{severity}] Struct log: {info_dict}")
                self.logger.log_struct = log_struct
        if gemini_location:
            os.environ["GOOGLE_CLOUD_LOCATION"] = gemini_location

    def register_feedback(self, feedback: dict[str, Any]) -> None:
        """Collect and log feedback."""
        feedback_obj = Feedback.model_validate(feedback)
        self.logger.log_struct(feedback_obj.model_dump(), severity="INFO")

    def register_operations(self) -> dict[str, list[str]]:
        """Registers the operations of the Agent."""
        operations = super().register_operations()
        operations[""] = [*operations.get("", []), "register_feedback"]
        return operations

    def clone(self) -> "AgentEngineApp":
        """Returns a clone of the Agent Runtime application."""
        return self


gemini_location = os.environ.get("GOOGLE_CLOUD_LOCATION")
logs_bucket_name = os.environ.get("LOGS_BUCKET_NAME")
agent_runtime = AgentEngineApp(
    app=adk_app,
    artifact_service_builder=lambda: (
        GcsArtifactService(bucket_name=logs_bucket_name)
        if logs_bucket_name
        else InMemoryArtifactService()
    ),
)
