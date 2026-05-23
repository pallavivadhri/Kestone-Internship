import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils.errors import ServerError
from agent import AgentB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentBExecutor(AgentExecutor):
    """Executor for AgentB: Answers PDF-based questions using vector similarity search."""

    def __init__(self):
        pdf_path = r"C:\Users\palla\Downloads\sample_pdf.pdf"
        faiss_path = r"C:\Users\palla\OneDrive\Desktop\Intern\Langchain\Faiss Vector\faiss_storage"
        self.agent = AgentB(pdf_path, faiss_path)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if not context.task_id or not context.context_id:
            raise ValueError("RequestContext must have task_id and context_id")
        if not context.message:
            raise ValueError("RequestContext must have a message")

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            await updater.submit()
        await updater.start_work()

        query = context.get_user_input()

        try:
            response_text = self.agent.invoke(query)

            parts = [Part(root=TextPart(text=response_text))]
            await updater.add_artifact(parts, name="pdf_chat_response")
            await updater.complete()

        except Exception as e:
            logger.error(f"AgentB error: {e}")
            raise ServerError(error=InternalError()) from e

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())
