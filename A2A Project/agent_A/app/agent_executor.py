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
from agent import AgentA,save_memory_to_txt,llm_search_txt_file  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentAExecutor(AgentExecutor):
    """Executor for AgentA, handles web research and summarization."""

    def __init__(self):
        self.agent = AgentA()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
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
            memory_result = llm_search_txt_file(query)
            if memory_result:
                logger.info("Memory hit. Returning cached summary.")
                output = f"[Memory-Based Response]\n\n{memory_result}"
            else:
                result = self.agent.invoke(query)
                retrieved = result.get("retrieved_content", "")
                scraped = result.get("scraped_content", "")
                final_summary = result.get("final_summary", "")
                output = (
                    " Retrieved Content \n\n"
                    f"{retrieved.strip()}\n\n"
                    "---\n\n"
                    " Scraped Content \n\n"
                    f"{scraped.strip()}"
                    "---\n\n"
                    " Final Summary \n\n"
                    f"{final_summary.strip()}\n\n"
                )
                save_memory_to_txt(query, retrieved, scraped, final_summary)

            print(output)
            parts = [Part(root=TextPart(text=output))]
            await updater.add_artifact(parts, name="agentA_summary")
            await updater.complete()

        except Exception as e:
            logger.error(f"An error occurred during AgentA execution: {e}")
            raise ServerError(error=InternalError()) from e

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())
