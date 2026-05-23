import logging
import os
import sys

import httpx
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from a2a.types import (
    AgentCapabilities,
    AgentCard,
)
from agent import AgentB
from agent_executor import AgentBExecutor
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Starts AgentB_PDFChat server."""
    host = "127.0.0.1"
    port = 5001  

    try:
        capabilities = AgentCapabilities(
            text=True,
            tools=False,
            streaming=False
        )

        agent_card = AgentCard(
            id="pdf_context_agent",
            name="AgentB_PDFChat",
            description="Answers questions based on PDF content using FAISS vector search.",
            url=f"http://{host}:{port}",
            version="1.0.0",
            llm={"provider": "openai", "model": "gpt-4o-mini"},
            capabilities=capabilities,
            defaultInputModes=AgentB.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=AgentB.SUPPORTED_CONTENT_TYPES,
            skills=[],
        )
        
        request_handler = DefaultRequestHandler(
            agent_executor=AgentBExecutor(),
            task_store=InMemoryTaskStore(),
        )

        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler
        )

        uvicorn.run(server.build(), host=host, port=port)

    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
