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
    AgentSkill,
)
from agent import AgentA
from agent_executor import AgentAExecutor
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""

def main():
    """Starts AgentA_SocialListening server."""
    host = "127.0.0.1"
    port = 5000
    try:

        capabilities = AgentCapabilities(
            text=True,
            tools=True,
            streaming=False
        )

        agent_card = AgentCard(
            id="tavily_wiki_agent",
            name="AgentA_SocialListening",
            description="Answers questions about any topic by using a combination of Wikipedia and Tavily web search.",
            url=f"http://{host}:{port}",
            version="1.0.0",
            llm={"provider": "openai", "model": "gpt-4o-mini"},
            capabilities=capabilities,
            defaultInputModes=AgentA.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=AgentA.SUPPORTED_CONTENT_TYPES,
            skills=[],
        )

        request_handler = DefaultRequestHandler(
            agent_executor=AgentAExecutor(),
            task_store=InMemoryTaskStore(),
        )

        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler
        )

        uvicorn.run(server.build(), host=host, port=port)

    except MissingAPIKeyError as e:
        logger.error(f"Missing API Key: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
