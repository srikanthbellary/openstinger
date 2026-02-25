"""
Agents module — multi-agent namespace lifecycle management.

Named agents have their own temporal graphs (openstinger_temporal_<id>).
Anonymous/task agents get read-only access to shared knowledge.
"""
from openstinger.agents.namespace import create_namespace, archive_namespace, list_namespaces
from openstinger.agents.context import AnonymousAgentContext

__all__ = [
    "create_namespace",
    "archive_namespace",
    "list_namespaces",
    "AnonymousAgentContext",
]
