# Backward-compat shim.
# The engine has moved to the standalone `domain-ingestion-engine` package.
# This file is kept so that any code using:
#   from rag_bench.engine.domain_ingestion_engine import DomainIngestionEngine
# continues to work. Do not add logic here.
from domain_ingestion_engine import DomainIngestionEngine  # noqa: F401
