# Backward-compat shim: the engine now lives in the standalone `domain-ingestion-engine` package.
# Import from there so existing code that does `from rag_bench.engine import DomainIngestionEngine`
# continues to work without changes.
from domain_ingestion_engine import DomainIngestionEngine  # noqa: F401
