"""Intern grind som spärrar direktanrop till dataladdarens interna datakälla.

Syftet är att teknisk göra det svårt att kringgå den gemensamma loadern i
research.oos_loader: dess interna fetch-funktion kontrollerar att den
körs inifrån loaderns egen kontext (loader_context) innan den gör något.
"""
import contextvars

_active = contextvars.ContextVar("research_loader_active", default=False)


class LoaderBypassError(RuntimeError):
    """Höjs när kod försöker läsa in data utan att gå via den gemensamma loadern."""


def require_loader_active() -> None:
    if not _active.get():
        raise LoaderBypassError(
            "Direkt datainläsning är förbjuden. All inläsning måste ske via "
            "research.oos_loader.load_market_data() — se docs/INSTRUKTION.md, avsnitt 2."
        )


class loader_context:
    """Context manager som markerar att vi befinner oss innanför den gemensamma loadern."""

    def __enter__(self):
        self._token = _active.set(True)
        return self

    def __exit__(self, exc_type, exc, tb):
        _active.reset(self._token)
        return False
