"""
db/connection.py — SQLAlchemy engine and session factory.

This module owns the single ``engine`` instance used by every other
package to talk to PostgreSQL. The engine is configured with a small
connection pool (size 5, overflow 10) and ``pool_pre_ping=True`` so
that connections dropped by the database server are silently recycled
on the next use.

Usage
-----
The :class:`Session` class is a thin context-manager wrapper that
commits on success and rolls back on exception::

    from db.connection import Session

    with Session() as s:
        rows = s.query(Price).filter(Price.ticker == "SPY").all()
        # commit happens automatically on context exit
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import config

#: Module-level SQLAlchemy engine. Imported by every other module that
#: needs raw DB access (e.g. ``db.models.create_all_tables``).
engine = create_engine(
    config.DB_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    echo=False,
)

_SessionFactory = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Session:
    """Context-managed SQLAlchemy session with auto-commit / rollback.

    On normal exit the underlying session is committed and closed. On
    exception the session is rolled back and closed. Outside a ``with``
    block, attribute access is forwarded to the underlying session so
    the class is also usable as a plain handle.
    """

    def __init__(self) -> None:
        self._s = _SessionFactory()

    def __getattr__(self, name: str):
        return getattr(self._s, name)

    def __enter__(self):
        return self._s

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type:
            self._s.rollback()
            self._s.close()
        else:
            self._s.commit()
            self._s.close()
