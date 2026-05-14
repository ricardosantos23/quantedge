from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import config

engine = create_engine(
    config.DB_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    echo=False,
)

_SessionFactory = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Session:
    def __init__(self):
        self._s = _SessionFactory()

    def __getattr__(self, name):
        return getattr(self._s, name)

    def __enter__(self):
        return self._s

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._s.rollback()
            self._s.close()
        else:
            self._s.commit()
            self._s.close()