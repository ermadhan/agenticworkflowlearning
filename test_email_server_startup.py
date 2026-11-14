import importlib
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import email_server.email_database as database


class EmailServerStartupTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        self.sessions = sessionmaker(bind=self.engine)
        with patch.object(database, "engine", self.engine), patch.object(database, "SessionLocal", self.sessions):
            self.service = importlib.import_module("email_server.email_service")
        self.service.Base.metadata.create_all(bind=self.engine)
        self.session_patch = patch.object(self.service, "SessionLocal", self.sessions)
        self.session_patch.start()
        self.addCleanup(self.session_patch.stop)
        self.addCleanup(self.engine.dispose)

    def test_existing_email_is_preserved_on_restarts(self):
        with self.sessions() as db:
            db.add(self.service.Email(
                sender="sender@example.com",
                recipient="recipient@example.com",
                subject="Keep this",
                body="Existing email",
            ))
            db.commit()
        self.service.preload_emails()
        self.service.preload_emails()
        with self.sessions() as db:
            subjects = [email.subject for email in db.query(self.service.Email).all()]
        self.assertEqual(subjects, ["Keep this"])

    def test_empty_database_is_seeded_once(self):
        self.service.preload_emails()
        self.service.preload_emails()
        with self.sessions() as db:
            self.assertEqual(db.query(self.service.Email).count(), 6)

    def test_explicit_reset_still_requests_a_reset(self):
        with patch.object(self.service, "_seed_emails") as seed:
            self.service.reset_database()
        seed.assert_called_once_with(reset=True)


if __name__ == "__main__":
    unittest.main()
