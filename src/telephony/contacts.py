"""
Contact book backed by a YAML file.

Supports fuzzy name lookup so the agent can resolve
"老张" or "张哥" to the actual phone number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Contact:
    name: str
    phone: str
    aliases: list[str] = field(default_factory=list)

    @property
    def all_names(self) -> list[str]:
        return [self.name] + self.aliases


class ContactBook:
    """YAML-backed contact book with name/alias lookup."""

    def __init__(self, path: str | Path = "config/contacts.yaml"):
        self.path = Path(path)
        self.contacts: list[Contact] = []
        self._load()

    def _load(self):
        if not self.path.exists():
            logger.warning("Contacts file not found: %s", self.path)
            return

        with open(self.path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        for entry in data.get("contacts", []):
            self.contacts.append(
                Contact(
                    name=entry["name"],
                    phone=str(entry["phone"]),
                    aliases=entry.get("aliases", []),
                )
            )
        logger.info("Loaded %d contacts", len(self.contacts))

    def lookup(self, query: str) -> Contact | None:
        """Find a contact by name or alias (exact match)."""
        query = query.strip()
        for contact in self.contacts:
            if query in contact.all_names:
                return contact
        return None

    def fuzzy_lookup(self, query: str) -> list[Contact]:
        """Find contacts whose name or alias contains the query string."""
        query = query.strip()
        results = []
        for contact in self.contacts:
            for name in contact.all_names:
                if query in name or name in query:
                    results.append(contact)
                    break
        return results

    def add_contact(self, name: str, phone: str, aliases: list[str] | None = None) -> bool:
        """Add a new contact and persist to YAML."""
        if self.lookup(name):
            return False
        contact = Contact(name=name, phone=str(phone), aliases=aliases or [])
        self.contacts.append(contact)
        self._save()
        logger.info("Added contact: %s (%s)", name, phone)
        return True

    def remove_contact(self, name: str) -> bool:
        """Remove a contact by name and persist to YAML."""
        contact = self.lookup(name)
        if not contact:
            matches = self.fuzzy_lookup(name)
            contact = matches[0] if len(matches) == 1 else None
        if not contact:
            return False
        self.contacts.remove(contact)
        self._save()
        logger.info("Removed contact: %s", contact.name)
        return True

    def _save(self):
        """Persist current contacts back to YAML."""
        data = {
            "contacts": [
                {"name": c.name, "phone": c.phone, **({"aliases": c.aliases} if c.aliases else {})}
                for c in self.contacts
            ]
        }
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def list_all(self) -> list[dict]:
        """Return all contacts as dicts (for agent context)."""
        return [
            {"name": c.name, "phone": c.phone, "aliases": c.aliases}
            for c in self.contacts
        ]
