"""Client that pushes classified transactions into a self-hosted sure.am instance.

Instead of writing a CSV for manual upload, this posts each transaction directly
to the sure.am REST API (POST /api/v1/transactions). Three properties make it
safe to run unattended on a schedule:

  * Idempotency — every transaction carries a stable ``external_id`` (a hash of
    its date, amount, description and in-batch occurrence) plus a ``source`` tag.
    sure.am returns the existing transaction instead of creating a duplicate, so
    re-running the whole pipeline never double-counts. See ``_external_id`` for
    the (documented) limits of a content-based key.

  * Sign safety — the bank's amount sign decides ``nature`` (expense vs income)
    and we always send the absolute amount. sure.am then stores the correct
    direction regardless of its own default sign convention.

  * Fail-soft — a bad row (unknown category, HTTP error) is recorded and skipped,
    never aborting the run. The summary reports what failed.

Authentication uses an API key generated in sure.am under Settings → API Keys,
sent in the ``X-Api-Key`` header. The key needs the ``read_write`` scope.

Request body shape (everything nests under ``transaction``)::

    {"transaction": {
        "account_id": "<uuid>", "date": "2026-06-01", "amount": 199.5,
        "nature": "expense", "name": "REMA 1000", "currency": "DKK",
        "category_id": "<uuid>", "tag_ids": ["<uuid>"], "notes": "...",
        "external_id": "<sha256>", "source": "lsb-classifier"
    }}

Typical usage::

    client = SureClient(base_url="http://localhost:3000", api_key=KEY)
    result = client.push_dataframe(df_pred, config, account_name="LSB Checking")
    print(result.summary())
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import requests

from src.export.sure_am import _label_to_sure_map

# sure.am accepts "income"/"inflow" (stored negative) and "expense"/"outflow"
# (stored positive). We derive this from the bank's amount sign so the stored
# direction never depends on sure.am's default interpretation of a bare amount.
_NATURE_INCOME = "income"
_NATURE_EXPENSE = "expense"

_PER_PAGE = 100
_DEFAULT_TIMEOUT = 30  # seconds per HTTP request


def _external_id(date_iso: str, amount: float, description: str, occurrence: int) -> str:
    """Stable dedupe key for one transaction.

    The bank's simple export has no unique transaction id, so we hash the fields
    that identify a transaction plus its ``occurrence`` — a 1-based counter over
    identical (date, amount, description) rows *within the same export*. That
    keeps two genuinely-identical same-day purchases distinct, while re-running
    the same export reproduces the same ids exactly (idempotent).

    Known limit: if the very same real transaction shows up in two overlapping
    exports whose row ordering differs, the occurrence counter can shift and it
    may be inserted twice. Carrying the advanced export's transaction reference
    through the pipeline would remove this edge case entirely (future work).
    """
    raw = f"{date_iso}|{amount:.2f}|{description}|{occurrence}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class PushResult:
    pushed: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "─" * 42,
            f"  Pushed   {self.pushed:>4}",
            f"  Failed   {self.failed:>4}",
        ]
        for err in self.errors[:5]:
            lines.append(f"    - {err}")
        if len(self.errors) > 5:
            lines.append(f"    … and {len(self.errors) - 5} more")
        lines.append("─" * 42)
        return "\n".join(lines)


class SureClient:
    """Minimal REST client for pushing transactions to sure.am's API v1."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        source: str = "lsb-classifier",
        review_tag: str = "needs_review",
        timeout: int = _DEFAULT_TIMEOUT,
        dry_run: bool = False,
    ):
        if not base_url:
            raise ValueError("base_url is required (e.g. http://localhost:3000)")
        if not api_key and not dry_run:
            raise ValueError("api_key is required unless dry_run=True")

        self.base_url = base_url.rstrip("/")
        self.source = source
        self.review_tag = review_tag
        self.timeout = timeout
        self.dry_run = dry_run

        self._session = requests.Session()
        self._session.headers.update({
            "X-Api-Key": api_key or "",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

        # name → id maps, populated lazily on first push
        self._accounts: dict[str, str] = {}
        self._categories: dict[str, str] = {}
        self._tags: dict[str, str] = {}
        self._lookups_loaded = False

    # ── HTTP helpers ────────────────────────────────────────────────────────

    def _url(self, path: str) -> str:
        return f"{self.base_url}/api/v1/{path.lstrip('/')}"

    def _get_list(self, path: str, root_key: str) -> list[dict[str, Any]]:
        """GET a paginated collection and return all items across pages.

        Handles both a bare JSON array and the ``{root_key: [...], pagination:
        {...}}`` envelope the API normally returns.
        """
        items: list[dict[str, Any]] = []
        page = 1
        while True:
            resp = self._session.get(
                self._url(path),
                params={"page": page, "per_page": _PER_PAGE},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            batch = data if isinstance(data, list) else data.get(root_key, [])
            items.extend(batch)

            pagination = data.get("pagination") if isinstance(data, dict) else None
            if pagination:
                if pagination.get("page", page) >= pagination.get("total_pages", page):
                    break
            elif len(batch) < _PER_PAGE:
                break
            page += 1
        return items

    # ── Lookups ─────────────────────────────────────────────────────────────

    @staticmethod
    def _index_by_name(items: list[dict[str, Any]]) -> dict[str, str]:
        """Build a case-insensitive name → id map from an API collection."""
        return {
            str(it["name"]).strip().lower(): it["id"]
            for it in items
            if it.get("name") and it.get("id")
        }

    def load_lookups(self) -> None:
        """Fetch accounts, categories and tags once and cache name → id maps.

        In dry-run mode this degrades gracefully: if the server or key is not
        available it warns and continues with empty maps, so payloads can still
        be previewed offline.
        """
        if self._lookups_loaded:
            return

        try:
            self._accounts = self._index_by_name(self._get_list("accounts", "accounts"))
            self._categories = self._index_by_name(self._get_list("categories", "categories"))
        except requests.RequestException as exc:
            if not self.dry_run:
                raise
            print(f"  Note: could not reach sure.am ({exc}) — offline dry-run, ids blank.")
            self._lookups_loaded = True
            return

        # Tags are optional — some deployments may not expose the endpoint.
        try:
            self._tags = self._index_by_name(self._get_list("tags", "tags"))
        except requests.HTTPError:
            self._tags = {}
            print("  Note: /api/v1/tags unavailable — review flag will go to notes only.")

        self._lookups_loaded = True

    def resolve_account(self, name: str) -> str:
        self.load_lookups()
        account_id = self._accounts.get(name.strip().lower())
        if account_id is None:
            if self.dry_run:
                return "<dry-run-account>"
            known = ", ".join(sorted(self._accounts)) or "(none found)"
            raise ValueError(
                f"Account {name!r} not found in sure.am. Known accounts: {known}. "
                "Create it in the app first, spelled exactly the same."
            )
        return account_id

    # ── Payload construction ────────────────────────────────────────────────

    def _build_payload(
        self,
        row: pd.Series,
        account_id: str,
        label_to_sure: dict[str, str],
        occurrence: int,
    ) -> dict[str, Any]:
        amount = float(row["amount"])
        date_iso = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        description = str(row["description"])
        needs_review = bool(row["needs_review"])

        txn: dict[str, Any] = {
            "account_id": account_id,
            "date": date_iso,
            "amount": round(abs(amount), 2),
            "nature": _NATURE_INCOME if amount > 0 else _NATURE_EXPENSE,
            "name": description,
            "currency": str(row.get("currency") or "DKK"),
            "notes": _notes(row),
            "external_id": _external_id(date_iso, amount, description, occurrence),
            "source": self.source,
        }

        # Category — only for confidently auto-labeled rows. Rows needing review
        # are left uncategorized so they surface in sure.am's uncategorized view.
        if not needs_review and row.get("label"):
            sure_name = label_to_sure.get(row["label"])
            category_id = self._categories.get((sure_name or "").strip().lower())
            if category_id:
                txn["category_id"] = category_id
            else:
                txn["notes"] = (
                    f"{txn['notes']} (unmapped category: {sure_name})".strip()
                )

        # Tag rows that need a human look, if the review tag exists.
        if needs_review:
            tag_id = self._tags.get(self.review_tag.strip().lower())
            if tag_id:
                txn["tag_ids"] = [tag_id]

        return txn

    def _post_transaction(self, txn: dict[str, Any]) -> None:
        resp = self._session.post(
            self._url("transactions"),
            json={"transaction": txn},
            timeout=self.timeout,
        )
        resp.raise_for_status()

    # ── Public entry point ──────────────────────────────────────────────────

    def push_dataframe(
        self,
        df: pd.DataFrame,
        config: dict,
        account_name: str,
    ) -> PushResult:
        """Push every row of a predicted DataFrame to sure.am.

        ``df`` must be the output of ``Predictor.predict_dataframe`` (columns:
        date, amount, description, currency, label, needs_review, review_reason).
        """
        self.load_lookups()
        account_id = self.resolve_account(account_name)
        label_to_sure = _label_to_sure_map(config)

        result = PushResult()
        occurrences: dict[str, int] = {}  # (date|amount|desc) → count seen so far

        for _, row in df.iterrows():
            key = f"{pd.to_datetime(row['date']).date()}|{row['amount']}|{row['description']}"
            occurrences[key] = occurrences.get(key, 0) + 1

            try:
                txn = self._build_payload(row, account_id, label_to_sure, occurrences[key])
                if self.dry_run:
                    print(f"  [dry-run] {txn['date']}  {txn['nature']:<7} "
                          f"{txn['amount']:>10.2f}  {txn['name'][:32]}")
                else:
                    self._post_transaction(txn)
                result.pushed += 1
            except requests.HTTPError as exc:
                body = exc.response.text[:200] if exc.response is not None else ""
                result.failed += 1
                result.errors.append(f"{row.get('description', '?')}: {exc} {body}")
            except Exception as exc:  # noqa: BLE001 — never let one row abort the run
                result.failed += 1
                result.errors.append(f"{row.get('description', '?')}: {exc}")

        return result


def _notes(row: pd.Series) -> str:
    """Short note explaining why a row needs review (mirrors the CSV exporter)."""
    reason = row.get("review_reason", "")
    if reason == "low_confidence":
        return f"confidence: {float(row.get('confidence', 0)):.2f}"
    if reason == "pattern_match":
        return "awaiting manual label"
    return ""
