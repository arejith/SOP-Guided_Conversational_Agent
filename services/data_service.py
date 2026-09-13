"""
Data services for loading and accessing insurance fixture data.

This module defines:

    InsuranceDataService: Loads policyholders, claims, representatives,
        claim-field definitions, and document guidance.
"""

import json
from pathlib import Path
from typing import Any


class InsuranceDataService:
    """
    Load and query insurance fixture data.

    Attributes:
        fixture_dir: Directory containing the insurance JSON fixtures.
        policyholders: Policyholder records used for identity verification.
        claims: Claim records belonging to policyholders.
        representatives: Records describing people acting on behalf of
            policyholders. These records alone do not authorize access.
        claim_schema: Definitions of claim fields.
        document_guidance: Document submission and follow-up guidance.
    """

    def __init__(
        self,
        fixture_dir: Path | None = None,
    ) -> None:
        """
        Load fixture files when the service is created.

        Args:
            fixture_dir: Optional fixture directory. Defaults to the
                project's insurance claims fixture directory.
        """

        project_root = Path(__file__).resolve().parents[1]

        self.fixture_dir = (
            Path(fixture_dir)
            if fixture_dir is not None
            else project_root / "apps" / "insurance_claims" / "fixtures"
        )

        self.policyholders = self._load_fixture("policyholders.json")
        self.claims = self._load_fixture("claims.json")
        self.representatives = self._load_fixture("representatives.json")

        self.claim_schema = self._load_fixture("claim_schema.json")
        self.document_guidance = self._load_fixture("required_document_guideline.json")
        self.consent_scenarios = self._load_fixture("consent_scenarios.json")

    def _load_fixture(self, filename: str) -> Any:
        """
        Read and return one JSON fixture.

        Args:
            filename: Name of the fixture file.

        Returns:
            Parsed JSON data.
        """

        fixture_path = self.fixture_dir / filename

        with fixture_path.open("r", encoding="utf-8") as fixture_file:
            return json.load(fixture_file)

    def find_policyholder_by_policy(
        self,
        policy_number: str,
    ) -> dict[str, Any] | None:
        """
        Find a policyholder using a policy number.

        This lookup does not verify the customer's identity.

        Args:
            policy_number: Policy number supplied by the customer.

        Returns:
            Matching policyholder record, or None.
        """

        normalized_policy = policy_number.strip().upper()

        for policyholder in self.policyholders:
            if policyholder["policy_number"].strip().upper() == normalized_policy:
                return policyholder

        return None

    def find_claims_for_party(
        self,
        party_id: str,
    ) -> list[dict[str, Any]]:
        """
        Find all claims belonging to a policyholder.

        Args:
            party_id: Policyholder ID supplied by the controller after
                successful identity verification.

        Returns:
            Claim records belonging to the specified policyholder.
        """

        return [claim for claim in self.claims if claim["party_id"] == party_id]

    def find_claim_for_party(
        self,
        party_id: str,
        case_id: str,
    ) -> dict[str, Any] | None:
        """
        Find a specific claim belonging to a policyholder.

        Args:
            party_id: Verified policyholder ID supplied by the controller.
            case_id: Claim ID to look up.

        Returns:
            Matching claim, or None if it does not belong to the party
            or does not exist.
        """

        normalized_case_id = case_id.strip().upper()

        for claim in self.find_claims_for_party(party_id):
            if claim["case_id"].strip().upper() == normalized_case_id:
                return claim

        return None

    def find_representative(
        self,
        party_id: str,
        representative_name: str,
        relationship: str,
    ) -> dict[str, Any] | None:
        """
        Find a representative listed for a policyholder.

        Matching this record does not authorize claim access.

        Args:
            party_id: Policyholder ID established by identity verification.
            representative_name: Name supplied by the representative.
            relationship: Representative's stated relationship.

        Returns:
            One matching representative record, or None.
        """

        normalized_name = " ".join(representative_name.casefold().split())
        normalized_relationship = " ".join(relationship.casefold().split())

        matches = [
            representative
            for representative in self.representatives
            if (
                representative["buyer_party_id"] == party_id
                and " ".join(representative["rep_name"].casefold().split())
                == normalized_name
                and " ".join(representative["relationship"].casefold().split())
                == normalized_relationship
            )
        ]

        return matches[0] if len(matches) == 1 else None
