"""
Simulated representative authorization for the insurance demo.

This module defines:

    AuthorizationRequest: Tracks one representative authorization request.

    RepresentativeAuthorizationService: Matches representative records
        and simulates authorization using configured fixture scenarios.
"""

from dataclasses import dataclass
from uuid import uuid4

from services.data_service import InsuranceDataService


@dataclass
class AuthorizationRequest:
    """
    Store a representative authorization request.

    Attributes:
        request_id: Identifier for this authorization request.
        party_id: Policyholder associated with the request.
        representative_name: Representative listed in the fixture.
        relationship: Representative's listed relationship.
        status: Current authorization result.
        polls: Number of status checks performed.
    """

    request_id: str
    party_id: str
    representative_name: str
    relationship: str
    status: str = "pending"
    polls: int = 0


class RepresentativeAuthorizationService:
    """
    Simulate representative authorization using fixture data.

    Attributes:
        data_service: Access to representative and consent fixtures.
        status_sequence: Configured sequence of simulated responses.
        requests: Authorization requests indexed by request ID.
    """

    def __init__(
        self,
        data_service: InsuranceDataService,
        scenario: str = "default",
    ) -> None:
        """
        Initialize the authorization simulator.

        Args:
            data_service: Service providing authorization fixtures.
            scenario: Server-selected demo scenario, such as default
                or timeout. Must not be selected by customer messages.

        Raises:
            ValueError: If the scenario configuration is invalid.
        """

        self.data_service = data_service

        configuration = data_service.consent_scenarios.get(scenario)

        if not isinstance(configuration, dict):
            raise ValueError("Unknown authorization scenario.")

        sequence = configuration.get("status_sequence")

        if not isinstance(sequence, list) or not sequence:
            raise ValueError("Authorization sequence must be nonempty.")

        allowed_statuses = {"pending", "approved", "denied"}

        if any(
            not isinstance(status, str) or status not in allowed_statuses
            for status in sequence
        ):
            raise ValueError("Unsupported authorization status.")

        self.status_sequence = tuple(sequence)
        self.requests: dict[str, AuthorizationRequest] = {}

    def begin_request(
        self,
        verified_party_id: str,
        representative_name: str,
        relationship: str,
    ) -> str:
        """
        Create a pending authorization request for a listed representative.

        The controller must establish verified_party_id through identity
        verification before calling this method.

        Args:
            verified_party_id: Policyholder ID established by verification.
            representative_name: Name supplied by the representative.
            relationship: Relationship supplied by the representative.

        Returns:
            Identifier of the pending authorization request.

        Raises:
            ValueError: If no unambiguous representative record matches.
        """

        representative = self.data_service.find_representative(
            party_id=verified_party_id,
            representative_name=representative_name,
            relationship=relationship,
        )

        if representative is None:
            raise ValueError("No matching representative record was found.")

        request_id = str(uuid4())

        self.requests[request_id] = AuthorizationRequest(
            request_id=request_id,
            party_id=verified_party_id,
            representative_name=representative["rep_name"],
            relationship=representative["relationship"],
        )

        return request_id

    def check_status(
        self,
        request_id: str,
        verified_party_id: str,
    ) -> str:
        """
        Advance a request through its simulated status sequence.

        Args:
            request_id: Authorization request associated with the session.
            verified_party_id: Verified policyholder for the current session.

        Returns:
            pending, approved, denied, or timeout.

        Raises:
            ValueError: If the request is unknown or belongs to another
                policyholder.
        """

        request = self.requests.get(request_id)

        if request is None:
            raise ValueError("Unknown authorization request.")

        if request.party_id != verified_party_id:
            raise ValueError("Authorization request belongs to another policyholder.")

        if request.status != "pending":
            return request.status

        request.status = self.status_sequence[request.polls]
        request.polls += 1

        if request.status == "pending" and request.polls >= len(self.status_sequence):
            request.status = "timeout"

        return request.status

    def is_approved(
        self,
        request_id: str,
        verified_party_id: str,
        representative_name: str,
        relationship: str,
    ) -> bool:
        """
        Check approval for the exact policyholder and representative.

        This check does not advance the simulated status sequence.

        Args:
            request_id: Request stored by the controller for this session.
            verified_party_id: Verified policyholder ID.
            representative_name: Current representative's name.
            relationship: Current representative's relationship.

        Returns:
            True only for an approved request matching every supplied field.
        """

        request = self.requests.get(request_id)

        if request is None:
            return False

        def normalize(value: str) -> str:
            return " ".join(value.casefold().split())

        return (
            request.status == "approved"
            and request.party_id == verified_party_id
            and normalize(request.representative_name) == normalize(representative_name)
            and normalize(request.relationship) == normalize(relationship)
        )
