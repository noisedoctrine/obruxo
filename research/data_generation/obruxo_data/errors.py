from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: Severity
    message: str
    pointer: str | None = None
    parameter: str | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "pointer": self.pointer,
            "parameter": self.parameter,
            "context": self.context,
        }


@dataclass(frozen=True)
class ValidationReport:
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def valid(self) -> bool:
        return not any(item.severity == Severity.ERROR for item in self.diagnostics)

    def extend(self, *items: Diagnostic) -> "ValidationReport":
        return ValidationReport(self.diagnostics + items)

    def merge(self, *reports: "ValidationReport") -> "ValidationReport":
        return ValidationReport(self.diagnostics + tuple(item for report in reports for item in report.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "diagnostics": [item.to_dict() for item in self.diagnostics]}

    def require_valid(self) -> None:
        if not self.valid:
            raise ValidationError(self)


class ObruxoDataError(Exception):
    """Base class for structured data-generation failures."""


class ValidationError(ObruxoDataError):
    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__("; ".join(item.message for item in report.diagnostics if item.severity == Severity.ERROR))


class DependencyUnavailableError(ObruxoDataError):
    """Raised when an explicitly configured optional native dependency is unavailable."""


class OutputExistsError(ObruxoDataError):
    """Raised when a command would overwrite an output without explicit permission."""
