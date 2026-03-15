"""Confidence scoring and human review flagging.

Computes aggregate confidence scores for each extracted item by combining
the LLM's self-reported confidence with validation edit results, and
flags items that need human review.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import logging

from naaccr_pipeline.validation.cross_field import EditViolation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Priority ordering for sorting
# ---------------------------------------------------------------------------

_PRIORITY_ORDER = {
    "CRITICAL": 0,
    "HIGH": 1,
    "MEDIUM": 2,
    "LOW": 3,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ReviewItem:
    """A single item flagged for human review."""

    patient_id: str
    tumor_index: int
    item_number: int
    item_name: str
    extracted_value: str
    resolved_code: str
    confidence: float
    priority: str           # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    reason: str             # why flagged
    evidence_text: str
    source_chunk_type: str


# ---------------------------------------------------------------------------
# ConfidenceScorer
# ---------------------------------------------------------------------------

class ConfidenceScorer:
    """Compute aggregate confidence and flag items for human review.

    Combines LLM self-reported confidence with validation results to
    produce a final per-item confidence score and identify items that
    require registrar review.
    """

    # Key variables requiring the highest accuracy per NAACCR Gold
    # standard: primary site, histology, sex, age at diagnosis, county
    KEY_VARIABLES: set[int] = {400, 522, 220, 230, 90}

    # Additional high-importance items (required by most registries)
    REQUIRED_ITEMS: set[int] = {
        400,   # Primary Site
        522,   # Histologic Type ICD-O-3
        523,   # Behavior Code ICD-O-3
        220,   # Sex
        230,   # Age at Diagnosis
        390,   # Date of Diagnosis
        410,   # Laterality
        490,   # Diagnostic Confirmation
        90,    # County at DX Geocode
        764,   # Summary Stage 2018
        880,   # TNM Path T
        890,   # TNM Path N
        900,   # TNM Path M
        910,   # TNM Path Stage Group
        940,   # TNM Clin T
        950,   # TNM Clin N
        960,   # TNM Clin M
        970,   # TNM Clin Stage Group
        1290,  # RX Summ--Surg Prim Site
        1360,  # RX Summ--Radiation
        1390,  # RX Summ--Chemo
        1750,  # Date of Last Contact
        1760,  # Vital Status
    }

    # Confidence penalty for items involved in edit violations
    _ERROR_PENALTY = 0.3
    _WARNING_PENALTY = 0.15

    # Fuzzy match penalty multiplier
    _FUZZY_MATCH_MULTIPLIER = 0.8

    # Review thresholds
    _KEY_VAR_THRESHOLD = 0.9
    _REQUIRED_ITEM_THRESHOLD = 0.7
    _LOW_CONFIDENCE_THRESHOLD = 0.5

    def score_record(
        self,
        record: dict,
        violations: list[EditViolation],
    ) -> dict[int, float]:
        """Compute final confidence for each item.

        Parameters
        ----------
        record : dict
            Mapping of item_number (int) -> ExtractionResult.
        violations : list[EditViolation]
            All violations from cross-field and consistency checks.

        Returns
        -------
        dict[int, float]
            Mapping of item_number -> final confidence score [0.0, 1.0].

        Factors
        -------
        1. Base: LLM self-reported confidence from ExtractionResult.
        2. Penalty if item is involved in any EditViolation
           (error: -0.3, warning: -0.15).
        3. Code resolution quality: if resolved_code != extracted_value
           and base confidence < 1.0, multiply by 0.8 (fuzzy match penalty).
        4. Clamp to [0.0, 1.0].
        """
        # Build a map of item_number -> worst violation severity
        item_violations: dict[int, str] = {}
        for v in violations:
            for item_num in v.item_numbers:
                current = item_violations.get(item_num)
                # "error" is worse than "warning"
                if current is None:
                    item_violations[item_num] = v.severity
                elif v.severity == "error" and current == "warning":
                    item_violations[item_num] = "error"

        scores: dict[int, float] = {}

        for item_num, result in record.items():
            # 1. Base confidence from LLM
            base_conf = getattr(result, "confidence", 0.0)
            if base_conf is None:
                base_conf = 0.0

            score = base_conf

            # 2. Penalty for violations
            severity = item_violations.get(item_num)
            if severity == "error":
                score -= self._ERROR_PENALTY
            elif severity == "warning":
                score -= self._WARNING_PENALTY

            # 3. Fuzzy match penalty
            extracted = getattr(result, "extracted_value", "") or ""
            resolved = getattr(result, "resolved_code", "") or ""

            if (extracted and resolved
                    and resolved != extracted
                    and base_conf < 1.0):
                score *= self._FUZZY_MATCH_MULTIPLIER

            # 4. Clamp
            score = max(0.0, min(1.0, score))
            scores[item_num] = round(score, 4)

        return scores

    def flag_for_review(
        self,
        record: dict,
        confidence_scores: dict[int, float],
        patient_id: str = "",
        tumor_index: int = 0,
    ) -> list[ReviewItem]:
        """Flag items needing human review.

        Parameters
        ----------
        record : dict
            Mapping of item_number (int) -> ExtractionResult.
        confidence_scores : dict[int, float]
            Final confidence scores from :meth:`score_record`.
        patient_id : str
            Patient identifier for the review item.
        tumor_index : int
            Tumor sequence number.

        Returns
        -------
        list[ReviewItem]
            Items needing review, sorted by priority (CRITICAL first),
            then confidence ascending (least confident first).

        Priority levels
        ---------------
        - CRITICAL: Key variables (site, histology, sex, age, county)
          with confidence < 0.9
        - HIGH: Required items with confidence < 0.7
        - MEDIUM: Any item involved in a validation violation (even if
          confidence is high)
        - LOW: Any item with confidence < 0.5
        """
        review_items: list[ReviewItem] = []
        # Track which items have already been flagged to avoid duplicates
        flagged: set[int] = set()

        for item_num, result in record.items():
            conf = confidence_scores.get(item_num, 0.0)

            item_name = getattr(result, "item_name", "") or f"Item {item_num}"
            extracted = getattr(result, "extracted_value", "") or ""
            resolved = getattr(result, "resolved_code", "") or ""
            evidence = getattr(result, "evidence_text", "") or ""
            chunk_type = getattr(result, "source_chunk_type", "") or ""

            priority: Optional[str] = None
            reason: Optional[str] = None

            # CRITICAL: key variables below threshold
            if item_num in self.KEY_VARIABLES and conf < self._KEY_VAR_THRESHOLD:
                priority = "CRITICAL"
                reason = (
                    f"Key variable (NAACCR Gold) with confidence {conf:.2f} "
                    f"< {self._KEY_VAR_THRESHOLD} threshold"
                )

            # HIGH: required items below threshold (only if not already CRITICAL)
            elif item_num in self.REQUIRED_ITEMS and conf < self._REQUIRED_ITEM_THRESHOLD:
                priority = "HIGH"
                reason = (
                    f"Required item with confidence {conf:.2f} "
                    f"< {self._REQUIRED_ITEM_THRESHOLD} threshold"
                )

            # LOW: any item with very low confidence
            elif conf < self._LOW_CONFIDENCE_THRESHOLD:
                priority = "LOW"
                reason = (
                    f"Low confidence {conf:.2f} "
                    f"< {self._LOW_CONFIDENCE_THRESHOLD} threshold"
                )

            if priority is not None and reason is not None:
                review_items.append(ReviewItem(
                    patient_id=patient_id,
                    tumor_index=tumor_index,
                    item_number=item_num,
                    item_name=item_name,
                    extracted_value=extracted,
                    resolved_code=resolved,
                    confidence=conf,
                    priority=priority,
                    reason=reason,
                    evidence_text=evidence[:500],
                    source_chunk_type=chunk_type,
                ))
                flagged.add(item_num)

        # MEDIUM: items involved in violations that haven't been flagged yet
        # We need to check violations -- they are not passed directly here,
        # but we can infer from confidence penalties. Items whose score was
        # reduced by violations will have conf < base confidence.
        # A cleaner approach: accept violations as optional parameter.
        # For now, flag items whose confidence dropped (heuristic).
        for item_num, result in record.items():
            if item_num in flagged:
                continue

            conf = confidence_scores.get(item_num, 0.0)
            base_conf = getattr(result, "confidence", 0.0) or 0.0

            # If final confidence is noticeably lower than base, a penalty
            # was applied -- which means a violation was involved
            if base_conf > 0 and (base_conf - conf) >= (self._WARNING_PENALTY - 0.01):
                item_name = getattr(result, "item_name", "") or f"Item {item_num}"
                extracted = getattr(result, "extracted_value", "") or ""
                resolved = getattr(result, "resolved_code", "") or ""
                evidence = getattr(result, "evidence_text", "") or ""
                chunk_type = getattr(result, "source_chunk_type", "") or ""

                review_items.append(ReviewItem(
                    patient_id=patient_id,
                    tumor_index=tumor_index,
                    item_number=item_num,
                    item_name=item_name,
                    extracted_value=extracted,
                    resolved_code=resolved,
                    confidence=conf,
                    priority="MEDIUM",
                    reason=(
                        f"Item involved in validation violation "
                        f"(confidence dropped from {base_conf:.2f} to {conf:.2f})"
                    ),
                    evidence_text=evidence[:500],
                    source_chunk_type=chunk_type,
                ))
                flagged.add(item_num)

        # Sort: priority order first (CRITICAL < HIGH < MEDIUM < LOW),
        # then confidence ascending (least confident first)
        review_items.sort(
            key=lambda ri: (_PRIORITY_ORDER.get(ri.priority, 99), ri.confidence)
        )

        return review_items

    def flag_for_review_with_violations(
        self,
        record: dict,
        confidence_scores: dict[int, float],
        violations: list[EditViolation],
        patient_id: str = "",
        tumor_index: int = 0,
    ) -> list[ReviewItem]:
        """Flag items needing human review, with explicit violation list.

        This is the preferred entry point when violations are available,
        as it can directly identify MEDIUM-priority items from the
        violation list rather than relying on confidence-drop heuristics.

        Parameters
        ----------
        record : dict
            Mapping of item_number (int) -> ExtractionResult.
        confidence_scores : dict[int, float]
            Final confidence scores from :meth:`score_record`.
        violations : list[EditViolation]
            All violations from validation checks.
        patient_id : str
            Patient identifier for the review item.
        tumor_index : int
            Tumor sequence number.

        Returns
        -------
        list[ReviewItem]
            Items needing review, sorted by priority then confidence.
        """
        review_items: list[ReviewItem] = []
        flagged: set[int] = set()

        # Collect items involved in violations
        violation_items: set[int] = set()
        for v in violations:
            for item_num in v.item_numbers:
                violation_items.add(item_num)

        for item_num, result in record.items():
            conf = confidence_scores.get(item_num, 0.0)

            item_name = getattr(result, "item_name", "") or f"Item {item_num}"
            extracted = getattr(result, "extracted_value", "") or ""
            resolved = getattr(result, "resolved_code", "") or ""
            evidence = getattr(result, "evidence_text", "") or ""
            chunk_type = getattr(result, "source_chunk_type", "") or ""

            priority: Optional[str] = None
            reason: Optional[str] = None

            # CRITICAL
            if item_num in self.KEY_VARIABLES and conf < self._KEY_VAR_THRESHOLD:
                priority = "CRITICAL"
                reason = (
                    f"Key variable (NAACCR Gold) with confidence {conf:.2f} "
                    f"< {self._KEY_VAR_THRESHOLD} threshold"
                )

            # HIGH
            elif item_num in self.REQUIRED_ITEMS and conf < self._REQUIRED_ITEM_THRESHOLD:
                priority = "HIGH"
                reason = (
                    f"Required item with confidence {conf:.2f} "
                    f"< {self._REQUIRED_ITEM_THRESHOLD} threshold"
                )

            # MEDIUM: involved in a violation
            elif item_num in violation_items:
                priority = "MEDIUM"
                # Find the most severe violation for this item
                relevant = [
                    v for v in violations if item_num in v.item_numbers
                ]
                if relevant:
                    worst = max(relevant, key=lambda v: 1 if v.severity == "error" else 0)
                    reason = (
                        f"Involved in validation edit '{worst.edit_name}' "
                        f"({worst.severity}): {worst.message[:200]}"
                    )
                else:
                    reason = "Involved in validation violation"

            # LOW
            elif conf < self._LOW_CONFIDENCE_THRESHOLD:
                priority = "LOW"
                reason = (
                    f"Low confidence {conf:.2f} "
                    f"< {self._LOW_CONFIDENCE_THRESHOLD} threshold"
                )

            if priority is not None and reason is not None:
                review_items.append(ReviewItem(
                    patient_id=patient_id,
                    tumor_index=tumor_index,
                    item_number=item_num,
                    item_name=item_name,
                    extracted_value=extracted,
                    resolved_code=resolved,
                    confidence=conf,
                    priority=priority,
                    reason=reason,
                    evidence_text=evidence[:500],
                    source_chunk_type=chunk_type,
                ))
                flagged.add(item_num)

        # Sort: priority order first, then confidence ascending
        review_items.sort(
            key=lambda ri: (_PRIORITY_ORDER.get(ri.priority, 99), ri.confidence)
        )

        return review_items
