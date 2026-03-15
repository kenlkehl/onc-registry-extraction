import re
import uuid
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str               # unique ID for audit trail
    text: str                   # the chunk text
    chunk_type: str             # "pathology", "radiology", "operative", "discharge_summary",
                                # "progress_note", "consult", "lab", "mixed"
    document_date: str          # date from the input document
    source_doc_index: int       # index of source document in PatientDocumentSet
    token_estimate: int         # rough token count


class ClinicalChunker:
    """Classifies and chunks individual clinical documents."""

    # Note type classification patterns
    NOTE_TYPE_PATTERNS = {
        'pathology': [
            r'(?i)(?:PATHOLOGY|SURGICAL\s+PATH|CYTOLOGY|AUTOPSY)\s+REPORT',
            r'(?i)FINAL\s+(?:PATHOLOGIC\s+)?DIAGNOSIS',
            r'(?i)(?:GROSS|MICROSCOPIC)\s+DESCRIPTION',
            r'(?i)(?:SYNOPTIC|CAP\s+PROTOCOL|CANCER\s+CHECKLIST)',
            r'(?i)SPECIMEN\s+(?:RECEIVED|SUBMITTED)',
        ],
        'radiology': [
            r'(?i)(?:RADIOLOGY|IMAGING|DIAGNOSTIC\s+IMAGING)\s+REPORT',
            r'(?i)(?:CT|MRI|PET|PET/CT|ULTRASOUND|X-RAY|MAMMOGRA)',
            r'(?i)(?:FINDINGS|IMPRESSION).*(?:cm\s+mass|lesion|nodule|tumor)',
        ],
        'operative': [
            r'(?i)(?:OPERATIVE|PROCEDURE|SURGICAL)\s+(?:NOTE|REPORT)',
            r'(?i)(?:PRE-?OPERATIVE|POST-?OPERATIVE)\s+DIAGNOSIS',
            r'(?i)OPERATION\s+PERFORMED',
        ],
        'discharge_summary': [
            r'(?i)DISCHARGE\s+SUMMARY',
            r'(?i)(?:ADMISSION|DISCHARGE)\s+(?:DATE|DIAGNOSIS)',
        ],
        'progress_note': [
            r'(?i)(?:PROGRESS|CLINIC|OFFICE|FOLLOW.?UP)\s+NOTE',
            r'(?i)(?:ONCOLOGY|MEDICAL\s+ONCOLOGY)\s+(?:NOTE|VISIT|CONSULT)',
        ],
        'consult': [
            r'(?i)(?:CONSULTATION|CONSULT)\s+(?:NOTE|REPORT)',
            r'(?i)REASON\s+FOR\s+CONSULTATION',
        ],
        'lab': [
            r'(?i)(?:LABORATORY|LAB)\s+(?:RESULTS?|REPORT)',
            r'(?i)(?:CHEMISTRY|HEMATOLOGY|CBC|BMP|CMP)\s+PANEL',
        ],
    }

    # Section boundary patterns within a document
    SECTION_PATTERNS = [
        r'(?m)^(?:ASSESSMENT|IMPRESSION|DIAGNOSIS|DIAGNOSES)[:\s]',
        r'(?m)^(?:PLAN|RECOMMENDATIONS|TREATMENT\s+PLAN)[:\s]',
        r'(?m)^(?:HISTORY\s+OF\s+PRESENT\s+ILLNESS|HPI)[:\s]',
        r'(?m)^(?:PAST\s+MEDICAL\s+HISTORY|PMH|MEDICAL\s+HISTORY)[:\s]',
        r'(?m)^(?:PHYSICAL\s+EXAM(?:INATION)?|PE|EXAM)[:\s]',
        r'(?m)^(?:LABORATORY|LABS?|LAB\s+RESULTS)[:\s]',
        r'(?m)^(?:IMAGING|RADIOL(?:OGY|OGIC))[:\s]',
        r'(?m)^(?:PATHOLOG(?:Y|IC)\s+(?:DIAGNOSIS|FINDINGS|STAGING))[:\s]',
        r'(?m)^(?:SURGICAL|OPERATIVE)\s+(?:FINDINGS|PROCEDURE)[:\s]',
        r'(?m)^(?:STAGING|TNM|AJCC|STAGE)[:\s]',
        r'(?m)^(?:FINAL\s+DIAGNOSIS|FINAL\s+PATHOLOGIC\s+DIAGNOSIS)[:\s]',
        r'(?m)^(?:GROSS\s+DESCRIPTION|MICROSCOPIC(?:\s+DESCRIPTION)?)[:\s]',
        r'(?m)^(?:SYNOPTIC|CANCER\s+CHECKLIST|CAP\s+PROTOCOL)[:\s]',
        r'(?m)^(?:IMMUNOHISTOCHEMI|IHC|MOLECULAR|GENOMIC|NGS|FISH)[:\s]',
        r'(?m)^(?:REVIEW\s+OF\s+SYSTEMS|ROS)[:\s]',
        r'(?m)^(?:MEDICATIONS|CURRENT\s+MEDICATIONS)[:\s]',
        r'(?m)^(?:ALLERGIES)[:\s]',
        r'(?m)^(?:SOCIAL\s+HISTORY|FAMILY\s+HISTORY)[:\s]',
    ]

    # Priority orderings for different extraction passes
    _PASS_PRIORITIES: dict[str, list[str]] = {
        'cancer_id': [
            'pathology', 'consult', 'discharge_summary', 'progress_note',
            'operative', 'radiology', 'lab', 'mixed',
        ],
        'staging': [
            'pathology', 'radiology', 'operative', 'consult',
            'discharge_summary', 'progress_note', 'lab', 'mixed',
        ],
        'treatment': [
            'operative', 'discharge_summary', 'progress_note', 'consult',
            'pathology', 'radiology', 'lab', 'mixed',
        ],
        'followup': [
            'progress_note', 'consult', 'discharge_summary', 'operative',
            'pathology', 'radiology', 'lab', 'mixed',
        ],
    }

    def __init__(self, chunk_target_tokens: int = 12000,
                 chunk_overlap_tokens: int = 500):
        self._target_tokens = chunk_target_tokens
        self._overlap_tokens = chunk_overlap_tokens

    def chunk_documents(self, documents: list) -> list[Chunk]:
        """Process a list of Document objects into Chunks.

        For each document:
        1. Classify note type
        2. If fits in budget, wrap as single Chunk
        3. If too large, split on section boundaries then paragraphs
        """
        all_chunks: list[Chunk] = []

        for doc in documents:
            text = doc.text.strip()
            if not text:
                continue

            note_type = self._classify_note_type(text)
            token_est = self._estimate_tokens(text)

            if token_est <= self._target_tokens:
                # Document fits in a single chunk
                chunk = Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=text,
                    chunk_type=note_type,
                    document_date=doc.date,
                    source_doc_index=doc.doc_index,
                    token_estimate=token_est,
                )
                all_chunks.append(chunk)
            else:
                # Document too large -- split on section boundaries
                logger.debug(
                    "Document %d exceeds target (%d tokens > %d), splitting",
                    doc.doc_index, token_est, self._target_tokens,
                )
                sections = self._split_into_sections(text)
                merged = self._merge_small_sections(sections)

                for piece in merged:
                    piece_tokens = self._estimate_tokens(piece)
                    if piece_tokens <= self._target_tokens:
                        chunk = Chunk(
                            chunk_id=str(uuid.uuid4()),
                            text=piece,
                            chunk_type=note_type,
                            document_date=doc.date,
                            source_doc_index=doc.doc_index,
                            token_estimate=piece_tokens,
                        )
                        all_chunks.append(chunk)
                    else:
                        # Section itself is too large, split further on paragraphs
                        sub_pieces = self._split_large_section(piece)
                        for sp in sub_pieces:
                            sp_tokens = self._estimate_tokens(sp)
                            chunk = Chunk(
                                chunk_id=str(uuid.uuid4()),
                                text=sp,
                                chunk_type=note_type,
                                document_date=doc.date,
                                source_doc_index=doc.doc_index,
                                token_estimate=sp_tokens,
                            )
                            all_chunks.append(chunk)

        logger.info(
            "Chunked %d documents into %d chunks",
            len(documents), len(all_chunks),
        )
        return all_chunks

    def _classify_note_type(self, text: str) -> str:
        """Classify document type by matching against NOTE_TYPE_PATTERNS.
        Check first 1000 chars for pattern matches. Return type with most matches.
        Default to 'mixed' if no clear match."""
        header = text[:1000]
        scores: dict[str, int] = {}

        for note_type, patterns in self.NOTE_TYPE_PATTERNS.items():
            match_count = 0
            for pattern in patterns:
                if re.search(pattern, header):
                    match_count += 1
            if match_count > 0:
                scores[note_type] = match_count

        if not scores:
            return 'mixed'

        # Return the type with the highest match count
        best_type = max(scores, key=scores.get)
        return best_type

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate: len(text) / 3.5"""
        return max(1, int(len(text) / 3.5))

    def _split_into_sections(self, text: str) -> list[tuple[str, str]]:
        """Split document into sections using SECTION_PATTERNS.
        Returns list of (section_header, section_text) tuples.
        If no sections found, return [("", full_text)]."""
        # Find all section boundary positions
        boundaries: list[tuple[int, str]] = []

        for pattern in self.SECTION_PATTERNS:
            for match in re.finditer(pattern, text):
                boundaries.append((match.start(), match.group().strip()))

        if not boundaries:
            return [("", text)]

        # Sort boundaries by position in the text
        boundaries.sort(key=lambda x: x[0])

        # Deduplicate boundaries at the same position (keep the first)
        deduped: list[tuple[int, str]] = []
        prev_pos = -1
        for pos, header in boundaries:
            if pos != prev_pos:
                deduped.append((pos, header))
                prev_pos = pos
        boundaries = deduped

        sections: list[tuple[str, str]] = []

        # If text exists before the first section header, include it as a preamble
        first_pos = boundaries[0][0]
        if first_pos > 0:
            preamble = text[:first_pos].strip()
            if preamble:
                sections.append(("", preamble))

        # Extract each section: from one boundary to the next
        for i, (pos, header) in enumerate(boundaries):
            if i + 1 < len(boundaries):
                end_pos = boundaries[i + 1][0]
            else:
                end_pos = len(text)

            section_text = text[pos:end_pos].strip()
            if section_text:
                sections.append((header, section_text))

        return sections if sections else [("", text)]

    def _merge_small_sections(self, sections: list[tuple[str, str]]) -> list[str]:
        """Merge adjacent small sections up to target token count."""
        if not sections:
            return []

        merged: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for _header, section_text in sections:
            section_tokens = self._estimate_tokens(section_text)

            if current_tokens + section_tokens <= self._target_tokens:
                # Fits in current merged chunk
                current_parts.append(section_text)
                current_tokens += section_tokens
            else:
                # Flush current merged chunk if non-empty
                if current_parts:
                    merged.append("\n\n".join(current_parts))
                # Start new merged chunk with this section
                current_parts = [section_text]
                current_tokens = section_tokens

        # Flush remaining
        if current_parts:
            merged.append("\n\n".join(current_parts))

        return merged

    def _split_large_section(self, text: str) -> list[str]:
        """Split a section that exceeds budget on paragraph boundaries.
        If a paragraph still exceeds, split on sentences with overlap."""
        # Split on double newlines (paragraph boundaries)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return [text]

        chunks: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)

            if para_tokens > self._target_tokens:
                # Paragraph itself is too large -- split on sentences with overlap
                if current_parts:
                    chunks.append("\n\n".join(current_parts))
                    current_parts = []
                    current_tokens = 0

                sentence_chunks = self._split_on_sentences(para)
                chunks.extend(sentence_chunks)
                continue

            if current_tokens + para_tokens <= self._target_tokens:
                current_parts.append(para)
                current_tokens += para_tokens
            else:
                # Flush current and start new, applying overlap
                if current_parts:
                    chunks.append("\n\n".join(current_parts))

                    # Build overlap from tail of current parts
                    overlap_parts: list[str] = []
                    overlap_tokens = 0
                    for part in reversed(current_parts):
                        part_tokens = self._estimate_tokens(part)
                        if overlap_tokens + part_tokens <= self._overlap_tokens:
                            overlap_parts.insert(0, part)
                            overlap_tokens += part_tokens
                        else:
                            break

                    current_parts = overlap_parts + [para]
                    current_tokens = overlap_tokens + para_tokens
                else:
                    current_parts = [para]
                    current_tokens = para_tokens

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return chunks

    def _split_on_sentences(self, text: str) -> list[str]:
        """Split text on sentence boundaries with overlap for very large paragraphs."""
        # Split on sentence-ending punctuation followed by whitespace
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return [text]

        chunks: list[str] = []
        current_sents: list[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self._estimate_tokens(sent)

            if current_tokens + sent_tokens <= self._target_tokens:
                current_sents.append(sent)
                current_tokens += sent_tokens
            else:
                if current_sents:
                    chunks.append(" ".join(current_sents))

                    # Build overlap from tail of current sentences
                    overlap_sents: list[str] = []
                    overlap_tokens = 0
                    for s in reversed(current_sents):
                        s_tokens = self._estimate_tokens(s)
                        if overlap_tokens + s_tokens <= self._overlap_tokens:
                            overlap_sents.insert(0, s)
                            overlap_tokens += s_tokens
                        else:
                            break

                    current_sents = overlap_sents + [sent]
                    current_tokens = overlap_tokens + sent_tokens
                else:
                    # Single sentence exceeds target -- just emit it
                    current_sents = [sent]
                    current_tokens = sent_tokens

        if current_sents:
            chunks.append(" ".join(current_sents))

        return chunks

    def prioritize_chunks(self, chunks: list[Chunk], pass_type: str) -> list[Chunk]:
        """Reorder chunks by relevance to the extraction pass.

        pass_type -> priority order of chunk_types:
        - "cancer_id": pathology, consult, discharge_summary, progress_note, ...
        - "staging": pathology, radiology, operative, consult, ...
        - "treatment": operative, discharge_summary, progress_note, consult, ...
        - "followup": (reverse chronological by document_date)
        """
        if pass_type == 'followup':
            # For followup, sort by date descending (most recent first)
            return sorted(
                chunks,
                key=lambda c: c.document_date or '',
                reverse=True,
            )

        priority_order = self._PASS_PRIORITIES.get(pass_type)
        if priority_order is None:
            logger.warning(
                "Unknown pass_type '%s', returning chunks in original order",
                pass_type,
            )
            return list(chunks)

        # Build a rank map: chunk_type -> priority rank (lower = higher priority)
        rank_map = {ct: idx for idx, ct in enumerate(priority_order)}
        default_rank = len(priority_order)

        return sorted(
            chunks,
            key=lambda c: (
                rank_map.get(c.chunk_type, default_rank),
                c.document_date or '',
                c.source_doc_index,
            ),
        )
