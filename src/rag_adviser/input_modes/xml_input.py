"""XML input parser for ragadvisor — load all answers from a structured XML file."""

from __future__ import annotations

from pathlib import Path

import defusedxml.ElementTree as ET  # noqa: N817

from rag_adviser.models import (
    AnswerType,
    BudgetTier,
    ContentType,
    DeploymentTarget,
    HardwareProfile,
    InvalidInputError,
    LatencyBudget,
    PrivacyLevel,
    QueryComplexity,
    QueryType,
    UpdateFrequency,
    UseCase,
    UserAnswers,
)


class XmlInputParser:
    """Parse a structured XML file into UserAnswers.

    Expected XML schema:

    ```xml
    <?xml version="1.0" encoding="UTF-8"?>
    <ragadvisor>
      <document_discovery>
        <document_path>/path/to/corpus</document_path>
        <future_languages>true</future_languages>
        <content_type>prose</content_type>
      </document_discovery>
      <use_case_constraints>
        <use_case>question_answering</use_case>
        <deployment>local</deployment>
        <latency>&lt;2s</latency>
        <hardware>cpu_only</hardware>
        <ram_gb>16</ram_gb>
        <vram_gb>0</vram_gb>
        <budget>free</budget>
        <privacy>none</privacy>
        <embedding_provider>huggingface</embedding_provider>
        <llm_provider>none</llm_provider>
        <implementation_lang>python</implementation_lang>
        <preferred_libraries>
          <library>langchain</library>
          <library>chromadb</library>
        </preferred_libraries>
      </use_case_constraints>
      <query_patterns>
        <query_type>natural_questions</query_type>
        <update_frequency>never</update_frequency>
        <ground_truth>false</ground_truth>
        <ground_truth_path></ground_truth_path>
        <use_llm_verification>false</use_llm_verification>
      </query_patterns>
    </ragadvisor>
    ```
    """

    def __init__(self, xml_path: Path) -> None:
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise InvalidInputError(f"XML file not found: {self.xml_path}")

    def parse(self) -> UserAnswers:
        """Parse the XML file and return UserAnswers."""
        try:
            tree = ET.parse(self.xml_path)
        except ET.ParseError as e:
            raise InvalidInputError(f"Failed to parse XML: {e}") from e

        root = tree.getroot()
        if root.tag != "ragadvisor":
            raise InvalidInputError(
                f"Expected root element 'ragadvisor', got '{root.tag}'"
            )

        answers = UserAnswers()

        # Phase 1: Document Discovery
        discovery = root.find("document_discovery")
        if discovery is not None:
            self._parse_discovery(discovery, answers)

        # Phase 2: Use Case & Constraints
        constraints_el = root.find("use_case_constraints")
        if constraints_el is not None:
            self._parse_constraints(constraints_el, answers)

        # Phase 3: Query Patterns
        patterns = root.find("query_patterns")
        if patterns is not None:
            self._parse_query_patterns(patterns, answers)

        return answers

    def _parse_discovery(self, el: ET.Element, answers: UserAnswers) -> None:
        """Parse the document_discovery section."""
        doc_path = self._get_text(el, "document_path")
        if doc_path:
            answers.document_path = Path(doc_path)

        future_langs = self._get_text(el, "future_languages")
        if future_langs:
            answers.future_languages = self._parse_bool(future_langs)

        content_type = self._get_text(el, "content_type")
        if content_type:
            try:
                answers.content_type_override = ContentType(content_type)
            except ValueError as e:
                raise InvalidInputError(
                    f"Invalid content_type: '{content_type}'. "
                    f"Valid options: {[ct.value for ct in ContentType]}"
                ) from e

    def _parse_constraints(self, el: ET.Element, answers: UserAnswers) -> None:
        """Parse the use_case_constraints section."""
        use_case = self._get_text(el, "use_case")
        if use_case:
            try:
                answers.use_case = UseCase(use_case)
            except ValueError as e:
                raise InvalidInputError(
                    f"Invalid use_case: '{use_case}'. "
                    f"Valid options: {[uc.value for uc in UseCase]}"
                ) from e

        hw = answers.constraints

        deployment = self._get_text(el, "deployment")
        if deployment:
            try:
                hw.environment = DeploymentTarget(deployment)
            except ValueError as e:
                raise InvalidInputError(
                    f"Invalid deployment: '{deployment}'. "
                    f"Valid options: {[d.value for d in DeploymentTarget]}"
                ) from e

        latency = self._get_text(el, "latency")
        if latency:
            try:
                hw.latency_budget = LatencyBudget(latency)
            except ValueError as e:
                raise InvalidInputError(
                    f"Invalid latency: '{latency}'. "
                    f"Valid options: {[lb.value for lb in LatencyBudget]}"
                ) from e

        hardware = self._get_text(el, "hardware")
        if hardware:
            try:
                hw.hardware = HardwareProfile(hardware)
            except ValueError as e:
                raise InvalidInputError(
                    f"Invalid hardware: '{hardware}'. "
                    f"Valid options: {[hp.value for hp in HardwareProfile]}"
                ) from e

        ram = self._get_text(el, "ram_gb")
        if ram:
            try:
                hw.ram_gb = float(ram)
            except ValueError as e:
                raise InvalidInputError(f"Invalid ram_gb: '{ram}' (must be a number)") from e

        vram = self._get_text(el, "vram_gb")
        if vram:
            try:
                hw.vram_gb = float(vram)
            except ValueError as e:
                raise InvalidInputError(f"Invalid vram_gb: '{vram}' (must be a number)") from e

        budget = self._get_text(el, "budget")
        if budget:
            try:
                hw.budget = BudgetTier(budget)
            except ValueError as e:
                raise InvalidInputError(
                    f"Invalid budget: '{budget}'. "
                    f"Valid options: {[b.value for b in BudgetTier]}"
                ) from e

        privacy = self._get_text(el, "privacy")
        if privacy:
            try:
                hw.privacy = PrivacyLevel(privacy)
            except ValueError as e:
                raise InvalidInputError(
                    f"Invalid privacy: '{privacy}'. "
                    f"Valid options: {[p.value for p in PrivacyLevel]}"
                ) from e

        embedding_provider = self._get_text(el, "embedding_provider")
        if embedding_provider:
            answers.embedding_provider = embedding_provider

        llm_provider = self._get_text(el, "llm_provider")
        if llm_provider:
            answers.llm_provider = llm_provider

        impl_lang = self._get_text(el, "implementation_lang")
        if impl_lang:
            answers.implementation_lang = impl_lang

        # Preferred libraries
        libs_el = el.find("preferred_libraries")
        if libs_el is not None:
            answers.preferred_libraries = [
                lib.text.strip()
                for lib in libs_el.findall("library")
                if lib.text and lib.text.strip()
            ]

    def _parse_query_patterns(self, el: ET.Element, answers: UserAnswers) -> None:
        """Parse the query_patterns section."""
        query_type = self._get_text(el, "query_type")
        if query_type:
            try:
                answers.query_type = QueryType(query_type)
            except ValueError as e:
                raise InvalidInputError(
                    f"Invalid query_type: '{query_type}'. "
                    f"Valid options: {[qt.value for qt in QueryType]}"
                ) from e

        query_complexity = self._get_text(el, "query_complexity")
        if query_complexity:
            try:
                answers.query_complexity = QueryComplexity(query_complexity)
            except ValueError as e:
                raise InvalidInputError(
                    f"Invalid query_complexity: '{query_complexity}'. "
                    f"Valid options: {[qc.value for qc in QueryComplexity]}"
                ) from e

        answer_type = self._get_text(el, "expected_answer_type")
        if answer_type:
            try:
                answers.expected_answer_type = AnswerType(answer_type)
            except ValueError as e:
                raise InvalidInputError(
                    f"Invalid expected_answer_type: '{answer_type}'. "
                    f"Valid options: {[at.value for at in AnswerType]}"
                ) from e

        # Sample queries
        samples_el = el.find("sample_queries")
        if samples_el is not None:
            answers.sample_queries = [
                q.text.strip()
                for q in samples_el.findall("query")
                if q.text and q.text.strip()
            ]

        update_freq = self._get_text(el, "update_frequency")
        if update_freq:
            try:
                answers.update_frequency = UpdateFrequency(update_freq)
            except ValueError as e:
                raise InvalidInputError(
                    f"Invalid update_frequency: '{update_freq}'. "
                    f"Valid options: {[uf.value for uf in UpdateFrequency]}"
                ) from e

        ground_truth = self._get_text(el, "ground_truth")
        if ground_truth:
            answers.has_ground_truth = self._parse_bool(ground_truth)

        gt_path = self._get_text(el, "ground_truth_path")
        if gt_path:
            answers.ground_truth_path = Path(gt_path)

        use_llm = self._get_text(el, "use_llm_verification")
        if use_llm:
            answers.use_llm_verification = self._parse_bool(use_llm)

    @staticmethod
    def _get_text(parent: ET.Element, tag: str) -> str | None:
        """Get text content of a child element, or None."""
        child = parent.find(tag)
        if child is not None and child.text:
            return child.text.strip()
        return None

    @staticmethod
    def _parse_bool(value: str) -> bool:
        """Parse a boolean string."""
        return value.lower() in ("true", "yes", "1", "on")
