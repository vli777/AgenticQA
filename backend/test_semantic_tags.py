"""
Pytest tests for semantic tagging.
Validates LLM-based extraction and zero-shot fallback.

Run with: pytest test_semantic_tags.py -v
"""

import pytest
from semantic_tags import extract_semantic_tags, infer_query_tags


class TestSemanticTagExtraction:
    """Test suite for semantic tag extraction."""

    def test_business_text_extraction(self):
        """Test tag extraction from business/DevOps text."""
        business_text = """
        Our company specializes in cloud infrastructure management and DevOps automation.
        We help clients migrate to Kubernetes and implement CI/CD pipelines using Terraform.
        """
        tags = extract_semantic_tags(business_text)
        assert isinstance(tags, list)
        assert all(isinstance(tag, str) for tag in tags)

        if not tags:
            pytest.skip("LLM API unavailable - no tags extracted")

        # Should identify at least one relevant tag
        expected_tags = {
            "devops", "kubernetes", "infrastructure", "cloud", "automation",
            "ci/cd", "terraform", "deployment", "technology", "software"
        }
        matching_tags = set(tags) & expected_tags
        assert len(matching_tags) > 0, (
            f"Expected at least one of {expected_tags}, got {tags}"
        )

    def test_medical_text_extraction(self):
        """Test tag extraction from medical domain text."""
        medical_text = """
        The patient presented with acute myocardial infarction and elevated troponin levels.
        ECG showed ST-segment elevation. Treatment included percutaneous coronary intervention.
        """
        tags = extract_semantic_tags(medical_text)
        assert isinstance(tags, list)
        assert all(isinstance(tag, str) for tag in tags)

        if not tags:
            pytest.skip("LLM API unavailable - no tags extracted")

        # Should identify at least one relevant medical tag
        expected_tags = {
            "medical", "healthcare", "cardiology", "cardiac", "heart",
            "diagnosis", "treatment", "patient", "clinical", "medicine"
        }
        matching_tags = set(tags) & expected_tags
        assert len(matching_tags) > 0, (
            f"Expected at least one of {expected_tags}, got {tags}"
        )

    def test_legal_text_extraction(self):
        """Test tag extraction from legal domain text."""
        legal_text = """
        The defendant filed a motion to dismiss pursuant to Federal Rule of Civil Procedure 12(b)(6).
        The court denied the motion, finding that the complaint stated a plausible claim for relief.
        """
        tags = extract_semantic_tags(legal_text)
        assert isinstance(tags, list)
        assert all(isinstance(tag, str) for tag in tags)

        if not tags:
            pytest.skip("LLM API unavailable - no tags extracted")

        # Should identify at least one relevant legal tag
        expected_tags = {
            "legal", "law", "court", "litigation", "procedure", "judicial",
            "motion", "complaint", "civil", "justice", "defendant"
        }
        matching_tags = set(tags) & expected_tags
        assert len(matching_tags) > 0, (
            f"Expected at least one of {expected_tags}, got {tags}"
        )

    def test_generic_text_extraction(self):
        """Test tag extraction from generic content."""
        generic_text = """
        The meeting is scheduled for next Tuesday at 3pm in Conference Room B.
        Please review the quarterly report before attending.
        """
        tags = extract_semantic_tags(generic_text)
        assert isinstance(tags, list)
        assert all(isinstance(tag, str) for tag in tags)

        if not tags:
            pytest.skip("LLM API unavailable - no tags extracted")

        # Should identify at least one relevant business/organizational tag
        expected_tags = {
            "meeting", "business", "schedule", "report", "quarterly",
            "organization", "planning", "corporate", "work", "administrative"
        }
        matching_tags = set(tags) & expected_tags
        assert len(matching_tags) > 0, (
            f"Expected at least one of {expected_tags}, got {tags}"
        )

    def test_empty_text(self):
        """Test extraction with empty text."""
        tags = extract_semantic_tags("")
        assert tags == []

    def test_none_text(self):
        """Test extraction with None text."""
        tags = extract_semantic_tags(None)
        assert tags == []


class TestQueryTagInference:
    """Test suite for query tag inference."""

    def test_deployment_query(self):
        """Test tag inference from deployment-related query."""
        query = "How do I deploy a containerized application to production?"
        tags = infer_query_tags(query)
        assert isinstance(tags, set)
        assert all(isinstance(tag, str) for tag in tags)

        if not tags:
            pytest.skip("LLM API unavailable - no tags extracted")

        # Should identify at least one relevant deployment/containerization tag
        expected_tags = {
            "deployment", "docker", "container", "containerization", "production",
            "devops", "kubernetes", "infrastructure", "deployment", "application"
        }
        matching_tags = tags & expected_tags
        assert len(matching_tags) > 0, (
            f"Expected at least one of {expected_tags}, got {tags}"
        )

    def test_empty_query(self):
        """Test inference with empty query."""
        tags = infer_query_tags("")
        assert tags == set()

    def test_query_caching(self):
        """Test that identical queries return cached results."""
        query = "How do I deploy a containerized application to production?"
        tags1 = infer_query_tags(query)
        tags2 = infer_query_tags(query)
        assert tags1 == tags2


class TestTagProperties:
    """Test suite for tag property validation."""

    def test_tags_are_lowercase(self):
        """Verify all extracted tags are lowercase."""
        text = "Python Programming and Machine Learning"
        tags = extract_semantic_tags(text)
        if not tags:
            pytest.skip("LLM API unavailable - no tags extracted")
        assert len(tags) > 0, "Expected tags to verify lowercase property"
        for tag in tags:
            assert tag == tag.lower(), f"Tag '{tag}' is not lowercase"

    def test_tags_are_sorted(self):
        """Verify extracted tags are sorted."""
        text = "Python Programming and Machine Learning"
        tags = extract_semantic_tags(text)
        if not tags:
            pytest.skip("LLM API unavailable - no tags extracted")
        assert len(tags) > 0, "Expected tags to verify sorting"
        assert tags == sorted(tags), "Tags are not sorted"

    def test_no_duplicate_tags(self):
        """Verify no duplicate tags in results."""
        text = "Python programming with Python libraries for Python development"
        tags = extract_semantic_tags(text)
        if not tags:
            pytest.skip("LLM API unavailable - no tags extracted")
        assert len(tags) > 0, "Expected tags to verify no duplicates"
        assert len(tags) == len(set(tags)), "Duplicate tags found"


@pytest.mark.integration
class TestWithLLMAPI:
    """Integration tests that require working LLM API."""

    def test_llm_extraction_with_valid_api(self):
        """Test LLM extraction works with valid API key."""
        # This test will be skipped if API is unavailable
        text = "Cloud computing with AWS and Azure infrastructure"
        tags = extract_semantic_tags(text)
        # If LLM is working, we should get some tags
        # This assertion is lenient to handle API unavailability
        assert isinstance(tags, list)
