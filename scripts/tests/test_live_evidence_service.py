"""Offline structural tests for TruthLens live evidence adapters."""
import unittest
from unittest.mock import patch

from backend.services import live_evidence_service as live


class LiveEvidenceTests(unittest.TestCase):
    @patch.object(live, "_get_json")
    def test_gdelt_results_are_marked_as_discovery(self, mock_get):
        mock_get.return_value = {
            "articles": [{
                "title": "Example article",
                "url": "https://example.org/story",
                "domain": "example.org",
                "seendate": "20261102T120000Z",
                "language": "English",
                "sourcecountry": "India",
            }]
        }
        results = live.search_gdelt_news("example claim")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source_type"], "news_discovery")
        self.assertIn("not evidence", results[0]["interpretation_note"])

    @patch.object(live, "_get_json")
    def test_wikidata_results_are_entities_not_evidence(self, mock_get):
        mock_get.return_value = {
            "search": [{
                "id": "Q123",
                "label": "Example",
                "description": "An example entity",
            }]
        }
        results = live.search_wikidata_entities("Example claim")
        self.assertEqual(results[0]["entity_id"], "Q123")
        self.assertEqual(results[0]["source_type"], "knowledge_graph_entity")

    @patch.object(live, "_get_html")
    def test_bbc_results_are_structured_discovery_leads(self, mock_get_html):
        mock_get_html.return_value = """
        <html><main>
          <div data-testid="newport-card">
            <a href="/news/articles/example123">
              <div data-testid="card-headline">Example BBC headline</div>
              <p>Example article summary for testing.</p>
              <time datetime="2026-11-02">2 Nov 2026</time>
            </a>
          </div>
        </main></html>
        """
        results = live.search_bbc_news("Example claim")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Example BBC headline")
        self.assertEqual(results[0]["summary"], "Example article summary for testing.")
        self.assertEqual(results[0]["url"], "https://www.bbc.com/news/articles/example123")
        self.assertEqual(results[0]["source_type"], "news_discovery")
        self.assertIn("not proof", results[0]["interpretation_note"])

    @patch.object(live, "_get_html", return_value="<html><main></main></html>")
    def test_bbc_empty_results_return_empty_list(self, _mock_get_html):
        self.assertEqual(live.search_bbc_news("Example claim"), [])

    @patch.dict("os.environ", {}, clear=True)
    def test_google_factcheck_is_optional_without_key(self):
        self.assertEqual(live.search_google_fact_checks("Example claim"), [])

    @patch.dict("os.environ", {"GOOGLE_FACT_CHECK_API_KEY": "test-key", "ENABLE_GDELT": "true"}, clear=True)
    @patch.object(live, "search_bbc_news", return_value=[])
    @patch.object(live, "search_google_fact_checks", side_effect=RuntimeError("unexpected"))
    @patch.object(live, "search_gdelt_news", return_value=[])
    @patch.object(live, "search_wikidata_entities", return_value=[])
    def test_provider_failures_do_not_crash_aggregate(self, _wd, _gdelt, _google, _bbc):
        result = live.search_live_sources("A sufficiently long test claim")
        self.assertIn("google_fact_check", result["provider_errors"])
        self.assertEqual(result["news_leads"], [])

    @patch.object(live, "search_bbc_news", return_value=[{"title": "BBC story", "source_type": "news_discovery"}])
    @patch.object(live, "search_wikidata_entities", return_value=[])
    @patch.dict("os.environ", {"ENABLE_GDELT": "false"}, clear=True)
    def test_aggregate_includes_bbc_and_skips_gdelt_by_default(self, _wd, _bbc):
        result = live.search_live_sources("A sufficiently long test claim")
        self.assertEqual(len(result["news_leads"]), 1)
        self.assertIn("bbc_search", result["provider_status"])
        self.assertTrue(result["provider_status"]["gdelt"].startswith("SKIPPED"))


if __name__ == "__main__":
    unittest.main()
