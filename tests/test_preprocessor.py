"""Tests for the Indonesian text preprocessor."""

import pytest

from narameter.data.preprocessor import TextPreprocessor, INDONESIAN_STOPWORDS
from narameter.config import PreprocessingConfig


@pytest.fixture
def preprocessor():
    """Create a preprocessor with default config but stemming disabled for speed."""
    config = PreprocessingConfig(stem=False)
    return TextPreprocessor(config)


@pytest.fixture
def preprocessor_with_stemming():
    """Create a preprocessor with stemming enabled."""
    config = PreprocessingConfig(stem=True)
    return TextPreprocessor(config)


class TestURLRemoval:
    def test_removes_http_urls(self, preprocessor):
        text = "Berita terbaru http://example.com tentang ekonomi"
        result = preprocessor.preprocess(text)
        assert "http" not in result
        assert "example" not in result

    def test_removes_https_urls(self, preprocessor):
        text = "Lihat https://www.news.co.id/artikel untuk info"
        result = preprocessor.preprocess(text)
        assert "https" not in result


class TestMentionRemoval:
    def test_removes_mentions(self, preprocessor):
        text = "@user123 mengatakan bahwa ekonomi membaik"
        result = preprocessor.preprocess(text)
        assert "@user123" not in result
        assert "user123" not in result


class TestHashtagHandling:
    def test_removes_hashtag_symbol_keeps_text(self, preprocessor):
        text = "Trending #EkonomiIndonesia hari ini"
        result = preprocessor.preprocess(text)
        assert "#" not in result
        assert "ekonomiindonesia" in result  # lowercase


class TestStopwordRemoval:
    def test_removes_indonesian_stopwords(self, preprocessor):
        text = "yang dan di dari untuk dengan akan pada"
        result = preprocessor.preprocess(text)
        # All common stopwords should be removed
        assert result == ""

    def test_keeps_content_words(self, preprocessor):
        text = "ekonomi Indonesia mengalami pertumbuhan signifikan"
        result = preprocessor.preprocess(text)
        assert "ekonomi" in result
        assert "pertumbuhan" in result


class TestLowercasing:
    def test_converts_to_lowercase(self, preprocessor):
        text = "Presiden MEMBUKA Sidang Tahunan"
        result = preprocessor.preprocess(text)
        assert result == result.lower()


class TestMinLength:
    def test_filters_short_texts(self, preprocessor):
        result = preprocessor.preprocess("ab")
        assert result == ""

    def test_keeps_long_enough_texts(self, preprocessor):
        result = preprocessor.preprocess("ekonomi pertumbuhan nasional")
        assert len(result) >= preprocessor.config.min_length


class TestBatchProcessing:
    def test_batch_returns_same_length(self, preprocessor):
        texts = ["ekonomi naik", "politik berubah", "kesehatan penting"]
        results = preprocessor.preprocess_batch(texts, show_progress=False)
        assert len(results) == len(texts)

    def test_batch_processes_correctly(self, preprocessor):
        texts = [
            "Harga beras naik di Jakarta",
            "@user ini https://link.com berita",
            "ekonomi",
        ]
        results = preprocessor.preprocess_batch(texts, show_progress=False)
        assert len(results) == 3


class TestStopwordsCompleteness:
    def test_common_stopwords_present(self):
        common = ["yang", "dan", "di", "dari", "untuk", "dengan", "ini", "itu"]
        for word in common:
            assert word in INDONESIAN_STOPWORDS, f"'{word}' missing from stopwords"

    def test_stopwords_are_lowercase(self):
        for word in INDONESIAN_STOPWORDS:
            assert word == word.lower(), f"Stopword '{word}' should be lowercase"
