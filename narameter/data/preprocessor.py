"""
Indonesian text preprocessor for Narameter pipeline.

Pipeline: cleaning → normalization → tokenization → stopword removal → stemming.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from narameter.config import PreprocessingConfig

logger = logging.getLogger(__name__)

# --- Indonesian Stopwords ---
# Comprehensive list of Indonesian stopwords
INDONESIAN_STOPWORDS = {
    "ada", "adalah", "adanya", "adapun", "agak", "agaknya", "agar", "akan",
    "akankah", "akhir", "akhirnya", "aku", "akulah", "amat", "amatlah",
    "anda", "andalah", "antar", "antara", "antaranya", "apa", "apaan",
    "apabila", "apakah", "apalagi", "apatah", "artinya", "asal", "asalkan",
    "atas", "atau", "ataukah", "ataupun", "awal", "awalnya",
    "bagai", "bagaikan", "bagaimana", "bagaimanakah", "bagaimanapun", "bagi",
    "bagian", "bahkan", "bahwa", "bahwasanya", "baik", "bakal", "bakalan",
    "balik", "banyak", "bapak", "baru", "bawah", "beberapa", "begini",
    "beginian", "beginikah", "beginilah", "begitu", "begitukah", "begitulah",
    "begitupun", "bekas", "belakang", "belakangan", "belum", "belumlah",
    "benar", "benarkah", "benarlah", "berada", "berakhir", "berakhirlah",
    "berakhirnya", "berapa", "berapakah", "berapalah", "berapapun", "berarti",
    "berawal", "berbagai", "berdatangan", "beri", "berikan", "berikut",
    "berikutnya", "berjumlah", "berkali", "berkenaan", "berlainan",
    "berlalu", "berlangsung", "berlebihan", "bermacam", "bermaksud",
    "bermula", "bersama", "bersamaan", "berturut", "bertutur", "berujar",
    "berupa", "besar", "besok", "betul", "betulkah", "biasa", "biasanya",
    "bila", "bilakah", "bisa", "bisakah", "boleh", "bolehkah", "bolehlah",
    "buat", "bukan", "bukankah", "bukanlah", "bukannya", "bulan", "bung",
    "cara", "caranya", "cukup", "cukupkah", "cukuplah", "cuma",
    "dahulu", "dalam", "dan", "dapat", "dari", "daripada", "datang",
    "dekat", "demi", "demikian", "demikianlah", "dengan", "depan", "di",
    "dia", "diakhiri", "diakhirinya", "dialah", "diantara", "diantaranya",
    "diberi", "diberikan", "diberikannya", "dibuat", "dibuatnya", "didapat",
    "didatangkan", "digunakan", "diibaratkan", "diibaratkannya", "diingat",
    "diingatkan", "diinginkan", "dijumpai", "dilakukan", "dilalui",
    "dilihat", "dimaksud", "dimaksudkan", "dimaksudkannya", "dimaksudnya",
    "diminta", "dimulai", "dimulailah", "dimulainya", "dimungkinkan",
    "dini", "dipastikan", "diperbuat", "diperbuatnya", "dipergunakan",
    "diperkirakan", "diperlihatkan", "diperlukan", "diperlukannya",
    "dipersoalkan", "dipertanyakan", "dipunyai", "diri", "dirinya",
    "disampaikan", "disebut", "disebutkan", "disebutkannya", "disini",
    "disinilah", "ditambahkan", "ditandaskan", "ditanya", "ditanyai",
    "ditanyakan", "ditegaskan", "ditujukan", "ditunjuk", "ditunjuki",
    "ditunjukkan", "ditunjukkannya", "ditunjuknya", "dituturkan",
    "dituturkannya", "diucapkan", "diucapkannya", "diungkapkan", "dong",
    "dua", "dulu",
    "empat", "enggak", "enggaknya",
    "entah", "entahlah",
    "guna", "gunakan",
    "hal", "hampir", "hanya", "hanyalah", "hari", "harus", "haruslah",
    "harusnya", "hendak", "hendaklah", "hendaknya", "hingga",
    "ia", "ialah", "ibarat", "ibaratkan", "ibaratnya", "ibu", "ikut",
    "ingat", "ingatkah", "ingin", "inginkah", "ini", "inikah", "inilah",
    "itu", "itukah", "itulah",
    "jadi", "jadilah", "jadinya", "jangan", "jangankan", "janganlah",
    "jauh", "jawab", "jawaban", "jawabnya", "jelas", "jelaskan",
    "jelaslah", "jelasnya", "jika", "jikalau", "juga", "jumlah",
    "jumlahnya", "justru",
    "kala", "kalau", "kalaulah", "kalaupun", "kalian", "kami", "kamilah",
    "kamu", "kamulah", "kan", "kapan", "kapankah", "kapanpun", "karena",
    "karenanya", "kasus", "kata", "katakan", "katakanlah", "katanya",
    "ke", "keadaan", "kebetulan", "kecil", "kedua", "keduanya",
    "keinginan", "kelamaan", "kelihatan", "kelihatannya", "kelima",
    "keluar", "kembali", "kemudian", "kemungkinan", "kemungkinannya",
    "kenapa", "kepada", "kepadanya", "kesampaian", "keseluruhan",
    "keseluruhannya", "keterlaluan", "ketika", "khususnya", "kini",
    "kinilah", "kira", "kiranya", "kita", "kitalah", "kok",
    "kurang", "lagi", "lagian", "lah", "lain", "lainnya", "lalu",
    "lama", "lamanya", "lanjut", "lanjutnya", "lebih", "lewat", "lima",
    "luar", "macam", "maka", "makanya", "makin", "malah", "malahan",
    "mampu", "mampukah", "mana", "manakala", "manalagi", "masa",
    "masalah", "masalahnya", "masih", "masihkah", "masing", "mau",
    "maupun", "melainkan", "melakukan", "melalui", "melihat", "memang",
    "memastikan", "memberi", "memberikan", "membuat", "memerlukan",
    "memihak", "meminta", "memintakan", "memisalkan", "memperbuat",
    "mempergunakan", "memperkirakan", "memperlihatkan", "mempersiapkan",
    "mempersoalkan", "mempertanyakan", "mempunyai", "memulai",
    "memungkinkan", "menaiki", "menambahkan", "menandaskan", "menanti",
    "menantikan", "menanya", "menanyai", "menanyakan", "mendapat",
    "mendapatkan", "mendatang", "mendatangi", "mendatangkan", "menegaskan",
    "mengakhiri", "mengapa", "mengatakan", "mengatakannya", "mengenai",
    "mengerjakan", "mengetahui", "menggunakan", "menghendaki",
    "mengibaratkan", "mengibaratkannya", "mengingat", "mengingatkan",
    "menginginkan", "mengira", "mengucapkan", "mengucapkannya",
    "mengungkapkan", "menjadi", "menjawab", "menjelaskan", "menuju",
    "menunjuk", "menunjuki", "menunjukkan", "menunjuknya", "menurut",
    "menuturkan", "menyampaikan", "menyangkut", "menyatakan",
    "menyebutkan", "menyeluruh", "menyiapkan", "merasa", "mereka",
    "merekalah", "merupakan", "meski", "meskipun", "meyakini",
    "meyakinkan", "minta", "mirip", "misal", "misalkan", "misalnya",
    "mula", "mulai", "mulailah", "mulanya", "mungkin", "mungkinkah",
    "nah", "naik", "namun", "nanti", "nantinya", "nyaris", "nyatanya",
    "oleh", "olehnya",
    "pada", "padahal", "padanya", "pak", "paling", "panjang", "pantas",
    "para", "pasti", "pastilah", "penting", "pentingnya", "per",
    "percuma", "perlu", "perlukah", "perlunya", "pernah", "persoalan",
    "pertama", "pertanyaan", "pertanyakan", "pihak", "pihaknya", "pukul",
    "pula", "pun", "punya",
    "rasa", "rasanya", "rata", "rupanya",
    "saat", "saatnya", "saja", "sajalah", "saling", "sama", "sambil",
    "sampai", "sana", "sangat", "sangatlah", "satu", "saya", "sayalah",
    "se", "sebab", "sebabnya", "sebagai", "sebagaimana", "sebagainya",
    "sebagian", "sebaik", "sebanyak", "sebegini", "sebegitu", "sebelum",
    "sebelumnya", "sebenarnya", "seberapa", "sebesar", "sebetulnya",
    "sebisanya", "sebuah", "sebut", "sebutlah", "sebutnya", "secara",
    "secukupnya", "sedang", "sedangkan", "sedemikian", "sedikit",
    "sedikitnya", "seenaknya", "segala", "segalanya", "segera",
    "seharusnya", "sehingga", "seingat", "sejak", "sejauh", "sejenak",
    "sejumlah", "sekadar", "sekadarnya", "sekali", "sekalian",
    "sekaligus", "sekalipun", "sekarang", "sekaranglah", "sekecil",
    "seketika", "sekiranya", "sekitar", "sekitarnya", "sekurang",
    "selain", "selaku", "selalu", "selama", "selamanya", "selanjutnya",
    "seluruh", "seluruhnya", "semacam", "semakin", "semampu", "semata",
    "sempat", "semua", "semuanya", "semula", "sendiri", "sendirinya",
    "seolah", "seorang", "sepanjang", "sepantasnya", "seperlunya",
    "seperti", "sepertinya", "sepihak", "sering", "seringnya", "serta",
    "serupa", "sesaat", "sesama", "sesampai", "sesegera", "sesekali",
    "seseorang", "sesuatu", "sesuatunya", "sesudah", "sesudahnya",
    "setelah", "setempat", "setengah", "seterusnya", "setiap", "setiba",
    "setidaknya", "setinggi", "seusai", "sewaktu", "siap", "siapa",
    "siapakah", "siapapun", "sini", "sinilah", "soal", "soalnya",
    "suatu", "sudah", "sudahkah", "sudahlah", "supaya",
    "tadi", "tadinya", "tahu", "tahun", "tak", "tambah", "tambahnya",
    "tampak", "tampaknya", "tandas", "tandasnya", "tanpa", "tanya",
    "tanyakan", "tanyanya", "tapi", "tenang", "tentu", "tentulah",
    "tentunya", "tepat", "terakhir", "terasa", "terbanyak", "terdahulu",
    "terdapat", "terdiri", "terhadap", "terhadapnya", "teringat",
    "terjadi", "terjadilah", "terjadinya", "terkira", "terlalu",
    "terlebih", "terlihat", "termasuk", "ternyata", "tersampaikan",
    "tersebut", "tersebutlah", "tertentu", "tertuju", "terus",
    "terutama", "tetap", "tetapi", "tiap", "tiba", "tidak", "tidakkah",
    "tidaklah", "tiga", "tinggi", "toh", "tunjuk",
    "turut", "tutur", "tuturnya",
    "ucap", "ucapnya", "ujar", "ujarnya", "umum", "umumnya", "ungkap",
    "ungkapnya", "untuk", "usah", "usai",
    "waduh", "wah", "wahai", "waktu", "waktunya", "walau", "walaupun",
    "wong",
    "yaitu", "yakin", "yakni", "yang",
}


class TextPreprocessor:
    """
    Indonesian text preprocessor for narrative analysis.

    Pipeline steps (configurable):
    1. Remove URLs
    2. Remove mentions (@user)
    3. Remove hashtag symbols (keep text)
    4. Remove special characters and extra whitespace
    5. Lowercase normalization
    6. Tokenization
    7. Stopword removal (Indonesian)
    8. Stemming (Indonesian via Sastrawi)
    9. Length filtering
    10. Deduplication
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the preprocessor.

        Args:
            config: Preprocessing configuration. Uses defaults if None.
        """
        self.config = config or PreprocessingConfig()
        self._stemmer = None
        self._stopwords = INDONESIAN_STOPWORDS.copy()

    @property
    def stemmer(self):
        """Lazy-load the Sastrawi stemmer."""
        if self._stemmer is None and self.config.stem:
            try:
                from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
                factory = StemmerFactory()
                self._stemmer = factory.create_stemmer()
                logger.info("Sastrawi Indonesian stemmer loaded")
            except ImportError:
                logger.warning(
                    "Sastrawi not installed. Stemming disabled. "
                    "Install with: pip install Sastrawi"
                )
                self.config.stem = False
        return self._stemmer

    def preprocess(self, text: str) -> str:
        """
        Preprocess a single text string.

        Args:
            text: Raw text input.

        Returns:
            Preprocessed text string.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # 1. Remove URLs
        if self.config.remove_urls:
            text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # 2. Remove mentions
        if self.config.remove_mentions:
            text = re.sub(r"@\w+", "", text)

        # 3. Remove hashtag symbols (keep the tag text)
        if self.config.remove_hashtag_symbol:
            text = re.sub(r"#(\w+)", r"\1", text)

        # 4. Remove RT prefix (retweet marker)
        text = re.sub(r"^RT\s+", "", text)

        # 5. Remove numbers
        if self.config.remove_numbers:
            text = re.sub(r"\d+", "", text)

        # 6. Remove punctuation and special characters
        if self.config.remove_punctuation:
            text = re.sub(r"[^\w\s]", " ", text)

        # 7. Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # 8. Lowercase
        if self.config.lowercase:
            text = text.lower()

        # 9. Tokenize, remove stopwords, and stem
        tokens = text.split()

        if self.config.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]

        if self.config.stem and self.stemmer is not None:
            tokens = [self.stemmer.stem(t) for t in tokens]

        text = " ".join(tokens)

        # 10. Length check
        if len(text) < self.config.min_length:
            return ""

        return text

    def preprocess_batch(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> list[str]:
        """
        Preprocess a batch of texts.

        Args:
            texts: List of raw text strings.
            show_progress: Whether to show a progress bar.

        Returns:
            List of preprocessed text strings (empty strings for filtered texts).
        """
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(texts, desc="Preprocessing", unit="text")
            except ImportError:
                iterator = texts
        else:
            iterator = texts

        results = [self.preprocess(text) for text in iterator]

        # Statistics
        total = len(texts)
        non_empty = sum(1 for r in results if r)
        filtered = total - non_empty
        logger.info(
            f"Preprocessed {total} texts: {non_empty} kept, {filtered} filtered out"
        )

        return results

    def preprocess_dataframe(
        self,
        df: "pd.DataFrame",
        text_col: str = "text",
        output_col: str = "text_clean",
    ) -> "pd.DataFrame":
        """
        Preprocess texts in a DataFrame.

        Args:
            df: Input DataFrame.
            text_col: Column containing raw text.
            output_col: Column name for preprocessed text.

        Returns:
            DataFrame with added preprocessed column, filtered for non-empty.
        """
        import pandas as pd

        df = df.copy()
        df[output_col] = self.preprocess_batch(df[text_col].tolist())

        # Remove rows where preprocessing resulted in empty string
        before = len(df)
        df = df[df[output_col].str.len() > 0].reset_index(drop=True)
        after = len(df)

        if before != after:
            logger.info(f"Removed {before - after} empty rows after preprocessing")

        # Deduplication
        if self.config.deduplicate:
            before_dedup = len(df)
            df = df.drop_duplicates(subset=[output_col]).reset_index(drop=True)
            deduped = before_dedup - len(df)
            if deduped > 0:
                logger.info(f"Removed {deduped} duplicate texts")

        return df
