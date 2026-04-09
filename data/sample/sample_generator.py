"""
Synthetic data generator for Narameter pipeline testing.

Generates fake Indonesian text data across multiple time periods
with deliberately injected narrative shifts at specific boundaries.
"""

from __future__ import annotations

import os
import random
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

# --- Narrative Topics & Templates ---
# Each "topic" represents a distinct narrative frame.
# Shifts happen when the dominant topic changes between periods.

TOPICS = {
    "ekonomi": {
        "keywords": [
            "ekonomi", "harga", "inflasi", "pasar", "bisnis", "keuangan",
            "investasi", "anggaran", "belanja", "pajak", "APBN", "rupiah",
            "pertumbuhan", "ekspor", "impor", "perdagangan", "industri",
        ],
        "templates": [
            "Harga {item} mengalami kenaikan signifikan di pasar {lokasi}",
            "Pertumbuhan ekonomi kuartal ini mencapai {angka} persen",
            "Pemerintah mengalokasikan anggaran {angka} triliun untuk program {program}",
            "Inflasi bulan ini tercatat {angka} persen menurut BPS",
            "Investasi asing masuk sebesar {angka} miliar dolar ke sektor {sektor}",
            "Rupiah menguat terhadap dolar AS di level {angka} per dolar",
            "Ekspor komoditas {item} meningkat {angka} persen dibanding tahun lalu",
            "Pelaku UMKM mendapat bantuan modal usaha dari pemerintah daerah",
            "Bank Indonesia mempertahankan suku bunga acuan di level {angka} persen",
            "Sektor {sektor} menjadi penopang utama pertumbuhan ekonomi nasional",
        ],
    },
    "politik": {
        "keywords": [
            "politik", "pemilu", "partai", "presiden", "DPR", "pemerintah",
            "kebijakan", "demokrasi", "calon", "koalisi", "oposisi",
            "regulasi", "undang-undang", "parlemen", "menteri",
        ],
        "templates": [
            "Partai {partai} menyatakan dukungan terhadap kebijakan {kebijakan}",
            "DPR mengesahkan undang-undang tentang {topik} setelah perdebatan panjang",
            "Survei terbaru menunjukkan elektabilitas calon {nama} mencapai {angka} persen",
            "Koalisi pemerintah menggelar rapat darurat terkait isu {topik}",
            "Masyarakat menuntut transparansi dalam proses {proses}",
            "Presiden mengeluarkan peraturan baru mengenai {topik}",
            "Oposisi mengkritik kebijakan pemerintah terkait {kebijakan}",
            "Pemilu serentak dijadwalkan berlangsung pada {waktu}",
            "Menteri {kementerian} mengundurkan diri karena alasan personal",
            "Demonstrasi berlangsung damai di depan gedung {gedung}",
        ],
    },
    "kesehatan": {
        "keywords": [
            "kesehatan", "rumah sakit", "vaksin", "pandemi", "penyakit",
            "obat", "dokter", "pasien", "BPJS", "layanan kesehatan",
            "virus", "wabah", "imunisasi", "gizi", "stunting",
        ],
        "templates": [
            "Kasus {penyakit} meningkat {angka} persen di wilayah {lokasi}",
            "Program vaksinasi {vaksin} telah menjangkau {angka} juta masyarakat",
            "Pemerintah membangun {angka} rumah sakit baru di daerah terpencil",
            "BPJS Kesehatan mencatat {angka} juta peserta aktif",
            "Kampanye pencegahan {penyakit} digalakkan di seluruh Indonesia",
            "Tenaga kesehatan mendapat insentif tambahan sebesar {angka} juta rupiah",
            "Prevalensi stunting turun menjadi {angka} persen berkat program gizi nasional",
            "Layanan telemedicine membantu masyarakat di daerah terpencil",
            "Peneliti Indonesia menemukan terapi baru untuk penyakit {penyakit}",
            "Kementerian Kesehatan meluncurkan aplikasi monitoring kesehatan digital",
        ],
    },
    "pendidikan": {
        "keywords": [
            "pendidikan", "sekolah", "universitas", "kurikulum", "guru",
            "siswa", "mahasiswa", "beasiswa", "ujian", "literasi",
            "digital", "pelatihan", "kampus", "riset", "teknologi",
        ],
        "templates": [
            "Kurikulum {kurikulum} resmi diterapkan di seluruh sekolah di Indonesia",
            "Pemerintah memberikan {angka} ribu beasiswa untuk mahasiswa berprestasi",
            "Tingkat literasi digital di kalangan pelajar meningkat {angka} persen",
            "Guru honorer menuntut peningkatan kesejahteraan dan sertifikasi",
            "Universitas {universitas} meraih peringkat {angka} di tingkat Asia",
            "Program kampus merdeka mendorong {angka} ribu mahasiswa magang di industri",
            "Ujian nasional digantikan dengan asesmen kompetensi minimum",
            "Sekolah di daerah terpencil mendapat bantuan infrastruktur digital",
            "Riset perguruan tinggi Indonesia mendapat pendanaan {angka} miliar rupiah",
            "Pelatihan coding untuk anak muda digelar di {angka} kota besar",
        ],
    },
    "lingkungan": {
        "keywords": [
            "lingkungan", "iklim", "polusi", "hutan", "banjir", "kekeringan",
            "emisi", "karbon", "energi terbarukan", "sampah", "daur ulang",
            "deforestasi", "konservasi", "biodiversitas", "laut",
        ],
        "templates": [
            "Deforestasi di {lokasi} mencapai {angka} ribu hektar tahun ini",
            "Pemerintah menargetkan pengurangan emisi karbon sebesar {angka} persen",
            "Banjir bandang melanda {lokasi} akibat curah hujan ekstrem",
            "Program energi terbarukan berhasil memasok {angka} persen kebutuhan listrik",
            "Kualitas udara di {lokasi} masuk kategori tidak sehat selama {angka} hari",
            "Proyek konservasi mangrove berhasil menanam {angka} juta bibit",
            "Sampah plastik di laut Indonesia diperkirakan mencapai {angka} ton per tahun",
            "Masyarakat adat menjaga kelestarian hutan di wilayah {lokasi}",
            "Indonesia menjadi tuan rumah konferensi perubahan iklim regional",
            "Teknologi daur ulang sampah elektronik dikembangkan oleh startup lokal",
        ],
    },
}

# Filler words/phrases
ITEMS = ["beras", "minyak goreng", "gula", "BBM", "LPG", "cabai", "daging sapi", "telur"]
LOCATIONS = ["Jakarta", "Surabaya", "Bandung", "Medan", "Makassar", "Kalimantan", "Papua", "Sulawesi"]
PARTIES = ["Golkar", "PDIP", "Gerindra", "PKB", "NasDem", "PKS", "Demokrat"]
SECTORS = ["teknologi", "pertanian", "manufaktur", "pariwisata", "perikanan", "energi"]
DISEASES = ["DBD", "malaria", "TBC", "diabetes", "hipertensi", "ISPA"]
UNIVERSITIES = ["UI", "ITB", "UGM", "ITS", "Unair", "IPB", "Unpad"]


def _fill_template(template: str) -> str:
    """Fill a template with random values."""
    result = template
    result = result.replace("{item}", random.choice(ITEMS))
    result = result.replace("{lokasi}", random.choice(LOCATIONS))
    result = result.replace("{partai}", random.choice(PARTIES))
    result = result.replace("{sektor}", random.choice(SECTORS))
    result = result.replace("{penyakit}", random.choice(DISEASES))
    result = result.replace("{universitas}", random.choice(UNIVERSITIES))
    result = result.replace("{angka}", str(random.randint(1, 99)))
    # Generic fills
    result = result.replace("{kebijakan}", random.choice(["fiskal", "moneter", "sosial", "digital"]))
    result = result.replace("{topik}", random.choice(["pendidikan", "kesehatan", "ekonomi", "lingkungan"]))
    result = result.replace("{program}", random.choice(["bantuan sosial", "infrastruktur", "digitalisasi"]))
    result = result.replace("{proses}", random.choice(["pemilihan", "legislasi", "anggaran"]))
    result = result.replace("{nama}", random.choice(["A", "B", "C"]))
    result = result.replace("{waktu}", random.choice(["April 2024", "Oktober 2024"]))
    result = result.replace("{kementerian}", random.choice(["Keuangan", "Pendidikan", "Kesehatan"]))
    result = result.replace("{gedung}", random.choice(["DPR", "Istana Negara", "Balai Kota"]))
    result = result.replace("{kurikulum}", random.choice(["Merdeka", "2023", "Digital"]))
    result = result.replace("{vaksin}", random.choice(["COVID-19", "campak", "polio"]))
    return result


def generate_synthetic_data(
    n_periods: int = 8,
    docs_per_period: int = 80,
    output_path: Optional[str] = "data/sample/synthetic.csv",
    start_date: str = "2024-01-01",
    shift_at: Optional[list[int]] = None,
    noise_ratio: float = 0.05,
) -> pd.DataFrame:
    """
    Generate synthetic Indonesian text data with known narrative shifts.

    Default shift pattern (8 periods):
      ekonomi, ekonomi, ekonomi, politik, politik, kesehatan, lingkungan, lingkungan
    Shifts at: period 4 (ekonomi->politik), 6 (politik->kesehatan), 7 (kesehatan->lingkungan).
    """
    random.seed(42)

    topic_names = list(TOPICS.keys())

    if shift_at is None:
        period_topics = []
        assignment = [0, 0, 0, 1, 1, 2, 4, 4]
        for i in range(n_periods):
            idx = assignment[i] if i < len(assignment) else random.randint(0, len(topic_names) - 1)
            period_topics.append(topic_names[idx])
    else:
        current_topic_idx = 0
        period_topics = []
        for i in range(n_periods):
            if i in shift_at:
                current_topic_idx = (current_topic_idx + 1) % len(topic_names)
            period_topics.append(topic_names[current_topic_idx])

    base_date = datetime.strptime(start_date, "%Y-%m-%d")
    records = []

    for period_idx in range(n_periods):
        topic_name = period_topics[period_idx]
        topic = TOPICS[topic_name]
        templates = topic["templates"]

        n_main = int(docs_per_period * (1.0 - noise_ratio))
        n_noise = docs_per_period - n_main

        period_start = base_date + timedelta(weeks=period_idx)

        for j in range(n_main):
            n_sentences = random.choice([2, 3])
            sentences = [_fill_template(random.choice(templates)) for _ in range(n_sentences)]
            text = ". ".join(sentences)
            timestamp = period_start + timedelta(
                days=random.randint(0, 6),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
            )
            records.append({"text": text, "timestamp": timestamp, "topic": topic_name})

        other_topics = [t for t in topic_names if t != topic_name]
        for j in range(n_noise):
            noise_topic = random.choice(other_topics)
            template = random.choice(TOPICS[noise_topic]["templates"])
            text = _fill_template(template)
            timestamp = period_start + timedelta(
                days=random.randint(0, 6),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
            )
            records.append({"text": text, "timestamp": timestamp, "topic": noise_topic})

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Generated {len(df)} synthetic documents across {n_periods} periods")
    print(f"Topic assignment per period:")
    for i, topic in enumerate(period_topics):
        marker = " ⚡ SHIFT" if i > 0 and period_topics[i] != period_topics[i - 1] else ""
        print(f"  Period {i + 1}: {topic}{marker}")

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

    return df


if __name__ == "__main__":
    generate_synthetic_data()


