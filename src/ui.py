"""Streamlit presentation for live grounded screening results."""

from __future__ import annotations

import os
from datetime import datetime
from urllib.parse import urlsplit

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import pandas as pd
import streamlit as st

from src.cache import DailyLimitExceeded, SQLiteAnalysisCache
from src.config import (
    ALL_MODELS,
    BODY_TYPES,
    CACHE_DB_PATH,
    LOCATIONS,
    MODELS_BY_BRAND,
    PLATE_CODES,
    TRANSMISSION_TYPES,
    DISCLAIMER,
    SOLID_COMPARABLES,
)
from src.pipeline import PipelineError, ScreeningPipelineResult, run_screening
from src.scoring import negotiation_advice_text, price_banner_warning, seller_questions_for_cheap
from src.search_provider import GeminiGroundingProvider


_COPY_SUMMARY_BUTTON = st.components.v2.component(
        "screening_copy_summary",
        html="""
        <button id="copy-summary" type="button">Salin ringkasan</button>
        <span id="copy-status" role="status" aria-live="polite"></span>
        """,
        js="""
        export default function (component) {
            const { data, parentElement } = component
            const button = parentElement.querySelector("#copy-summary")
            const status = parentElement.querySelector("#copy-status")
            if (!button || !status) return
            button.onclick = async () => {
                const text = data?.text ?? ""
                try {
                    await navigator.clipboard.writeText(text)
                    status.textContent = "Ringkasan tersalin"
                } catch (error) {
                    const textarea = document.createElement("textarea")
                    textarea.value = text
                    textarea.style.position = "fixed"
                    parentElement.appendChild(textarea)
                    textarea.select()
                    const copied = document.execCommand("copy")
                    textarea.remove()
                    status.textContent = copied ? "Ringkasan tersalin" : "Penyalinan gagal."
                }
            }
        }
        """,
)


def _format_price(price: int) -> str:
    return f"Rp {price:,.0f}".replace(",", ".")


def _format_price_compact(price: int | float | None) -> str:
    """Format price in millions (jt) or billions (M) to avoid UI clipping."""
    if price is None:
        return "-"
    if price >= 1_000_000_000:
        val = price / 1_000_000_000
        return f"Rp {val:.2f} M".replace(".00", "").replace(".", ",")
    val = price / 1_000_000
    if val.is_integer():
        return f"Rp {int(val)} jt"
    return f"Rp {val:.1f} jt".replace(".", ",")


def _daily_limit() -> int:
    value = os.environ.get("DAILY_ANALYSIS_LIMIT")
    if not value:
        try:
            value = str(st.secrets.get("DAILY_ANALYSIS_LIMIT", "10"))
        except Exception:
            value = "10"
    try:
        limit = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("DAILY_ANALYSIS_LIMIT harus berupa bilangan bulat.") from error
    if limit < 1:
        raise ValueError("DAILY_ANALYSIS_LIMIT harus minimal 1.")
    return limit


def _read_listing_form() -> dict:
    with st.sidebar:
        st.header("Data listing")
        brand = st.selectbox("Merek *", options=sorted(MODELS_BY_BRAND))
        model_options = MODELS_BY_BRAND.get(brand, ALL_MODELS)
        model = st.selectbox("Model *", options=model_options)
        variants = ["E", "G", "V", "VX", "GR", "Prime", "Style"]
        variant = st.selectbox("Varian/tipe *", options=variants)
        body_type = st.selectbox("Jenis bodi *", options=BODY_TYPES)
        year = st.number_input(
            "Tahun *", min_value=1980, max_value=datetime.now().year, value=min(2020, datetime.now().year), step=1
        )
        transmission = st.selectbox("Transmisi *", options=TRANSMISSION_TYPES)
        kilometer = st.number_input("Kilometer (KM) *", min_value=0, max_value=1_500_000, value=80_000, step=1000)
        listing_price = st.number_input(
            "Harga listing (Rp) *", min_value=1_000_000, max_value=20_000_000_000,
            value=220_000_000, step=1_000_000,
        )
        location = st.selectbox("Lokasi *", options=LOCATIONS)
        plate_code = st.selectbox("Kode plat", options=PLATE_CODES)
        description = st.text_area(
            "Deskripsi penjual (opsional)",
            height=140,
            placeholder="Tempel informasi kondisi, riwayat servis, pajak, dan dokumen jika tersedia.",
        )
        search_mode_label = st.radio(
            "Metode Pencarian *",
            options=[
                "Pencarian Langsung Marketplace (Default - Bebas Kuota AI)",
                "Google Search Grounding (Gemini AI)",
            ],
            index=0,
            help="Pencarian Langsung membaca katalog Carsome & Carmudi secara otomatis tanpa memakan kuota AI. Google Search Grounding mencari via web search.",
        )
        search_mode = "direct" if "Langsung" in search_mode_label else "grounding"

        manual_urls = st.text_area(
            "Tempel URL listing pembanding (opsional, maks 10)",
            height=90,
            placeholder="Biarkan kosong untuk pencarian otomatis, atau tempel URL listing khusus per baris.",
        )
        analyze = st.button("Analisa", type="primary", width="stretch", icon=":material/search:")
    return {
        "brand": brand,
        "model": model,
        "variant": variant,
        "body_type": body_type,
        "year": int(year),
        "transmission": transmission,
        "kilometer": int(kilometer),
        "listing_price": int(listing_price),
        "location": location,
        "plate_code": plate_code,
        "deskripsi": description.strip(),
        "manual_listing_urls": manual_urls.strip(),
        "search_mode": search_mode,
    }, analyze


def _summary_text(result: ScreeningPipelineResult, user_input: dict) -> str:
    stats = result.statistics
    n_count = stats["count"]
    is_indicative = n_count < SOLID_COMPARABLES and n_count >= 2
    range_label = "min/max" if n_count < SOLID_COMPARABLES else "P25/P75"
    price_range = "Belum cukup data" if stats["price_median"] is None else (
        f"{_format_price(stats['price_low'])} / {_format_price(stats['price_median'])} / "
        f"{_format_price(stats['price_high'])} ({range_label})"
    )
    lines = [
        f"AI Car Screening: {user_input['brand']} {user_input['model']} {user_input['variant']} {user_input['year']}",
        f"Listing: {_format_price(user_input['listing_price'])} | {user_input['kilometer']:,} km | {user_input['transmission']} | {user_input['location']}",
        f"Skor: {result.screening.get('total_score_display', result.screening.get('total_score'))} ({result.screening['category_label']})",
        f"Harga rendah / median / tinggi: {price_range}",
        f"Kesimpulan: {result.final_analysis.kesimpulan}",
        f"Keyakinan: {result.confidence['level']}. {result.confidence['reason']}",
        f"Pembanding nyata: {n_count}",
        f"Model AI: {result.model_used or 'Gemini'}",
        f"Panggilan grounding: {result.usage['grounding_calls']}; token input/output: {result.usage['input_tokens']}/{result.usage['output_tokens']}",
    ]
    if is_indicative:
        lines.insert(3, "Mode: INDIKATIF - data sangat sedikit")
    nego = result.final_analysis.saran_nego
    if nego.awal is not None:
        lines.append(f"Negosiasi awal/target/batas: {_format_price(nego.awal)} / {_format_price(nego.realistis)} / {_format_price(nego.batas)}")
    lines.extend(f"Sumber: {source.url}" for source in result.sources)
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def _render_market_chart(result: ScreeningPipelineResult, listing_price: int) -> None:
    prices = [{"price_million": item.price / 1_000_000} for item in result.comparables]
    if not prices:
        st.info("Histogram belum tersedia karena tidak ada pembanding nyata yang lolos validasi.")
        return
    n_count = len(prices)
    if n_count >= 8:
        # Histogram untuk n >= 8
        spec = {
            "layer": [
                {
                    "data": {"values": prices},
                    "mark": {"type": "bar", "color": "#4b806a"},
                    "encoding": {
                        "x": {"field": "price_million", "bin": {"maxbins": 10}, "type": "quantitative", "title": "Harga listing pembanding (juta Rp)"},
                        "y": {"aggregate": "count", "type": "quantitative", "title": "Jumlah listing"},
                        "tooltip": [{"aggregate": "count", "type": "quantitative", "title": "Jumlah listing"}],
                    },
                },
                {
                    "data": {"values": [{"listing_million": listing_price / 1_000_000}]},
                    "mark": {"type": "rule", "color": "#c3493f", "strokeWidth": 3},
                    "encoding": {"x": {"field": "listing_million", "type": "quantitative"}},
                },
            ],
            "height": 250,
        }
    else:
        # Grafik titik horizontal untuk n < 8
        dot_data = [{"label": f"Pembanding {i+1}", "price_million": p["price_million"]} for i, p in enumerate(prices)]
        spec = {
            "layer": [
                {
                    "data": {"values": dot_data},
                    "mark": {"type": "circle", "size": 120, "color": "#4b806a", "opacity": 0.8},
                    "encoding": {
                        "x": {"field": "price_million", "type": "quantitative", "title": "Harga (juta Rp)"},
                        "y": {"value": 0},
                        "tooltip": [
                            {"field": "label", "type": "nominal", "title": "Unit"},
                            {"field": "price_million", "type": "quantitative", "title": "Harga (jt)", "format": ",.1f"},
                        ],
                    },
                },
                {
                    "data": {"values": [{"listing_million": listing_price / 1_000_000}]},
                    "mark": {"type": "rule", "color": "#c3493f", "strokeWidth": 3, "strokeDash": [4, 2]},
                    "encoding": {"x": {"field": "listing_million", "type": "quantitative"}},
                },
                {
                    "data": {"values": [{"listing_million": listing_price / 1_000_000}]},
                    "mark": {"type": "text", "color": "#c3493f", "dy": -15, "fontSize": 11},
                    "encoding": {
                        "x": {"field": "listing_million", "type": "quantitative"},
                        "text": {"value": "Harga listing"},
                    },
                },
            ],
            "height": 80,
        }
    st.vega_lite_chart(spec, width="stretch", alt="Sebaran harga pembanding dengan garis harga listing pengguna")


def _render_comparables(result: ScreeningPipelineResult, listing_price: int) -> None:
    if getattr(result, "tier_label", None):
        st.info(f"ℹ️ **Status Tingkatan Data**: Pembanding diambil dari {result.tier_label}.")

    rows = []
    for item in result.comparables:
        # Hanya sertakan baris dari halaman detail
        if getattr(item, "page_type", "detail") != "detail":
            continue
        diff_pct = (item.price - listing_price) / listing_price * 100 if listing_price else 0
        domain = urlsplit(item.source_url).netloc if item.source_url else "-"
        tier_val = getattr(item, "tier", 1)
        tier_display = f"Tier {tier_val}"
        if getattr(item, "is_twin_model", False):
            tier_display += " (Kembar)"
        weight_val = getattr(item, "weight", 1.0)
        note_val = getattr(item, "notes", "") or "-"
        rows.append({
            "Merek/Model/Varian": f"{item.brand} {item.model} {item.variant}".strip(),
            "Tier": tier_display,
            "Bobot": f"{weight_val:.1f}",
            "Tahun": item.year if item.year is not None else "tidak diketahui",
            "KM": f"{item.kilometer:,}" + (" (est)" if getattr(item, "is_km_estimated", False) else ""),
            "Lokasi": item.location,
            "Harga": _format_price(item.price),
            "Selisih %": diff_pct,
            "Catatan": note_val,
            "Sumber": item.source_url,
            "Tanggal ambil": item.fetched_at[:10] if item.fetched_at else "-",
        })
    if not rows:
        st.info("Tidak ada listing pembanding nyata yang lolos validasi. Tidak ada data dummy yang digunakan.")
        return
    table = pd.DataFrame(rows)
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        alt="Listing pembanding ter-grounding dari halaman detail, harga, selisih, tautan, dan tanggal pengambilan",
        column_config={
            "Selisih %": st.column_config.NumberColumn(format="%+.1f%%"),
            "Sumber": st.column_config.LinkColumn(display_text="Buka listing"),
        },
    )



def _render_analysis(result: ScreeningPipelineResult, user_input: dict, copy_button) -> None:
    stats = result.statistics
    screening = result.screening
    st.subheader("Ringkasan unit yang dianalisis")
    st.write(
        f"**{user_input['brand']} {user_input['model']} {user_input['variant']} {user_input['year']}** | "
        f"{user_input['body_type']} | {user_input['transmission']} | {user_input['kilometer']:,} km | "
        f"{user_input['location']} | Plat {user_input['plate_code']}"
    )
    st.write(f"Harga listing: **{_format_price(user_input['listing_price'])}**")

    if getattr(result, "funnel_metrics", None):
        fm = result.funnel_metrics
        st.markdown("##### 📊 Funnel Pengambilan Data Pembanding")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Katalog Dibuka", fm.get("catalog_urls_opened", 0))
        f2.metric("Tautan Ditemukan", fm.get("detail_links_found", 0))
        f3.metric("Detail Terbaca", fm.get("detail_pages_parsed", 0))
        f4.metric("Lolos Validasi", fm.get("valid_comparables", 0))

    model_display = result.model_used or "Gemini"
    if result.warning:
        st.error(result.warning, icon=":material/warning:")
    if result.ai_error:
        st.warning(result.ai_error)
    if result.cache_hit:
        st.caption(f"🤖 Model AI: **{model_display}** (Data listing dimuat dari cache SQLite 24 jam).")
    else:
        st.caption(f"🤖 Model AI yang digunakan: **{model_display}**")

    score_col, price_col = st.columns([1, 2])
    with score_col:
        score_label = screening.get("total_score_display") or (
            f"{screening['total_score']}/100" if screening.get("total_score") is not None else "Tidak dapat dinilai"
        )
        st.metric("Skor screening", score_label)
        st.write(screening["category_label"])
        if screening.get("is_indicative"):
            st.caption("🔶 Skor indikatif - data sangat sedikit")
    with price_col:
        st.metric("Label harga", screening["price_label"])
        if stats["price_median"] is not None:
            n_count = stats["count"]
            range_label = "min/max" if n_count < SOLID_COMPARABLES else "P25/P75"
            low_col, median_col, high_col = st.columns(3)
            low_col.metric(
                f"Rendah ({range_label.split('/')[0]})",
                _format_price_compact(stats["price_low"]),
                help=_format_price(stats["price_low"]),
            )
            median_col.metric(
                "Tengah (median)",
                _format_price_compact(stats["price_median"]),
                help=_format_price(stats["price_median"]),
            )
            high_col.metric(
                f"Tinggi ({range_label.split('/')[1]})",
                _format_price_compact(stats["price_high"]),
                help=_format_price(stats["price_high"]),
            )
        else:
            st.caption("Rentang harga belum tersedia tanpa pembanding tervalidasi.")
        st.caption(f"Rentang dihitung dari {stats['count']} listing nyata yang lolos pembersihan.")

    with st.expander("Alasan skor dan kelengkapan input", expanded=True):
        for component in screening["components"]:
            score_display = component.get("score_display", f"{component['score']:.0f}" if component.get('assessed', True) else "-")
            st.write(f"**{component['name']}** ({component['weight'] * 100:.0f}%): {score_display}/100")
            st.caption(component["reason"])
        filled = ["merek", "model", "varian", "tahun", "transmisi", "kilometer", "harga listing", "lokasi"]
        if user_input.get("plate_code") != "Lainnya":
            filled.append("kode plat")
        if user_input.get("deskripsi"):
            filled.append("deskripsi penjual")
        missing = ["deskripsi penjual"] if not user_input.get("deskripsi") else []
        st.markdown("**Terisi:** " + ", ".join(filled))
        st.markdown("**Belum tersedia:** " + (", ".join(missing) if missing else "tidak ada"))

    st.subheader("Kesimpulan akhir")
    st.write(result.final_analysis.kesimpulan)
    st.subheader("Saran negosiasi")
    nego = result.final_analysis.saran_nego
    price_median = stats.get("price_median") or 0
    listing_price = int(user_input.get("listing_price") or 0)
    price_diff_pct = (listing_price - price_median) / price_median if price_median > 0 else 0
    nego_warning = negotiation_advice_text(price_diff_pct, stats["count"])
    if nego_warning is not None:
        st.info(nego_warning)
    elif nego.awal is None:
        st.info("Saran nominal belum tersedia karena tidak ada median pembanding nyata.")
    else:
        opening, target, maximum = st.columns(3)
        opening.metric("Tawar awal (estimasi)", _format_price_compact(nego.awal), help=_format_price(nego.awal))
        target.metric("Target realistis (estimasi)", _format_price_compact(nego.realistis), help=_format_price(nego.realistis))
        maximum.metric("Batas maksimal (estimasi)", _format_price_compact(nego.batas), help=_format_price(nego.batas))
        st.caption("Estimasi dari harga listing pembanding dikoreksi 3-10%. Bukan harga transaksi terverifikasi.")

    st.subheader("Posisi di pasar")
    n_count = stats["count"]
    if n_count == 0:
        st.info("Posisi pasar belum dapat ditentukan karena belum ada pembanding yang valid.")
    elif n_count < SOLID_COMPARABLES and n_count >= 1:
        # Teks deskriptif untuk data sedikit, bukan persentil
        prices = sorted([item.price for item in result.comparables if item.price > 0])
        below = sum(1 for p in prices if listing_price > p)
        if below == n_count:
            st.write(f"Harga listing **lebih tinggi dari semua {n_count} pembanding**.")
        elif below == 0:
            st.write(f"Harga listing **lebih rendah dari semua {n_count} pembanding**.")
        else:
            st.write(f"Harga listing **lebih tinggi dari {below} dan lebih rendah dari {n_count - below}** dari {n_count} pembanding.")
        _render_market_chart(result, listing_price)
    elif stats.get("listing_percentile") is not None:
        st.write(f"Harga listing berada pada **persentil ke-{stats['listing_percentile']:.0f}** dari {n_count} pembanding.")
        _render_market_chart(result, listing_price)

    st.subheader("Tabel pembanding")
    _render_comparables(result, user_input["listing_price"])

    # Saran tindakan saat 0 pembanding
    if n_count == 0 and getattr(result, "manual_search_urls", None):
        st.markdown("#### 🔍 Buka Pencarian Manual di Marketplace (Filter Terisi)")
        st.write("Belum ada pembanding valid yang lolos filter otomatis. Anda dapat membuka pencarian manual di marketplace berikut dengan filter yang sudah terpasang:")
        cols = st.columns(3)
        for idx, (m_name, m_url) in enumerate(result.manual_search_urls.items()):
            cols[idx % 3].link_button(f"Buka {m_name}", m_url, use_container_width=True)
        st.info("💡 **Tips Pelonggaran**: Coba kosongkan nama varian saat screening, atau pilih model kembar sekelas jika tersedia.")

    st.subheader("Analisis KM")
    if n_count == 0 or stats.get("average_km_per_year") is None:
        st.info("Rata-rata KM/tahun pasar belum dapat dihitung karena belum ada pembanding yang valid.")
    else:
        current_year = datetime.now().year
        listing_km_per_year = user_input["kilometer"] / max(1, current_year - user_input["year"])
        difference = (listing_km_per_year - stats["average_km_per_year"]) / stats["average_km_per_year"] * 100 if stats["average_km_per_year"] else 0
        st.write(
            f"Unit: **{listing_km_per_year:,.0f} km/tahun**; rata-rata pembanding: "
            f"**{stats['average_km_per_year']:,.0f} km/tahun** ({difference:+.1f}%). Ini estimasi dari odometer dan tahun tercatat."
        )

    st.subheader("Tingkat keyakinan")
    st.write(f"**{result.confidence['level']}**. {result.confidence['reason']}")

    flags = screening["rule_flags"] + screening["ai_flags"]
    st.subheader("Red flag")
    if flags:
        for flag in flags:
            source = "Aturan" if flag["source"] == "rule" else "AI"
            st.warning(f"[{source}] {flag['message']}")
    else:
        st.success("Tidak ada red flag yang terdeteksi.")

    st.subheader("Pertanyaan untuk penjual")
    for question in result.final_analysis.pertanyaan_untuk_penjual:
        st.write(f"- {question}")
    st.subheader("Checklist inspeksi")
    for category, checks in result.final_analysis.checklist_inspeksi.model_dump().items():
        st.markdown(f"**{category.title()}**")
        for check in checks:
            st.checkbox(check, key=f"inspection-{category}-{check}")

    st.subheader("Catatan biaya dan analisa")
    st.write("Periksa pajak tahunan di Samsat. Jika plat berasal dari luar wilayah, tanyakan biaya balik nama/mutasi; cek juga kebutuhan servis setelah inspeksi. Biaya pasti tidak dapat ditentukan dari data listing ini.")
    for note in result.final_analysis.catatan:
        st.write(f"- {note}")

    with st.expander("Sumber grounding, katalog, dan data yang dibuang", expanded=False):
        if getattr(result, "catalog_urls_opened", None):
            st.markdown("**Halaman Katalog Marketplace yang Dibuka (URL Terfilter)**")
            for cat in result.catalog_urls_opened:
                status_icon = "✅" if cat.get("status") == "success" else "⚠️"
                st.markdown(f"- {status_icon} **{cat.get('domain', '')}** (Hal. {cat.get('page', 1)}): [{cat.get('url', '')}]({cat.get('url', '')}) — *{cat.get('links_found', 0)} tautan detail*")

        st.markdown("**Kueri Google Search**")
        if result.queries:
            for query in result.queries:
                st.write(f"- `{query}`")
        else:
            st.caption("Tidak ada kueri grounding (mode katalog langsung / cache).")

        st.markdown("**URL Sumber Grounding / Detail**")
        if result.sources:
            for source in result.sources:
                domain = urlsplit(source.url).netloc
                st.markdown(f"- [{domain}]({source.url}) - `{source.url}`")
        else:
            st.caption("Metadata grounding tidak mengembalikan URL sumber.")

        st.markdown("**Listing yang Dibuang per Alasan (Dikelompokkan)**")
        if getattr(result, "rejected_by_reason", None):
            for reason, urls in result.rejected_by_reason.items():
                st.markdown(f"- ❌ **{reason}** ({len(urls)} URL)")
                with st.expander(f"Daftar URL: {reason}", expanded=False):
                    for u in urls:
                        st.write(f"- `{u}`")
        elif result.rejected:
            grouped_rej: dict[str, list[str]] = {}
            for rej in result.rejected:
                r_text = rej.get("reason", "Tidak memenuhi kriteria validasi")
                grouped_rej.setdefault(r_text, []).append(rej.get("source_url", "-"))
            for reason, urls in grouped_rej.items():
                st.markdown(f"- ❌ **{reason}** ({len(urls)} URL)")
                with st.expander(f"Daftar URL: {reason}", expanded=False):
                    for u in urls:
                        st.write(f"- `{u}`")
        else:
            st.caption("Tidak ada data listing yang dibuang pada analisa ini.")


    usage = result.usage
    model_display = result.model_used or "Gemini"
    st.caption(
        f"Model AI: **{model_display}** | Analisa harian ke-{result.daily_count} | Panggilan grounding: {usage['grounding_calls']} | "
        f"Token input/output/total: {usage['input_tokens']}/{usage['output_tokens']}/{usage['total_tokens']}"
    )
    report = _summary_text(result, user_input)
    copy_col, download_col = st.columns(2)
    with copy_col:
        copy_button(data={"text": report}, key="copy-grounded-summary", width="stretch")
    with download_col:
        st.download_button(
            "Unduh ringkasan (TXT)",
            data=report,
            file_name=f"screening_{user_input['model']}_{user_input['year']}.txt",
            mime="text/plain",
            width="stretch",
        )
    st.caption(DISCLAIMER)
    updated_at = datetime.now()
    st.caption(f"Tanggal analisa: {updated_at.day} {updated_at.strftime('%B')} {updated_at.year}")


def _render_history_tab(cache: SQLiteAnalysisCache) -> None:
    st.subheader("Riwayat Analisa Mobil")
    st.caption("Hasil screening yang pernah dijalankan tersimpan di database SQLite lokal.")
    records = cache.list_screening_history(limit=50)
    if not records:
        st.info("Belum ada riwayat analisa tersimpan. Lakukan analisa unit baru untuk menyimpannya ke database.")
        return

    table_data = []
    for r in records:
        try:
            dt = datetime.fromisoformat(r["created_at"]).strftime("%d/%m/%Y %H:%M")
        except Exception:
            dt = r["created_at"][:16]

        score_display = f"{r['screening_score']}/100" if r["screening_score"] is not None else "Tidak dinilai"
        median_display = _format_price(r["price_median"]) if r["price_median"] else "-"
        unit_name = f"{r['brand']} {r['model']} {r['variant']} ({r['year']})"
        table_data.append({
            "ID": r["id"][:8],
            "Tanggal": dt,
            "Unit": unit_name,
            "Transmisi": r["transmission"],
            "Harga Listing": _format_price(r["listing_price"]),
            "Median Pasar": median_display,
            "Pembanding": r["comparables_count"],
            "Skor": score_display,
            "Rekomendasi": r["recommendation"] or "-",
            "Status": r["completion_status"],
            "_full_id": r["id"],
        })

    df = pd.DataFrame(table_data)
    display_df = df.drop(columns=["_full_id"])
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.write("#### 🔎 Lihat Detail Riwayat Analisa")
    options = {
        f"{r['Tanggal']} - {r['Unit']} ({r['Harga Listing']}) [ID: {r['ID']}]": r["_full_id"]
        for r in table_data
    }
    selected_label = st.selectbox("Pilih analisa untuk melihat laporan lengkap:", options=list(options.keys()))
    if selected_label:
        selected_id = options[selected_label]
        record = cache.get_screening_record(selected_id)
        if record and record.get("result_data"):
            data = record["result_data"]
            inp = data.get("user_input", {})
            st.success(f"Memuat laporan analisa untuk: **{inp.get('brand')} {inp.get('model')} {inp.get('year')}** (ID: `{selected_id}`)")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Harga Listing", _format_price(inp.get("listing_price", 0)))
            med = data.get("statistics", {}).get("price_median")
            c2.metric("Median Pasar", _format_price(med) if med else "Data terbatas")
            score = data.get("screening", {}).get("score")
            c3.metric("Skor Kelayakan", f"{score}/100" if score is not None else "Tidak dapat dinilai")
            c4.metric("Keyakinan Pasar", data.get("confidence", {}).get("level", "-"))

            if data.get("kesimpulan"):
                st.info(f"💡 **Kesimpulan AI:** {data['kesimpulan']}")

            nego = data.get("saran_nego", {})
            if isinstance(nego, dict) and any(nego.values()):
                st.write("**Saran Negosiasi:**")
                n1, n2, n3 = st.columns(3)
                if nego.get("awal"):
                    n1.metric("Tawaran Awal", _format_price(nego["awal"]))
                if nego.get("realistis"):
                    n2.metric("Target Realistis", _format_price(nego["realistis"]))
                if nego.get("batas"):
                    n3.metric("Batas Maksimal", _format_price(nego["batas"]))

            comps = data.get("comparables", [])
            if comps:
                st.write(f"**Pembanding Valid ({len(comps)} unit):**")
                comp_rows = []
                for c in comps:
                    comp_rows.append({
                        "Model/Varian": f"{c.get('brand', '')} {c.get('model', '')} {c.get('variant', '')}".strip(),
                        "Tahun": c.get("year") or "-",
                        "KM": f"{c.get('kilometer'):,}" if c.get("kilometer") else "-",
                        "Transmisi": c.get("transmission", "-"),
                        "Lokasi": c.get("location", "-"),
                        "Harga": _format_price(c.get("price", 0)),
                        "URL Sumber": c.get("source_url", "-"),
                    })
                st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

            del_col, _ = st.columns([1, 4])
            with del_col:
                if st.button("🗑️ Hapus Analisa Ini", key=f"del_{selected_id}"):
                    cache.delete_screening_record(selected_id)
                    st.toast("Analisa berhasil dihapus.")
                    st.rerun()


def _render_database_tab(cache: SQLiteAnalysisCache) -> None:
    st.subheader("Database Listing Pasar Mobil Bekas")
    st.caption("Kumpulan listing pembanding riil dari berbagai marketplace yang berhasil diekstrak dan divalidasi.")

    total_count = cache.count_market_listings()
    m1, m2 = st.columns(2)
    m1.metric("Total Listing Tersimpan", f"{total_count:,} unit")

    all_listings = cache.list_market_listings(limit=500)
    if not all_listings:
        st.info("Database listing pasar masih kosong. Jalankan analisa unit untuk mulai mengumpulkan listing.")
        return

    brands = sorted(list({row["brand"] for row in all_listings if row.get("brand")}))
    domains = sorted(list({row["domain"] for row in all_listings if row.get("domain")}))

    f1, f2, f3 = st.columns(3)
    with f1:
        selected_brand = st.selectbox("Filter Merek", options=["Semua Merek"] + brands)
    with f2:
        selected_domain = st.selectbox("Filter Marketplace", options=["Semua Marketplace"] + domains)
    with f3:
        search_model = st.text_input("Cari Model (misal: Avanza, Xpander)")

    brand_filter = None if selected_brand == "Semua Merek" else selected_brand
    domain_filter = None if selected_domain == "Semua Marketplace" else selected_domain
    model_filter = search_model.strip() if search_model.strip() else None

    filtered_listings = cache.list_market_listings(
        brand=brand_filter,
        model=model_filter,
        domain=domain_filter,
        limit=500,
    )

    m2.metric("Listing Tampil", f"{len(filtered_listings):,} unit")

    if not filtered_listings:
        st.warning("Tidak ada listing yang cocok dengan filter yang dipilih.")
        return

    rows = []
    for item in filtered_listings:
        try:
            scraped_dt = datetime.fromisoformat(item["scraped_at"]).strftime("%d/%m/%Y")
        except Exception:
            scraped_dt = item["scraped_at"][:10]
        rows.append({
            "Merek": item["brand"],
            "Model": item["model"],
            "Varian": item["variant"] or "-",
            "Tahun": item["year"] or "-",
            "KM": f"{item['kilometer']:,}" if item.get("kilometer") is not None else "-",
            "Transmisi": item["transmission"] or "-",
            "Lokasi": item["location"] or "-",
            "Harga": _format_price(item["price"]),
            "Marketplace": item["domain"],
            "Tipe Penjual": item["seller_type"] or "-",
            "Tanggal": scraped_dt,
            "URL": item["source_url"],
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "URL": st.column_config.LinkColumn("Link Listing", display_text="Buka Listing"),
        },
    )

    csv_data = df.to_csv(index=False)
    st.download_button(
        label="📥 Unduh CSV Listing Pasar",
        data=csv_data,
        file_name="database_listing_pasar.csv",
        mime="text/csv",
    )


def render_app() -> None:
    cache = SQLiteAnalysisCache(CACHE_DB_PATH, ttl_seconds=86_400)
    st.title("AI Car Screening Jabodetabek")
    st.caption("Analisa harga dari listing web yang ter-grounding.")
    user_input, analyze = _read_listing_form()

    tab_screening, tab_history, tab_database = st.tabs([
        "🔍 Analisa Unit",
        "📋 Riwayat Analisa",
        "🏪 Database Listing Pasar",
    ])

    with tab_history:
        _render_history_tab(cache)

    with tab_database:
        _render_database_tab(cache)

    with tab_screening:
        retry_triggered = st.session_state.pop("retry_requested", False)
        should_run = analyze or retry_triggered

        if should_run:
            st.session_state["screening_result"] = None
            st.session_state["pipeline_error"] = None
            if not all(user_input[key] for key in ("brand", "model", "variant", "year", "transmission", "kilometer", "listing_price", "location")):
                st.error("Lengkapi semua data wajib sebelum analisa.")
            else:
                progress = st.progress(0, text="Menyiapkan analisa...")
                with st.status("🔍 Sedang menjalankan analisa harga & screening...", expanded=True) as status_box:
                    def handle_stage(value: int, text: str):
                        progress.progress(value, text=text)
                        status_box.write(f"**[{value}%]** {text}")
                        status_box.update(label=f"Proses ({value}%): {text}")

                    def handle_status(detail: str):
                        status_box.caption(f"↳ {detail}")

                    try:
                        load_dotenv(override=True)
                        provider = GeminiGroundingProvider(on_status=handle_status)
                        result = run_screening(
                            provider=provider,
                            cache=cache,
                            user_input=user_input,
                            daily_limit=_daily_limit(),
                            on_stage=handle_stage,
                            on_status=handle_status,
                        )
                        done_label = (
                            "✅ Analisa selesai dengan data."
                            if result.completion_status == "selesai dengan data"
                            else "⚠️ Analisa selesai tanpa data valid."
                        )
                        status_box.update(label=done_label, state="complete", expanded=False)
                        st.session_state["screening_result"] = result
                        st.session_state["screening_input"] = user_input
                        st.session_state["pipeline_error"] = None
                    except DailyLimitExceeded as error:
                        status_box.update(label="⚠️ Batas harian tercapai", state="error", expanded=True)
                        st.error(str(error))
                    except PipelineError as error:
                        status_box.update(label=f"❌ Gagal pada tahap {error.stage}", state="error", expanded=True)
                        if error.stage == "Pencarian":
                            st.session_state["pipeline_error"] = (
                                f"Tahap {error.stage} gagal: Semua model Gemini terkendala koneksi atau kapasitas server.\n\n"
                                f"Detail error: {error}"
                            )
                        elif error.stage == "Ekstraksi":
                            st.session_state["pipeline_error"] = (
                                f"Tahap {error.stage} gagal: Model Gemini berhasil merespons, "
                                f"tetapi data yang dikembalikan tidak dapat diproses ke skema listing.\n\n"
                                f"Hal ini biasanya bersifat sementara. Silakan coba lagi.\n\n"
                                f"Detail: {error}"
                            )
                        else:
                            st.session_state["pipeline_error"] = (
                                f"Tahap {error.stage} gagal: {error}"
                            )
                    except (ValueError, RuntimeError) as error:
                        status_box.update(label="❌ Analisa terhenti", state="error", expanded=True)
                        st.session_state["pipeline_error"] = (
                            f"Analisa terhenti: {error}\n\n"
                            "Silakan periksa konfigurasi model atau coba beberapa saat lagi."
                        )
                    except Exception as error:
                        status_box.update(label="❌ Analisa gagal", state="error", expanded=True)
                        st.session_state["pipeline_error"] = f"Analisa gagal karena error tak terduga: {error}"
                    finally:
                        progress.empty()

        if st.session_state.get("pipeline_error") and not st.session_state.get("screening_result"):
            st.error(st.session_state["pipeline_error"])
            st.info("Pencarian tidak menggunakan data tiruan (dummy). Anda dapat mencoba kembali proses analisa:")
            if st.button("🔄 Coba lagi", key="btn_retry_screening", type="primary"):
                st.session_state["retry_requested"] = True
                st.rerun()

        result = st.session_state.get("screening_result")
        saved_input = st.session_state.get("screening_input")
        if result and saved_input:
            _render_analysis(result, saved_input, _COPY_SUMMARY_BUTTON)
        elif not st.session_state.get("pipeline_error"):
            st.info("Isi data unit di sidebar dan tekan **Analisa**. Hasil hanya memakai listing web yang lolos validasi sumber.")

