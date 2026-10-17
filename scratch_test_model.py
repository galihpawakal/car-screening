from typing import Any
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator

class NegotiationAdvice(BaseModel):
    model_config = ConfigDict(extra="ignore")
    awal: int | None = None
    realistis: int | None = None
    batas: int | None = None

class InspectionChecklist(BaseModel):
    model_config = ConfigDict(extra="ignore")
    dokumen: list[str] = Field(default_factory=list)
    fisik: list[str] = Field(default_factory=list)
    transmisi: list[str] = Field(default_factory=list)

class FinalAnalysis(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kesimpulan: str = Field(description="Kesimpulan keputusan 2-3 kalimat.")
    saran_nego: NegotiationAdvice = Field(default_factory=NegotiationAdvice)
    red_flag: list[str] = Field(default_factory=list)
    pertanyaan_untuk_penjual: list[str] = Field(default_factory=list)
    checklist_inspeksi: InspectionChecklist = Field(default_factory=InspectionChecklist)
    catatan: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def unwrap_hasil_analisis(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for wrap_key in ("hasil_analisis", "final_analysis", "analisis", "hasil"):
                if wrap_key in data and isinstance(data[wrap_key], dict):
                    return data[wrap_key]
        return data

    @field_validator("catatan", mode="before")
    @classmethod
    def coerce_catatan_to_list(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            clean = v.strip()
            return [clean] if clean else []
        if isinstance(v, dict):
            res = []
            for k, val in v.items():
                if val is not None and str(val).strip():
                    res.append(f"{k}: {val}")
            return res
        if isinstance(v, (list, tuple)):
            res = []
            for item in v:
                if item is None:
                    continue
                if isinstance(item, str):
                    clean = item.strip()
                    if clean:
                        res.append(clean)
                elif isinstance(item, dict):
                    for k, val in item.items():
                        if val is not None and str(val).strip():
                            res.append(f"{k}: {val}")
                else:
                    res.append(str(item).strip())
            return res
        return [str(v)]

# Test 1: Wrapped in hasil_analisis with dict catatan
t1 = {
    "hasil_analisis": {
        "kesimpulan": "Kondisi unit cukup baik. Periksa transmisi dan kelengkapan dokumen sebelum bayar.",
        "catatan": {"pajak": "Pajak hidup", "estimasi": "Biaya mutasi Rp 2jt"}
    }
}
res1 = FinalAnalysis.model_validate(t1)
print("Test 1 success:", res1.kesimpulan, "| catatan:", res1.catatan)

# Test 2: Catatan as a string
t2 = {
    "kesimpulan": "Mobil terawat.",
    "catatan": "Harga di bawah pasar."
}
res2 = FinalAnalysis.model_validate(t2)
print("Test 2 success:", res2.catatan)

# Test 3: Catatan as None
t3 = {
    "kesimpulan": "Mobil terawat.",
    "catatan": None
}
res3 = FinalAnalysis.model_validate(t3)
print("Test 3 success:", res3.catatan)
