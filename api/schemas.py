from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    umur: int = Field(..., ge=17, le=100)
    kelas_pekerja: str
    pendidikan: str
    jmlh_tahun_pendidikan: int = Field(..., ge=0, le=20)
    status_perkawinan: str
    pekerjaan: str
    jenis_kelamin: str
    keuntungan_kapital: float = Field(..., ge=0)
    kerugian_capital: float = Field(..., ge=0)
    jam_per_minggu: float = Field(..., ge=1, le=100)


class PredictResponse(BaseModel):
    predicted_kategori_gaji: str
    probabilitas_kelas_1: float
    probabilitas_kelas_0: float
    model_used: str = "Random Forest (class_weight=balanced)"
