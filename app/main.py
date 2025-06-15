from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import uvicorn
from agents.flow import run_log_analysis_flow  # Your agent logic

app = FastAPI()

# Input schema
class LogRecord(BaseModel):
    id: int
    dur: float
    proto: str
    service: str
    state: str
    spkts: int
    dpkts: int
    sbytes: int
    dbytes: int
    rate: float
    sttl: int
    dttl: int
    sload: float
    dload: float
    sloss: int
    dloss: int
    sinpkt: float
    dinpkt: float
    sjit: float
    djit: float
    swin: int
    stcpb: int
    dtcpb: int
    dwin: int
    tcprtt: float
    synack: float
    ackdat: float
    smean: int
    dmean: int
    trans_depth: int
    response_body_len: int
    ct_srv_src: int
    ct_state_ttl: int
    ct_dst_ltm: int
    ct_src_dport_ltm: int
    ct_dst_sport_ltm: int
    ct_dst_src_ltm: int
    is_ftp_login: int
    ct_ftp_cmd: int
    ct_flw_http_mthd: int
    ct_src_ltm: int
    ct_srv_dst: int
    is_sm_ips_ports: int
    attack_cat: Optional[str]
    label: int

class LogInput(BaseModel):
    logs: List[LogRecord]

@app.get("/")
def root():
    return {"status": "ok", "info": "use POST /analyze to detect threats"}

@app.post("/analyze")
def analyze_logs(batch: LogInput):
    try:
        log_dicts = [log.model_dump() for log in batch.logs]
        log_dicts = json.dumps(log_dicts)
        result = run_log_analysis_flow(log_dicts)
        return {"decision": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

