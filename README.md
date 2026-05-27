# MediQ AI v3

MediQ AI 서버는 구조화된 증상 입력 JSON을 기반으로 후보 질환을 추론하고,
Spring Boot 백엔드에서 호출 가능한 FastAPI `/analyze` 엔드포인트를 제공합니다.

## Current Inference Version

- Base Model: Qwen/Qwen3-8B
- Fine-tuning: QLoRA
- Adapter: qwen3_8b_mediq_qlora_adapter_v3
- Inference Mode: CSV-first tiered hybrid selection
- Free-text Mode: enabled
- Red-flag Safety Filter: enabled

## Run Server

```bash
conda activate kmbert_gpu
D:
cd "D:\학교\졸작 데이터\MediQ_AI_v3"
uvicorn inference_server_v3_3_free:app --host 0.0.0.0 --port 8000

API

Swagger: http://127.0.0.1:8000/docs
Health check: http://127.0.0.1:8000/
Analyze endpoint: POST /analyze