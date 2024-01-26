## start for dev
```bash
RAY_DEDUP_LOGS=0 CUDA_VISIBLE_DEVICES=3 python -u vllm/entrypoints/deepseek/uvicorn_server.py --host 0.0.0.0 --port 37900 \
    --served-model-name DeepSeek-Coder-5.7B --model \
    /3fs-jd/prod/deepseek/shared/jiashi/coder_5d7B \
    --trust-remote-code --max-num-seqs 512 --max-num-batched-tokens 16384 --max-model-len 16384 \
    --engine-use-ray --enforce-eager --gpu-memory-utilization 0.5 --pipeline-parallel-size 1 --tensor-parallel-size 1

curl -X POST http://127.0.0.1:37900/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"token_ids":  [32013, 32013, 2042, 417, 245, 9396, 20391, 13, 13518, 3649, 3475, 25, 185, 17535, 0, 185, 13518, 21289, 25, 185], "request_id": "1123"}'
```
