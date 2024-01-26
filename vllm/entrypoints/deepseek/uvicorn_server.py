import os
import uvicorn

from vllm.entrypoints.deepseek.main import app, args, TIMEOUT_KEEP_ALIVE

# start api server
if __name__ == "__main__":
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=os.getenv('LOG_LEVEL', 'info'),
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
