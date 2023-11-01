"""
推理后端使用的 Loadbalancer API
单文件+配置文件的形式嵌入到推理后端代码中, 不能依赖任何此 codebase 中的其他模块
"""

import logging
import json
from redis import Redis
from munch import Munch


__all__ = ['init_load_balance', 'report_load_score']


# redis keys
LB_SERVER_QUEUE_REDIS_KEY_PREFIX = 'lb_server_queue:v2'
LB_SERVER_SCORE_REDIS_KEY = 'lb_server_score:v2'
LB_READY_SERVER_SET_REDIS_KEY = 'lb_ready_server_set:v2'
LB_SERVER_LAST_REQ_ID_REDIS_KEY_PREFIX = 'lb_server_req_id:v2'
LB_SERVER_INFO_REDIS_KEY_PREFIX = 'lb_server_info:v2'

_worker_addr = None
_redis_client = None


def _init_redis_client(redis_config_file: str):
    config = Munch.fromYAML(open(redis_config_file))

    redis_config = config.redis
    return Redis(
        host=redis_config.host,
        port=redis_config.port,
        db=redis_config.db,
        password=redis_config.password,
    )


def init_load_balance(worker_addr: str, redis_config_file: str):
    global _worker_addr
    _worker_addr = worker_addr

    global _redis_client
    _redis_client = _init_redis_client(redis_config_file)


def report_load_score(
    score: float,
    current_max_request_id: str
):
    """
    向负载均衡上报当前 server 的负载分数和接收到的最大请求 ID.
    应当在 负载分数 或 接收到的最大请求ID 中任何一个发生变化时上报最新数据.

    :param worker_addr: 当前后端的 worker_addr.
    :type worker_addr: str
    :param score: 当前后端的最新负载分数.
    :type score: float
    :param current_max_request_id: 当前后端接收到的最大请求 ID.
    :type current_max_request_id: str
    """

    if not _worker_addr:
        return

    # 已经 unregister 的机器仍然可能仍在进行生成并上报负载, 忽略
    if (_redis_client.sismember(LB_READY_SERVER_SET_REDIS_KEY, _worker_addr)) == 0:
        return

    server_info = _redis_client.get(f'{LB_SERVER_INFO_REDIS_KEY_PREFIX}_{_worker_addr}')
    if server_info is None:
        logging.warn('cannot get server model_type')
        return
    model_type = json.loads(server_info)['model_type']

    pipeline = _redis_client.pipeline()
    # 更新负载分数
    pipeline.zadd(LB_SERVER_SCORE_REDIS_KEY, {_worker_addr: score})

    request_key = f'{LB_SERVER_LAST_REQ_ID_REDIS_KEY_PREFIX}_{_worker_addr}'
    server_last_id = _redis_client.get(request_key)
    if server_last_id is not None:
        if server_last_id.decode() == current_max_request_id:
            pipeline.delete(request_key)
        else:
            # 设置了 last_id 且当前请求不是 last id 时, 不重新加入队列
            pipeline.execute()
            return

    # 加入待选队列, 或更新权重
    if score >= 3.0:    # 完全满载, 直接删除这个 worker_addr
        pipeline.zrem(f'{LB_SERVER_QUEUE_REDIS_KEY_PREFIX}_{model_type}', _worker_addr)
    else:
        pipeline.zadd(f'{LB_SERVER_QUEUE_REDIS_KEY_PREFIX}_{model_type}', {_worker_addr: score})
    pipeline.execute()
