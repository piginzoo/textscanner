#!/usr/bin/env python
import time,os,datetime,logging

logger = logging.getLogger("Util")

def timestamp_s():
    s = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    return s


def get_checkpoint(dir):
    if not os.path.exists(dir):
        logger.info("找不到最新的checkpoint文件")
        return None

    list = os.listdir(dir)
    if len(list)==0:
        logger.info("找不到最新的checkpoint文件")
        return None

    list.sort(key=lambda fn:os.path.getmtime( os.path.join(dir,fn) ))

    latest_model_name = os.path.join(dir,list[-1])

    logger.debug("在目录%s中找到最新的模型文件：%s",dir,latest_model_name)

    return latest_model_name
