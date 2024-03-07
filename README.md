# LMCenter

#### 开源大模型部署方案

#### ModelController
- 模型控制器，提供模型worker注册、worker心跳检测、同名模型负载均衡、通用对外API接口等功能

#### ModelWorker
- 模型工作者，通过配置文件启动不同的大模型worker,自动注册到Controller中
- 支持Baichuan、ChatGLM3、Qwen1.5系列等模型


#### 快速开始
docker-compose 一键启动服务，自带容器健康检测，自动重启

1. 准备好大模型的相关文件

2. 拉取源代码
```commandline
git clone https://github.com/yuemengrui/LMCenter.git
```

3. 在项目中有一个docker-compose.yml文件，修改相关配置项，例如:
```commandline
  baichuan2_13b_server:  // 修改服务名
    container_name: model_worker_baichuan2_13b_server  // 修改容器名
    image: registry.cn-beijing.aliyuncs.com/yuemengrui/ai:pytorch2.1.2-cuda12.1-cudnn8-ubuntu20.04-py311-base
    command: [ "/bin/bash", "-c", "/workspace/ModelWorker/docker_run.sh" ]
    depends_on:
      model_controller:
        condition: service_healthy
        restart: true
    volumes:
      - ./ModelWorker:/workspace/ModelWorker
      - ./ModelWorker/configs/baichuan2_13b_configs.py:/workspace/ModelWorker/configs/configs.py  // 挂载配置文件
      - ./DATA/Models/Baichuan2-13B-Chat:/workspace/Models/Baichuan2-13B-Chat  // 挂载模型文件夹
    ports:
      - "24621:24621"  // 端口映射
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: [ "1" ]  // GPU卡号
              capabilities: [ gpu ]

```

4. 在ModelWorker/configs/目录下创建一个新的配置文件，例如baichuan2_13b_configs.py, 可从configs.py.template复制一份，修改相关配置项, 只需修改ModelWorkerConfig中的部分配置项即可
```commandline
########################

ModelWorkerConfig = {
    "controller_addr": "http://model_controller:24620",  // controller地址
    "worker_addr": f"http://xxx:{FASTAPI_PORT}", // worker地址，将xxx替换为你的容器名
    "worker_id": WORKER_ID,
    "worker_type": "",  # vllm or others // worker类型，可以选择是否启用vllm
    "model_type": "xx",  # ['Baichuan', 'ChatGLM3', 'Qwen2']  // 支持的模型类型
    "model_path": "xxx",  // 模型路径，为上面docker-compose.yml中挂载的模型文件夹, 例如：/workspace/Models/Baichuan2-13B-Chat
    "model_name": "xxx",  // 模型名
    "limit_worker_concurrency": 5,
    "multimodal": False,  // 是否是多模态模型
    "device": "cuda",
    "dtype": "float16",  # ["float32", "float16", "bfloat16"], // 模型精度，推荐bfloat16
    "gpu_memory_utilization": 0.9
}


########################
```

5. 启动服务
```commandline
sudo docker compose up -d
```

6. 静待几分钟, 查看服务健康情况, 不出意外的话，所有服务应该是(healthy)的状态
```commandline
sudo docker compose ps
```

7. 服务启动成功，可以通过 http://controller地址:24620/redoc 查看API文档

8. 除了docker compose单独启动本项目外，也可以将服务编排到你现有docker-compose中，方便统一管理
