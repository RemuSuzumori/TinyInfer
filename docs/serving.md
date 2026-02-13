# TinyInfer Serving

## 构建

```bash
cmake -S . -B build -DTINYINFER_BUILD_SERVING=ON -DBUILD_TESTING=OFF
cmake --build build -j
```

产物：

- `build/bin/tinyinfer-serving`
- `build/bin/client_demo`

## 启动 TCP 服务

```bash
./build/bin/tinyinfer-serving --port 9000 --queue-capacity 256 --compute-workers 4
```

## 协议（小端，length-prefixed）

### RequestHeader（20 bytes）

- `uint32 magic = 0x53464954` (`"TIFS"`)
- `uint16 version = 1`
- `uint16 msg_type = 1`
- `uint64 request_id`
- `uint32 payload_len`

后接 `payload_len` 字节 payload。

### ResponseHeader（22 bytes）

- `uint32 magic`
- `uint16 version`
- `uint16 msg_type = 2`
- `uint64 request_id`
- `uint16 status`（0=ok,1=bad_request,2=busy,3=internal）
- `uint32 result_len`

后接 `result_len` 字节 result。

## 示例客户端

```bash
./build/bin/client_demo --host 127.0.0.1 --port 9000 --payload hello
```

预期：`status=0 result=...`

## 压测（阶段 3）

```bash
python3 tools/load_test.py --host 127.0.0.1 --port 9000 --threads 20 --loops 500
```

预期：

- 服务持续运行，不崩溃
- 输出中可见 `busy`（队列满）
- 服务端 metrics 周期打印总请求、busy、队列峰值

## 真实 MNIST 推理（阶段 4）

服务启动：

```bash
./build/bin/tinyinfer-serving --port 9000 --real-mnist --model-dir models
```

payload 准备（模型输入已是 3136 bytes float32）：

```bash
python3 tools/make_mnist_payload.py --input models/input.bin --output /tmp/mnist.payload
```

当前 `client_demo` 发送字符串 payload，真实 mnist 二进制可用 `tools/load_test.py` 改为读取文件或后续扩展客户端发送。

服务端返回：`label=<n>,score=<float>`。

## 可观测性与治理（阶段 5）

- 周期 metrics：总请求、成功、busy、bad_request、internal、队列当前/峰值、平均耗时
- 读超时：连接建立后超过 `--read-timeout-ms` 未收到完整包会关闭
- 每连接 in-flight 限制：`--max-inflight`
