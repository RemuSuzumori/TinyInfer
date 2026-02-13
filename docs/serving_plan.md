# TinyInfer Serving Plan（阶段 0）

## 1) 现有推理入口与调用链

当前仓库没有单一 `infer()` 函数，推理入口由模型层组合完成：

1. `tiny_infer::Linear::forward(const Tensor&)`
2. `tiny_infer::ReLU::forward(const Tensor&)`
3. 通过示例中的顺序调用得到最终输出（logits）

MNIST 示例调用链：

- `h1 = fc1.forward(input)`
- `h2 = relu.forward(h1)`
- `output = fc2.forward(h2)`

因此 serving 层将封装一个最小推理函数（例如 `infer_mnist`），内部复用上述链路。

## 2) 输入格式要求（shape / dtype）

从 `Tensor` 与 `Linear` 的实现可知，数据类型固定为 `float`（C++ `float32`）。

MNIST 最小模型输入约束：

- shape: `[1, 784]`（即 28×28 拉平后单样本 batch）
- dtype: `float32`
- 内存布局：连续行主序（`Tensor` 内部 `std::vector<float>`）

服务协议阶段会先定义 `raw_tensor_mnist`：

- payload = 784 个 `float32`（小端）
- 总字节数 = `784 * 4 = 3136`

## 3) 返回结果格式

现有推理直接输出最后一层 logits（`[1, 10]`）。

Serving 最小可用返回格式计划：

- `top1_label: uint32`
- `top1_score: float32`（对应 top1 的 logit）

可扩展：后续支持完整 logits（10 个 `float32`）。

## 4) 线程安全评估

- `Linear::forward` / `ReLU::forward` 对输入进行只读，返回新 `Tensor`。
- 但 `Linear` 持有可变权重张量，接口未显式标注 const-thread-safe；框架中也无读写锁。
- 在模型参数加载完成后，仅执行只读 forward，按当前实现可近似视为“读多并发可用”。

稳妥策略（serving 采用）：

- I/O 线程不做推理。
- 计算线程域串行访问共享模型实例（先确保正确性）。
- 若后续压测确认并发前向安全，再扩展为每 worker 独立模型副本。

## 5) 与 ThreadPool 的对齐

仓库已有 `tiny_infer::ThreadPool` 和全局 `Context::instance().thread_pool()`。

Serving 将采用：

- 异步 I/O（Boost.Asio）
- 有界任务队列 + 背压
- 计算 worker 从队列取任务，再将响应通过 `asio::post` 切回 I/O 执行器写回

该方案满足 “I/O 与推理解耦” 的硬约束。
