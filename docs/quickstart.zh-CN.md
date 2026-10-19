# 从数据开始：一条提示词完成工作流

未发布候选版的推荐路径见 [WAN 工作流](training-candidate.md)。以下普通工作流保留
稳定 0.2 请求行为；安装候选版不代表新服务能力已经上线。

在仓库中执行 `pip install '.[data]'`（Python 3.11+）。通过不回显的方式设置
`TENON_API_KEY`，按邀请设置 `TENON_BASE_URL`。API key 不代表默认地址已经开放，
也不代表所有机器人布局都有可用模型。无需服务即可完成本地检查和准备。

## 1. 先检查，再选择导入路径

```bash
tenon-pi data inspect ./my-data
```

- 已准备的 bundle：检查会重新验证。上传整个根目录，不能只上传 `dataset/`。
- LeRobot v2.0/v2.1 RGB PNG 或 v3.0 canonical RGB video：使用明确提供的 profile。
- 按顺序记录的 JSONL + PNG：使用 `data import-jsonl`，随后用同一 profile 准备。
- 其他原始数据：先读元数据和样本，提出明确的转换规则，保留原文件并逐项核对
  state/action、图像、时间戳、episode 边界及动作顺序。单位和标定不明时必须询问。
- LIBERO：使用[公开数据获取与仿真示例](libero-demo.md)及其固定 schema；不要把
  LIBERO 的时间语义套用到任意机器人记录。

```bash
tenon-pi data prepare ./my-data --profile ./profile.json \
  --input-space native --repo-id owner/task -o .tenon/prepared
tenon-pi data verify .tenon/prepared
tenon-pi data manifest .tenon/prepared -o .tenon/bundle.json
```

[数据与 profile 约定](robot-preparation.zh-CN.md)说明了支持范围。本地 profile 有效，
不等于服务已支持该布局；服务的验证、报价和 admission 才是实际可用性的依据。

## 2. agent 先负责安装和预检

仿真目标在花费额度前运行 `tenon-pi agent setup-simulator --display window`。
它负责隔离 Python、固定依赖/源码、真实 CPU renderer 和窗口预检，不要求用户已配置
simulator Python。v2 receipt 绑定实际 SDK/依赖/LIBERO 字节；旧版或漂移记录会在
API 调用前拒绝，显式 `--python` 也不能跳过新的真实预检。失败时按 [setup 指南](simulator-setup.md)处理缺少的主机权限，
不要创建付费资源来替代预检。配置好的机器人使用 [agent start-robot](configured-robot.md)。

## 3. 只训练一次，中断后接着做

```bash
tenon-pi account capabilities
tenon-pi account credits
tenon-pi agent train .tenon/prepared \
  --schema .tenon/prepared/robot-schema.json --name 'my task' \
  --method full_ft --max-steps 100 --max-runtime-seconds 7200 \
  --checkpoint-interval-steps 100 --max-cost-microcredits 5000000 \
  --context .tenon/run.json
```

以上数值仅为命令示例，不是效果或速度承诺。LIBERO 使用其 materialized 数据根目录和
生成的 schema；agent 命令自动选择对应公开 recipe。其他 profile 交给服务解析
`balanced-v1` 默认 preset，不能把成功准备数据当成服务接受模型的证据。

命令等待真实数据 `ready`、展示报价，再确认花费。只有用户已授权预算时才用 `--yes`。
LoRA 使用 `--method lora`，并要求服务提供对应的命名 profile；不添加 rank/alpha 字段。

继续运行时使用**相同命令和 `.tenon/run.json`**。文件以私有权限原子保存输入摘要、
schema、账号/地址身份、spec、资源 ID、报价、幂等 key 和精确文件完成 receipt，不保存
API key 或签名 URL。它必须在上传目录之外，且不得进入 Git。POSIX 文件锁阻止两个
agent 同时写同一工作流。PUT 中断且尚无完整 receipt、报价过期、或超过服务幂等保留
窗口的模糊结果需要显式核对；不要删除文件后盲目重新上传/训练。

失败任务不会自动重建。训练成功后仍需检查 ready checkpoint 的 dataset、schema、
transform、normalization、method、model 和 artifact 元数据。这不等于客户端独立
下载验证全部模型权重。

## 4. 部署并启动真实运行

在总授权预算内为控制会话单独留出预算：

```bash
tenon-pi agent deploy --context .tenon/run.json --name 'my policy' \
  --prompt 'place the block in the tray' --execution-mode async_aligned \
  --duration-seconds 300 --max-cost-microcredits 1000000
```

命令复用 deployment/session ID，重新核对 checkpoint 来源，并要求会话实际 `active`。
`deployment.ready` 不等于 runtime 已 warm。失败或过期会话不会静默替换。

LIBERO 环境安装官方 simulator 与此客户端后：

```bash
tenon-pi agent start-simulator --context .tenon/run.json \
  --setup .tenon/simulator/setup.json --display window --task-suite libero_10 --task-id 0
```

`--display window` 在本机打开 Tk 实时窗口，展示 simulator 的 agent/wrist 相机帧；
关闭窗口会停止本地 stepping。GUI 不可用时明确失败，不以 MP4 替代可见窗口。
`--display auto` 尝试窗口并如实报告 headless fallback；`--display headless` 明确不打开
窗口。启动和结果 JSON 记录实际模式、原因和呈现帧数。可见 demo 应保留 `window`。
Mac/Linux 的 Tk 与 renderer 配置见 [LIBERO 指南](libero-demo.md#desktop-window-setup)。

一次性连接 token 仅在内存中使用，WSS 地址来自服务响应。缺失依赖会明确报错。
结束后通过 `session get` 获取当前版本，再用 `session close --resource-version VERSION`
关闭会话，避免继续计费。再次运行需显式申请新会话，不恢复旧 buffer。

硬件运行必须提供可信驱动、实际标定、各通道 bounds/slew、同步采集和现场停止/碰撞
约束。参考[本地 bridge](../examples/robot_control_boundary.py)：驱动必须真实 connected
且明确 armed。agent 可以接入已有驱动，不能生成不存在的硬件。

如实记录 episode 结果和资源来源。`evaluation.ready` 仅表示结构验证通过；它不代表
任务成功、策略提升或网络时延保证。
