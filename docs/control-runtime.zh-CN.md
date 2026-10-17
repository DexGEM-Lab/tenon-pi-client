# 本地 index-aligned control runtime

`AsyncAlignedController` 负责原生 `tenon-control-v1` 的 action chronology；调用方
负责 control clock 和最终 simulator/actuator 调用。类名为兼容保留，但构造时必须
显式选择 `async_aligned` 或 `guided_rtc`。客户端不包含 guided sampler、VJP、guidance
weight 或私有模型数学，只实现冻结 wire、index/buffer、actuator truth 与 reset/rearm。
`trained_rtc` 和 mode 冒充都会拒绝。

## Motion 前的 admission

`warm_start()` 在第一条 motion command 前测量完整 turnaround。只有以下条件满足才
arm：

```text
required_steps = ceil(max_warm_turnaround × controller_hz) + safety_steps
2 × required_steps + 1 <= action_horizon_steps
```

warm admission 只是对已观测样本的本地检查，不是 WAN 时延保证。guided warm 请求的
prefix 为空，测得的是普通 sampler turnaround，不能代表非空 guided conditioning 的
额外开销。仿真控制频率是仿真时间轴配置，不等于墙钟实时运行能力。

## 显式 mode 与 committed prefix

handshake、WebSocket backend 和 controller 必须使用同一个 expected mode。hello 或
action response 的 mode 不一致会失败。

- `async_aligned` 禁止发送 `committed_prefix`；只保留本地 committed index，并 splice
  仍未执行的 future suffix。
- `guided_rtc` 使用现有冻结字段，prefix 必须精确覆盖
  `[target_start_action_index, committed_end_action_index_exclusive)`。
- hash 是 row-major little-endian float32 action payload 的 SHA-256；warm 的零长度
  prefix 也显式发送。
- guided response 的 mode、consumed count、prefix hash 与
  `sampler_conditioning="guided"` 必须和请求完全一致。
- `trained_rtc`、`synchronous_chunked`、guided 缺 prefix、async 带 prefix，以及把
  guided response 改叫 async 都会明确失败。

始终只有一个 inference request in flight。错误 epoch、request、observation、start、
shape、dimension、mode/prefix evidence、非有限值或超出 deployment physical bounds 的
action 都会整体丢弃/触发 fault。客户端不静默 clip native response。

## requested command 与实际 applied command

`next_action()` 返回 requested command。下一 tick 前，调用方必须把 simulator 或
actuator **实际返回**的 command 传给 `record_applied()`。客户端不能自行知道硬件实际
执行了什么，也不声称额外的 tracking guarantee。

profile 声明的 canonical 与 native 单位可能采用不同 command representation，
不能用有损 float32 往返后的数值直接判断是否改写。
`CommandEncodingReceipt` 会在任何可变 limiter/driver 调用前快照精确 canonical
command token 及其 native encoding；`NativeCommandTrace` 独立快照 post-limiter
实际提交的 native payload 与 driver 返回的 native payload：

- 若提交 payload 等于 receipt encoding 且 driver 原样 echo，则确认原 canonical token，
  不做第二次有损转换；
- limiter 或 driver 真正改变 native payload 时，按该实际值转换，并单独保留精确
  `native_changed` signal；
- bridge 使用 `record_applied(..., command_changed=True)` 传递该信号，因此即使不同
  native float32 反量化成同一个 canonical float32，guided work 仍会失效；
- prefix hash/bounds 不使用 tolerance，也不会伪造 canonical vector 来强行触发失效。

receipt 证明的是 command representation continuity，不是物理关节状态；后者仍需真实
安装的 telemetry。

如果 guided committed row 被本地 limiter/driver 改写，基于该 prefix 的 cloud suffix
已经失效。客户端会立即清空它、丢弃相关 in-flight result、进入 `rearm_required`，并在
唯一 in-flight 请求之后串行发送现有 reset。reset ACK 必须让 epoch 加一，之后才能用
新的 observation 重新 warm。fallback truth 使用实际 applied command，绝不把 raw
prediction 伪装成 applied。async mode 没有 value prefix，但同样记录 actuator truth。

## 串行 reset ownership

普通 reset 与 guided override reset 使用同一个 backend/socket lane。inference 尚未完成时，
客户端继续持有它的 `Future`，旧 response 只由原 backend call 消费。唯一 reset task 在其
后等待，再发送一次现有 reset frame 并接收一次 ACK；绝不会并发调用 socket `recv`。

`mark_disconnected()` 不会把 `Future.cancel()` 当成 running receive 已停止的证据。
无法取消的 receive 会继续被显式持有；warm 会被阻止，之后的 reset 必须等待该 owner，
即使调用方注入了 multiworker executor 也一样。

`reset(reason, timeout_seconds=5)` 会立即让本地 disarm。若调用方等待时间先耗尽，抛出
`ResetPending`，且 `reset_pending` 保持 true；timeout 不等于完成，epoch 不变。pending
期间再次调用 `reset` 只等待同一个 reset index/task，不会发送第二帧。只有精确
`old_epoch + 1` 的 ACK 才完成 reset 并允许重新 warm。失败或错误 ACK 会保留 unknown
server epoch 的显式错误，不能伪造本地 epoch 前进。

即使 backend receive timeout 更长，本地默认只等 5 秒；这限制调用方等待时间，但不
丢弃 in-flight ownership。

## Disconnect、fallback 与 rearm

网络断开后，已接受的 buffer 可以 drain。之后：

- delta/cartesian semantics：motion 归零，只保留配置 channel（例如 gripper）的最后
  actual applied 值；
- absolute joint-position semantics：重复完整的最后 actual applied target。

hold 次数有限，耗尽即停。普通 reconnect 必须接受 server-authoritative 新 epoch，
清理旧 cloud work，发送 fresh-observation rearm，再重新 warm；绝不恢复旧 response。

## LIBERO

先完成 setup，随后由 launcher 在 API 调用前复核 receipt，并在内存中获取一次性连接 token：

```bash
tenon-pi agent start-simulator --context .tenon/run.json \
  --setup .tenon/simulator/setup.json --display window \
  --task-suite libero_10 --task-id 0
```

只有服务支持时才在 `agent deploy --execution-mode guided_rtc` 选择 guided；launcher
读取 context 中实际 admitted mode，不会降级或冒充。显式 `--python` 也会先执行新的
真实本地 preflight，不能绕过资格检查。结果是 episode evidence，不是质量 benchmark。
检查范围见 [setup receipt 指南](simulator-setup.md)。

## 显式 profile 的本地机器人边界

`examples/robot_control_boundary.py` 使用准备后的 schema 和调用方提供的
RobotController。native/canonical state/action 维度、虚拟坐标、夹爪端点和 model delta
映射全部来自 profile，state/action 维度可以不同。驱动必须真实 connected/armed；
各通道 bounds/slew、相机同步限制、控制时钟、工作空间/碰撞约束和停止行为必须明确。

bridge 将 canonical requested command 按 profile.action 编码，经过本地 limiter/driver，
再用精确 receipt 报告 driver 返回的命令与独立 native_changed 信号。异常会 fault 并调用
adapter.stop()；多执行器 adapter 需负责部分失败后停止所有自己控制的执行器。
receipt 不等于物理关节 telemetry。

相机尺寸来自 profile，PNG request 的大小限制来自 hello，而不是新增 RobotSchema 字段。
服务必须实际实现对应 profile 的模型转换和输出约束，本地准备成功不能证明服务已有该
adapter。详见[数据准备](robot-preparation.zh-CN.md)。
