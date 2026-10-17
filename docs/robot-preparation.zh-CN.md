# 显式 profile 驱动的数据准备

`RobotProfile` 是经过验证的 JSON 数据，不执行表达式或加载代码。它描述布局和转换，
并不声明服务已有对应模型。参考 [JSON schema](../schemas/robot-profile.schema.json)
和两个合成示例：[旋转平面机构](../examples/profiles/planar.json)、
[直线机构及独立负载传感器](../examples/profiles/gantry.json)。合成标定不能用于真机。

## 必须明确的内容

- `profile_version`、`profile_id`、实际 `calibration_digest`。
- 独立的 state/action 映射：native 通道的名字、单位、测得的 bounds，以及 canonical
  坐标顺序。每个物理坐标必须恰好保留一次，可重排并添加常量虚拟坐标。
- 转换仅支持 identity、角度到弧度、毫米到米、夹爪 closedness、常量。identity 的单位
  必须一致；夹爪需提供不同的 `native_closed` / `native_open` 标定端点，输出 `[0,1]`
  闭合程度。单位不明就报错，不猜测。
- `model_delta_state_indices` 明确模型空间中哪些绝对动作要减去哪个 captured-state
  坐标；null 保持绝对值。上传数据仍保留 canonical 物理动作，不提前改成模型 delta。
- 模型 state/action padding 宽度、相机尺寸/ID/序列号/安装位置/时钟/朝向/标定摘要、
  image transform、控制频率、action horizon 和 committed prefix 上限。

native/canonical feature names 必须与 profile 的顺序完全一致。准备过程不裁剪数值、
不猜比例、不重采样时间、不静默重新编码图像；生成的 RobotSchema 不增加 wire 字段。

## 支持范围

- v2.0：RGB PNG，native 或 canonical 向量，global `meta/stats.json`。
- v2.1：RGB PNG，native 或 canonical 向量，`meta/episodes_stats.jsonl`。
- v3.0：仅 canonical 向量、RGB video、默认模板和完整连续 episodes。
- 明确的 JSONL + PNG：先导入 v2.1，再准备。

v2 检查每个 episode 的 Parquet、连续索引、task/episode ownership 和 Arrow 类型标签。
只转换 state/action 列及其 metadata；所有 numeric feature 和真实 PNG pixel 的统计均
重新计算。可选的旧统计文件也会更新，不添加空的跨版本统计文件。按 profile 频率和
100 微秒容差检查相邻时间戳，原值不变。

v3 原始 rows、视频、task/episode metadata 和时间轴保持不变。支持跨 episode 共用
视频、非 keyframe offset，以及 data/video 各自独立换文件。检查 ownership、offset、
duration 和逐行时间；本地验证不能替代服务的真实视频解码验证。不支持 native-v3、
v2 video/JPEG、混合/depth 或自定义 v3 模板。LIBERO 使用独立路径和时间语义。

```bash
tenon-pi data prepare ./source --profile ./profile.json \
  --input-space native --repo-id owner/recording -o .tenon/prepared
tenon-pi data verify .tenon/prepared
tenon-pi data manifest .tenon/prepared -o .tenon/bundle.json
```

只有数据已经符合 canonical 布局/单位时才用 `--input-space canonical`。`--repo-id`
是稳定的逻辑 `namespace/name`，不能是路径或 URL。输出必须是源目录之外的新目录。
上传/训练入口必须指向整个 bundle，保留 dataset、assets、schema 和 preparation manifest。
验证会重新生成统计和绑定；仅修改哈希不能使错误统计通过。

每行的 horizon 包含同一 episode 中连续的动作，仅尾部用最后动作补齐。先做 float32
转换，再逐行做一次 float64 state 更新和一次 action-window 更新。LeRobot feature
metadata 与模型 normalization 是两种不同统计，不能混用。

## 原始 JSONL

每行恰好包含以下字段，坐标和相机名必须匹配自己的 profile：

```json
{"episode_index":0,"frame_index":0,"timestamp":0.0,"state":[0,0,50],"action":[1,2,40],"task":"place block","images":{"view_0":"frames/000.png"}}
```

```bash
tenon-pi data import-jsonl ./raw/rows.jsonl --profile ./profile.json \
  --input-space native -o .tenon/imported
```

episode/frame 从零连续，保持原行顺序。PNG 相对 JSONL 所在目录寻址，禁止 traversal
和 symlink，图像原始字节直接嵌入。state/action 必须能被 float32 精确表示，否则需
先明确批准精度转换，不能隐式 downcast；时间戳保留 float64。源 JSONL/图像摘要写入
provenance，发布前再次核对源文件。

其他原始格式需先确定语义再写 converter，逐项证明值、像素、顺序和 episode 的保留。
单纯通过文件格式检查不是无损证明。

## 真机边界

`RobotController` 需要明确的 bounds/slew、实际 connected/armed driver，并以 driver
返回的实际命令为准。失败会 fault 并调用 adapter.stop()；多执行器 adapter 必须在部分
失败后停止所有自己负责的执行器。SDK 不会凭空提供物理 telemetry、现场碰撞限制或
多设备急停能力。详见[控制语义](control-runtime.zh-CN.md)。

## 明确的模型输出投影

`output_projection` 可选 `strict`（默认）或 `clip_bounds_v1`，进入 canonical descriptor
和 transform digest。仅对模型反归一化输出生效：先检查所有输入坐标有限，再裁去
padding、按 captured state 逆 delta、恢复虚拟坐标，再检查结果有限，最后按声明选择
是否用 float32 bounds 投影。NaN/Inf 和逆变换 overflow 不能被裁剪掩盖。strict 的越界
native target 会拒绝；`canonical_target(..., project=False)` 保留未投影 raw 输出。
示教数据、输入 observation、普通单位转换和已收到的 canonical command 都不静默裁剪。
新的未发布 bundle 需重新生成，历史已接受的 artifact 身份不变。
