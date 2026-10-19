# Tenon Pi Client

**Unpublished 0.3.0a1 contract-pairing candidate.** New training and trained-RTC
features require a qualified amended endpoint; installing this candidate does
not enable them. Published 0.2 remains usable for ordinary runs. Install candidate
bytes from the reviewed checkout/wheel, not an unpublished tag. See the
[recommended WAN workflow](docs/training-candidate.md).

A Python SDK, CLI and coding-agent workflow for hosted VLA training and inference:
**local demonstrations → validated dataset → full fine-tuning or LoRA → checkpoint
→ deployment → simulator or your configured robot controller**.

The client is robot-agnostic. A supplied, validated profile declares the channel
layout, units, calibration, cameras and model transforms. Hosted model support
is determined by the service, not by whether a local profile can be prepared.

## Clone, configure, give your agent one prompt

```bash
git clone https://github.com/DexGEM-Lab/tenon-pi-client.git
cd tenon-pi-client
python -m venv .venv
source .venv/bin/activate
pip install '.[data]'
```

Set `TENON_API_KEY` through your terminal's secret-input mechanism. Set
`TENON_BASE_URL` to your invited service endpoint if different from the configured
default, `https://tenon.dex-gem.ai/v1`. This repository does **not** establish that
the default endpoint is currently available to your account. No service is needed
for offline inspection/preparation. This source candidate is `0.3.0a1`; availability
on a package index or release page is not implied.

Open your coding agent in the repository. `AGENTS.md` discovers the workflow;
Claude Code uses `CLAUDE.md`. For a qualified candidate endpoint, paste this,
replacing the local data path, user effective batch and budget:

> Read AGENTS.md and use Tenon Pi. My data are in `./my-data`; the API key is in the environment.
> Use my supplied calibrated adapter or requested simulator; pass its local preflight before spending.
> My effective batch is **[BATCH]**. Use the WAN-trained workflow only with a qualified endpoint and checkpoint.
> I authorize at most **[TOTAL] microcredits in total**: show quotes, stay within it and reuse `.tenon/run.json`.
> Prepare, train, verify and deploy; open the live simulator or run my supplied calibrated adapter. Ask only for irreducible missing information and report real outcomes.

The agent can infer a supported import route from metadata. It cannot infer
unidentified units, manufacture calibration or create a hardware connection.
For a hardware-free demo, use [the exact LIBERO prompt](docs/demo-prompt.md).

## Offline smoke test

```bash
tenon-pi data fixture --layout gantry -o .tenon/example
tenon-pi data inspect .tenon/example/native-v2
tenon-pi data prepare .tenon/example/native-v2 \
  --profile .tenon/example/profile.json --input-space native \
  --repo-id example/gantry -o .tenon/prepared
tenon-pi data verify .tenon/prepared
tenon-pi data manifest .tenon/prepared -o .tenon/bundle.json
```

The generated data/calibration are **synthetic mechanics fixtures**, never robot
commissioning values or policy-quality evidence. Two abstract layouts exercise
rotary and linear mechanisms, unequal state/action dimensions, virtual slots,
different cameras, rates and horizons.

## Documentation

- [English quickstart](docs/quickstart.md) · [中文快速开始](docs/quickstart.zh-CN.md)
- [Profiles and data preparation](docs/robot-preparation.md) · [数据与配置](docs/robot-preparation.zh-CN.md)
- [Public LIBERO data acquisition and simulator](docs/libero-demo.md)
- [Control chronology, receipts and reset](docs/control-runtime.md) · [控制语义](docs/control-runtime.zh-CN.md)
- [Agent skill](skills/tenon-pi-agent/SKILL.md) · [Integration/migration contract](docs/client-integration.md)

`tenon-control-v1` supports explicit action chronology. `async_aligned` and
`guided_rtc` are distinct; requesting one never silently selects the other.
`openpi-ws-v1` is a separate compatibility path. There is no WAN latency SLA or
claim of compatibility with a gated hosted API. Successful publication,
session admission and a successful task episode are separate observations.

## 中文：数据 + API key，一条提示词

Tenon Pi 是通用托管 VLA 客户端，不绑定某一种机器人。把原始示教数据、明确的
状态/动作含义及必要的标定交给 coding agent；它会检查、转换、上传、训练、检查
checkpoint 来源、部署，并启动仿真或你已配置且真实连接/armed 的控制器。

按上面的步骤克隆和安装，把 key 放入 `TENON_API_KEY`，按邀请配置
`TENON_BASE_URL`。默认地址的存在不代表服务已对你的账号开放。然后告诉 agent：

> 阅读 AGENTS.md 并使用 Tenon Pi；数据在 `./my-data`，API key 已在环境变量中。
> 请负责安装隔离的 SDK/仿真环境，先通过 CPU renderer 和可见窗口检查，再创建付费资源。
> 我授权的**总预算上限为 [预算] microcredits**；展示报价、不超支，复用 `.tenon/run.json`。
> 完成准备、训练、核验和部署，打开本机实时仿真窗口或运行我提供的已标定 adapter。仅询问不可推断的信息，如实报告结果。

详见[中文快速开始](docs/quickstart.zh-CN.md)。
