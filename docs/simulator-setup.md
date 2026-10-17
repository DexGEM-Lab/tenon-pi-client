# Agent-owned simulator setup before spending

The coding agent owns this step. A user with data and an API key does not need to
arrive with a simulator Python already configured. Run from the cloned client:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install '.[data]'
tenon-pi agent setup-simulator --directory .tenon/simulator --display window
```

The helper provisions an isolated Python 3.11.11 with uv 0.6.17, or accepts an
existing Python3.11 via `--python`. It installs this SDK non-editably, CPU PyTorch
2.2.2 (CPU wheel index on Linux), MuJoCo3.2.3, robosuite1.4.1, NumPy1.26.4,
Numba0.59.1 and the direct pins in
`src/tenon_pi/resources/simulator-requirements.txt`. It fetches LIBERO at
`f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`, the submodule recorded by OpenPI
`215abfb217dbac7d5f1273282331b9b1866c0479`. No OpenPI model package, checkpoint
weights, server or billable resource is created.

The pinned upstream LIBERO wheel omits its namespace root. The helper explicitly
exposes only its owned pinned source checkout through a local `.pth`, and writes
its own `LIBERO_CONFIG_PATH` rather than changing `~/.libero` or prompting on
first import. The SDK itself remains non-editable. API credentials and inherited
PYTHONPATH are removed from installer/preflight subprocess environments.

## Actual preflight, not import-only readiness

Setup creates the real LIBERO-10 task-0 environment, restores its initial state,
renders both256×256 cameras, presents frames in the requested viewer and performs
three declared dummy stabilization steps. No policy is loaded. Only after these
operations succeed does `.tenon/simulator/setup.json` record `preflight=passed`,
actual package versions, camera shapes, selected display and environment/Python,
plus a `tenon-simulator-receipt-v2` byte inventory.
A failed attempt removes any prior success receipt; inspect `setup.log` and fix
the cause before retrying. An owned partial directory can resume, but foreign
existing directories are rejected.

- **Linux desktop:** automatic renderer selection uses the pinned robosuite GLX path when DISPLAY exists,
  with `LIBGL_ALWAYS_SOFTWARE=1` and Mesa GLX vendor selection for CPU preflight.
  It requires an observed llvmpipe/softpipe/software GL renderer before success.
  The pinned robosuite rewrites `glfw` to EGL, so the helper rejects that spelling. Matching GL/Tk libraries and
  X11/XWayland are host requirements. Headless machines can explicitly use
  `--display headless --renderer osmesa` when OSMesa is installed. For CPU CI with
  Xvfb and GLX already available, run the helper under `xvfb-run -a` with
  `--renderer glx`. A virtual window is not proof of a user's physical desktop.
- **macOS:** the helper selects native CGL and the supplied/provisioned Python must
  have usable matching Tk. Linux EGL/OSMesa settings are rejected. Actual macOS
  qualification is not claimed by Linux tests. If the chosen Python lacks Tk,
  the agent should select/install a compatible user-local interpreter, or ask
  for necessary host permission; do not assert readiness or silently use a movie.
- No sudo/system-package installation occurs automatically. Missing host GL/Tk,
  build tools, display access or network permission is an explicit blocker that
  needs host permission, not an excuse to create a paid training/session resource.

Once preflight passes, acquire/materialize the recommended **native v2** LIBERO
source, then use `agent train` and `agent deploy`. The technical starter recipe is
100 full-FT steps,7200-second runtime cap,checkpoint interval100 and a300-second
async-aligned session. These are workflow defaults, not quality/performance
promises. The agent checks current capabilities/quotes and partitions the user's
one approved total ceiling between training/control caps; it may not invent a
ceiling or bypass quote approval. Reuse the already-authorized run context.

```bash
tenon-pi agent start-simulator --context .tenon/run.json \
  --setup .tenon/simulator/setup.json --display window --task-id 0
```

The launcher reads the preflight's Python/environment; the user need not choose
an internal preset or a simulator executable. `window` requires an actual window,
`auto` reports fallback, and `headless` explicitly has none. Re-run setup/preflight
after changing dependencies, renderer or machine. Close the admitted control
session after the episode. Hosted native LIBERO-v3 is not supported merely because
local v3 materialization exists.

## Receipt freshness and coverage

A success receipt binds the actual non-editable SDK package tree and distribution
files, every installed file listed by each pinned direct simulator dependency,
the observed GLFW/PyOpenGL distribution files, the pinned LIBERO revision/tree
and runtime source/resource bytes, configuration bytes, and selected Python
binary/version/environment. Actual file contents are hashed; package versions,
RECORD entries or mtimes alone are not treated as content proof. SDK file additions
and ordinary same-version module/dependency/source changes invalidate the receipt.
Editable SDK/dependency installations are rejected; the pinned LIBERO checkout is
covered separately. Unlisted third-party file additions, OS/user-site injection
and packages outside the selected environment are not covered as installed files.
The optional installed-wheel hash is corroborating metadata; the actual installed
bytes remain authoritative.

Preflight compares the inventory before and after creating/closing the real
environment/window. The parent launcher validates in the selected interpreter
before API use, and the child rechecks before requesting or reading a connection
token. Old v1, incomplete, edited or drifted receipts fail explicitly. Re-run
`agent setup-simulator` after drift; do not repair hashes by hand. An advanced
`--python` override performs a new real credential-free preflight every time.

Inventory checks occur at setup/startup boundaries, never per control frame. They
cover the named package/source set, not the whole host: OS drivers, Python stdlib,
unselected transitive packages and display permissions are not content-attested.
Actual environment/window creation still precedes token use. Receipts are local
qualification records, not signatures or protection against an administrator who
can rewrite both the code and the receipt. No API credential, service endpoint or signed
URL enters the receipt or qualification subprocess arguments/logs.

## 中文

agent 应先运行 `agent setup-simulator --display window`，负责建立隔离 SDK/仿真环境，
而不是要求用户提前配置 Python。它固定 Python/仿真依赖和 LIBERO 版本，在 CPU 上创建
真实环境、渲染两路相机并检查可见窗口，成功后才生成 setup.json；不加载模型、不创建
任何付费资源。缺少系统 GL/Tk、显示器访问或安装权限时明确询问，不以 headless 视频
冒充窗口。通过后使用该 receipt 启动仿真。用户只需批准一个总预算，步骤/时限等技术
默认值由本 runbook 和 agent 管理。推荐托管流程走 native v2；本地 v3 导出不代表服务
已经支持 native LIBERO-v3。Mac 的实际运行资格不能由 Linux 测试代替。

新的 v2 receipt 绑定实际 SDK、指定依赖、LIBERO 源码/资源与配置的字节，而非仅版本号。
启动前会在所选解释器中复核，child 在申请/读取连接 token 前再次检查。旧版、不完整、
修改过或环境漂移后的 receipt 会明确拒绝，需重跑 setup 和真实 preflight。显式
`--python` 也必须先做新的实际预检，不能绕过。检查仅在 setup/startup 发生，不逐帧
全环境扫描，也不声称能抵抗可同时改写程序与 receipt 的管理员。
