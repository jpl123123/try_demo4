# bug 目录：kvpress-ascend / SqueezeAttention-ascend 适配实战全清单

> 本目录收录实战中**所有** bug（含琐碎项），每条含：症状 → 根因一句话 → 修复 → 类目（对应 SKILL.md §3.4 的 Gx/Kx 模式）→ 发现途径 → 回归测试。
> 新项目请按同格式追加自己的 bug（REGRESS 纪律：每个修复必须能在这里找到一行记录 + 一个回归测试）。

## A. 开发期（L0/L1/L2 模拟测试发现）

| # | 症状 | 根因 | 修复 | 类目 | 发现途径 | 回归测试 |
|---|---|---|---|---|---|---|
| A1 | pytest 静默挂死（无 traceback，表现为超时） | registry 非重入锁：`with _lock:` 内调用会再取同一把锁的 `seams_summary()` | 锁内只取数据快照，日志输出移到锁外 | G10 | 测试超时排查 | test_heartbeat |
| A2 | topk 保留集偏移 | 打分 gather 用 `m*bs` 个槽（含尾块 padding），padding 内容污染分数 | `repeat(row, bs)[:orig_len]`，只取真实 token 槽 | K3 | L2 保留集断言 | test_l2_invariants |
| A3 | 行重写二次执行后内容错乱 | `rewrite_row` 非幂等，却每步重放 | 一次性重写 + `row_rewritten` 标志 + `num_blocks_per_row` 缩减为 `k+(valid−m)` | K4 | L2 compact 不变量 | test_l2_invariants |
| A4 | 窗口注意力 matmul 维度错/结果乱 | keys `unsqueeze(1).transpose(1,2)` 顺序错 | `transpose(0,1).unsqueeze(1)` → `(kvh,1,k_len,hd)` | K5 | 模拟器报错 | simulate/snapkv |
| A5 | 单步完成整个 prompt 的请求永远"未完成" | 完成判定 `0 < before` 漏掉 `before==0` 首步 | `before < prompt and before + n_sched >= prompt` | G1 | L1 测试 | test_l1_capture |
| A6 | `TypeError: 'NoneType' object is not subscriptable` | 未捕获层的 `rc.queries.get(layer)[:n]` 直接切片 | 先判 None 再切片 | K6 | 模拟器报错 | test_l1_capture |
| A7 | 强制保留尾块时丢错块 | "让位"用 `argmin(全部块分)` 而非 `argmin(已选块分)` | `argmin(bscores[bl])` | K7 | L2 保留集断言 | test_l2_invariants |
| A8 | 多步 decode 参考集断言莫名失败 | 测试参考集只含当前步新 token，漏累积 | 参考集跨步累积全部新 token | G6 | 不变量断言 | test_l2_invariants |
| A9 | 心跳单测偶发失败 | `registry._last_step` 全局守卫被前序测试污染 | 测试内重置 `_last_step = -1` | G11 | 测试隔离排查 | test_heartbeat |
| A10 | pytest `import file mismatch` 收集失败 | 两包测试文件同名（test_l0_kvcore.py） | basename 全局唯一（改名 test_l0_window_math.py） | G12 | pytest 收集报错 | — |
| A11 | S7 位移不生效 | `_shift_positions` 读 engine 模块级 `ctx`，测试未注入 | 测试设 `engine.ctx = mgr` | G11 | 断言失败 | test_l2_invariants |
| A12 | 测试注意力参考 einsum 报错 | `einsum("skh,skh->kh")` 用 2D attn 当 3D；softmax 维度错 | 参考实现改为 `"sk,skh->kh"`，softmax 沿 S 维 | G24 | 测试报错 | test_l2_invariants |
| A13 | 模拟器 `index out of bounds` | fake 物理块 id（1000+）超出 fake 缓存张量尺寸（256 块） | fake id 用 `r*100+b` 落在尺寸内 | G24 | 模拟器报错 | simulate |
| A14 | 模拟器误删布局（`layout_dropped_recompute` 乱涨） | driver 不维护 fake 的 num_computed，recompute 检测误判 | driver 每步按 sample_tokens 时序更新 num_computed | G24 | 模拟器统计 | simulate |
| A15 | ubatch 场景错改/崩溃 | `attn_metadata` 是 `list[dict]` 时 `.items()` 失败 | `isinstance(list/tuple)` → 跳过该步（`skipped_ubatch`） | G13 | 源码审查 | test_failsoft |
| A16 | 空窗口 softmax 空张量 → NaN/崩溃 | `k_len − w < 1` 未守卫 | 返回全 1 分数（等价不剪枝） | G14 | 边界推演 | presses 单测 |
| A17 | n_kept 突然变 48（应 36），视图多读一个尾块 | 整理代码时把 `orig_len − b*bs` 写成 `orig_len − int(b)`（块下标当起始位置） | 改回 `orig_len − int(b)*bs` | G23 | **L2 不变量当场抓住**（`slot 0` 尾项） | test_l2_invariants |

## B. 真机反馈（NPU 日志驱动）

| # | 症状 | 根因 | 修复 | 类目 | 发现途径 | 回归测试 |
|---|---|---|---|---|---|---|
| B1 | 心跳永远 `FAIL=S6_compress_pass` | 逻辑 seam（压缩 pass 寄生 S5 钩子）从未 mark_installed；`seams=N/M` 混用命中数 | 逻辑 seam 随宿主钩子一起标记；口径改 installed + 独立 hit | G16 | 真机心跳 | test_activation_policy |
| B2 | 16×262144-token prompt、KV 91.5%、`completed=0` 永远、压缩从不触发 | 完成时触发的设计与资源在完成前耗尽 = 鸡生蛋（抢占循环） | 渐进式 mid-prefill 压缩（预算锚点推进 + 完成时重锚定） | G17 | 真机日志统计 | test_l2_mid_prefill |
| B3 | 改了源码重跑测试仍是旧行为（AttributeError 找不到新属性） | pip 安装的 `.pth` 在解释器启动时把旧包注册进 sys.modules | 开发循环先 `pip uninstall`；发布前重装并核验 `__file__` | G18 | 测试环境排查 | — |
| B4 | `can't convert npu:N device type tensor to numpy` | 设备张量直接 `.numpy()`/`.tolist()`（CPU mock 测不出） | `_as_numpy()` 统一 `detach().cpu().numpy()` | G19 | 真机 mid-prefill 报错 | _as_numpy 单测 |
| B5 | compact 元数据设备不匹配 | per-layer `seq_lens` 是 CPU 张量、delta 在 NPU，直接相减 | `delta_t.cpu()` 后再减 | G19 | 源码审查 | — |
| B6 | 请求已进 decode 但 `completed=0`（step9 prefilling=1 → step10 prefilling=0 无完成步） | 末块跨步时严格判定漏检（MTP async 计数口径） | 补检 `last_before < prompt <= before` + `_compressed_done` 去重 | G20 | 真机心跳序列 | test_l1_capture |
| B7 | mid-prefill 失败：`'NoneType' object has no attribute 'shape'` | 某层 `kv_cache=(None,None)`，`not kc` 拦不住元组元素 | 显式元素检查 + `skipped_no_kv` + **逐层 fail-soft**（坏一层不毁整请求） | G21 | 真机 mid-prefill 报错 | test_failsoft |
| B8 | AIV `IndexCheckKernel::CheckUpperBound` 断言（index 341748 > 923136） | 非法块 id 打进设备 gather，污染 NPU 流（try/except 救不回） | G11 守卫：行 id ∈ [0,num_cache_blocks)、槽 ∈ [0,num_blocks·bs)、保留块同界；`skipped_bad_row` 诊断 | G9 | 真机 AIV 断言 | test_failsoft |
| B10 | 每轮压缩都打 `layer mtp.layers.0.self_attn.attn scoring failed: 'NoneType' ... 'shape'` | step3.5 的 MTP draft 层混进 group-0 层列表且 kv_cache 未绑定；逐层 fail-soft 兜住但刷屏 | 结构性排除 draft 层（drafter.attn_layer_names + `.mtp.`/`.draft.` 启发式，`layers_excluded_draft` 计数）；kv_cache 缺失告警每层只报一次 | G25 | 真机日志 | test_failsoft |
| B12 | 压缩健康运行 ~400 步后 AIV 越界崩 worker（索引值跨 run 确定性重复 ~341k） | 疑似坏视图布局（保留块/视图长度越界）到达 FIA；旧代码只校验了保留块 id | 元数据改写**原子化**（快照原值、异常恢复，杜绝半改写状态）+ **全量自检**（m/klen≤max_blocks、view_len∈[1,true_len]），任一违规丢弃布局并 `skipped_bad_view` 诊断；坏视图**永远到不了 FIA** | G9/G21 延伸 | 真机 AIV | test_failsoft |
| B11 | `ArgSortKernelNpuOpApi.cpp` WARNING：kernel [ArgSort] 不支持 int32/int64，降级 AiCpu | 设备端 `topk(...).indices.sort()`（int64 索引） | 排序移到 numpy（`np.sort(_as_numpy(...))`），topk 留在设备端 | G26 | 真机启动/运行期 WARNING | 既有 L2 套件 |
| B9 | 让位路径打 `ERROR installed with FAILED seams`（还引用没打印的 summary） | apply() 把"主动让位"当"安装失败" | `DEFERRED_REASON` 标记 → INFO；真实失败才 summary+ERROR；ASCII 破折号 | G22 | 真机启动日志 | test_activation_policy |

## 统计

- 共 **29 条**（开发期 17 + 真机 12）；其中模拟测试发现 15、真机日志驱动 9、代码审查/推演 2。
- 类目分布：通用模式 G1–G24 覆盖 19 条；KV 压缩案例模式 K3–K7 覆盖 5 条。
- 每条真机 bug（B 组）都有对应回归测试入库；开发期 A 组 15 条有测试/模拟器断言覆盖。

## 二期增补（2025-08，kvpress-ascend / SqueezeAttention-ascend）

- 新增 C1–C12（视图行物理 id、窗口滑动 sync 分支、hook 闭包晚绑定、owner 竞态、
  registry 双身份、kv_heads 未解析、重要性方向语义、首层 residual、view_len 锚点语义、
  集合 vs 顺序不变量、env 地板值、lazy seam 口径）——全部有回归测试。
- 通用模式新增推论：G23 扩展「物理/逻辑块 id」、G2 扩展「循环内闭包晚绑定」、
  G16 扩展「lazy seam 口径」、G11 扩展「双包 .pth 启动竞态与 owner 标记」。

## C. kvpress-ascend / SqueezeAttention-ascend 适配新增（2025-08 第二期）

| # | 症状 | 根因 | 修复 | 类目 | 发现途径 | 回归测试 |
|---|---|---|---|---|---|---|
| C1 | 视图行内容错乱（多请求时保留块被读成物理块 id 错位） | 视图 buffer 行写入的是**逻辑块下标**（kept 数组值），FIA 需要**物理块 id**（`row[kept]`） | buffer 填充改为 `row[kept]`（同步时从真行取值） | G23 变体 | **L2 端到端不变量**（view slots == 参考槽集合） | test_l2_invariant（两包） |
| C2 | 窗口滑动后视图读错块（append 分支在 recent_first 前移时执行） | squeeze `_sync_window_row` 的 append-only 分支没校验 `recent_first` 未变 | 分支条件改为 `recent_first == marker.recent_first` 才 append；前移走全量重装 | G23 变体 | L2 不变量跨块滑动 | test_l2_window_invariant |
| C3 | 层 hook 全部拿到最后一层的 layer_name/forward（cos-sim 全错） | 循环内闭包**晚绑定**（layer_name/orig/residual_pos 捕获变量） | 默认参数绑定 `_layer=_layer` 等 | G2 变体 | stats 层数/均值断言 | test_layer_hook_measures |
| C4 | 双包同开时所有权竞态（谁先装取决于对方策略检查是否 eager import） | `.pth` 启动期互相 find_spec+import，`_APPLIED` 检查时机不定 | 共享 `KV_ASCEND_OWNER` 进程标记 + 策略只读 env/sys.modules，**不跨包 import** | G11 变体 | 安装链路双包测试 | test_engine_defers_* |
| C5 | 测试进程内 re-import 包导致 registry 双身份（计数器全空、断言假败） | gate-off 测试 pop sys.modules 后重 import，harness 引用旧模块对象 | gate-off 测试改子进程验证；测试内禁止 re-import 包 | G11/G24 | 全量跑批 | test_gate_off_zero_imports |
| C6 | `kv_heads=0` → gather reshape 崩溃（snapkv 全层跳过） | `_resolve_target_layers` 只解析了 block_size，未解析 num_kv_heads/head_size | 缓存张量 shape 为地面真值（`key_cache.shape[2]`）；spec 属性兜底 | G2 变体 | simulate CLI | test_l2（snapkv） |
| C7 | 层重要性方向断言反了 | cos-sim 低 = 表示变化大 = 更重要（上游 class3 = 最高 cos 拿最小预算） | 语义核对上游代码后修正断言与注释 | 语义 | 上游对照 | test_squeeze_mode_clustering |
| C8 | 首层 cos-sim 永远捕获不到（residual=None） | 首层 residual 为 None，hook 只查 residual 参数 | residual 缺失时回退 hidden_states（首层输入即 residual） | G14 变体 | stats 键数断言 | test_layer_hook_measures |
| C9 | 锚点时刻视图长度=全量（压缩“不生效”） | `view_len` 初版 `true_len<=orig 返回 true_len`，anchor 时读全量 | 语义修正：`view_len = kept_tokens + max(0, true_len-orig)`（anchor 时=kept_tokens） | 公式语义 | **L0 断言当场抓住** | test_l0_core |
| C10 | L2 端到端不变量用 `array_equal` 误报 | 视图按块序读、参考按真实序读：槽**集合**相同、顺序不同（注意力与顺序无关） | 不变量改集合比较 + 长度断言（绝不读未写 padding）；参考集用**步前布局快照**（anchor 步 S4 先于 S5，视图下一步才生效） | G24/时序 | L2 跑批 | test_l2_invariant |
| C11 | 小预算测试配置被 env 地板值吞掉（`max(1024,...)`） | envs 层给 mid/decode 预算设了过高下限 | 地板降到 16；harness 的 `make_runner` 先清空本包全部 env 再设新值（防跨测试污染） | G11 | 跑批 | test_l2（budget=40） |
| C12 | 心跳/汇总把惰性安装的 seam 报 FAILED | layer_hook 在首个推理步才安装（模型加载后才存在），安装期汇总误判 | summary 只对 eager seams 判 FAIL（`EAGER_SEAMS` 口径），lazy seam 标记为声明项 | G16 变体 | L1 安装测试 | test_l1_seams |

## D. 三期：真机 no_q / compose 组合模式（2025-08）

| # | 症状 | 根因 | 修复 | 类目 | 发现途径 | 回归测试 |
|---|---|---|---|---|---|---|
| D1 | 真机心跳 `hit=94560` 巨大但 `viewed=0`、`skipped_no_q=337280` | `mark_hit("backend_forward")` 在 append 循环**外无条件执行**——hit 增长只证明 forward 被调用，不证明捕获成功（诊断误导） | `mark_hit` 只在 append 成功后计数；`maybe_capture_query` 每个提前返回分支打 `cap_*` 分原因计数（cap_guarded/cap_len_mismatch/cap_layer_miss/...），心跳 `cap=app:.. skip:..` 一眼定位断点 | 可观测性（G16 变体） | 真机日志 | test_capture_fallback |
| D2 | snapkv 每层打分都 `no_q`（捕获被真机环境阻断，具体 guard 待 cap_* 定位） | 捕获链路任一 guard（capturing/draft/图回放/长度错位）命中即整层跳过 → 压缩全部失败 | `KVPRESS_ASCEND_PRESS_FALLBACK=streaming`（默认）：无窗口时降级 positional 打分，压缩照常发生（`fallback_streaming` 计数）；`=none` 关闭 | G14/G21 变体（兜底优先于跳过） | 真机日志 | test_snapkv_falls_back_to_streaming_without_window |
| D3 | compose 模式下 kvpress 完成压缩看不到 squeeze 预算 | 包装嵌套顺序 = 安装顺序（晚装者最外层、其 finally/pass 最后跑）；真机 kvpress 先装（最内层）→ 其 pass 先于 squeeze 聚类跑 | 完成延迟一拍：`_compose_completion_deferred`（squeeze 激活且预算未就绪且 `compose_defer_count<2`）→ 延迟，G20 补检下一步重触发；上限防死锁 | 时序（L2 状态时序） | 设计推演（真机序模拟） | test_compose_completion_defer_decision |
| D4 | compose 让位判定读对方模块状态（`kvpress_ascend._APPLIED`）→ 测试污染、潜在启动竞态 | 跨包状态检测违背"绝不跨包 import/读模块状态"纪律 | `_compose_defers_views` 改**纯 env**（两个 policy 都 == compose）；跨包数据走 runner 属性桥 | G11 变体（组合模式版） | 测试跑批 | test_compose_* |
| D5 | 新增 compose 测试后，既有 L2 窗口不变量失败（vs 52 vs 53） | compose 测试把 kvpress 包装（class 级）/env（kvpress 系列变量）/模块状态残留到共享 FakeRunner 类与进程 → 后续测试被 kvpress S4 改写污染 | 组合测试用**独立 runner 子类**装 kvpress 包装；两个 harness 的 make_runner 统一**清两包全部 env** | G11/G24 | 全量跑批 | test_compose（隔离）+ 全套件 |
| D6 | `class _Cfg: policy = policy` → `NameError: name 'policy' is not defined` | Python **类体不参与外层函数作用域闭包**（class body 只查 类命名空间→全局→builtins） | 改用 `types.SimpleNamespace(policy=policy)` | Python 语言陷阱 | L1 测试编写 | test_l1_seams |
| D7 | bash heredoc 里写多行 Python 字符串 → `SyntaxError: unterminated string literal`；尾随空格 → `assert old in s` 静默失败 | heredoc 中字符串字面量跨物理行/行尾残留空格与文件内容不匹配 | 字符串一律单物理行（显式 `\n`）；编辑用 edit 工具/先 repr 核对锚点 | 工具链 | 编辑期 | — |

> 三期方法论沉淀：多机制**同时兼容**不是把逻辑写进一个包，而是按逻辑链分工
> （§2.3d compose 原则）——层维度预算 / token 维度选择 / 视图表达各归一个机制，
> 唯一写者 + 运行期桥 + 纯 env 检测 + 延迟一拍 + fallback 兜底。
