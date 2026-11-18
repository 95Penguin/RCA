# 上游实现存放约定

已保留 13 个固定版本的源码快照，来源、commit 与逐文件哈希见上级 `sources.lock.json`。仅包含源码、配置和说明，不带示例原始数据、权重、字节码或嵌套 Git 仓库。LTFP、Eadro、ART 已被当前框架适配器调用；其他来源只完成固定版本，尚未接入运行。

后续源码优先按 `<method_id>/` 放在这里，TORAI 的作者集成放在 `rcaeval/`，不虚构独立 TORAI 仓库。获取代码时先记录精确 commit、来源 URL、原始使用说明和许可证；保留作者实现，数据和评价适配放在旁边的 `adapters/`。

避免把多套嵌套 Git 仓库、训练数据和权重直接纳入主工程版本控制。下载源码时再选择固定 commit 的外部 checkout 或保留来源记录的快照，并针对实际目录添加忽略规则；本次不改主工程 `.gitignore`。

LTFP 已有本机副本但有未跟踪文件，只能按固定 commit 读取，不复制整个工作树。`/Users/95penguin/project/InstantOps` 的 origin 是用户自己的仓库，不是本次已确认的 InstantOps 作者发布。
