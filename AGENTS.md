# Course Weight Optimizer — Agent 指引

这份文件是给 Codex、Claude Code、Cursor 等 coding agent 读取的项目说明。用户只需要把仓库目录路径交给 agent，并要求读取本文件，agent 就应当能够完成本地初始化、配置检查和运行。

## 项目目标

这是一个东北大学投权选课的本地 Python 命令行工具。它根据课程偏好和全局选课快照，输出建议投权向量以及保守、中性、激进三种情景下的代理录取概率。代理概率用于比较策略，不是真实录取承诺。

## 在用户电脑上部署

1. 将工作目录切换到仓库根目录（包含本文件的目录）。
2. 确认 Python 3.9 或更高版本：`python3 --version`。
3. 创建并使用隔离环境（如果用户已有项目环境，可以沿用）：

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate       # Windows: .venv\Scripts\activate
   ```

4. 本项目没有必需的第三方依赖；直接使用标准库即可。若希望表格在终端中更好地显示，可选安装 `wcwidth`：`python -m pip install wcwidth`。
5. 程序入口位于 `Course_Weight-Optimizer/main.py`。运行前切换到该目录，因为输入 JSON 使用相对路径：

   ```bash
   cd Course_Weight-Optimizer
   python main.py
   ```

6. 首次运行先使用仓库内的示例 `desired_courses.json` 和 `global_state.json` 验证环境，再询问用户是否要替换成自己的数据。不要擅自覆盖用户的 JSON 文件。

## 输入与输出

- `desired_courses.json` 的 `preferences` 数组包含 `course_id` 和 `utility`。
- `global_state.json` 必须包含正整数 `grade_size`，以及带有 `course_id`、`capacity`、`bidders` 的 `courses` 数组。
- 如果用户提供了其他位置的 JSON，先复制到 `Course_Weight-Optimizer/`，或在运行前明确修改 `main.py` 顶部的路径常量；修改后要告知用户。
- 结果直接打印到终端。把结果解释为模型建议和风险区间，不要把代理概率表述成学校系统的真实概率。

## Agent 工作约束

- 先读取 `README.md`、`main.py`、`utils.py`，再执行命令或修改文件。
- 只在用户明确要求时修改算法、输入样例或投权参数；默认只做环境初始化和运行。
- 运行前检查当前工作树，保留用户已有改动，不执行破坏性 Git 操作。
- 任何代码修改后至少运行：

  ```bash
  python -m py_compile main.py utils.py
  python main.py
  ```

- 如果输入数据缺失、格式不合法或 Python 不可用，说明具体原因和下一步，不要编造课程数据或结果。

## 可直接复制给 Agent 的提示词

```text
请把 <仓库目录路径> 作为工作目录。先读取该目录下的 AGENTS.md、README.md、main.py 和 utils.py，按 AGENTS.md 的步骤在我电脑上创建隔离环境并运行示例。运行成功后告诉我实际执行的命令、Python 版本和输出位置；如果需要我的课程数据，再明确列出所需 JSON 字段。不要覆盖我的输入文件，也不要把代理录取概率当成真实保证。
```
