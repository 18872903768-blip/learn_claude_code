import ast
import json
import os
import sys
import subprocess
import time
from pathlib import Path
import platform
import anthropic
from anthropic import Anthropic
from dotenv import load_dotenv
import yaml
"""
s08_context_compact.py - Context Compact

Four-layer compaction pipeline inserted before LLM calls:

    L1: snip_compact      — trim middle messages when count > 50
    L2: micro_compact     — replace old tool_results with placeholders
    L3: tool_result_budget — persist large results to disk
    L4: compact_history   — LLM full summary (1 API call)

    Emergency: reactive_compact — when API still returns prompt_too_long

    ┌─────────────────────────────────────────────────────────────┐
    │  messages[]                                                 │
    │    ↓                                                        │
    │  L3 budget ─→ L1 snip ─→ L2 micro ─→ [token > threshold?]  │
    │                                      ├─ No  → LLM          │
    │                                      └─ Yes → L4 summary   │
    │                                              ↓              │
    │                                          LLM call           │
    │                                    [prompt_too_long?]        │
    │                                      └─ Yes → reactive      │
    └─────────────────────────────────────────────────────────────┘

Core principle: cheap first, expensive last.
Execution order matches CC source: budget → snip → micro → auto.

Builds on s07 (skill loading). Usage:

    python s08_context_compact/code.py
    Needs: pip install anthropic python-dotenv + ANTHROPIC_API_KEY in .env
"""
#一.初始化llm
#导入配置
# 获取当前操作系统名称，例如 "Windows"、"Linux"、"Darwin" (macOS)
OS_NAME = platform.system()
load_dotenv("config.env",override=True)
#创建llm对话
client = Anthropic(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL")
    )
WORKDIR = Path.cwd()
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
TOOL_RESULTS_DIR=WORKDIR / ".task_outputs" / "tool-results"
SKILLS_DIR = WORKDIR / "skills"
MODEL=os.getenv("MODEL")
#现在需要把skills加载到system里面
# SYSTEM=(
#     f"You are a coding agent at {WORKDIR}. "
#     f"Ensure all shell commands you execute via 'bash' are fully compatible with {OS_NAME}. "
#     "For complex sub-problems, use the task tool to spawn a subagent."
# )
# s06: subagent gets its own system prompt — no task, no recursion
# s07: subagent gets its own system prompt — no skill loading, no task
SUB_SYSTEM=(
    f"You are a coding agent at {WORKDIR}. "
    f"Ensure all shell commands you execute via 'bash' are fully compatible with {OS_NAME}. "
    "Complete the task you were given, then return a concise summary. "
    "Do not delegate further."
)
#s07 skill使用分两步：
# 一、初始时将skill列表加载进提示词

#s07 解析skill的信息，将skill的前言与正文分离
def _parse_frontmatter(text:str)->tuple[dict,str]:
    #因为一般的skill文件，前言用----包裹，以此与正文分隔开
    #1.检查是否有----
    if not text.startswith("---"):
        return {},text
    #2.根据----将文件分割为三部分，---前面为空白，----和----之间为前言，----后为正文
    parts =text.split("---",2)
    #3.判断是否分割为三部分了
    if len(parts)<3:
        return {},text
    #4.确认判断三部分之后
    try:
        #安全的将yaml数据解析为python对象
        meta=yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as e:
        meta={}
    return meta,parts[2].strip()
#harness 启动时调用 _scan_skills() 扫描 skills/ 目录，解析每个 SKILL.md 的 YAML frontmatter（name、description），
# 存入 SKILL_REGISTRY 字典。list_skills() 从注册表生成目录，注入 SYSTEM prompt。Agent 每轮都能看到"我有哪些技能可用"，不花额外 API 调用：
SKILL_REGISTRY:dict[str,dict]={}
def _scan_skills():
    #扫描特定目录文件
    if not SKILLS_DIR.exists():
        return
    for d in sorted(SKILLS_DIR.iterdir()):
        #如果不是文件夹
        if not d.is_dir():
            continue
        #是文件夹：
        # 在当前技能文件夹d下，定位名为SKILL.md的文件。
        #因为Claude code的Skills 开放标准就是一个文件夹一个skill.md文件
        manifest = d/"SKILL.md"
        if manifest.exists():
            #获取SKILL.md的内容
            raw = manifest.read_text()
            #获取skill的前言和内容
            meta,body = _parse_frontmatter(raw)
            name = meta.get("name",d.name)
           #如果没有description，用第一行的标题来代替
            desc = meta.get("description",raw.split("\n")[0].lstrip("#").strip())
            SKILL_REGISTRY[name]={"name":name,"description":desc,"content":raw}
#获取skills列表
def list_skills()->str:
    #SKILL_REGISTRY.values()只要值
    return "\n".join(f"- **{s['name']}**:{s['description']}" for s in SKILL_REGISTRY.values())
#构建新的提示词，带skills

def build_system()->str:
    catalog=list_skills()
    return (f"You are a coding agent at {WORKDIR}."
            f"Skills available:\n{catalog}\n"
            "Use load_skill to get full details when needed.")
#启动扫描（导入或直接运行该脚本时都会执行）
#原理：开头不是def的没有缩进的代码，Python 遇到这些代码时会立即执行它们。
_scan_skills()
#现在需要把skills加载到system里面
SYSTEM=build_system()

    #获取文本
#二、通过load_skill[name]调用

#二.初始化工具

#s05只规划，不执行
def normalize_todos(todos):
    #检查传入的todos是否是字符串类型
    if isinstance(todos,str):
        try:
            #将todos反序列化为json格式
            todos = json.loads(todos)
        #捕获 JSON 解析失败的异常
        except json.JSONDecodeError:
            try:
                #大语言模型有时会生成 Python 风格的列表字符串，
                #ast.literal_eval 能够非常安全地解析 Python 的字面量
                todos = ast.literal_eval(todos)
            #捕获 ast.literal_eval() 也无法解析时的异常。
            #SyntaxError：说明字符串本身的括号不匹配或结构严重破损
            #ValueError：说明包含不合法的、无法被安全评估的值。
            except (SyntaxError,ValueError):
                return None,"Error: todos must be a list"
    # 检查传入的todos是否是列表类型
    if not isinstance(todos,list):
        return None,"Error:todos must be a list"
    #
    for i,t in enumerate(todos):
        #检查是否是字典
        if not isinstance(t,dict):
            return None,f"Error:todos{i} must be an object "

        #检查t是否合法
        if "content" not in t or "status" not in t:
            return None,f"Error:todos{i} missing content or status"
        #检查t的状态是否合法
        if t['status'] not in ("pending", "in_progress", "completed"):
            return None,f"Error:todos{i} has invalid status '{t['status']}'"
    return todos,None


# 工具执行
def run_bash(command:str)->str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "错误：危险指令已上锁"
    # 💡 新增：动态识别系统编码。Windows 中文版采用 gbk，Linux/macOS 采用 utf-8
    system_encoding = "gbk" if sys.platform == "win32" else "utf-8"
    try:
        r = subprocess.run(command,shell=True,cwd=WORKDIR,
                           capture_output=True,text=True,
                           encoding=system_encoding,errors="replace",timeout = 120)
        out = (r.stdout+r.stderr).strip()
        return out[:50000] if out else "(没有输出)"
    except subprocess.TimeoutExpired:
        return "错误：超时(120秒)"
    except (FileNotFoundError,OSError) as e:
        return f"错误:{e}"
#四个新工具
#路径安全检查
def safe_path(p:str)->Path:
    #拼接两个路径：
    path = (WORKDIR/p).resolve()
    #检查解析后的路径是否在WORKDIR目录或子目录下：
    if not path.is_relative_to(WORKDIR):
        #抛出逃逸异常
        raise ValueError(f"Path escapes workspace: {p}")
    return path

#安全获取本地文件内容
#limit表示可选参数：int或None

def run_read(path:str,limit:int|None = None)->str:
    try:
        #read_text()：pathlib 的方法，将整个文件内容以字符串形式一次性读取到内存中
        #.splitlines()：将读取到的完整文本按行切分，返回一个字符串列表 list[str]，同时自动去除了每行末尾的换行符
        lines = safe_path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        #行数截断
        if limit and limit <len(lines):
            lines = lines[:limit]+[f"...({len(lines)-limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"

#向本地文件安全写入内容
def run_write(path:str,content:str)-> str:
    try:
        #安全路径解析
        file_path = safe_path(path)
        #file_path.parent：获取文件所在的父目录。例如，如果 path 是 logs/2026/info.log，其父目录就是 logs/2026。
        #parents=True：递归创建目录。如果 logs 和 2026 目录都不存在，它们都会被创建（相当于 Linux 中的 mkdir -p）。
        #exist_ok=True：如果目录已经存在，不会报错。这保证了多次写入同一个文件夹时程序能顺利运行。
        file_path.parent.mkdir(parents=True,exist_ok=True)
        #将content写入目标文件
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error:{e}"

#替换指定内容
def run_edit(path:str,old_text:str,new_text:str)->str:
    try:
        file_path=safe_path(path)
        text = file_path.read_text()
        if old_text not in text :
            return f"{old_text}不在{text}里面！"
        file_path.write_text(text.replace(old_text,new_text,1))
        return f"成功修改了{path}的内容！"
    except Exception as e:
        return f"出错了：{e}！"

#搜索指定格式的文件
def run_glob(pattern:str)->str:
    #glob 专门用于支持类似 Unix 终端的通配符路径匹配
    import glob  as g
    try:
        results = []
        #开启 recursive=True 允许智能体使用 **/*.py 搜索子目录
        for match in g.glob(pattern,root_dir=WORKDIR,recursive=True):
            #检查解析后的真实绝对路径，是否依然处于 WORKDIR 目录（或其子目录）内部
            if (WORKDIR/match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "没有匹配的文件！"
    except Exception as e:
        return f"出错了：{e}!"
#s05 to do write做之前规划
CURRENT_TODOS:list[dict] = []
#查看当前任务列表的函数？
def run_todo_write(todos:list)->str:
    #global声明后面的变量是全局变量而不是局部变量
    global CURRENT_TODOS
    todos,error=normalize_todos(todos)
    if error:
        return error
    CURRENT_TODOS = todos
    lines=["\n## Current Tasks"]
    for t in CURRENT_TODOS:
        icon ={"pending": " ", "in_progress": "\033[36m▸\033[0m", "completed": "\033[32m✓\033[0m"}[t["status"]]
        lines.append(f"[{icon}]{t['content']}")
    print("\n".join(lines))
    return f"Updated {len(CURRENT_TODOS)} tasks"

# ═══════════════════════════════════════════════════════════
#  NEW in s08: Four-Layer Compaction Pipeline
# ═══════════════════════════════════════════════════════════
CONTENT_LIMIT=70000
KEEP_RECENT=3
PERSIST_THERSHOLD = 15000

#获取消息长度
def estimate_size(msgs):return len(str(msgs))
#判断块是否为列表类型：区分block是text还是tool_use
def _block_type(block):
    return block.get("type") if isinstance(block,dict) else getattr(block,"type",None)
#判断消息类型是否为tool_use
def _message_has_tool_use(msg):
    if msg.get("role")!="assistant":
        return False
    content = msg.get("content")
    if not isinstance(content,list):
        return False
    return any(_block_type(block)=="tool_use" for block in content)
#判断消息类型是否为tool_result
def _is_tool_result_message(msg):
    if msg.get("role")!="user":
        return False
    content = msg.get("content")
    if not isinstance(content,list):
        return False
    return any(_block_type(block)=="tool_result" for block in content)
# L1: snipCompact — trim middle messages
def snip_compact(messages,max_messages=50):
    #messages<max_messages不用切割
    if len(messages)<max_messages:
        return messages
    #找到切割的头尾
    head_end=3
    tail_start=len(messages)-max_messages+3

    #判断是否有孤立的tool_use
    if head_end>0 and _message_has_tool_use(messages[head_end-1]):
        while head_end<len(messages) and _is_tool_result_message(messages[head_end]):
            head_end=head_end+1
    #判断是否有孤立的tool_result
    if (tail_start>0 and tail_start<len(messages) and _is_tool_result_message(messages[tail_start]) and _message_has_tool_use(messages[tail_start-1])):
        tail_start-=1
    #判断head<start
    if head_end<tail_start:
        #拼接切割结果
        snipped = tail_start - head_end
        messages=messages[:head_end]+[{"role":"user","content":f"[snipped {snipped} messages]"}]+messages[tail_start:]
        #返回切割结果
    return messages
# L2: microCompact — old result placeholders、
#只保留最新的三条工具结果消息
def collect_tool_results(messages):
    #存放结果，存储消息索引、块索引、消息块
    blocks=[]
    for mi,msg in enumerate(messages):
        #过滤不是tool_result的部分
        if msg.get("role")!="user" or not isinstance(msg.get("content"),list):continue
        for bi,block in enumerate(msg["content"]):
            if isinstance(block,dict) and block.get("type")=="tool_result":
                blocks.append((mi,bi,block))

    return blocks

def micro_compact(messages):
    tool_results=collect_tool_results(messages)
    if len(tool_results)<=KEEP_RECENT:
        return messages
    for _,_,block in tool_results[:-KEEP_RECENT]:
        if len(block.get("content"))>120:
            block["content"]="[Earlier tool result compacted. Re-run if needed.]"
            print("[micro_compact]")
    return messages
# L3: toolResultBudget — persist large results to disk
#判断输出内容是否大于阈值，是则将其写入磁盘
def persist_large_output(tool_use_id,output):
    #文件内容小于阈值，则不管
    if len(output)<PERSIST_THERSHOLD:
        return output
    # 文件内容大于阈值，创建一个文件，
    TOOL_RESULTS_DIR.mkdir(parents=True,exist_ok=True)
    #将内容写进去，
    #这一步只是创建一个path对象，磁盘上没有内容
    path=TOOL_RESULTS_DIR/f"{tool_use_id}.txt"
    #所以第一次检查磁盘是否存在内容，得出结果是false，故而会触发写入，给磁盘真正创建文件
    if not path.exists():
        path.write_text(output,encoding="utf-8")
    # 并返回文件路径和部分内容
    return f"<persisted-output>\nFull output: {path}\npreview:{output[:2000]}</persisted-output>"
#对传入的聊天消息列表（messages）进行分析和瘦身。
def tool_result_budget(messages,max_bytes=40000):
    #1.安全获得最新的一次消息
    last=messages[-1] if messages else None
    #2.判断这条消息是不是我们要的
    #我们要的是：user的
    #tool_result的
    if not last or last.get("role")!="user" or not isinstance(last.get("content"),list):
        return messages
    #3.现在找到了我们要的最后一条消息，把里面type==tool_result的block找出来
    blocks=[(i,b) for i,b in enumerate(last["content"]) if isinstance(b,dict) and b.get("type")=="tool_result"]
    #4.现在所有的工具块和它的索引都被我们找到了
    # 计算当前所有工具结果内容（content）的字符/字节总大小
    total=sum(len(str(b.get("content",""))) for _,b in blocks)
    #小于阈值就返回
    if total<=max_bytes:
        return messages
    #大于阈值就需要压缩
    #4.按content从大到小排序这些块
    ranked=sorted(blocks,key=lambda p:len(str(p[1].get("content"))),reverse=True)
    #5.循环处理大块的内容
    for _,b in ranked:
        # 如果总大小已经达标，提前退出循环
        if total<=max_bytes:
            break
        #小于PERSIST_THERSHOLD的就不用压缩
        content = str(b.get("content",""))
        if len(content)<PERSIST_THERSHOLD:
            continue
        # 大于于PERSIST_THERSHOLD的就要压缩
        tid = b.get("tool_use_id","unknown")
        # 核心步骤：直接在原地修改 block 字典中的 "content"（因为 Python 字典是引用传递，修改 block 会直接同步到原 messages 列表中）
        # 将完整内容持久化，并在消息中替换为“路径 + 预览”的形式
        b["content"]=persist_large_output(tid,content)
        # 重新计算修改后的所有工具结果总大小，用于下一轮循环的条件判定
        total=sum(len(str(b.get("content",""))) for _,b in blocks)
    # 8. 返回瘦身完成（部分过大结果已被保存至本地并替换为预览）的 messages
    return messages
# L4: autoCompact — LLM full summary
#将全部历史消息写入磁盘
def write_transcript(messages):
    #1.创建目录文件夹,若不存在则递归创建
    TRANSCRIPT_DIR.mkdir(parents=True,exist_ok=True)
    #2.创建文件对象
    path=TRANSCRIPT_DIR/f"transcript_{int(time.time())}.jsonl"
    #内容写入磁盘
    # 3. 以写入模式（"w"）打开该路径对应的文件
    with path.open("w") as f:
    # 4. 遍历每一条消息，并将其序列化为 JSON 字符串写入文件，每条消息占一行
    # default=str 的作用是：如果消息中包含无法被直接转为 JSON 的对象（如 datetime），则将其转换为字符串，避免报错
        for msg in messages:
            f.write(json.dumps(msg,default=str)+"\n")
    return path
#生成全量摘要
def summarize_history(messages):
    # 1. 将完整的消息列表序列化为 JSON 字符串，并通过切片 [:80000] 限制最大长度为 80,000 个字符
    #*****这里为什么要将消息序列化为json字符串呢？*****
    #*****1.现在的messages是python对象，llm识别不了
    # *****2.llm经常处理json格式数据，对它胃口，转成str也行，没有json规范，
    # 且一些python特有的数据结构转成str会有问题，比如：datetime.datetime(...)
    #    这样做可以防止历史记录本身过于庞大，超出总结模型本身的单次输入限制
    conversation=json.dumps(messages,default=str)[:8000]
    # 2. 构建提示词（Prompt），指导模型如何进行压缩，并明确要求保留以下 5 类核心信息：
    #    - current goal (当前目标)
    #    - key findings/decisions (重要发现/决定)
    #    - files read/changed (读取或修改过的文件)
    #    - remaining work (剩余待办工作)
    #    - user constraints (用户的限制条件)

    prompt=("Summarize this coding-agent conversation so work can continue.\n"
              "Preserve: 1. current goal, 2. key findings/decisions, 3. files read/changed, "
              "4. remaining work, 5. user constraints.\nBe compact but concrete.\n\n" + conversation)
    # 3. 调用大模型 API（从格式看采用的是 Anthropic Claude 的 Messages API）
    response = client.messages.create(
        model=MODEL,
        messages=[{"role":"user","content":prompt}],
        max_tokens=2000
    )
    # 4. 解析模型的返回结果。遍历返回的每一个内容块（block），
    #    如果是文本类型（"text"），则获取其 text 属性，并用换行符 "\n" 拼接。
    #    如果最终生成的总结为空，则返回默认占位符 "(empty summary)"
    return "\n".join(
        getattr(block,"text","")
        for block in response.content
        if getattr(block,"type",None)=="text"
    ).strip() or "(empty summary)"
#历史消息写入加全量摘要替换
def compact_history(messages):
    # 1. 调用 write_transcript 函数将当前所有的完整对话保存到本地，得到保存路径
    transcript_path = write_transcript(messages)

    # 2. 在控制台打印提示，告知开发者原始完整记录已成功备份，便于日后排查
    print(f"[transcript saved: {transcript_path}]")

    # 3. 调用 summarize_history 函数，让大模型生成当前对话的精简总结
    summary = summarize_history(messages)
    # 4. 【关键步骤】丢弃传入的所有 messages 历史，
    #    只返回一个全新的、只有一个元素的列表，模拟一次带有 "[Compacted]" 标记的新输入。
    #    由于历史被彻底替换为了这一个 summary，上下文占用的 Token 量瞬间从几万甚至十几万降低到了几百个。
    return [{"role":"user","content":f"[Compacted]\n\n{summary}"}]
# Emergency: reactiveCompact — on API error
#如果全量摘要也不能使得上下文小于阈值，就得采取非常措施
#
def reactive_compact(messages):
    # 1. 调用 write_transcript 函数将当前所有的完整对话保存到本地，得到保存路径
    transcript_path = write_transcript(messages)
    tail_start = max(0, len(messages) - 5)
    if (tail_start > 0 and tail_start < len(messages)
            and _is_tool_result_message(messages[tail_start])
            and _message_has_tool_use(messages[tail_start - 1])):
        tail_start -= 1
    # 2. 调用 summarize_history 函数，让大模型生成当前对话的精简总结
    summary = summarize_history(messages[:tail_start])
    return [{"role": "user", "content": f"[Reactive compact]\n\n{summary}"}, *messages[tail_start:]]

#1.找到所有的工具，封装成消息索引，
# ═══════════════════════════════════════════════════════════
#  NEW in s07: load_skill — runtime full content loading
# ═══════════════════════════════════════════════════════════
def load_skill(name:str)->str:
    skill = SKILL_REGISTRY.get(name)
    if not skill:
        return f"Skill not found: {name}"
    return skill["content"]
# 工具定义
TOOLS=[{
    # 工具名称
    "name":"bash",
    # 工具描述
    "description":"Run a shell command.",
    #输入参数的约束条件
    "input_schema":{
        # 输入的参数必须是键值对
        "type":"object",
        #对象中有一个名为“command”的属性，其类型必须是字符串
        "properties":{"command":{"type":"string"}},
        #command是必填参数
        "required":["command"]
    }

},{
    "name":"read_file",
    "description":"Read file contents.",
    "input_schema":{
        "type":"object",
        "properties":{"path":{"type":"string"},"limit":{"type":"integer"}},
        "required":["path"]
    }
},{
    "name":"write_file",
    "description":"Write content to a file.",
    "input_schema":{
        "type":"object",
        "properties":{"path":{"type":"string"},"content":{"type":"string"}},
        "required":["path","content"]
    }
},{
    "name":"edit_file",
    "description":"Replace exact text in a file once.",
    "input_schema":{
        "type":"object",
        "properties":{"path":{"type":"string"},"old_text":{"type":"string"},"new_text":{"type":"string"}},
        "required":["path","old_content","new_content"]
    }
},{
    "name":"glob",
    "description":"Find files matching a glob pattern.",
    "input_schema":{
        "type":"object",
        "properties":{"pattern":{"type":"string"}},
        "required":["pattern"]
    },
},
    {
        "name":"todo_write",
        "description":"Create and manage a task list ...",
        "input_schema":{
            "type":"object",
            "properties":{
                "todos":{
                    "type":"array",
                    "item":{
                        "type":"object",
                        "properties":{
                            "content":{"type":"string"},
                            "status":{"type":"string","enum": ["pending", "in_progress", "completed"]},
                        },
                    },
                },

            },
        },
    },{
        "name":"load_skill",
        "description":"Load the full content of a skill by name.",
        "input_schema":{
            "type":"object",
            "properties":{
                "name":{"type":"string"}
            },
            "required":["name"]

        }
    },
    {
        "name":"compact",
        "description":"Summarize earlier conversation to free context space.",
        "input_schema":{
            "type":"object",
            "properties":{"focus": {"type": "string"}},
        }
    },
]

#工具分发映射
TOOL_HANDLERS= {
    "bash":run_bash,
    "read_file":run_read,
    "write_file":run_write,
    "edit_file":run_edit,
    "glob":run_glob,
    "todo_write":run_todo_write,
    "load_skill":load_skill,
}

# ═══════════════════════════════════════════════════════════
#  NEW in s06: Subagent — fresh messages[], summary only
# ═══════════════════════════════════════════════════════════
# NO "task" tool — prevent recursive spawning
SUB_TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in a file once.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files matching a glob pattern.",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
]
SUB_HANDLERS={
    "bash":run_bash,
    "read_file":run_read,
    "write_file":run_write,
    "edit_file":run_edit,
    "glob":run_glob,
}
#安全获取子智能体的工作返回摘要
def extract_text(content)->str:
    #不是list类型就转为str返回
    if not isinstance(content,list):
        return str(content)
    #是list类型，提取其中的text返回
    return "\n".join(getattr(b,"text","") for b in content if getattr(b,"type",None)=="text")
#创建子智能体并使用
def spawn_subagent(description:str)->str:
    """Spawn a subagent with fresh messages[], return summary only."""
    print(f"\n\033[35m [subagent spawned]\033[0m")
    #定义上下文并初始化
    messages=[{"role":"user","content":description}]
    #设置30轮对话为子智能体的极限
    for _ in range(30):
        #创建大模型对话
        response = client.messages.create(
            system=SUB_SYSTEM,
            model=MODEL,
            messages=messages,
            tools=SUB_TOOLS,
            max_tokens=8000
        )
        #将对话存到上下文中
        message=response.content
        messages.append({"role":"assistant","content":message})
        #如果结束的原因不是使用工具，代表已经完成了任务，可以退出了
        if response.stop_reason!="tool_use":
            break
        #用来记录工具调用结果
        results=[]
        #否则进入工具调用
        for block in message:
            #获取工具列表
            if block.type=="tool_use":
                #工具执行前的钩子-工具权限检查等...
                blocked=trigger_hooks("PreToolUse",block)
                # 工具权限检查未通过，不执行工具函数，记录结果
                if blocked:
                    results.append({"type":"tool_result","tool_use_id":block.id,"content":str(blocked)})
                    continue
                # 工具权限检查通过，执行工具函数
                hander = SUB_HANDLERS.get(block.name)
                output=hander(**block.input) if hander else f"Unknown: {block.name}"
                trigger_hooks("PostToolUse",block,output)
                print(f"  \033[36m [sub]{block.name}: {str(output)[:100]}\033[0m")
                results.append({"type":"tool_result","tool_use_id":block.id,"content":output})
        messages.append({"role":"user","content":results})
    # Issue 5: fallback if safety limit hit during tool_use
    #获取最后一次子智能体的文本，用来返回摘要
    result = extract_text(messages[-1]["content"])
    #如果为空，说明最后一次是调用工具，往上追溯对话
    if not result:
        for m in reversed(messages):
            #找到大模型的回答
            if m["role"]=="assistant":
                result=extract_text(m["content"])
                if result:
                    break

        if not result:
            result = "Subagent stopped after 30 turns without final answer."
    print(f"\033[35m[Subagent done]\033[0m")
    return result


        #记录执行结果
#给主agent添加task作为工具来调用子智能体
TOOLS.append({
    "name":"task",
    "description":"Launch a subagent to handle a complex subtask. Returns only the final conclusion.",
    "input_schema":{
        "type":"object",
         "properties":{"description":{"type":"string"}},
         "required":["description"]
    }

})
TOOL_HANDLERS["task"] = spawn_subagent
#day03权限验证部分
#大门1-硬拒绝表-任何情况都禁止
DENY_LIST =["rm -rf /","sudo","shutdown","reboot","mkfs","dd if=","> /dev/sda"]
DESTRUCTIVE=["rm ","> /etc/","chmod 777","del ","rd "]
#拒绝函数
def deny_list(command:str)->str|None:
    for pattern in DENY_LIST:
        if pattern in command:
            return f"{pattern}指令是非法的！"
    return None
#s04.1定义钩子集
HOOKS={
    "UserPromptSubmit":[],
    "PreToolUse":[],
    "PostToolUse":[],
    "Stop":[],
}
#s04.2定义钩子注册函数
#这里callback不用str格式是因为，python里面函数作为参数是对象形式而不是字符串
def register_hook(event :str,callback):
    HOOKS[event].append(callback)

#s04.3定义钩子挂载函数
#*args代表参数打包与解包运算符，将多个参数打包为一个，统一参数形式，可以使多个触发钩子函数写成一种形式
def trigger_hooks(event:str,*args)->str|None:
    for callback in HOOKS[event]:
        result = callback(*args)
        if result:
            return result
    #返回None代表没通过hook，不能执行后续操作
    return None
#s04.4定义具体的hook函数
#1."UserPromptSubmit"，用户输入之后，调用大模型之前
def content_inject_hook(query:str)->str|None:
    '''Inject current working directory info into every prompt.'''
    print(f"\033[90m [HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")
    return None
#hook定义完就注册hook
register_hook("UserPromptSubmit",content_inject_hook)
#2.权限检查hook，工具调用之前
def permission_hook(block)->str|None:

    if block.name == "bash":
        # 1.检查是否有危险指令
        for command in DENY_LIST:
            if command in block.input.get("command",""):
                return f"Permission denied by deny list:{command}"
        # 2.检查是否有风险指令
        for kw in DESTRUCTIVE:
            if kw in block.input.get("command",""):
                print(f"\n\033[33m⚠  Potentially destructive command\033[0m")
                print(f"   Tool: {block.name}({block.input})")
                choice = input("   Allow? [y/N] ").strip().lower()
                if choice not in ("y", "yes"):
                    return "Permission denied by user"


    if block.name in ["edit_file","write_file"]:
        path = block.input.get("path","")
        if not (WORKDIR/path).resolve().is_relative_to(WORKDIR):
            choice = input("Allow [y/N]").strip().lower()
            if choice not in ["y","yes"]:
                return f"Permission denied by user"
    return None
# PreToolUse: 日志
def log_hook(block):
    print(f"[HOOK]:{block.name}(...)")
register_hook("PreToolUse",permission_hook)
register_hook("PreToolUse",log_hook)
#3.调用工具之后  大文件提醒
def large_output_hook(block,output):
    if len(str(output))>100000:
        print(f"[HOOK] ⚠ Large output from {block.name}")
register_hook("PostToolUse",large_output_hook)
#4.循环即将退出时触发,打印收尾统计：
def summary_hook(messages:list)->str|None:
    tool_count=sum(1 for m in messages
                   for b in(m.get("content") if isinstance(m.get("content"),list) else [])
                   if isinstance(b,dict) and b.get("type")=="tool_result")
    print(f" \033[90m[HOOK] Stop: session used {tool_count} tool calls\033[0m")
    return None
register_hook("Stop",summary_hook)
# ═══════════════════════════════════════════════════════════
#  agent_loop — s08 core: run compaction pipeline before LLM
# ═══════════════════════════════════════════════════════════
MAX_REACTIVE_RETRIES = 1  # retry limit for reactive compact

rounds_since_todo = 0
# 三.agent harness core is a loop
#agent_loop — same as s04 + nag reminder counter
def agent_loop(messages:list):
    reactive_retries = 0
    global rounds_since_todo
    while True:
        # s08 change: three preprocessors (0 API calls, cheap first)
        # Order matches CC source: budget → snip → micro
        # L3: persist large results first
        messages[:]=tool_result_budget(messages)
        # L1: trim middle
        messages[:]=snip_compact(messages)
        #L2: old result placeholders
        messages[:]=micro_compact(messages)

        # s08 change: tokens still over threshold → LLM summary (1 API call)
        if estimate_size(messages)>CONTENT_LIMIT:
            print("[auto compact]")
            messages[:]=compact_history(messages)
        try:
            # 将接收message发送给agent，并期望回复
            response = client.messages.create(
                model=MODEL,
                messages=messages,
                system=SYSTEM,
                tools=TOOLS,
                max_tokens=8000
            )
            # reset on successful API call
            reactive_retries=0
        except Exception as e:
            if ("prompt_too_long" in str(e).lower() or "too many tokens" in str(e).lower()) and reactive_retries<MAX_REACTIVE_RETRIES:
                print("[reactive compact]")
                # 使用切片赋值（Slice Assignment）。这样可以在不改变messages
                # 列表内存地址的前提下，清空并用压缩后的新数据替换原列表内容。这在多处引用同一个列表对象时非常有用。
                messages[:] = reactive_compact(messages)
                reactive_retries += 1

                continue
            # 将异常原样向上抛出，交由上层调用者处理。
            raise



        # s05: nag reminder — inject if model hasn't updated todos for 3 rounds
        if rounds_since_todo>=3 and messages:
            messages.append({"role":"user","content":"<reminder>Update your todos.</reminder>"})
            rounds_since_todo = 0


        #将大模型的回答加入到消息里
        message=response.content
        messages.append({"role":"assistant","content":message})
        #判断是否调用工具,如果不调用工具而结束，说明回答结束了
        if response.stop_reason!="tool_use":
            force=trigger_hooks("Stop",messages)
            if force:
                messages.append({"role":"user","content":force})
                continue
            return
        #寻找并调用工具
        rounds_since_todo+=1
        results=[]
        for block in message:
            if block.type!="tool_use":
                continue
            # 控制台打印出要执行的命令
            print(f"\033[33m$ {block.name}\033[0m")
            # s08: compact tool triggers compact_history, not a no-op string
            if block.name=="compact":
                messages[:] = compact_history(messages)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": "[Compacted. Conversation history has been summarized.]"})
                messages.append({"role": "user", "content": results})
                break  # end current turn, start fresh with compacted context

                #控制台打印出要执行的命令
            print(f"\033[33m$ {block.name}\033[0m")
            blocked=trigger_hooks("PreToolUse",block)
            if blocked:
                results.append({
                    "type":"tool_result",
                    "content":str(blocked),
                    "tool_use_id":block.id

                })
                continue

                #.get() 方法：比直接用 TOOL_HANDLERS[block.name] 更安全。
                # 如果模型生成了一个不存在的工具名（例如 "delete_file"），直接用中括号会触发 KeyError 导致程序崩溃；
                # 而 .get() 会安全地返回 None。
            hander=TOOL_HANDLERS.get(block.name)
                #双星号解包 **（Dictionary Unpacking）：
                #block.input 通常是一个包含参数的字典，例如：{"path": "config.txt", "limit": 10}。
                #加了双星号 ** 后，Python 会把字典中的键值对打散，转换为函数的关键字参数。
                #即：run_read(**{"path": "config.txt", "limit": 10}) 在底层等价于：run_read(path="config.txt", limit=10)
            output=hander(**block.input) if hander else f"未知的：{block.name}！"
            trigger_hooks("PostToolUse",block,output)
            # s05: reset nag counter when todo_write is called
            if block.name=="todo_write":
                rounds_since_todo=0
            print(output[:200])
            results.append({
                    "type":"tool_result",
                    "content":output,
                    "tool_use_id":block.id

                })

        messages.append({"role":"user","content":results})


if __name__ == '__main__':
    #四.搭建服务端
    #1.欢迎语
    print("s08: Context Compact — four-layer compaction pipeline")
    print("输入问题，回车发送。输入q 退出。\n")
    #2.上下文
    history=[]
    #3.循环对话
    while True:
        try:
            #用户输入
            query = input("\033[036ms07 >> \033[0m")
            trigger_hooks("UserPromptSubmit",query)
        except (EOFError,KeyboardInterrupt):
            break
        #退出出口
        if query.strip().lower() in ("q","exit",""):
            print("我走啦，再见！")
            break
        #用户输入加入上下文
        history.append({"role":"user","content":query})
        #上下文发给llm
        agent_loop(history)
        #提取llm最新的回答
        response_content = history[-1]["content"]
        #校验回答格式
        if isinstance(response_content,list):
            #提取有效回答
            for block in response_content:
                if getattr(block,"type",None)=="text":
                    print(block.text)
        print()


