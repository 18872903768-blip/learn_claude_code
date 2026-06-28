import ast
import json
import os
import sys
import subprocess
from pathlib import Path
import platform
import anthropic
from anthropic import Anthropic
from dotenv import load_dotenv
"""
s06: Subagent — spawn sub-agents with fresh messages[] for context isolation.
  Parent Agent                           Subagent
  +------------------+                  +------------------+
  | messages=[...]   |                  | messages=[task]  | <-- fresh
  |                  |   dispatch       |                  |
  | tool: task       | ---------------> | own while loop   |
  |   prompt="..."   |                  |   bash/read/...  |
  |                  |   summary only   |   (max 30 turns) |
  | result = "..."   | <--------------- | return last text |
  +------------------+                  +------------------+
        ^                                      |
        |       intermediate results DISCARDED  |
        +--------------------------------------+
  Subagent tools: bash, read, write, edit, glob (NO task — no recursion)
Changes from s05:
  + task tool + spawn_subagent() with fresh messages[]
  + Safety limit: max 30 turns per subagent
  + extract_text() helper
  Subagent cannot spawn sub-subagents (no task tool in sub_tools).
  Main loop unchanged: task auto-dispatches via TOOL_HANDLERS.
Run: python s06_subagent/code.py
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
MODEL=os.getenv("MODEL")
SYSTEM=(
    f"You are a coding agent at {WORKDIR}. "
    f"Ensure all shell commands you execute via 'bash' are fully compatible with {OS_NAME}. "
    "For complex sub-problems, use the task tool to spawn a subagent."
)
# s06: subagent gets its own system prompt — no task, no recursion
SUB_SYSTEM=(
    f"You are a coding agent at {WORKDIR}. "
    f"Ensure all shell commands you execute via 'bash' are fully compatible with {OS_NAME}. "
    "Complete the task you were given, then return a concise summary. "
    "Do not delegate further."
)
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
        lines = safe_path(path).read_text().splitlines()
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
        "properties":{"path":{"type":"string"},"old_content":{"type":"string"},"new_content":{"type":"string"}},
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


rounds_since_todo = 0
# 三.agent harness core is a loop
#agent_loop — same as s04 + nag reminder counter
def agent_loop(messages:list):
    global rounds_since_todo
    while True:
        # s05: nag reminder — inject if model hasn't updated todos for 3 rounds
        if rounds_since_todo>=3 and messages:
            messages.append({"role":"user","content":"<reminder>Update your todos.</reminder>"})
            rounds_since_todo = 0

        # 将接收message发送给agent，并期望回复
        response=client.messages.create(
            model=MODEL,
            messages=messages,
            system=SYSTEM,
            tools=TOOLS,
            max_tokens=8000
        )
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
    print("s06: TodoWrite — plan before execute, nag if you forget")
    print("输入问题，回车发送。输入q 退出。\n")
    #2.上下文
    history=[]
    #3.循环对话
    while True:
        try:
            #用户输入
            query = input("\033[036ms06 >> \033[0m")
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


