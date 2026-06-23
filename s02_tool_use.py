import os
import subprocess
from pathlib import Path
import anthropic
from anthropic import Anthropic
from dotenv import load_dotenv
"""
s02: Tool Use — 在 s01 基础上新增 4 个工具 + 分发映射。

运行: python s02_tool_use/code.py
需要: pip install anthropic python-dotenv + .env 中配置 ANTHROPIC_API_KEY

本文件 = s01 的全部代码 + 以下新增:
  + run_read / run_write / run_edit / run_glob 四个工具实现
  + TOOL_HANDLERS 分发映射（替代 s01 中硬编码的 run_bash 调用）
  + safe_path 路径安全校验

循环本身（agent_loop）与 s01 完全一致。
"""
#一.初始化llm
#导入配置
load_dotenv("config.env",override=True)
#创建llm对话
client = Anthropic(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL")
    )
MODEL=os.getenv("MODEL")
SYSTEM=f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."
WORKDIR = Path.cwd()
#二.初始化工具


# 工具执行
def run_bash(command:str)->str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "错误：危险指令已上锁"
    try:
        r = subprocess.run(command,shell=True,cwd=WORKDIR,
                           capture_output=True,text=True,
                           encoding="utf-8",errors="replace",timeout = 120)
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
def run_edit(path:str,old_content:str,new_content:str)->str:
    try:
        file_path=safe_path(path)
        text = file_path.read_text()
        if old_content not in text :
            return f"{old_content}不在{text}里面！"
        file_path.write_text(text.replace(old_content,new_content,1))
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
        "properties":{"path":{"type":"string"},"content":{"type":"string"}},
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
    }
}
]
#工具分发映射
TOOL_HANDLERS= {
    "bash":run_bash,
    "read_file":run_read,
    "write_file":run_write,
    "edit_file":run_edit,
    "glob":run_glob,
}
# 三.agent harness core is a loop
def agent_loop(messages:list):
    while True:

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
            return
        #寻找并调用工具
        results=[]
        for block in message:
            if block.type=="tool_use":
                #控制台打印出要执行的命令
                print(f"\033[33m$ {block.name}\033[0m")
                #.get() 方法：比直接用 TOOL_HANDLERS[block.name] 更安全。
                # 如果模型生成了一个不存在的工具名（例如 "delete_file"），直接用中括号会触发 KeyError 导致程序崩溃；
                # 而 .get() 会安全地返回 None。
                hander=TOOL_HANDLERS.get(block.name)
                #双星号解包 **（Dictionary Unpacking）：
                #block.input 通常是一个包含参数的字典，例如：{"path": "config.txt", "limit": 10}。
                #加了双星号 ** 后，Python 会把字典中的键值对打散，转换为函数的关键字参数。
                #即：run_read(**{"path": "config.txt", "limit": 10}) 在底层等价于：run_read(path="config.txt", limit=10)
                output=hander(**block.input) if hander else f"未知的：{block.name}！"
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
    print("s02: Tool Use — 在 s01 基础上加了 4 个工具")
    print("输入问题，回车发送。输入q 退出。\n")
    #2.上下文
    history=[]
    #3.循环对话
    while True:
        try:
            #用户输入
            query = input("\033[037ms02 >> \033[0m")
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


