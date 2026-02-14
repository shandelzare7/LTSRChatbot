"""
EmotionalChatBot V5.0 图形窗口
- 聊天区：仅显示「你:」「Bot:」对话，无其他信息
- 日志区：整个 graph 运行过程（detection、reasoner、stage 等）的 log
"""
import asyncio
import queue
import sys
import threading
from datetime import datetime
from pathlib import Path

_root_dir = Path(__file__).resolve().parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))
try:
    from utils.env_loader import load_project_env
    load_project_env(_root_dir)
except Exception:
    pass

import tkinter as tk
from tkinter import font as tkfont

try:
    from main import DEFAULT_USER_ID, DEFAULT_BOT_ID, _make_initial_state
except Exception:
    DEFAULT_USER_ID = "local_user_5128d1c1"
    DEFAULT_BOT_ID = "4d803b5a-cb30-4d14-89eb-88d259564610"
    from main import _make_initial_state  # 再试一次

from langchain_core.messages import HumanMessage
from app.graph import build_graph
from app.state import AgentState


class QueueWriter:
    """把 print 输出写入队列，供主线程显示到 log 区。"""
    def __init__(self, log_queue: queue.Queue, original_stdout):
        self._q = log_queue
        self._out = original_stdout
        self._buf = ""

    def write(self, s: str):
        if s:
            self._buf += s
            while "\n" in self._buf or "\r" in self._buf:
                line, self._buf = self._buf.split("\n", 1) if "\n" in self._buf else self._buf.split("\r", 1)
                line = line.strip()
                if line:
                    self._q.put(line)
        if self._out:
            self._out.write(s)

    def flush(self):
        if self._buf.strip():
            self._q.put(self._buf.strip())
            self._buf = ""
        if self._out:
            self._out.flush()


def run_one_turn(
    user_text: str,
    user_id: str,
    bot_id: str,
    result_queue: queue.Queue,
):
    """在后台线程中跑一轮图（会临时重定向 stdout 到 log_queue）。"""
    async def _invoke():
        app = build_graph()
        state = _make_initial_state(user_id, bot_id)
        now = datetime.now().isoformat()
        state["messages"] = [HumanMessage(content=user_text, additional_kwargs={"timestamp": now})]
        state["current_time"] = now
        try:
            result = await app.ainvoke(state, config={"recursion_limit": 50})
            result_queue.put(("ok", result))
        except Exception as e:
            result_queue.put(("err", str(e)))

    asyncio.run(_invoke())


def run_gui(user_id: str = DEFAULT_USER_ID, bot_id: str = DEFAULT_BOT_ID):
    log_queue = queue.Queue()
    result_queue = queue.Queue()
    real_stdout = sys.stdout

    root_win = tk.Tk()
    root_win.title("EmotionalChatBot V5.0")
    root_win.geometry("880x620")
    root_win.minsize(400, 400)
    root_win.configure(bg="#f2f2f2")

    # 打印 Tcl/Tk 版本；macOS 上 Tcl 8.5 或异常构建会导致控件不显示
    try:
        tcl_ver = root_win.tk.call("info", "patchlevel")
        if real_stdout:
            print(f"[GUI] Tcl/Tk version: {tcl_ver}", file=real_stdout, flush=True)
        if str(tcl_ver).startswith("8.5") and real_stdout:
            print("[GUI] 检测到 Tcl 8.5，macOS 上可能导致界面空白。建议：用 python.org 的 Python 3.10+，或先 brew install tcl-tk 再重装 Python。", file=real_stdout, flush=True)
    except Exception:
        pass

    # macOS 上部分 Tk 构建下 grid/pack 不分配空间，只有 Button 等能显示。
    # 改用 place() 为每块区域指定像素位置和尺寸，强制分配并绘制。
    pad = 8
    w = 880 - pad * 2
    h_chat = 220
    h_log = 280
    h_input = 48
    y_chat = pad
    y_log = y_chat + h_chat + pad
    y_input = y_log + h_log + pad

    # 对话区（像素定位）
    chat_frame = tk.LabelFrame(root_win, text="对话", padx=4, pady=4, bg="#f2f2f2", fg="#111")
    chat_frame.place(x=pad, y=y_chat, width=w, height=h_chat)
    chat_text = tk.Text(
        chat_frame,
        wrap=tk.WORD,
        state=tk.DISABLED,
        font=("", 11),
        bg="#ffffff",
        fg="#111111",
        insertbackground="#111111",
        bd=1,
        relief="solid",
        highlightthickness=1,
        highlightbackground="#c8c8c8",
    )
    chat_sb = tk.Scrollbar(chat_frame, command=chat_text.yview)
    chat_text.configure(yscrollcommand=chat_sb.set)
    sb_w = 18
    chat_text.place(x=0, y=0, width=w - sb_w, height=h_chat - 36)
    chat_sb.place(x=w - sb_w, y=0, width=sb_w, height=h_chat - 36)

    # Graph 过程日志（像素定位）
    log_frame = tk.LabelFrame(root_win, text="Graph 过程日志", padx=4, pady=4, bg="#f2f2f2", fg="#111")
    log_frame.place(x=pad, y=y_log, width=w, height=h_log)
    log_text = tk.Text(
        log_frame,
        wrap=tk.WORD,
        state=tk.DISABLED,
        font=("", 10),
        bg="#ffffff",
        fg="#111111",
        insertbackground="#111111",
        bd=1,
        relief="solid",
        highlightthickness=1,
        highlightbackground="#c8c8c8",
    )
    log_sb = tk.Scrollbar(log_frame, command=log_text.yview)
    log_text.configure(yscrollcommand=log_sb.set)
    log_text.place(x=0, y=0, width=w - sb_w, height=h_log - 36)
    log_sb.place(x=w - sb_w, y=0, width=sb_w, height=h_log - 36)

    # 输入行（像素定位，保证输入框和按钮可见）
    input_frame = tk.Frame(root_win, bg="#f2f2f2")
    input_frame.place(x=pad, y=y_input, width=w, height=h_input)
    entry = tk.Entry(
        input_frame,
        font=("", 14),
        bg="#ffffff",
        fg="#111111",
        insertbackground="#111111",
        bd=1,
        relief="solid",
        highlightthickness=1,
        highlightbackground="#7a7a7a",
    )
    entry.place(x=0, y=8, width=w - 90, height=32)
    send_btn = tk.Button(input_frame, text="发送", bd=1, relief="ridge")
    send_btn.place(x=w - 82, y=8, width=74, height=32)

    def append_chat(line: str):
        chat_text.configure(state=tk.NORMAL)
        chat_text.insert(tk.END, line + "\n")
        chat_text.see(tk.END)
        chat_text.configure(state=tk.DISABLED)

    def append_log(line: str):
        log_text.configure(state=tk.NORMAL)
        log_text.insert(tk.END, line + "\n")
        log_text.see(tk.END)
        log_text.configure(state=tk.DISABLED)

    def drain_log():
        try:
            while True:
                line = log_queue.get_nowait()
                append_log(line)
        except queue.Empty:
            pass

    def on_send():
        text = entry.get().strip()
        if not text:
            return
        entry.delete(0, tk.END)
        append_chat("你: " + text)
        send_btn.configure(state=tk.DISABLED)

        # 重定向 stdout，在后台线程跑图
        writer = QueueWriter(log_queue, None)
        sys.stdout = writer

        def thread_target():
            run_one_turn(text, user_id, bot_id, result_queue)
            sys.stdout = real_stdout
            root_win.after(0, on_turn_done)

        threading.Thread(target=thread_target, daemon=True).start()

        def poll():
            drain_log()
            root_win.after(80, poll)

        poll()

    def on_turn_done():
        drain_log()
        try:
            status, payload = result_queue.get_nowait()
        except queue.Empty:
            send_btn.configure(state=tk.NORMAL)
            return
        if status == "ok":
            result = payload
            reply = result.get("final_response") or ""
            if not reply and result.get("final_segments"):
                reply = " ".join(result["final_segments"])
            if not reply:
                reply = result.get("draft_response") or "（无回复）"
            append_chat("Bot: " + reply)
        else:
            append_chat("Bot: [出错] " + str(payload))
        send_btn.configure(state=tk.NORMAL)

    send_btn.configure(command=on_send)
    entry.focus_set()

    def on_return(_):
        on_send()
    entry.bind("<Return>", on_return)

    append_chat(f"已连接 (user_id={user_id}, bot_id={bot_id})。输入内容后按发送或回车。")
    root_win.update_idletasks()
    root_win.update()
    root_win.mainloop()


if __name__ == "__main__":
    import os
    user_id = os.environ.get("USER_ID", DEFAULT_USER_ID)
    bot_id = os.environ.get("BOT_ID", DEFAULT_BOT_ID)
    run_gui(user_id=user_id, bot_id=bot_id)
