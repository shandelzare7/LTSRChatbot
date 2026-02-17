"""
Memory Retriever 节点：根据 reasoner 输出的 search_spec 执行检索。
检索严格受 search_spec 控制，mode 只控制检索预算，不允许 search 自己发散。
"""
from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Callable, Dict, List

from app.state import AgentState
from utils.tracing import trace_if_enabled


def _run_async(coro):
    """在同步节点里执行 async DB 调用。若运行在已有 event loop 中，请改用异步入口。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("Detected running event loop; please use an async graph entry (ainvoke) for DB operations.")


def create_memory_retriever_node(memory_service: Any) -> Callable[[AgentState], dict]:
    """
    创建 Memory Retriever 节点。
    根据 reasoner 输出的 search_spec 执行检索，严格受 search_spec 控制。
    """
    
    @trace_if_enabled(
        name="Memory/Retriever",
        run_type="chain",
        tags=["node", "memory", "retrieval"],
        metadata={"state_outputs": ["retrieved_memories", "retrieval_ok"]},
    )
    def memory_retriever_node(state: AgentState) -> Dict[str, Any]:
        """
        根据 search_spec 执行检索。
        
        规则：
        - mute_mode: 不检索（直接 no-op）
        - cold_mode: 如果 search_spec.enabled 仍为 true，则 top_k 降到 2~3；否则不检索
        - normal_mode: 按 search_spec 执行（top_k 6~10）
        - 禁止自动扩写 query，只能使用 search_spec.query_seeds
        """
        # 获取 mode_id
        mode = state.get("current_mode")
        mode_id = None
        if isinstance(mode, dict):
            mode_id = mode.get("id")
        elif mode:
            mode_id = getattr(mode, "id", None)
        mode_id = mode_id or "normal_mode"
        
        # mute_mode: 不检索（直接 no-op）
        if mode_id == "mute_mode":
            print("[Memory Retriever] mute_mode: 跳过检索")
            return {"retrieved_memories": []}
        
        def _norm(x: Any) -> str:
            return (str(x) if x is not None else "").strip()

        def _is_greeting(text: str) -> bool:
            s = _norm(text)
            if not s:
                return False
            return bool(re.match(r"^\s*(hi|hello|hey|你好|您好|嗨|哈喽|在吗|早上好|中午好|晚上好|晚安)\s*[!！。.]?\s*$", s, re.IGNORECASE))

        def _extract_seeds_from_text(text: str, max_seeds: int = 4) -> List[str]:
            """
            最低配 seeds 提取：只从已有外部文本切分，不做"语义扩写"。
            目的：即使没有 response_plan，也能基于用户话语/最近对话窗口做一次检索。
            """
            s = _norm(text)
            if not s:
                return []
            # 先按标点/空白切分，再筛选长度>=2 的片段（中文/英文都可）
            parts = re.split(r"[，。！？、；：\s,\.!?;:\n]+", s)
            out: List[str] = []
            for p in parts:
                p = _norm(p)
                if not p:
                    continue
                # 避免把整句过长直接塞进去
                if len(p) > 24:
                    p = p[:24]
                if len(p) < 2:
                    continue
                if p not in out:
                    out.append(p)
                if len(out) >= max_seeds:
                    break
            # 兜底：如果切不出，就取前 12 字
            if not out:
                out = [s[:12]]
            return out[:max_seeds]

        # 获取 response_plan 和 search_spec
        response_plan = state.get("response_plan")
        # P0：检索不应被 response_plan 绑死。
        # 没有 response_plan 时，仍然用 user_msg + stage + relationship_state + last_k_turns 做一次最低配检索。
        fallback_enabled = False
        fallback_query_seeds: List[str] = []
        if not response_plan:
            latest_user_text = _norm(state.get("external_user_text") or state.get("user_input") or "")
            stage = _norm(state.get("current_stage") or "")
            rel = state.get("relationship_state") or {}
            # last_k_turns：从 chat_buffer 取最近几条外部文本窗口
            chat_buf = state.get("chat_buffer") or []
            last_window: List[str] = []
            try:
                for m in list(chat_buf)[-6:]:
                    t = _norm(getattr(m, "content", str(m)))
                    if t:
                        last_window.append(t)
            except Exception:
                last_window = []

            # greeting 时也允许检索，但 seeds 必须更克制，避免"问候就召回一堆无关长记忆"
            seeds = []
            if latest_user_text:
                seeds.extend(_extract_seeds_from_text(latest_user_text, max_seeds=3 if _is_greeting(latest_user_text) else 4))
            if stage:
                seeds.append(stage)
            # 关系状态只取粗粒度标签（不把数值扩写成解释）
            try:
                closeness = float((rel or {}).get("closeness", 0.0) or 0.0)
                trust = float((rel or {}).get("trust", 0.0) or 0.0)
                seeds.append("low_trust" if trust < 0.35 else "trust")
                seeds.append("low_closeness" if closeness < 0.35 else "closeness")
            except Exception:
                pass
            # 追加最近对话窗口的少量片段（仍属于"已有文本切分"，非自动扩写）
            if last_window:
                seeds.extend(_extract_seeds_from_text(" ".join(last_window[-2:]), max_seeds=2))

            # 去重+截断
            merged: List[str] = []
            for s in seeds:
                s = _norm(s)
                if not s:
                    continue
                if s not in merged:
                    merged.append(s)
                if len(merged) >= 8:
                    break
            fallback_query_seeds = merged
            fallback_enabled = bool(fallback_query_seeds)
            print(f"[Memory Retriever] 无 response_plan：启用最低配检索 fallback_seeds={fallback_query_seeds}")
        
        # 提取主 plan（weight 最高的）
        plans = []
        if isinstance(response_plan, dict):
            plans = response_plan.get("plans", [])
        elif isinstance(response_plan, list):
            plans = response_plan
        
        if not plans:
            # 没有 plans 时，也走 fallback（如果可用）
            if fallback_enabled:
                plans = []
                main_plan = {}
                search_spec = {"enabled": True, "query_seeds": fallback_query_seeds}
            else:
                print("[Memory Retriever] 无 plans，跳过检索")
                return {}
        
        if plans:
            main_plan = max(plans, key=lambda p: float(p.get("weight", 0) or 0))
            search_spec = main_plan.get("search_spec", {})
        else:
            # fallback 分支已在上面设置
            main_plan = {}
            search_spec = {"enabled": True, "query_seeds": fallback_query_seeds}
        
        if not isinstance(search_spec, dict):
            if fallback_enabled:
                search_spec = {"enabled": True, "query_seeds": fallback_query_seeds}
            else:
                print("[Memory Retriever] search_spec 不是 dict，跳过检索")
                return {}
        
        # 检查 search_spec.enabled
        enabled = search_spec.get("enabled", False)
        if isinstance(enabled, str):
            enabled = enabled.lower() in ("true", "1", "yes", "on")
        else:
            enabled = bool(enabled)
        
        if not enabled:
            if fallback_enabled:
                enabled = True
            else:
                print("[Memory Retriever] search_spec.enabled=False，跳过检索")
                return {"retrieved_memories": []}
        
        # 获取 query_seeds（禁止自动扩写，只能使用 search_spec.query_seeds）
        query_seeds = search_spec.get("query_seeds", [])
        if not isinstance(query_seeds, list):
            query_seeds = []
        query_seeds = [str(x).strip() for x in query_seeds if str(x).strip()]
        if (not query_seeds) and fallback_enabled:
            query_seeds = list(fallback_query_seeds)

        # 反思 patch -> 检索增强：把上一轮 LATS 的 search_patch 真正接入 retriever（跨轮生效）
        # 注意：仍然遵守"禁止自动扩写"，这里只接受结构化 patch 明确给出的 seeds/entities
        try:
            tree = state.get("lats_tree") or {}
            ap = tree.get("active_patch") if isinstance(tree, dict) else None
            sp = ap.get("search_patch") if isinstance(ap, dict) else None
            if isinstance(sp, dict) and sp:
                add = sp.get("add_query_seeds", [])
                rm = sp.get("remove_query_seeds", [])
                ents = sp.get("strengthen_entities", [])
                add_list = [str(x).strip() for x in (add or []) if str(x).strip()] if isinstance(add, list) else []
                ent_list = [str(x).strip() for x in (ents or []) if str(x).strip()] if isinstance(ents, list) else []
                rm_set = {str(x).strip() for x in (rm or []) if str(x).strip()} if isinstance(rm, list) else set()

                merged = [q for q in query_seeds if q and q not in rm_set]
                for x in add_list + ent_list:
                    if x and x not in merged and x not in rm_set:
                        merged.append(x)
                query_seeds = merged
        except Exception:
            pass
        
        if not query_seeds:
            print("[Memory Retriever] query_seeds 为空，跳过检索")
            return {"retrieved_memories": []}
        
        # 根据 mode 确定 top_k
        if mode_id == "cold_mode":
            top_k = 3  # cold_mode: top_k 降到 2~3
        elif mode_id == "normal_mode":
            top_k = 8  # normal_mode: top_k 6~10（取中间值 8）
        else:
            top_k = 6  # 默认值
        
        # 构建 query（只使用 query_seeds，不自动扩写）
        query = " ".join(query_seeds).strip()
        
        print(f"[Memory Retriever] mode={mode_id}, enabled={enabled}, query_seeds={query_seeds}, top_k={top_k}")
        
        # 执行检索
        user_id = state.get("user_id") or "default_user"
        bot_id = state.get("bot_id") or (state.get("bot_basic_info") or {}).get("name") or "default_bot"
        relationship_id = state.get("relationship_id")
        
        retrieved: List[str] = []
        had_error = False
        
        try:
            # 尝试使用 DB
            if os.getenv("DATABASE_URL"):
                from app.core.database import DBManager
                db = DBManager.from_env()
                if db and relationship_id:
                    try:
                        notes = _run_async(db.search_notes(relationship_id=str(relationship_id), query=query, limit=top_k))
                        trans = _run_async(db.search_transcripts(relationship_id=str(relationship_id), query=query, limit=top_k))
                        merged_items = list(notes) + list(trans)
                        seen: set[str] = set()
                        for it in merged_items:
                            if it.get("store") == "B":
                                line = f"[B/{it.get('note_type') or 'note'}] {it.get('content') or ''} (src={it.get('source_pointer') or ''})"
                            else:
                                ctx = it.get("short_context") or it.get("topic") or ""
                                u = str(it.get("user_text") or "")[:60]
                                b = str(it.get("bot_text") or "")[:60]
                                line = f"[A/{it.get('created_at')}] {ctx} U:{u} B:{b} (id=transcript:{it.get('id')})"
                            line = line.strip()
                            if not line or line in seen:
                                continue
                            seen.add(line)
                            retrieved.append(line)
                            if len(retrieved) >= top_k:
                                break
                    except Exception as e:
                        print(f"[Memory Retriever] DB 检索失败: {e}")
                        had_error = True
            
            # 如果 DB 失败或未配置，使用 LocalStore
            if not retrieved:
                from app.core.local_store import LocalStoreManager
                store = LocalStoreManager()
                notes = store.search_notes(str(user_id), str(bot_id), query, limit=top_k)
                trans = store.search_transcripts(str(user_id), str(bot_id), query, limit=top_k)
                merged_items = list(notes) + list(trans)
                seen: set[str] = set()
                for it in merged_items:
                    if it.get("store") == "B":
                        line = f"[B/{it.get('note_type') or 'note'}] {it.get('content') or ''} (src={it.get('source_pointer') or ''})"
                    else:
                        ctx = it.get("short_context") or it.get("topic") or ""
                        u = str(it.get("user_text") or "")[:60]
                        b = str(it.get("bot_text") or "")[:60]
                        line = f"[A/{it.get('created_at')}] {ctx} U:{u} B:{b} (id=transcript:{it.get('id')})"
                    line = line.strip()
                    if not line or line in seen:
                        continue
                    seen.add(line)
                    retrieved.append(line)
                    if len(retrieved) >= top_k:
                        break
        except Exception as e:
            print(f"[Memory Retriever] 检索失败: {e}")
            had_error = True
        
        print(f"[Memory Retriever] 检索完成: {len(retrieved)} 条记忆")

        # 最小策略：每轮覆盖写，避免 DB 检索失败时拼入旧 retrieved（制造噪声）
        # 若本轮发生错误（尤其是 event loop 问题），则视为 retrieval_ok=False，直接清空 retrieved。
        retrieval_ok = (not had_error) and bool(retrieved)
        return {
            "retrieved_memories": (retrieved[:top_k] if retrieval_ok else []),
            "retrieval_ok": bool(retrieval_ok),
        }
    
    return memory_retriever_node
