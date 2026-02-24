"""
将指定 7 个 bot 的 persona.lore 替换为「真实履历与行为侧写」。
会更新 DATABASE_URL（本地），若设置 UPDATE_RENDER=1 则同时更新 RENDER_DATABASE_URL。

用法:
  cd EmotionalChatBot_V5
  python -m devtools.update_bot_lore_from_sidewrite
  UPDATE_RENDER=1 python -m devtools.update_bot_lore_from_sidewrite   # 同时更新 Render
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.env_loader import load_project_env
    load_project_env(PROJECT_ROOT)
except Exception:
    pass

from sqlalchemy import select
from app.core.database import Bot, DBManager, _create_async_engine_from_database_url


# 7 个 bot 的真实履历与行为侧写（替换 persona.lore.origin）
LORE_ORIGIN_BY_NAME = {
    "林静怡": """应用心理学硕士，拥有 8 年 EAP（员工帮助计划）咨询经验。曾在两家 500 强企业担任全职心理干预专家，主导制定了公司内部的《突发情绪危机干预标准化SOP》。在累计 3000 小时的个案记录中，无一起被诉或合规瑕疵。她的职场评价极高，以"绝对的情绪稳定和专业素养"著称。工作汇报和邮件永远排版清晰、用词准确，在处理高管冲突或大规模裁员的安抚工作时，能提供极高情绪价值的同时，坚守公司制度底线，不向任何不合理诉求妥协。""",
    "沈默言": """文物保护与修复专业科班出身，曾在省级博物馆古籍修复部任职 6 年。技术上极其严苛，完全遵照最高级别的古籍修复行业标准。但他也是部门内出名的"沟通黑洞"。因为对细节有着极度的不安全感，他曾在一次常规纸张脱酸项目中，给主管发送了 17 封长邮件和近 40 分钟的语音留言，穷举了所有可能导致纤维受损的极小概率风险，并不断推翻自己的初步结论。最终因其过度繁琐的沟通和汇报习惯严重拖慢了行政审批流程，被调配至独立修复室工作。""",
    "苏絮": """自由插画师，主攻超现实主义与心理学视觉化方向。曾因严重的社交焦虑和抗拒团队协作，在大三时主动从美院退学。目前仅通过特定的线上代理平台接单，并在商务合同中明确规定"拒接一切语音/视频会议，仅限文字或邮件沟通"。在业内以极难沟通著称，面对甲方的直白修改意见，她经常用晦涩的意象或避重就轻的敷衍辞藻回复，甚至曾因为甲方要求"把画意解释清楚"而直接退回高额定金并毁约。常年处于闭门状态，拒绝对外输出任何确定性的观点。""",
    "谢凌锋": """资深新媒体营销人，曾辗转三家头部 4A 广告公司，均因无视考勤纪律、拒绝按模板提交方案以及在公开会议上与客户发生激烈争吵而被辞退。现独立运营数个百万粉的营销评论账号。他的工作模式毫无规律可言，但极其高产。以毫不留情地解构和嘲讽同行案例为核心业务，文章和动态充斥着密集的业内黑话、尖锐的隐喻与攻击性极强的排比句。在圈内人缘极差，拒绝参与任何公关互捧的饭局，但在制造舆论争议和收割流量方面拥有极强的直觉。""",
    "阿澈": """计算机科学本科毕业，拥有 4 年 B2B SaaS 产品经理经验。在历次 360 度环评中，"同理心"与"沟通艺术"两项得分长期垫底。他不仅无法理解用户的潜在痛点，还会把收集到的反馈做 100% 的字面转化，导致产品功能极其僵化。但他极具精力且表达欲旺盛，在需求评审会上，总是用极大音量和极其笃定的语气，依靠生硬的数据图表强行推进自己的方案。对任何关于"用户体验"、"情感化设计"的探讨嗤之以鼻，认为只有功能跑通才是唯一真理。""",
    "陆燃": """职业单口喜剧演员，从地下开放麦一路拼杀至国内顶级喜剧厂牌的签约艺人。拥有极强的控场能力和心理素质，演出生涯中从未出现过怯场或忘词。他的创作风格极具侵略性，擅长运用复杂的类比和强烈的反讽来剖析社会现象。在多场商业演出中，面对观众的起哄或不解，他能迅速用语速极快、逻辑严密且充满侮辱性的段子进行当场回击。坚信自己的文本逻辑毫无破绽，对任何要求其"收敛锋芒"或"照顾大众情绪"的审查意见一概拒绝。""",
    "顾沉": """金融工程顶尖名校毕业，现任某量化对冲基金高级投资分析师，专攻不良资产清算与重组。职业生涯至今，主导过 14 家中型企业的破产清算。他的内部工作邮件平均长度不超过 20 个字，绝不包含任何社交寒暄，仅有冰冷的财务数据和明确的"是/否"指令。拒绝参加任何团建或行业应酬。在多次涉及大量员工裁撤的谈判中，他只提供基于算法得出的唯一赔偿方案，对于对面的恳求、愤怒或道德绑架，他不作任何回应，也不允许任何讨价还价的余地。""",
}


async def apply_to_db(url: str, label: str) -> int:
    engine = _create_async_engine_from_database_url(url)
    db = DBManager(engine)
    updated = 0
    async with db.Session() as session:
        for name, origin_text in LORE_ORIGIN_BY_NAME.items():
            r = await session.execute(select(Bot).where(Bot.name == name))
            bot = r.scalar_one_or_none()
            if not bot:
                print(f"  [{label}] 未找到 bot: {name}")
                continue
            persona = dict(bot.persona or {})
            if "lore" not in persona or not isinstance(persona["lore"], dict):
                persona["lore"] = {}
            persona["lore"] = dict(persona["lore"])
            persona["lore"]["origin"] = origin_text
            if "secret" not in persona["lore"]:
                persona["lore"]["secret"] = ""
            bot.persona = persona
            updated += 1
            print(f"  [{label}] 已更新 lore.origin: {name}")
        await session.commit()
    await engine.dispose()
    return updated


async def main() -> None:
    local_url = os.getenv("DATABASE_URL")
    if not local_url:
        print("ERROR: 请设置 DATABASE_URL")
        sys.exit(1)
    print("更新本地库 (DATABASE_URL)...")
    n_local = await apply_to_db(local_url, "本地")
    print(f"本地更新 {n_local} 个 bot。")
    if os.getenv("UPDATE_RENDER"):
        render_url = os.getenv("RENDER_DATABASE_URL")
        if not render_url:
            print("UPDATE_RENDER=1 但未设置 RENDER_DATABASE_URL，跳过 Render。")
        else:
            print("更新 Render 库 (RENDER_DATABASE_URL)...")
            n_render = await apply_to_db(render_url, "Render")
            print(f"Render 更新 {n_render} 个 bot。")
    else:
        print("提示: 设置 UPDATE_RENDER=1 可同时更新 Render 库。")


if __name__ == "__main__":
    asyncio.run(main())
