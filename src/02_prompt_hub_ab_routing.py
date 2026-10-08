"""
Bước 2 — Prompt Hub & A/B Routing
===================================
NHIỆM VỤ:
  1. Viết 2 system prompt khác nhau (V1: ngắn gọn, V2: có cấu trúc)
  2. Push cả 2 lên LangSmith Prompt Hub qua client.push_prompt()
  3. Pull lại từ Hub qua client.pull_prompt()
  4. Implement A/B routing tất định: hash(request_id) % 2 → V1 hoặc V2
  5. Chạy 50 câu hỏi qua router → ≥ 50 LangSmith traces nữa

DELIVERABLE: 2 prompt version hiển thị trong Prompt Hub trên https://smith.langchain.com
"""
import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Tên Prompt trên Hub ─────────────────────────────────────────────────
PROMPT_V1_NAME = "tran-xuan-duc-rag-prompt-v1"
PROMPT_V2_NAME = "tran-xuan-duc-rag-prompt-v2"


# ── 2. Định nghĩa 2 Prompt Templates ──────────────────────────────────────
# V1 — ngắn gọn, bám sát context: ưu tiên độ trung thực (faithfulness)
SYSTEM_V1 = (
    "Bạn là trợ lý hỏi đáp kỹ thuật. Chỉ được dùng thông tin trong phần Context bên dưới, "
    "không thêm kiến thức bên ngoài. Trả lời trực tiếp vào câu hỏi trong 2-4 câu, "
    "giữ nguyên thuật ngữ và con số như trong Context. "
    "Nếu Context không chứa thông tin cần thiết, nói rõ là tài liệu không có thông tin. "
    "Trả lời bằng cùng ngôn ngữ với câu hỏi.\n\n"
    "Context:\n{context}"
)

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

# V2 — giọng chuyên gia, có cấu trúc: câu kết luận trước, sau đó giải thích chi tiết
SYSTEM_V2 = (
    "Bạn là chuyên gia AI/ML đang giải thích cho kỹ sư. Hãy đọc kỹ Context, "
    "xác định các sự kiện liên quan trực tiếp đến câu hỏi, rồi trả lời theo cấu trúc:\n"
    "1. Một câu kết luận trả lời thẳng câu hỏi.\n"
    "2. 2-4 câu giải thích: định nghĩa, cơ chế hoặc ví dụ lấy từ Context.\n"
    "Không suy đoán ngoài Context; nếu Context thiếu thông tin, nói rõ phần nào còn thiếu. "
    "Trả lời bằng cùng ngôn ngữ với câu hỏi.\n\n"
    "Context:\n{context}"
)

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])


# ── 3. Push Prompts lên Prompt Hub ─────────────────────────────────────────
def _push_prompt(client: Client, label: str, name: str, prompt, description: str):
    """Push 1 prompt; coi lỗi 409 "Nothing to commit" là bình thường (prompt chưa đổi)."""
    try:
        url = client.push_prompt(name, object=prompt, description=description)
        print(f"✅ Đã push {label} → {url}")
    except Exception as e:
        if "409" in str(e) and "Nothing to commit" in str(e):
            print(f"ℹ️  {label} không đổi so với commit mới nhất trên Hub — giữ nguyên phiên bản hiện tại")
        else:
            print(f"⚠️  {label} lỗi: {e}")


def push_prompts_to_hub(client: Client):
    """
    Upload cả 2 prompt templates lên LangSmith Prompt Hub.
    Hub tự tạo commit (phiên bản) mới mỗi khi nội dung prompt thay đổi.
    """
    _push_prompt(client, "V1", PROMPT_V1_NAME, PROMPT_V1, "V1 – ngắn gọn, bám sát context")
    _push_prompt(client, "V2", PROMPT_V2_NAME, PROMPT_V2, "V2 – chuyên gia, có cấu trúc")


# ── 4. Pull Prompts từ Prompt Hub ──────────────────────────────────────────
def pull_prompts_from_hub(client: Client) -> dict:
    """
    Tải 2 prompt từ LangSmith Prompt Hub.
    Fallback về template local nếu Hub không khả dụng.

    Gợi ý: client.pull_prompt(name) → ChatPromptTemplate

    Trả về: {name: ChatPromptTemplate}
    """
    prompts = {}

    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"↓ Đã pull '{PROMPT_V1_NAME}' từ Hub")
    except Exception as e:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"ℹ️  Dùng local fallback cho '{PROMPT_V1_NAME}' ({e})")

    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"↓ Đã pull '{PROMPT_V2_NAME}' từ Hub")
    except Exception as e:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"ℹ️  Dùng local fallback cho '{PROMPT_V2_NAME}' ({e})")

    return prompts


# ── 5. A/B Routing tất định ────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Xác định prompt version dựa trên MD5 hash của request_id.

    Quy tắc: hash chẵn → PROMPT_V1_NAME | hash lẻ → PROMPT_V2_NAME
    TÍNH CHẤT: cùng request_id LUÔN cho cùng kết quả (deterministic).

    Gợi ý:
        hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME
    """
    # Dùng MD5 thay vì hash() built-in: hash() của str bị random hóa giữa các tiến trình
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Traced A/B Query ────────────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Chạy RAG chain với prompt version được chọn bởi router.

    Bước:
      a) Retrieve top-3 docs từ retriever
      b) Ghép page_content thành context string
      c) Chạy (prompt | llm | StrOutputParser()).invoke({"context": ..., "question": ...})
      d) Trả về {"question": ..., "answer": ..., "version": ...}
    """
    docs    = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    answer  = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
    return {"question": question, "answer": answer, "version": version}


# ── 7. Setup Vectorstore (tái sử dụng logic Bước 1) ───────────────────────
def setup_vectorstore():
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 8. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    client = Client(api_key=config.LANGSMITH_API_KEY)

    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client)

    # Tạo vectorstore, retriever và LLM
    vectorstore = setup_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm         = get_llm()

    # Chạy A/B routing cho tất cả câu hỏi
    v1_count, v2_count = 0, 0
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id  = f"req-{i:04d}"

        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]

        result = ask_ab(retriever, llm, prompt, question, version_tag)

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1
        print(f"[{i+1:02d}] {request_id} [prompt-{version_tag}] {question[:55]}...")
        print(f"       A: {' '.join(result['answer'].split())[:100]}")

    print(f"\n📊 Routing: V1={v1_count} câu | V2={v2_count} câu | Tổng={len(SAMPLE_QUESTIONS)}")

    # Kiểm tra tính tất định: gọi lại router nhiều lần với cùng request_id
    sample_ids = [f"req-{i:04d}" for i in range(5)]
    deterministic = all(
        len({get_prompt_version(rid) for _ in range(10)}) == 1 for rid in sample_ids
    )
    print(f"🔁 Kiểm tra tất định (10 lần/req, {len(sample_ids)} request_id): "
          f"{'OK — cùng request_id luôn ra cùng version' if deterministic else 'FAIL'}")
    print("✅ Bước 2 hoàn thành! Kiểm tra Prompt Hub và traces trên LangSmith.")


if __name__ == "__main__":
    main()
