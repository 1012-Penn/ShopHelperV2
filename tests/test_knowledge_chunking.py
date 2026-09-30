from app.services.knowledge.chunking import split_markdown
from app.services.knowledge.content import build_embedding_text


def test_nested_heading_path_and_parent_category_are_preserved():
    chunks = split_markdown("returns.md", "# 退换货\n## 退货条件\n### 商品完好\n保持附件齐全。")
    assert len(chunks) == 1
    assert chunks[0].chapter_path == ["退换货", "退货条件", "商品完好"]
    assert chunks[0].questions == ["商品完好"]
    assert chunks[0].category == "退换货 / 退货条件"


def test_root_heading_uses_explicit_ecommerce_category():
    chunk = split_markdown("root.md", "# 配送\n费用以结算页显示为准。")[0]
    assert chunk.category == "电商客服"
    assert chunk.chapter_path == ["配送"]


def test_product_faq_question_heading_is_preserved_as_real_question():
    chunks = split_markdown("product.md", "# 商品 FAQ\n## 如何确认商品尺寸？\n请查看商品详情页的尺寸说明。")
    assert chunks[0].questions == ["如何确认商品尺寸？"]
    assert chunks[0].category == "商品 FAQ"


def test_long_paragraph_recursively_splits_and_uses_complete_sentence_overlap():
    source = "第一句说明配送范围。第二句说明计算方式。第三句说明特殊地区。第四句说明查询入口。"
    chunks = split_markdown("delivery.md", "# 配送\n" + source, max_chars=22, overlap_chars=10)
    assert len(chunks) > 1
    assert "第一句说明配送范围。" in chunks[0].answer
    for previous, current in zip(chunks, chunks[1:]):
        previous_sentences = set(previous.answer.split())
        current_sentences = set(current.answer.split())
        assert previous_sentences & current_sentences
        assert previous.answer.rstrip()[-1] in "。！？.!?；;"


def test_oversize_sentence_is_kept_whole_and_table_headers_repeat():
    long_sentence = "运费规则" + "非常重要" * 400 + "。"
    markdown = "# 配送\n\n" + long_sentence + "\n\n|区间|规则|\n|---|---|\n|A|甲|\n|B|乙|"
    chunks = split_markdown("shipping.md", markdown, max_chars=20, overlap_chars=5)
    assert any(long_sentence in chunk.answer for chunk in chunks)
    table_chunks = [chunk.answer for chunk in chunks if "|区间|规则|" in chunk.answer]
    assert len(table_chunks) == 2
    assert all("|---|---|" in chunk for chunk in table_chunks)


def test_chinese_and_english_sentence_boundaries_are_supported():
    text = "中文句子结束。English sentence ends! Another sentence? 最后一句。"
    chunks = split_markdown("mixed.md", "# 测试\n" + text, max_chars=25, overlap_chars=0)
    joined = "".join(chunk.answer for chunk in chunks)
    for sentence in ("中文句子结束。", "English sentence ends!", "Another sentence?", "最后一句。"):
        assert sentence in joined


def test_fenced_code_blocks_remain_balanced_and_source_keys_are_deterministic():
    markdown = "# 操作\n\n```text\n第一行\n第二行\n```\n\n确认页面状态。"
    first = split_markdown("guide.md", markdown, max_chars=16, overlap_chars=0)
    second = split_markdown("guide.md", markdown, max_chars=16, overlap_chars=0)
    assert [item.source_key for item in first] == [item.source_key for item in second]
    for chunk in first:
        assert chunk.answer.count("```") % 2 == 0


def test_embedding_text_contains_only_category_questions_and_answer():
    text = build_embedding_text("配送", ["邮费是多少", "如何查询运费"], "以结算页为准。")
    assert text == "配送\n邮费是多少\n如何查询运费\n以结算页为准。"
    assert "chapter_path" not in text and "policy" not in text
