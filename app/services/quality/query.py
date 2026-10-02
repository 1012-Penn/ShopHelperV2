"""Single-turn normalization preserving identifiers and negative conditions."""
import re
from functools import lru_cache
from dataclasses import dataclass
from pydantic import BaseModel


_IDENTIFIER_RE = re.compile(r'[A-Za-z][A-Za-z0-9_.-]*')
_CHINESE_NUMBER = r'[零〇一二三四五六七八九十百千万亿两]+(?:点[零〇一二三四五六七八九]+)?'
_FUZZY_NUMBER = r'(?:若干|数(?:十|百|千)?|十几|几十|几(?:十|百|千)?)'
_NUMBER_ATOM = rf'(?:\d+(?:\.\d+)?|{_FUZZY_NUMBER}|{_CHINESE_NUMBER})'
_QUANTITY_RE = re.compile(
    rf'(?P<currency>[$¥￥])?\s*(?P<sign>[+\-−＋－]|负)?\s*(?P<number>{_NUMBER_ATOM})\s*'
    r'(?P<unit>千瓦时|毫安时|平方公里|平方米|立方米|平方厘米|摄氏度|毫升|公斤|千克|毫克|毫米|厘米|公里|千瓦|毫安|星期|个月|小时|分钟|块钱|天|日|周|月|年|元|块|钱|件|次|个|折|秒|米|升|斤|吨|克|瓦|伏|安|度|℃|℉|倍|人|张|岁|台|款|[%％])?'
)
_MARKED_CHINESE_NUMBER_RE = re.compile(
    rf'(?:选项|金额|价格|余额|售价|总价|折扣|数量|数字|活动号|编号|第)\s*'
    rf'(?:[+\-−＋－]|负)?(?:{_FUZZY_NUMBER}|{_CHINESE_NUMBER})'
    rf'(?:(?:还是|或者|或|和|、)\s*(?:[+\-−＋－]|负)?(?:{_FUZZY_NUMBER}|{_CHINESE_NUMBER}))*'
)
_LEXICAL_ONE_RE = re.compile(r'一点(?![零〇一二三四五六七八九十百千万亿两])|一起|一样|同一|一般|一定|一直|一边|一些|一种|一旦|一下|一向|一共|一律')
_CLAUSE_SPLIT_RE = re.compile(r'[,，;；。！？!?]+|但是|然而|但')
_NEGATIVE_RE = re.compile(r'不|无|未|没|非(?!常)|(?<!分)别|(?<!是)(?<!能)否')


def _ordered_unique(values):
    return tuple(dict.fromkeys(values))


def _identifier_signature(text):
    return _ordered_unique(_IDENTIFIER_RE.findall(text))


def _quantity_signature(text):
    """Keep numeric occurrences and their units separate from model IDs."""
    identifier_spans = [match.span() for match in _IDENTIFIER_RE.finditer(text)]
    quantities = []
    for match in _QUANTITY_RE.finditer(text):
        number_start, number_end = match.span('number')
        if any(start <= number_start and number_end <= end for start, end in identifier_spans):
            continue
        number = match.group('number')
        unit = match.group('unit') or ''
        currency = match.group('currency') or ''
        sign = match.group('sign') or ''
        # A bare Chinese numeral is usually part of ordinary wording. Treat it
        # as a quantity only when a unit makes that reading explicit.
        if not unit and not currency and not re.search(r'\d', number) and '点' not in number:
            continue
        quantities.append(f'{currency}{sign}{number}{unit}')
    quantities.extend(match.group() for match in _MARKED_CHINESE_NUMBER_RE.finditer(text))
    return tuple(quantities)


def _unknown_quantity_clauses(text):
    """Lock clauses where a number is followed by an unrecognized Chinese unit."""
    identifier_spans = [match.span() for match in _IDENTIFIER_RE.finditer(text)]
    marked_spans = [match.span() for match in _MARKED_CHINESE_NUMBER_RE.finditer(text)]
    lexical_spans = [match.span() for match in _LEXICAL_ONE_RE.finditer(text)]
    locked = []
    offset = 0
    for clause in _CLAUSE_SPLIT_RE.split(text):
        clause_start = text.find(clause, offset) if clause else offset
        if clause_start < 0:
            clause_start = offset
        clause_end = clause_start + len(clause)
        offset = clause_end
        if not clause:
            continue
        for match in _QUANTITY_RE.finditer(clause):
            if match.group('unit'):
                continue
            number_start, number_end = match.span('number')
            start, end = clause_start + number_start, clause_start + number_end
            if any(left <= start and end <= right for left, right in identifier_spans):
                continue
            if any(left <= start and end <= right for left, right in marked_spans):
                continue
            if any(left <= start and end <= right for left, right in lexical_spans):
                continue
            following = clause[number_end:].lstrip()
            if following and '\u4e00' <= following[0] <= '\u9fff':
                locked.append(clause)
                break
    return tuple(locked)


def _negative_clauses(text):
    """Return literal clauses containing an actual negation marker.

    Chinese commas and sentence punctuation bound a clause. Contrastive
    connectors also split clauses so an adjacent positive clause can change
    while each negative clause remains byte-for-byte intact.
    """
    return tuple(
        clause for clause in _CLAUSE_SPLIT_RE.split(text)
        if clause and _NEGATIVE_RE.search(clause)
    )


@dataclass(frozen=True)
class QueryUnderstanding:
    raw: str
    canonical: str
    lexical: str
    downgrade_reason: str = ""


class NormalizedQuery(BaseModel):
    canonical: str


class QueryNormalizer:
    def __init__(self, rewrite=None):
        self.rewrite=rewrite

    @lru_cache(maxsize=1024)
    def normalize(self, raw):
        canonical=raw.strip()
        reason=""
        if self.rewrite:
            try:
                candidate=self.rewrite(raw)
                candidate=candidate.strip()
                identifiers_match=_identifier_signature(raw)==_identifier_signature(candidate)
                quantities_match=_quantity_signature(raw)==_quantity_signature(candidate)
                unknown_quantities_match=_unknown_quantity_clauses(raw)==_unknown_quantity_clauses(candidate)
                negative_clauses_match=_negative_clauses(raw)==_negative_clauses(candidate)
                if candidate and identifiers_match and quantities_match and unknown_quantities_match and negative_clauses_match:
                    canonical=candidate.strip()
                else:
                    reason='改写未完整保留标识、数值或否定范围，使用原话'
            except Exception:
                reason='改写服务或输出不可用，使用原话'
        expansions=[]
        for words in [('邮费','运费'),('退钱','退款'),('退东西','退货'),('不开机','无法开机'),('多久到','配送时效')]:
            if any(word in raw or word in canonical for word in words):
                expansions.extend(words)
        return QueryUnderstanding(raw,canonical,' '.join(dict.fromkeys([raw,canonical,*expansions])),reason)

    @classmethod
    def from_model(cls, model):
        structured=model.with_structured_output(NormalizedQuery,method='json_mode')
        def rewrite(raw):
            result=structured.invoke([{'role':'system','content':(
                '将本轮口语客服问题归一为清晰、简短的标准问法，只处理本轮，不做指代消解或多轮推断。'
                '逐项保留原问题中的每个型号及后缀（如 SE、Pro）、活动编号、金额和所有数字/数量及其单位、正负号与百分号、适用条件与范围。'
                '保留用户已经提供的状态和时间背景（如已经购买/使用、已经使用几个月），保留模糊数量和模糊时长（如几个月、若干天、数年）。'
                '比较问题必须保留每个不同的比较型号；可以合并重复提到的同一个型号，但不能删掉不同型号，也不能新增型号。'
                '保持原问题的目标、比较对象和比较关系；不得引入用户未问的计算规则或政策维度。'
                '否定条件和否定作用范围必须保留；任何含否定条件的子句必须逐字保持原文，不得移动、交换、弱化或新增否定。'
                '只在信息明确且上述内容均保真时归一；无法确认等价就原样返回用户问题。'
                '输出JSON：{"canonical":"标准问法"}。'
            )}, {'role':'user','content':raw}])
            return result.canonical
        return cls(rewrite)
