"""Allowlisted source rendering and section anchors shared with citation URLs."""
import hashlib
import html
import re
from urllib.parse import quote
from pathlib import Path


def section_anchor(section_path):
    return 'section-'+hashlib.sha256('/'.join(section_path).encode()).hexdigest()[:16]


def source_url(source_key,section_path):
    if not source_key.startswith('doc:'):return None
    name=source_key[4:].rsplit(':',1)[0]
    return '/api/v1/knowledge/source?source='+quote(name,safe='')+'#'+section_anchor(section_path)


class SourceDocuments:
    def __init__(self,root,registered):
        self.root=Path(root).resolve()
        self.registered=set(registered)

    def render(self,name):
        path=(self.root/name).resolve()
        if name not in self.registered or path.parent!=self.root or not path.is_file():
            raise FileNotFoundError('source unavailable')
        parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+html.escape(name)+'</title><style>body{max-width:900px;margin:40px auto;padding:24px;font:16px/1.8 system-ui}p{white-space:pre-wrap}h1,h2,h3,h4{scroll-margin-top:24px}:target{background:#fff0bd}</style><body>']
        stack=[]
        for line in path.read_text().splitlines():
            match=re.match(r'^(#{1,6})\s+(.+)',line)
            if match:
                level=len(match[1]);title=match[2].strip()
                while stack and stack[-1][0]>=level:stack.pop()
                stack.append((level,title))
                anchor=section_anchor([t for _,t in stack])
                parts.append(f'<h{level} id="{anchor}">'+html.escape(title)+f'</h{level}>')
            elif line.strip():parts.append('<p>'+html.escape(line)+'</p>')
        return ''.join(parts)+'</body></html>'
